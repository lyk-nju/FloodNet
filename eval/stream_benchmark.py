"""Unified stream benchmark runner (Task 002).

Usage::

    python eval/stream_benchmark.py \\
        --config configs/stream.yaml \\
        --ckpt outputs/step_460000.ckpt \\
        --vae_ckpt outputs/vae_1d_z4_step=300000.ckpt \\
        --raw_data_dir /path/to/raw_data \\
        --preset smoke \\
        --render_video
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
from lightning import seed_everything
from torch_ema import ExponentialMovingAverage
from omegaconf import OmegaConf

from utils.initialize import check_state_dict, instantiate, load_config
from utils.motion_process import StreamJointRecovery263, extract_root_trajectory_263
from utils.stream_rollout import (
    StreamTextSegment, StreamTextRolloutController,
    build_stream_step_model_input,
    build_stream_suffix_conditioning,
)
from utils.stream_traj import (
    StreamTrajectoryPlan,
    assign_uniform_timestamps,
    resample_polyline_by_arclength,
    sample_plan_future,
    sample_timestamped_trajectory,
    smoothstep01,
)
from utils.visualize import render_single_video
from eval.stream_benchmarks import get_cases
from eval.stream_metrics import build_plan_metrics

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── model loading ──────────────────────────────────────────────────────

def _load_vae(cfg, device):
    vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False,
                      **cfg.test_vae.params)
    ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in ckpt:
        vae.load_state_dict(ckpt["state_dict"], strict=True)
        ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(vae.parameters())
    else:
        vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _load_model(cfg, ckpt_path, device):
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False,
                        **cfg.model.params)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(ckpt["state_dict"].keys())
    cn_missing = not any(k.startswith("controlnet.") for k in ckpt_keys)
    strict = not cn_missing
    result = model.load_state_dict(ckpt["state_dict"], strict=strict)
    if not strict and result.missing_keys:
        if any("controlnet." in k for k in result.missing_keys):
            model.controlnet.init_from_backbone(model.model)
    if "ema_state" in ckpt:
        n_shadow = len(ckpt["ema_state"]["shadow_params"])
        ema_params = [p for p in model.parameters() if p.requires_grad]
        if len(ema_params) != n_shadow:
            ema_params = list(model.parameters())
        ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(ema_params)
    model.to(device).eval()
    return model


# ── sample loading ─────────────────────────────────────────────────────

def _load_humanml3d_sample(raw_data_dir, sample_id):
    data_dir = os.path.join(raw_data_dir, "HumanML3D")
    feat = np.load(os.path.join(data_dir, "new_joint_vecs", f"{sample_id}.npy")).astype(np.float32)
    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            text_data.append({
                "caption": parts[0],
                "tokens": parts[1].split(" ") if len(parts) > 1 else [],
                "f_tag": float(parts[2]) if len(parts) > 2 else 0.0,
                "to_tag": float(parts[3]) if len(parts) > 3 else 0.0,
            })
    traj_xyz = extract_root_trajectory_263(feat)
    token = np.load(os.path.join(
        data_dir, "TOKENS_20251030_085836_vae_wan_z4", f"{sample_id}.npy")).astype(np.float32)
    return {
        "name": sample_id, "dataset": "HumanML3D",
        "feature": torch.from_numpy(feat).float(), "feature_length": len(feat),
        "token": torch.from_numpy(token).float(), "token_length": len(token),
        "text": text_data[0]["caption"],
        "traj": torch.from_numpy(traj_xyz).float(), "traj_length": len(traj_xyz),
        "token_mask": torch.ones(len(token), dtype=torch.float32),
        "traj_mask": torch.ones(len(traj_xyz), dtype=torch.float32),
    }


def _load_babel_sample(raw_data_dir, sample_id):
    data_dir = os.path.join(raw_data_dir, "BABEL_streamed")
    feat = np.load(os.path.join(data_dir, "motions", f"{sample_id}.npy")).astype(np.float32)
    token = np.load(os.path.join(data_dir, "TOKENS_20251030_085836_vae_wan_z4",
                                 f"{sample_id}.npy")).astype(np.float32)
    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            ft = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
            tt = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0.0
            text_data.append({"caption": parts[0].strip(),
                              "f_tag": 0.0 if np.isnan(ft) else ft,
                              "to_tag": 0.0 if np.isnan(tt) else tt})
    return {"feature": feat, "token": token, "text_data": text_data, "name": sample_id}


def _merge_babel(raw_data_dir, sample_ids):
    parts = [_load_babel_sample(raw_data_dir, sid) for sid in sample_ids]
    feat = np.concatenate([p["feature"] for p in parts], axis=0)
    token = np.concatenate([p["token"] for p in parts], axis=0)
    tf, tt = len(feat), len(token)
    text_data, feat_ofs = [], 0
    feat_fps = 20.0
    for p in parts:
        for td in p["text_data"]:
            ft, ttag = td["f_tag"], td["to_tag"]
            if ft == 0.0 and ttag == 0.0:
                af, at = feat_ofs / feat_fps, (feat_ofs + len(p["feature"])) / feat_fps
            else:
                af, at = feat_ofs / feat_fps + ft, feat_ofs / feat_fps + ttag
            text_data.append({"caption": td["caption"], "f_tag": af, "to_tag": at})
        feat_ofs += len(p["feature"])
    texts, fte, cursor = [], [], 0
    for td in text_data:
        a_start = max(0, int(td["f_tag"] * feat_fps + 0.5))
        a_end = int(td["to_tag"] * feat_fps + 0.5) if td["to_tag"] > 0 else tf
        if a_end <= a_start:
            continue
        if a_start > cursor:
            texts.append(""); fte.append(min(a_start, tf)); cursor = a_start
        texts.append(td["caption"]); fte.append(min(a_end, tf)); cursor = a_end
    if cursor < tf:
        texts.append(""); fte.append(tf)
    if not texts:
        texts = [td["caption"] or "" for td in text_data] or [""]; fte = [tf]
    token_te = [max(0, min(tt, (ef - 1 + 3) // 4 + 1)) for ef in fte]
    traj = extract_root_trajectory_263(feat)
    return {
        "name": sample_ids[0].rsplit("_", 1)[0], "dataset": "BABEL_streamed",
        "feature": torch.from_numpy(feat).float(), "feature_length": tf,
        "token": torch.from_numpy(token).float(), "token_length": tt,
        "text": texts, "traj": torch.from_numpy(traj).float(), "traj_length": len(traj),
        "token_text_end": token_te, "feature_text_end": fte,
        "token_mask": torch.ones(tt, dtype=torch.float32),
        "traj_mask": torch.ones(len(traj), dtype=torch.float32),
    }


# ── case runners ───────────────────────────────────────────────────────

def _run_step(model, vae, sample, device, *, hl, nds, mode):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    _bs = {  # batch-style wrapper for build_stream_suffix_conditioning
        "traj": sample["traj"].unsqueeze(0),
        "token_length": torch.tensor([tl]),
        "traj_length": torch.tensor([sample["traj_length"]]),
        "token_mask": sample["token_mask"].unsqueeze(0),
        "traj_mask": sample["traj_mask"].unsqueeze(0),
    }
    sr = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, _roots = [], []
    fc = True
    for ci in range(tl):
        if mode == "step_no_traj":
            ti = None
        elif mode == "step_predroot":
            ti = build_stream_suffix_conditioning(_bs, ci, prefer_xyz=True)
            if ti is not None and len(_roots) > 0:
                pr_cur = _roots[-1]
                gt_ci = sample["traj"].numpy()[min(ci * 4, len(sample["traj"]) - 1)].astype(np.float32)
                offset = pr_cur.astype(np.float32) - gt_ci
                t = ti["traj"]
                if torch.is_tensor(t):
                    ti["traj"] = t + torch.from_numpy(offset).float().to(t)
        else:
            ti = build_stream_suffix_conditioning(_bs, ci, prefer_xyz=True)
        sp = build_stream_step_model_input(
            sample["text"] if isinstance(sample["text"], str) else sample["text"][0],
            traj_input=ti)
        out = model.stream_generate_step(sp, first_chunk=fc)
        dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                 first_chunk=fc)[0].float().cpu().numpy())
        fc = False
        for frm in dec:
            sr.process_frame(frm)
            _roots.append(sr.r_pos_accum.copy())
        decs.append(dec)
    vae.clear_cache()
    pm = np.concatenate(decs, axis=0)[:tfs] if decs else np.zeros((0, 263))
    pr = np.array(_roots, dtype=np.float32)[:tfs] if _roots else np.zeros((0, 3))
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    return pm, pr, gr


def _run_real(model, vae, sample, device, *, hl, nds, hz, tdt, wpdt, fps, mode):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gr_arr = sample["traj"].numpy()
    dur = (sample["feature_length"] - 1) / fps
    npt = max(2, int(dur / wpdt) + 1)
    plan_pts = resample_polyline_by_arclength(gr_arr, npt)
    plan_t = assign_uniform_timestamps(npt, wpdt)
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    sr = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, _roots, fc = [], [], True
    for ci in range(tl):
        if mode == "real_no_traj":
            ti = None
        else:
            cr = np.zeros(3, dtype=np.float32)
            if mode == "real_gtroot":
                cr = gr_arr[min(ci * 4, len(gr_arr) - 1)].astype(np.float32)
            else:
                cr[[0, 2]] = sr.r_pos_accum[[0, 2]].astype(np.float32)
            ft = sample_plan_future(StreamTrajectoryPlan(times=plan_t, points_xyz=plan_pts, start_commit_index=0, version=0, source="bench"), current_commit=ci, current_root_xyz=cr, horizon_tokens=hz, token_dt=tdt, reanchor_to_current_root=True)
            ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0),
                  "token_mask": torch.ones(1, hz)}
        sp = build_stream_step_model_input(
            sample["text"] if isinstance(sample["text"], str) else sample["text"][0],
            traj_input=ti)
        out = model.stream_generate_step(sp, first_chunk=fc)
        dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                 first_chunk=fc)[0].float().cpu().numpy())
        fc = False
        for frm in dec:
            sr.process_frame(frm)
            _roots.append(sr.r_pos_accum.copy())
        decs.append(dec)
    vae.clear_cache()
    pm = np.concatenate(decs, axis=0)[:tfs] if decs else np.zeros((0, 263))
    pr = np.array(_roots, dtype=np.float32)[:tfs] if _roots else np.zeros((0, 3))
    return pm, pr, gr, plan_t, plan_pts


def _rotate_xz(points, anchor, deg):
    pts = np.asarray(points, dtype=np.float32).copy()
    anc = np.asarray(anchor, dtype=np.float32).reshape(3)
    c, s = float(np.cos(np.deg2rad(deg))), float(np.sin(np.deg2rad(deg)))
    rel = pts[:, [0, 2]] - anc[[0, 2]][None, :]
    pts[:, 0], pts[:, 2] = anc[0] + c * rel[:, 0] - s * rel[:, 1], anc[2] + s * rel[:, 0] + c * rel[:, 1]
    return pts


def _run_turn(model, vae, sample, device, *, hl, nds, hz, tdt, wpdt, fps, mode, angle,
              delay_tokens=20, blend_tokens=4):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gr_arr = sample["traj"].numpy()
    dur = (sample["feature_length"] - 1) / fps
    npt = max(2, int(dur / wpdt) + 1)
    plan_pts = resample_polyline_by_arclength(gr_arr, npt)
    plan_t = assign_uniform_timestamps(npt, wpdt)
    split_tok, sf = 15, max(1, 1 + 4 * 14)
    rot_pts = np.concatenate(
        [plan_pts[:sf], _rotate_xz(plan_pts[sf:], plan_pts[sf - 1], angle)], axis=0)
    rot_t = np.arange(len(rot_pts), dtype=np.float32) * wpdt
    ed = int(delay_tokens) if isinstance(delay_tokens, (int, float)) else 20
    eb = int(blend_tokens) if isinstance(blend_tokens, (int, float)) else 4
    # Extend past the blend zone for post-turn observation.
    extra = max(0, split_tok + ed + eb - tl)
    total_tl = tl + extra + 8
    total_tfs = 1 + 4 * (total_tl - 1) if total_tl > 1 else 1
    # Extend plans to cover the full query horizon past the blend zone.
    _needed_wp = max(len(plan_pts), int((total_tl + hz) * tdt / wpdt) + 2)
    for _p_arr, _p_name in [(plan_pts, "plan"), (rot_pts, "rot")]:
        _n = len(_p_arr)
        if _needed_wp > _n:
            _start_wp = max(0, _n - 5)
            _vel = _p_arr[-1] - _p_arr[_start_wp]
            _denom = max(1, _n - 1 - _start_wp)  # actual intervals spanned
            _step = _vel / float(_denom)
            _n_extra = _needed_wp - _n
            _p_new = _p_arr[-1][None, :] + np.arange(1, _n_extra + 1, dtype=np.float32)[:, None] * _step[None, :]
            if _p_name == "plan":
                plan_pts = np.concatenate([plan_pts, _p_new.astype(np.float32)], axis=0)
            else:
                rot_pts = np.concatenate([rot_pts, _p_new.astype(np.float32)], axis=0)
    plan_t = assign_uniform_timestamps(len(plan_pts), wpdt)
    rot_t = np.arange(len(rot_pts), dtype=np.float32) * wpdt
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    sr = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, _roots, fc = [], [], True
    for ci in range(total_tl):
        cr = np.zeros(3, dtype=np.float32)
        cr[[0, 2]] = sr.r_pos_accum[[0, 2]].astype(np.float32)
        # 3-zone delayed blend: delay -> blend -> replace
        if ci < split_tok + ed:
            use, ut = plan_pts, plan_t
        elif ci < split_tok + ed + eb and eb > 0:
            w = smoothstep01(float(ci - split_tok - ed) / eb)
            use = (1.0 - w) * plan_pts + w * rot_pts
            ut = np.arange(len(use), dtype=np.float32) * wpdt
        else:
            use, ut = rot_pts, rot_t
        _p = StreamTrajectoryPlan(times=ut,
                                   points_xyz=np.asarray(use, dtype=np.float32),
                                   start_commit_index=0, version=0, source="bench")
        ft = sample_plan_future(_p, current_commit=ci, current_root_xyz=cr,
                                horizon_tokens=hz, token_dt=tdt, reanchor_to_current_root=True)
        ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0), "token_mask": torch.ones(1, hz)}
        sp = build_stream_step_model_input(
            sample["text"] if isinstance(sample["text"], str) else sample["text"][0],
            traj_input=ti)
        out = model.stream_generate_step(sp, first_chunk=fc)
        dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                 first_chunk=fc)[0].float().cpu().numpy())
        fc = False
        for frm in dec:
            sr.process_frame(frm)
            _roots.append(sr.r_pos_accum.copy())
        decs.append(dec)
    vae.clear_cache()
    pm = np.concatenate(decs, axis=0)[:total_tfs] if decs else np.zeros((0, 263))
    pr = np.array(_roots, dtype=np.float32)[:total_tfs] if _roots else np.zeros((0, 3))
    return pm, pr, gr, plan_t, rot_pts, total_tfs


def _run_babel(model, vae, sample, device, *, hl, nds, hz, tdt, wpdt, fps, mode):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gr_arr = sample["traj"].numpy()
    dur = (sample["feature_length"] - 1) / fps
    npt = max(2, int(dur / wpdt) + 1)
    plan_pts = resample_polyline_by_arclength(gr_arr, npt)
    plan_t = assign_uniform_timestamps(npt, wpdt)
    segs = [StreamTextSegment(text=t, token_end=te)
            for t, te in zip(sample["text"], sample["token_text_end"])]
    tc = StreamTextRolloutController(segs)
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    sr = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, _roots, fc = [], [], True
    for ci in range(tl):
        txt = tc.get_text_for_commit_index(ci)
        if mode == "babel_no_traj":
            ti = None
        elif mode == "babel_timestamped":
            cr = np.zeros(3, dtype=np.float32)
            cr[[0, 2]] = sr.r_pos_accum[[0, 2]].astype(np.float32)
            qt = (float(ci) * tdt + np.arange(hz, dtype=np.float32) * tdt)
            gt_t = np.arange(len(gr_arr), dtype=np.float32) / 20.0
            ft = sample_timestamped_trajectory(gt_t, gr_arr, qt)
            anc = sample_timestamped_trajectory(gt_t, gr_arr, np.asarray([qt[0]], dtype=np.float32))[0]
            ft = cr + (ft - anc.astype(np.float32))
            ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0), "token_mask": torch.ones(1, hz)}
        else:
            cr = np.zeros(3, dtype=np.float32)
            cr[[0, 2]] = sr.r_pos_accum[[0, 2]].astype(np.float32)
            ft = sample_plan_future(StreamTrajectoryPlan(times=plan_t, points_xyz=plan_pts, start_commit_index=0, version=0, source="bench"), current_commit=ci, current_root_xyz=cr, horizon_tokens=hz, token_dt=tdt, reanchor_to_current_root=True)
            ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0), "token_mask": torch.ones(1, hz)}
        sp = build_stream_step_model_input(txt, traj_input=ti)
        out = model.stream_generate_step(sp, first_chunk=fc)
        dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                 first_chunk=fc)[0].float().cpu().numpy())
        fc = False
        for frm in dec:
            sr.process_frame(frm)
            _roots.append(sr.r_pos_accum.copy())
        decs.append(dec)
    vae.clear_cache()
    pm = np.concatenate(decs, axis=0)[:tfs] if decs else np.zeros((0, 263))
    pr = np.array(_roots, dtype=np.float32)[:tfs] if _roots else np.zeros((0, 3))
    return pm, pr, gr, plan_t, plan_pts


# ── main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Unified stream benchmark runner")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--raw_data_dir", required=True)
    p.add_argument("--output_dir", default="outputs/stream_benchmark")
    p.add_argument("--preset", default="smoke")
    p.add_argument("--suites", default=None)
    p.add_argument("--render_video", action="store_true", default=False)
    p.add_argument("--history_length", type=int, default=30)
    p.add_argument("--traj_horizon_tokens", type=int, default=20)
    p.add_argument("--num_denoise_steps", type=int, default=10)
    p.add_argument("--waypoint_dt", type=float, default=0.05)
    p.add_argument("--token_dt", type=float, default=0.20)
    p.add_argument("--motion_fps", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--precomputed_text_emb_path", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(config_path=args.config)
    OmegaConf.update(cfg.config, "test_vae_ckpt", args.vae_ckpt)
    if args.precomputed_text_emb_path:
        OmegaConf.update(cfg.config, "model.params.use_precomputed_text_emb", True)
        OmegaConf.update(cfg.config, "model.params.precomputed_text_emb_path",
                         args.precomputed_text_emb_path)

    print(f"Loading VAE ...")
    vae = _load_vae(cfg, dev)
    print(f"Loading model ...")
    model = _load_model(cfg, args.ckpt, dev)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, run_id)
    vdir = os.path.join(out_root, "videos") if args.render_video else None
    os.makedirs(out_root, exist_ok=True)
    if vdir: os.makedirs(vdir, exist_ok=True)

    suites_list = [s.strip() for s in args.suites.split(",")] if args.suites else None
    cases = get_cases(suites=suites_list, preset=args.preset)
    print(f"{len(cases)} case(s)  preset={args.preset}  suites={suites_list}")

    all_recs = []
    for case in cases:
        print(f"\n--- {case.name} ({case.suite}/{case.mode}) ---")
        seed_everything(args.seed)

        if case.dataset == "babel" and case.sample_ids:
            sample = _merge_babel(args.raw_data_dir, case.sample_ids)
        elif case.dataset == "babel":
            sample = _load_babel_sample(args.raw_data_dir, case.sample_id)
        else:
            sample = _load_humanml3d_sample(args.raw_data_dir, case.sample_id)
        gr_base = extract_root_trajectory_263(sample["feature"].numpy())

        kw = dict(hl=args.history_length, nds=args.num_denoise_steps,
                  hz=args.traj_horizon_tokens, tdt=args.token_dt,
                  wpdt=args.waypoint_dt, fps=args.motion_fps)

        if case.suite == "step":
            pm, pr, gr = _run_step(model, vae, sample, dev, mode=case.mode, **kw)
            rec = build_plan_metrics(
                pr, original_gt_root=gr,
                plan_times=np.arange(len(gr), dtype=np.float32) / 20.0,
                plan_points_xyz=gr,
                target_frames=sample["feature_length"], motion_fps=args.motion_fps,
                motion_263=pm, target_source="original_gt_root",
            )
        elif case.suite == "real":
            pm, pr, gr, pt, pp = _run_real(model, vae, sample, dev, mode=case.mode, **kw)
            rec = build_plan_metrics(
                pr, original_gt_root=gr, plan_times=pt, plan_points_xyz=pp,
                target_frames=sample["feature_length"], motion_fps=args.motion_fps,
                motion_263=pm,
            )
        elif case.suite == "turn":
            ang = case.mode_kwargs.get("update_angle", 30.0)
            dt_val = case.mode_kwargs.get("mid_update_delay_tokens", 20)
            db_val = case.mode_kwargs.get("mid_update_blend_tokens", 4)
            if isinstance(dt_val, str):
                dt_val = int(dt_val.split(",")[0])
            pm, pr, gr, pt, pp, ttfs = _run_turn(model, vae, sample, dev, mode=case.mode,
                                           angle=ang, delay_tokens=int(dt_val),
                                           blend_tokens=int(db_val), **kw)
            rec = build_plan_metrics(
                pr, original_gt_root=gr, plan_times=pt, plan_points_xyz=pp,
                target_frames=ttfs, motion_fps=args.motion_fps,
                motion_263=pm,
            )
        elif case.suite == "babel":
            pm, pr, gr, pt, pp = _run_babel(model, vae, sample, dev, mode=case.mode, **kw)
            rec = build_plan_metrics(
                pr, original_gt_root=gr, plan_times=pt, plan_points_xyz=pp,
                target_frames=sample["feature_length"], motion_fps=args.motion_fps,
                motion_263=pm,
            )
        else:
            print(f"  SKIP: unknown suite {case.suite}")
            continue

        rec["suite"] = case.suite; rec["mode"] = case.mode
        rec["sample_id"] = case.sample_id; rec["case_name"] = case.name
        all_recs.append(rec)
        print(f"  ADE={rec.get('ADE', float('nan')):.4f}  FDE={rec.get('FDE', float('nan')):.4f}")

        if args.render_video and pm.size > 0:
            mp4 = os.path.join(vdir, f"{case.name}.mp4")
            render_single_video(motion=pm, save_path=mp4, dim=263, render_setting={})
            print(f"    video: {mp4}")

    summary = {"run_id": run_id, "config": args.config, "ckpt": args.ckpt,
               "vae_ckpt": args.vae_ckpt, "waypoint_dt": args.waypoint_dt,
               "traj_horizon_tokens": args.traj_horizon_tokens,
               "history_length": args.history_length, "records": all_recs}
    sp = os.path.join(out_root, "summary.json")
    with open(sp, "w") as f: json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary: {sp}")

    if all_recs:
        cp = os.path.join(out_root, "summary.csv")
        fields = ["suite", "mode", "sample_id", "ADE", "FDE", "path_arc",
                  "path_chamfer", "chamfer_type", "lateral_velocity_ratio",
                  "heading_path_error_deg", "target_source",
                  "ADE_vs_original_gt", "FDE_vs_original_gt"]
        with open(cp, "w", newline="") as fc:
            w = csv.DictWriter(fc, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for rec in all_recs: w.writerow(rec)
        print(f"CSV: {cp}")


if __name__ == "__main__":
    main()
