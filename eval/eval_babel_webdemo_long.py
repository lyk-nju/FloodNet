"""BABEL long-session web-demo-like evaluator.

Merges multiple BABEL present segments into one continuous session,
runs stream_generate_step() with text switching and timestamped trajectory
plans, and compares 4 modes: gt_suffix / timestamped_gt_plan /
duration_waypoints / no_traj.

Usage::

    python eval/eval_babel_webdemo_long.py \\
        --config configs/eval_babel_stream.yaml \\
        --ckpt /path/to/model.ckpt --vae_ckpt /path/to/vae.ckpt \\
        --sample_ids 9797_1,9797_2,9797_3 \\
        --mode all --waypoint_dt 0.05 --render_video
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

# Allow running from any directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
from lightning import seed_everything
from torch_ema import ExponentialMovingAverage

from omegaconf import OmegaConf

from utils.initialize import (
    check_state_dict,
    instantiate,
    load_config,
)
from utils.motion_process import StreamJointRecovery263, extract_root_trajectory_263
from utils.stream_rollout import (
    StreamTextRolloutController,
    StreamTextSegment,
    build_stream_step_model_input,
)
from utils.stream_traj import sample_timestamped_trajectory
from utils.traj_batch import root_to_traj_feats
from utils.visualize import render_single_video

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── model loading ──────────────────────────────────────────────────────

def _load_vae(cfg, device):
    vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False,
                      **cfg.test_vae.params)
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
        print(f"Loaded VAE from {cfg.test_vae_ckpt} with EMA")
    else:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        print(f"Loaded VAE from {cfg.test_vae_ckpt} w/o EMA")
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _load_model(cfg, ckpt_path, device):
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False,
                        **cfg.model.params)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(checkpoint["state_dict"].keys())
    controlnet_missing = not any(k.startswith("controlnet.") for k in ckpt_keys)
    strict = not controlnet_missing
    result = model.load_state_dict(checkpoint["state_dict"], strict=strict)
    if not strict and result.missing_keys:
        print(f"Loaded LDF strict=False. Missing: {result.missing_keys}")
        if any("controlnet." in k for k in result.missing_keys):
            model.controlnet.init_from_backbone(model.model)
            print("Re-init ControlNet from backbone")
    if "ema_state" in checkpoint:
        n_shadow = len(checkpoint["ema_state"]["shadow_params"])
        ema_params = [p for p in model.parameters() if p.requires_grad]
        if len(ema_params) != n_shadow:
            ema_params = list(model.parameters())
        assert len(ema_params) == n_shadow
        ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(ema_params)
        print(f"Loaded model with EMA ({n_shadow} params)")
    else:
        print(f"Loaded model w/o EMA")
    model.to(device).eval()
    return model


# ── BABEL sample loading & merging ─────────────────────────────────────

def _load_one_babel_sample(raw_data_dir: str, sample_id: str) -> dict:
    data_dir = os.path.join(raw_data_dir, "BABEL_streamed")
    feat = np.load(os.path.join(data_dir, "motions", f"{sample_id}.npy")).astype(np.float32)
    token_dir = os.path.join(data_dir, "TOKENS_20251030_085836_vae_wan_z4")
    token = np.load(os.path.join(token_dir, f"{sample_id}.npy")).astype(np.float32)

    text_data = []
    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            caption = parts[0].strip()
            tokens = parts[1].split(" ") if len(parts) > 1 else []
            f_tag = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
            to_tag = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0.0
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            text_data.append({"caption": caption, "tokens": tokens,
                              "f_tag": f_tag, "to_tag": to_tag})
    return {"feature": feat, "token": token, "text_data": text_data,
            "name": sample_id}


def _merge_babel_samples(raw_data_dir: str, sample_ids: List[str]) -> dict:
    """Load and merge multiple BABEL present segments into one session."""
    parts = [_load_one_babel_sample(raw_data_dir, sid) for sid in sample_ids]

    features = [p["feature"] for p in parts]
    tokens = [p["token"] for p in parts]
    merged_feat = np.concatenate(features, axis=0)
    merged_token = np.concatenate(tokens, axis=0)

    # Merge text_data with cumulative time offsets.
    text_data = []
    feat_offset = 0
    token_offset = 0
    feat_fps = 20.0
    for i, p in enumerate(parts):
        for td in p["text_data"]:
            f_tag, to_tag = td["f_tag"], td["to_tag"]
            if f_tag == 0.0 and to_tag == 0.0:
                abs_f = feat_offset / feat_fps
                abs_t = (feat_offset + len(p["feature"])) / feat_fps
            else:
                abs_f = feat_offset / feat_fps + f_tag
                abs_t = feat_offset / feat_fps + to_tag
            text_data.append({
                "caption": td["caption"], "tokens": td["tokens"],
                "f_tag": abs_f, "to_tag": abs_t,
            })
        feat_offset += len(p["feature"])
        token_offset += len(p["token"])

    # Build feature_text_end / token_text_end from merged text_data.
    texts: List[str] = []
    feature_text_end: List[int] = []
    cursor = 0
    total_frames = len(merged_feat)
    total_tokens = len(merged_token)
    for td in text_data:
        abs_start = max(0, int(td["f_tag"] * feat_fps + 0.5))
        abs_end = int(td["to_tag"] * feat_fps + 0.5) if td["to_tag"] > 0 else total_frames
        if abs_end <= abs_start:
            continue
        if abs_start > cursor:
            texts.append("")
            feature_text_end.append(min(abs_start, total_frames))
            cursor = abs_start
        texts.append(td["caption"])
        feature_text_end.append(min(abs_end, total_frames))
        cursor = abs_end
    if cursor < total_frames:
        texts.append("")
        feature_text_end.append(total_frames)
    if not texts:
        texts = [td["caption"] or "" for td in text_data] or [""]
        feature_text_end = [total_frames]

    token_text_end = []
    for ef in feature_text_end:
        last_frame = ef - 1
        tok_end = (last_frame + 3) // 4 + 1
        token_text_end.append(max(0, min(total_tokens, tok_end)))

    # Trajectory.
    traj_xyz = extract_root_trajectory_263(merged_feat)
    traj_features = root_to_traj_feats(traj_xyz)

    sample = {
        "name": sample_ids[0].rsplit("_", 1)[0],
        "dataset": "BABEL_streamed",
        "feature": torch.from_numpy(merged_feat).float(),
        "feature_length": total_frames,
        "token": torch.from_numpy(merged_token).float(),
        "token_length": total_tokens,
        "text": texts,
        "text_data": text_data,
        "traj": torch.from_numpy(traj_xyz).float(),
        "traj_length": len(traj_xyz),
        "traj_features": torch.from_numpy(traj_features).float(),
        "token_text_end": token_text_end,
        "feature_text_end": feature_text_end,
        "token_mask": torch.ones(total_tokens, dtype=torch.float32),
        "traj_mask": torch.ones(len(traj_xyz), dtype=torch.float32),
        "_parts": parts,
        "_sample_ids": sample_ids,
    }
    return sample


# ── trajectory modes ───────────────────────────────────────────────────

def _build_timestamped_input(sample: dict, commit_idx: int,
                             horizon_tokens: int, token_dt: float,
                             *, times=None, waypoints=None):
    """Sample future H tokens from timestamped trajectory.

    When *times* / *waypoints* are given, they define the trajectory plan
    (e.g. duration-sampled waypoints).  Otherwise the full GT root is used
    as the reference (timestamped_gt_plan mode).
    """
    token_length = sample["token_length"]
    if commit_idx >= token_length:
        return None
    if times is None:
        gt_root = sample["traj"].numpy()
        total_frames = len(gt_root)
        times = np.arange(total_frames, dtype=np.float32) / 20.0
        waypoints = gt_root
    n = min(horizon_tokens, token_length - commit_idx)
    query_times = np.arange(n, dtype=np.float32) * token_dt + commit_idx * token_dt
    future_traj = sample_timestamped_trajectory(times, waypoints, query_times)
    return {
        "traj": torch.from_numpy(future_traj).float().unsqueeze(0),
        "token_mask": torch.ones(1, n),
    }


def _build_duration_waypoint_input(sample: dict, commit_idx: int,
                                   horizon_tokens: int, token_dt: float,
                                   waypoint_dt: float):
    """Dense spatial waypoints sampled from GT at *waypoint_dt* stride."""
    gt_root = sample["traj"].numpy()
    total_frames = len(gt_root)
    wp_indices = np.arange(0, total_frames, max(1, int(waypoint_dt * 20.0)))
    wp_times = np.arange(len(wp_indices), dtype=np.float32) * waypoint_dt
    return _build_timestamped_input(
        sample, commit_idx, horizon_tokens, token_dt,
        times=wp_times, waypoints=gt_root[wp_indices],
    )


# ── metrics ────────────────────────────────────────────────────────────

def _compute_ade(pred_root, gt_root):
    n = min(len(pred_root), len(gt_root))
    if n == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(pred_root[:n, [0, 2]] - gt_root[:n, [0, 2]], axis=1)))


def _compute_fde(pred_root, gt_root):
    n = min(len(pred_root), len(gt_root))
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(pred_root[n - 1, [0, 2]] - gt_root[n - 1, [0, 2]]))


def _compute_path_length(root):
    if len(root) < 2:
        return 0.0
    xz = root[:, [0, 2]] if root.shape[1] >= 3 else root
    return float(np.sum(np.linalg.norm(np.diff(xz, axis=0), axis=1)))


def _compute_segment_metrics(pred_root, gt_root, segments):
    out = []
    for seg in segments:
        sf, ef = seg["start_frame"], seg["end_frame"]
        ef = min(ef, len(pred_root), len(gt_root))
        if ef <= sf:
            out.append(None)
            continue
        pr, gr = pred_root[sf:ef], gt_root[sf:ef]
        out.append({
            "text": seg["text"], "start_frame": sf, "end_frame": ef,
            "ADE": _compute_ade(pr, gr), "FDE": _compute_fde(pr, gr),
            "pred_path_length": _compute_path_length(pr),
            "gt_path_length": _compute_path_length(gr),
        })
    return out


# ── stream runner ──────────────────────────────────────────────────────

@torch.no_grad()
def _run_stream_session(model, vae, sample: dict, device,
                        *, history_length: int, num_denoise_steps: int,
                        horizon_tokens: int, token_dt: float,
                        mode: str, waypoint_dt: float = 0.05,
                        render_video: bool = False, out_dir: str | None = None):
    """Run one full stream_generate_step session, return metrics."""
    token_length = sample["token_length"]
    total_frames = 1 + 4 * (token_length - 1) if token_length > 1 else 1

    segments = [
        StreamTextSegment(text=t, token_end=te)
        for t, te in zip(sample["text"], sample["token_text_end"])
    ]
    text_ctrl = StreamTextRolloutController(segments)

    vae.clear_cache()
    model.init_generated(history_length, batch_size=1,
                         num_denoise_steps=num_denoise_steps)
    model.generated = model.generated.to(device)

    stream_recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    all_decoded = []
    all_pred_root = []
    first_chunk = True

    for commit_idx in range(token_length):
        current_text = text_ctrl.get_text_for_commit_index(commit_idx)

        if mode == "no_traj":
            traj_input = None
        elif mode == "gt_suffix":
            remain = token_length - commit_idx
            traj_input = {
                "traj": sample["traj"][4 * commit_idx:].unsqueeze(0).float(),
                "token_mask": torch.ones(1, max(1, remain)),
            }
        elif mode == "timestamped_gt_plan":
            traj_input = _build_timestamped_input(
                sample, commit_idx, horizon_tokens, token_dt)
        elif mode == "duration_waypoints":
            traj_input = _build_duration_waypoint_input(
                sample, commit_idx, horizon_tokens, token_dt, waypoint_dt)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        step_payload = build_stream_step_model_input(current_text, traj_input=traj_input)
        output = model.stream_generate_step(step_payload, first_chunk=first_chunk)
        generated = output["generated"]
        decoded = (vae.stream_decode(generated[0][None, :].to(device),
                                     first_chunk=first_chunk)[0]
                   .float().cpu().numpy())
        first_chunk = False

        for frame in decoded:
            stream_recovery.process_frame(frame)
        all_decoded.append(decoded)
        all_pred_root.append(extract_root_trajectory_263(decoded))

    vae.clear_cache()

    pred_motion = (np.concatenate(all_decoded, axis=0)[:total_frames]
                   if all_decoded else np.zeros((0, 263)))
    pred_root = (np.concatenate(all_pred_root, axis=0)[:total_frames]
                 if all_pred_root else np.zeros((0, 3)))
    gt_root = extract_root_trajectory_263(sample["feature"].numpy()[:total_frames])

    # Build per-segment info.
    seg_pairs = []
    for i, t in enumerate(sample["text"]):
        sf = sample["feature_text_end"][i - 1] if i > 0 else 0
        ef = sample["feature_text_end"][i]
        seg_pairs.append({"text": t, "start_frame": sf, "end_frame": ef})

    metrics = {
        "mode": mode,
        "ADE": _compute_ade(pred_root, gt_root),
        "FDE": _compute_fde(pred_root, gt_root),
        "pred_path_length": _compute_path_length(pred_root),
        "gt_path_length": _compute_path_length(gt_root),
        "path_length_ratio": (_compute_path_length(pred_root) /
                              max(_compute_path_length(gt_root), 1e-6)),
        "segments": _compute_segment_metrics(pred_root, gt_root, seg_pairs),
    }

    if render_video and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if pred_motion.size > 0:
            _mp4 = os.path.join(out_dir, "pred_motion.mp4")
            render_single_video(motion=pred_motion, save_path=_mp4,
                                dim=263, render_setting={})
            print(f"    [{mode}] video saved to {_mp4}")
        gt_motion = sample["feature"].numpy()[:total_frames]
        _gt_mp4 = os.path.join(out_dir, "gt_motion.mp4")
        render_single_video(motion=gt_motion, save_path=_gt_mp4,
                            dim=263, render_setting={})
        # Animated trajectory comparison.
        _n = min(len(pred_root), len(gt_root))
        if _n > 1:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.animation import FFMpegWriter
            _traj_mp4 = os.path.join(out_dir, "traj_compare.mp4")
            _fig, _ax = plt.subplots(figsize=(7, 7))
            _all_x = [gt_root[:_n, 0], pred_root[:_n, 0]]
            _all_z = [gt_root[:_n, 2], pred_root[:_n, 2]]
            _xlim = (min(a.min() for a in _all_x) - 0.5,
                     max(a.max() for a in _all_x) + 0.5)
            _zlim = (min(a.min() for a in _all_z) - 0.5,
                     max(a.max() for a in _all_z) + 0.5)
            _writer = FFMpegWriter(fps=20)
            _step_f = max(1, _n // 150)
            with _writer.saving(_fig, _traj_mp4, dpi=100):
                for _f in range(1, _n + 1, _step_f):
                    _ax.clear()
                    _ax.plot(gt_root[:min(_f, _n), 0],
                             gt_root[:min(_f, _n), 2],
                             "g-", linewidth=1.5, alpha=0.7, label="target")
                    _ax.plot(pred_root[:min(_f, _n), 0],
                             pred_root[:min(_f, _n), 2],
                             "r-", linewidth=1.5, alpha=0.7, label="pred")
                    _ax.plot(gt_root[0, 0], gt_root[0, 2], "go", markersize=6)
                    _ax.plot(pred_root[min(_f - 1, _n - 1), 0],
                             pred_root[min(_f - 1, _n - 1), 2], "r.", markersize=8)
                    for sp in seg_pairs:
                        ssf = sp["start_frame"]
                        if 0 < ssf < _n:
                            _ax.axvline(x=gt_root[ssf, 0], color="gray",
                                        linestyle=":", alpha=0.3)
                    _ax.set_xlim(_xlim)
                    _ax.set_ylim(_zlim)
                    _ax.set_aspect("equal")
                    _ax.legend(loc="upper right")
                    _ax.set_title(f"{mode}  frame {min(_f, _n)}/{_n}")
                    _writer.grab_frame()
            plt.close(_fig)
            print(f"    [{mode}] traj video saved to {_traj_mp4}")

    return pred_motion, pred_root, gt_root, metrics


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BABEL long-session web-demo-like evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--sample_ids", required=True,
                        help="Comma-separated BABEL sample IDs")
    parser.add_argument("--mode", default="all",
                        choices=["gt_suffix", "timestamped_gt_plan",
                                 "duration_waypoints", "no_traj", "all"])
    parser.add_argument("--out_dir", default="outputs_babel_webdemo_long")
    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument("--history_length", type=int, default=30)
    parser.add_argument("--traj_horizon_tokens", type=int, default=20)
    parser.add_argument("--num_denoise_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--waypoint_dt", type=float, default=0.05)
    parser.add_argument("--token_dt", type=float, default=0.20)
    parser.add_argument("--render_video", action="store_true", default=False)
    parser.add_argument("--precomputed_text_emb_path", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_ids = [s.strip() for s in args.sample_ids.split(",")]
    base_name = sample_ids[0].rsplit("_", 1)[0]
    out_root = os.path.join(args.out_dir, base_name)

    cfg = load_config(config_path=args.config)
    OmegaConf.update(cfg.config, "test_vae_ckpt", args.vae_ckpt)
    if args.precomputed_text_emb_path:
        OmegaConf.update(cfg.config, "model.params.use_precomputed_text_emb", True)
        OmegaConf.update(cfg.config, "model.params.precomputed_text_emb_path",
                         args.precomputed_text_emb_path)

    print(f"Loading VAE from {args.vae_ckpt} ...")
    vae = _load_vae(cfg, device)
    print(f"Loading model from {args.ckpt} ...")
    model = _load_model(cfg, args.ckpt, device)

    print(f"Merging samples: {sample_ids}")
    sample = _merge_babel_samples(args.raw_data_dir, sample_ids)
    print(f"  frames={sample['feature_length']}, tokens={sample['token_length']}, "
          f"texts={len(sample['text'])}")
    for i, t in enumerate(sample["text"]):
        print(f"    [{i}] '{t[:60]}' -> frame {sample['feature_text_end'][i]}")

    all_modes = ["gt_suffix", "timestamped_gt_plan", "duration_waypoints", "no_traj"]
    modes_to_run = all_modes if args.mode == "all" else [args.mode]

    all_metrics = []
    for mode in modes_to_run:
        print(f"\n=== {mode} ===")
        _md = os.path.join(out_root, mode) if args.render_video else None
        pred_motion, pred_root, gt_root, metrics = _run_stream_session(
            model, vae, sample, device,
            history_length=args.history_length,
            num_denoise_steps=args.num_denoise_steps,
            horizon_tokens=args.traj_horizon_tokens,
            token_dt=args.token_dt,
            mode=mode, waypoint_dt=args.waypoint_dt,
            render_video=args.render_video, out_dir=_md,
        )
        print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}  "
              f"path_ratio={metrics['path_length_ratio']:.4f}")
        for seg in (metrics.get("segments") or []):
            if seg is None:
                continue
            print(f"    [{seg['text'][:40]}] ADE={seg['ADE']:.4f}")

        _md = os.path.join(out_root, mode)
        os.makedirs(_md, exist_ok=True)
        np.save(os.path.join(_md, "pred_motion.npy"), pred_motion)
        np.save(os.path.join(_md, "pred_root.npy"), pred_root)
        np.save(os.path.join(_md, "gt_root.npy"), gt_root)
        np.save(os.path.join(_md, "gt_motion.npy"),
                sample["feature"].numpy()[:pred_motion.shape[0]])
        with open(os.path.join(_md, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        all_metrics.append(metrics)

    summary = {
        "base_name": base_name, "sample_ids": sample_ids,
        "config": args.config, "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt,
        "history_length": args.history_length,
        "traj_horizon_tokens": args.traj_horizon_tokens,
        "waypoint_dt": args.waypoint_dt, "token_dt": args.token_dt,
        "modes": {m["mode"]: {k: v for k, v in m.items() if k != "segments"}
                  for m in all_metrics},
        "text_schedule": [
            {"text": t, "end_frame": int(fe), "end_token": int(te)}
            for t, fe, te in zip(sample["text"], sample["feature_text_end"],
                                 sample["token_text_end"])
        ],
    }
    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    print(f"\n{'Mode':<25} {'ADE':>8} {'FDE':>8} {'path_ratio':>10}")
    print("-" * 55)
    for m in all_metrics:
        print(f"{m['mode']:<25} {m['ADE']:>8.4f} {m['FDE']:>8.4f} "
              f"{m['path_length_ratio']:>10.4f}")


if __name__ == "__main__":
    main()
