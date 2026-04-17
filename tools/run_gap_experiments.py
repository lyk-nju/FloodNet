"""
FloodNet 实验：定位 forward / generate 差距根因
================================================

实验 1  误差累积（时间维度）
  - 对每个样本生成完整序列，按 20 帧分段 & 前缀计算 MSE
  - 找"首次发散区间"

实验 2  forward 采样策略偏差
  - 方案 A：当前随机采样（1 次）
  - 方案 B：均匀扫描所有窗口位置后平均
  - 比较 B-A，验证是否系统低估

实验 3  TF vs SF-proxy（teacher-forcing vs self-forcing）
  - TF:       x["feature"] = GT token（现有 forward 逻辑）
  - SF-proxy: x["feature"] = model.generate() 生成的 latent（替换历史上下文）
  - 对比两者 active-window control loss，判断 TF bias 贡献

实验 4  CFG / denoise_steps 扫描
  - 扫 cfg_scale_text ∈ {1.0, 2.0, 3.0, 5.0}（如果模型支持），固定 seed=1234
  - 也扫 noise_steps（如果支持 num_denoise_steps override）

实验 5  ControlNet 残差强度 × generate loss 相关性
  - 记录每样本 residual L2 norm（层均值）
  - 与 generate loss 做 Pearson / Spearman 相关

Usage:
    python tools/run_gap_experiments.py --config configs/ldf.yaml --exps 1 2 3 5
    python tools/run_gap_experiments.py --config configs/ldf.yaml --exps 4
    python tools/run_gap_experiments.py --config configs/ldf.yaml --exps 1 2 3 4 5
"""
import argparse
import os
import random
import sys
import types
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.initialize import Config, get_function, instantiate
from utils.motion_process import extract_root_trajectory_263_torch


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/ldf.yaml")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 2345, 3456, 4567, 5678])
    p.add_argument("--exps", type=int, nargs="+", default=[1, 2, 3, 5],
                   help="Which experiments to run (1-5). Default: 1 2 3 5")
    p.add_argument("--seg_size", type=int, default=20,
                   help="Segment size in frames for Exp 1 (default 20)")
    p.add_argument("--exp2_n_fwd", type=int, default=10,
                   help="Number of fwd passes per sample for method A in Exp2 (default 10)")
    p.add_argument("--exp3_n_fwd", type=int, default=10,
                   help="Number of fwd passes per sample per mode in Exp3 (default 10)")
    p.add_argument("--cfg_scales", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0])
    p.add_argument("--denoise_steps_list", type=int, nargs="+", default=[10, 20])
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _load_model(cfg, ckpt_path, device, use_ema):
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    strict = not (
        cfg.model.params.get("use_traj_cond", False)
        or cfg.model.params.get("use_controlnet_traj", False)
    )
    model.load_state_dict(ckpt["state_dict"], strict=strict)
    if not strict and getattr(model, "controlnet", None) is not None:
        from eval_control_loss import _load_model as _lm  # reuse logic
    if use_ema and "ema_state" in ckpt:
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
        try:
            ema.load_state_dict(ckpt["ema_state"])
            ema.copy_to(model.parameters())
            print("[load] Applied EMA weights.")
        except ValueError as e:
            print(f"[load] EMA skip: {e}")
    model.to(device).eval()
    return model


def _load_vae(cfg, device):
    vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False, **cfg.test_vae.params)
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
    if "ema_state" in vae_ckpt:
        ve = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        ve.load_state_dict(vae_ckpt["ema_state"])
        ve.copy_to(vae.parameters())
    vae.to(device).eval()
    return vae


def _build_model_batch(batch, device):
    mb = {k: v for k, v in batch.items()}
    mb["feature"] = batch["token"]
    mb["feature_length"] = batch["token_length"]
    if "token_text_end" in batch:
        mb["feature_text_end"] = batch["token_text_end"]
    return _to_device(mb, device)


def _extract_xz_from_latent(latent, vae, device):
    """(T_tok, z_dim) → (T_frame, 2) xz"""
    with torch.no_grad():
        decoded = vae.decode(latent.unsqueeze(0).to(device))[0].float()  # (T, 263)
    traj = extract_root_trajectory_263_torch(decoded.unsqueeze(0))  # (1, T, 3)
    return traj[0, :, [0, 2]]  # (T, 2)


def _xz_mse_masked(pred_xz, gt_xz, mask):
    """All (T,2) or (T,2). mask: (T,). Returns (sum_loss, n_valid)."""
    T = min(len(pred_xz), len(gt_xz), len(mask))
    if T == 0:
        return 0.0, 0
    sq = ((pred_xz[:T] - gt_xz[:T]) ** 2).sum(dim=-1)  # (T,)
    m = mask[:T].float()
    return float((m * sq).sum().item()), float(m.sum().item())


def _forward_control_loss_one_pass(model, mb, vae, device, chunk_size):
    """Single forward pass → control loss on active window (same as train).
    Returns (loss_sum, n_valid) floats or (None, None) if no control_aux."""
    with torch.no_grad():
        out = model(mb)
    if "control_aux" not in out:
        return None, None
    pred_list = out["control_aux"]["pred_x0_latent_list"]
    traj = mb["traj"]
    traj_mask = mb["traj_mask"]
    traj_length = mb["traj_length"]
    token_to_frame = 4

    total_loss_sum = 0.0
    total_n = 0.0
    for i, pred_latent in enumerate(pred_list):
        t_tok = pred_latent.size(0)
        if t_tok > chunk_size:
            start_tok = t_tok - chunk_size
            start_f = 0 if start_tok == 0 else 4 * start_tok - 3
            end_f = t_tok * token_to_frame
        else:
            start_f, end_f = 0, None

        with torch.no_grad():
            decoded = vae.decode(pred_latent.unsqueeze(0))[0].float()
        l_motion = decoded.size(0)
        l_gt_total = min(int(traj_length[i].item()), traj.shape[1])

        pred_sl = slice(min(start_f, l_motion), min(end_f, l_motion) if end_f else l_motion)
        gt_sl   = slice(min(start_f, l_gt_total), min(end_f, l_gt_total) if end_f else l_gt_total)
        l = min(pred_sl.stop - pred_sl.start, gt_sl.stop - gt_sl.start)
        if l <= 0:
            continue

        pred_traj = extract_root_trajectory_263_torch(decoded.unsqueeze(0))
        pred_xz = pred_traj[:, pred_sl, :][:, :l, [0, 2]]
        gt_xz   = traj[i, gt_sl, :][:l, [0, 2]].unsqueeze(0).to(pred_xz.device, dtype=pred_xz.dtype)
        mask    = traj_mask[i, gt_sl][:l].unsqueeze(0).to(pred_xz.device, dtype=pred_xz.dtype)
        sq_err  = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
        total_loss_sum += float((mask * sq_err).sum().item())
        total_n        += float(mask.sum().item())

    return total_loss_sum, total_n


# ──────────────────────────────────────────────────────────────────────────────
# Exp 1: Error Accumulation
# ──────────────────────────────────────────────────────────────────────────────

def run_exp1(model, vae, loader, device, seeds, seg_size):
    print("\n" + "=" * 65)
    print("实验 1  误差累积（时间维度）")
    print("=" * 65)

    # Collect per-sample data for one seed (use first seed)
    # Then report across seeds for key samples
    all_rows = []  # {name, seed, seg_mse_list, prefix_mse_list, first_drift_seg}

    for seed in seeds:
        _set_seed(seed)
        for batch in loader:
            name = batch.get("name", ["?"])[0]
            if "traj" not in batch:
                continue
            mb = _build_model_batch(batch, device)
            with torch.no_grad():
                out = model.generate(mb)
            if "generated" not in out:
                continue
            generated = out["generated"][0]  # (T_tok, z_dim)

            pred_xz = _extract_xz_from_latent(generated, vae, device).cpu()  # (T, 2)
            gt_traj = batch["traj"][0].float()  # (T, 3) xyz
            gt_xz = gt_traj[:, [0, 2]]
            mask = batch["traj_mask"][0].float()

            T = min(len(pred_xz), len(gt_xz), len(mask))

            # Segment MSE (non-overlapping windows of seg_size frames)
            seg_mse = []
            n_segs = (T + seg_size - 1) // seg_size
            for s in range(n_segs):
                sf, ef = s * seg_size, min((s + 1) * seg_size, T)
                p_s = pred_xz[sf:ef]
                g_s = gt_xz[sf:ef]
                m_s = mask[sf:ef]
                n = float(m_s.sum().item())
                if n > 0:
                    sq = ((p_s - g_s) ** 2).sum(dim=-1)
                    seg_mse.append(float((m_s * sq).sum().item()) / n)
                else:
                    seg_mse.append(float("nan"))

            # Prefix MSE at 20, 40, 60, ... T
            prefix_mse = []
            for ef in range(seg_size, T + 1, seg_size):
                p_p = pred_xz[:ef]
                g_p = gt_xz[:ef]
                m_p = mask[:ef]
                n = float(m_p.sum().item())
                if n > 0:
                    sq = ((p_p - g_p) ** 2).sum(dim=-1)
                    prefix_mse.append(float((m_p * sq).sum().item()) / n)
                else:
                    prefix_mse.append(float("nan"))

            # First drift segment (first segment MSE > 0.01)
            DRIFT_THRESH = 0.01
            first_drift = None
            for s, mse in enumerate(seg_mse):
                if mse == mse and mse > DRIFT_THRESH:
                    first_drift = f"{s*seg_size}:{(s+1)*seg_size}"
                    break

            all_rows.append({
                "name": name, "seed": seed,
                "seg_mse": seg_mse, "prefix_mse": prefix_mse,
                "first_drift": first_drift or "none",
                "max_seg_mse": max((x for x in seg_mse if x == x), default=float("nan")),
            })

    # Print template table
    print(f"\n  segment_size={seg_size}帧, 发散阈值={0.01}")
    print(f"  {'sample':<10} {'seed':>6} {'first_drift_seg':<16} {'max_seg_mse':>12} "
          f"{'prefix_mse@40':>14} {'prefix_mse@80':>14}")
    print("  " + "-" * 80)
    for r in all_rows:
        pm40 = r["prefix_mse"][1] if len(r["prefix_mse"]) > 1 else float("nan")
        pm80 = r["prefix_mse"][3] if len(r["prefix_mse"]) > 3 else float("nan")
        print(f"  {r['name']:<10} {r['seed']:>6} {r['first_drift']:<16} "
              f"{r['max_seg_mse']:>12.5f} {pm40:>14.5f} {pm80:>14.5f}")

    # Per-sample segment MSE curve (first seed only)
    print(f"\n  --- 分段 MSE 曲线（seed={seeds[0]}）---")
    seen = set()
    for r in all_rows:
        if r["seed"] != seeds[0] or r["name"] in seen:
            continue
        seen.add(r["name"])
        segs_str = "  ".join(
            f"[{i*seg_size}:{(i+1)*seg_size}]={v:.4f}" if v == v else f"[{i*seg_size}:-]=nan"
            for i, v in enumerate(r["seg_mse"])
        )
        print(f"  {r['name']}: {segs_str}")

    return all_rows


# ──────────────────────────────────────────────────────────────────────────────
# Exp 2: Forward Sampling Bias
# ──────────────────────────────────────────────────────────────────────────────

def run_exp2(model, vae, loader, device, seed, n_fwd_a, chunk_size):
    print("\n" + "=" * 65)
    print("实验 2  forward 采样策略偏差")
    print("=" * 65)

    _set_seed(seed)
    rows = []

    for batch in loader:
        name = batch.get("name", ["?"])[0]
        if "traj" not in batch:
            continue
        mb = _build_model_batch(batch, device)

        # ── 方案 A：随机采样 n_fwd_a 次，平均 ──────────────────────────────
        a_sums, a_ns = [], []
        for _ in range(n_fwd_a):
            ls, n = _forward_control_loss_one_pass(model, mb, vae, device, chunk_size)
            if ls is not None and n > 0:
                a_sums.append(ls)
                a_ns.append(n)
        loss_A = sum(a_sums) / sum(a_ns) if sum(a_ns) > 0 else float("nan")

        # ── 方案 B：均匀扫描所有整数 window 位置 ──────────────────────────
        tok_len = int(batch["token_length"][0].item())
        n_windows = max(tok_len // chunk_size, 1)

        import torch as _torch
        b_sums, b_ns = [], []
        for w in range(1, n_windows + 1):
            # Force time_step so active window ends at token position w*chunk_size
            # We do this by temporarily patching forward to use fixed time_step
            # Simpler: create a truncated batch with seq_len = w*chunk_size
            t_end = min(w * chunk_size, tok_len)
            # Frame count for t_end tokens: causal VAE convention N tokens → 4*(N-1)+1 frames
            traj_frames = 4 * (t_end - 1) + 1 if t_end > 1 else 1
            # Keys whose sequence dimension is in frames (not tokens)
            frame_level_keys = {"traj_features", "traj_mask", "feature_raw"}
            trunc_mb = {}
            for k, v in mb.items():
                if _torch.is_tensor(v) and v.dim() >= 2:
                    if k in frame_level_keys:
                        # Frame-level tensor: truncate to traj_frames
                        trunc_end = min(traj_frames, v.shape[1])
                        trunc_mb[k] = v[:, :trunc_end]
                    elif v.shape[1] >= tok_len:
                        # Token-level tensor: truncate to t_end tokens
                        trunc_mb[k] = v[:, :t_end]
                    else:
                        trunc_mb[k] = v
                elif _torch.is_tensor(v) and v.dim() == 1 and v.shape[0] == mb["feature_length"].shape[0]:
                    # scalar per-sample fields like traj_length, feature_length
                    trunc_mb[k] = _torch.tensor([t_end], device=v.device, dtype=v.dtype)
                else:
                    trunc_mb[k] = v
            # override feature_length to t_end
            trunc_mb["feature_length"] = _torch.tensor([t_end], device=device)
            # Also truncate traj_length
            if "traj_length" in trunc_mb:
                orig_tl = int(batch["traj_length"][0].item())
                trunc_mb["traj_length"] = _torch.tensor(
                    [min(traj_frames, orig_tl)], device=device, dtype=_torch.long
                )
            ls, n = _forward_control_loss_one_pass(model, trunc_mb, vae, device, chunk_size)
            if ls is not None and n > 0:
                b_sums.append(ls)
                b_ns.append(n)
        loss_B = sum(b_sums) / sum(b_ns) if sum(b_ns) > 0 else float("nan")

        delta = (loss_B - loss_A) if (loss_B == loss_B and loss_A == loss_A) else float("nan")
        rows.append({"name": name, "A": loss_A, "B": loss_B, "delta": delta,
                     "n_windows": n_windows, "tok_len": tok_len})

    print(f"\n  seed={seed}, A方案重复{n_fwd_a}次随机, B方案扫描全部{chunk_size}帧窗口")
    print(f"  {'sample':<10} {'A_forward':>12} {'B_uniform':>12} {'delta(B-A)':>12} {'n_windows':>10}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['name']:<10} {r['A']:>12.5f} {r['B']:>12.5f} {r['delta']:>12.5f} {r['n_windows']:>10}")

    valid = [(r["A"], r["B"]) for r in rows if r["A"] == r["A"] and r["B"] == r["B"]]
    if valid:
        mean_A = np.mean([x[0] for x in valid])
        mean_B = np.mean([x[1] for x in valid])
        print(f"\n  overall: A_mean={mean_A:.5f}  B_mean={mean_B:.5f}  delta_mean={mean_B-mean_A:.5f}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Exp 3: TF vs SF-proxy
# ──────────────────────────────────────────────────────────────────────────────

def run_exp3(model, vae, loader, device, seed, n_fwd, chunk_size):
    print("\n" + "=" * 65)
    print("实验 3  TF vs SF-proxy（teacher-forcing vs self-forcing）")
    print("=" * 65)

    _set_seed(seed)
    rows = []

    for batch in loader:
        name = batch.get("name", ["?"])[0]
        if "traj" not in batch:
            continue
        mb = _build_model_batch(batch, device)

        # ── TF: forward with GT token ───────────────────────────────────────
        tf_sums, tf_ns = [], []
        for _ in range(n_fwd):
            ls, n = _forward_control_loss_one_pass(model, mb, vae, device, chunk_size)
            if ls is not None and n > 0:
                tf_sums.append(ls)
                tf_ns.append(n)
        loss_TF = sum(tf_sums) / sum(tf_ns) if sum(tf_ns) > 0 else float("nan")

        # ── Generate: full autoregressive ──────────────────────────────────
        with torch.no_grad():
            gen_out = model.generate(mb)
        if "generated" not in gen_out:
            rows.append({"name": name, "TF": loss_TF, "SF": float("nan"),
                         "GEN": float("nan"), "gap_gen_TF": float("nan"),
                         "gap_gen_SF": float("nan")})
            continue
        generated_latents = gen_out["generated"]  # List[(T_tok, z_dim)]

        # Compute generate control loss (full sequence)
        traj_arr = mb["traj"]
        traj_mask_arr = mb["traj_mask"]
        traj_length_arr = mb["traj_length"]
        gen_sums, gen_ns = 0.0, 0.0
        for i, gen_lat in enumerate(generated_latents):
            with torch.no_grad():
                dec = vae.decode(gen_lat.unsqueeze(0))[0].float()
            pred_traj = extract_root_trajectory_263_torch(dec.unsqueeze(0))
            l_motion = pred_traj.size(1)
            l_gt = min(int(traj_length_arr[i].item()), traj_arr.shape[1])
            l = min(l_motion, l_gt)
            if l <= 0:
                continue
            pred_xz = pred_traj[:, :l, [0, 2]]
            gt_xz = traj_arr[i, :l, [0, 2]].unsqueeze(0).to(pred_xz.device, dtype=pred_xz.dtype)
            mask = traj_mask_arr[i, :l].unsqueeze(0).to(pred_xz.device, dtype=pred_xz.dtype)
            sq = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
            gen_sums += float((mask * sq).sum().item())
            gen_ns += float(mask.sum().item())
        loss_GEN = gen_sums / gen_ns if gen_ns > 0 else float("nan")

        # ── SF-proxy: replace GT token with generated latent in forward ─────
        # Build a fake batch where "feature" = generated latent (instead of GT)
        tok_len = int(batch["token_length"][0].item())
        z_dim = generated_latents[0].shape[-1]
        sf_feature = torch.zeros(1, tok_len, z_dim, device=device)
        gen_lat = generated_latents[0]  # (T_gen_tok, z_dim)
        copy_len = min(gen_lat.shape[0], tok_len)
        sf_feature[0, :copy_len] = gen_lat[:copy_len]

        sf_mb = {k: v for k, v in mb.items()}
        sf_mb["feature"] = sf_feature  # SF history context

        sf_sums, sf_ns = [], []
        for _ in range(n_fwd):
            ls, n = _forward_control_loss_one_pass(model, sf_mb, vae, device, chunk_size)
            if ls is not None and n > 0:
                sf_sums.append(ls)
                sf_ns.append(n)
        loss_SF = sum(sf_sums) / sum(sf_ns) if sum(sf_ns) > 0 else float("nan")

        gap_gen_TF = loss_GEN - loss_TF if (loss_GEN == loss_GEN and loss_TF == loss_TF) else float("nan")
        gap_gen_SF = loss_GEN - loss_SF if (loss_GEN == loss_GEN and loss_SF == loss_SF) else float("nan")
        rows.append({
            "name": name, "TF": loss_TF, "SF": loss_SF,
            "GEN": loss_GEN, "gap_gen_TF": gap_gen_TF, "gap_gen_SF": gap_gen_SF,
        })

    print(f"\n  seed={seed}, 每种模式各 {n_fwd} 次 forward pass 平均")
    print(f"  {'sample':<10} {'TF_fwd':>10} {'SF_proxy':>10} {'generate':>10} "
          f"{'gap_gen-TF':>12} {'gap_gen-SF':>12}")
    print("  " + "-" * 68)
    for r in rows:
        print(f"  {r['name']:<10} {r['TF']:>10.5f} {r['SF']:>10.5f} {r['GEN']:>10.5f} "
              f"{r['gap_gen_TF']:>12.5f} {r['gap_gen_SF']:>12.5f}")

    valid = [r for r in rows if all(r[k] == r[k] for k in ["TF", "SF", "GEN"])]
    if valid:
        for k in ["TF", "SF", "GEN", "gap_gen_TF", "gap_gen_SF"]:
            print(f"  mean_{k:<12} = {np.mean([r[k] for r in valid]):.5f}")

    print("\n  解读：")
    print("  - 若 gap_gen-TF >> gap_gen-SF：TF 偏差贡献大（teacher forcing 是主因）")
    print("  - 若 gap_gen-SF 仍很大：TF 偏差不是主因（另有其他问题如对齐、残差爆炸）")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Exp 4: CFG / denoise_steps 扫描
# ──────────────────────────────────────────────────────────────────────────────

def run_exp4(model, vae, loader, device, seed, cfg_scales, denoise_steps_list):
    print("\n" + "=" * 65)
    print("实验 4  CFG / denoise_steps 扫描")
    print("=" * 65)

    # Check if model supports cfg_scale_text override
    has_cfg = hasattr(model, "cfg_scale_text")
    orig_cfg = getattr(model, "cfg_scale_text", None)
    print(f"  model.cfg_scale_text 可配置: {has_cfg}  (当前: {orig_cfg})")

    rows = []
    for steps in denoise_steps_list:
        for cfg in cfg_scales:
            if has_cfg:
                model.cfg_scale_text = cfg
            _set_seed(seed)
            total_loss, total_n = 0.0, 0.0
            for batch in loader:
                if "traj" not in batch:
                    continue
                mb = _build_model_batch(batch, device)
                with torch.no_grad():
                    try:
                        gen_out = model.generate(mb, num_denoise_steps=steps)
                    except Exception as e:
                        gen_out = model.generate(mb)  # fallback

                if "generated" not in gen_out:
                    continue
                for i, gen_lat in enumerate(gen_out["generated"]):
                    with torch.no_grad():
                        dec = vae.decode(gen_lat.unsqueeze(0))[0].float()
                    pred_traj = extract_root_trajectory_263_torch(dec.unsqueeze(0))
                    traj = mb["traj"]
                    traj_mask_arr = mb["traj_mask"]
                    traj_length_arr = mb["traj_length"]
                    l_gt = min(int(traj_length_arr[i].item()), traj.shape[1])
                    l = min(pred_traj.size(1), l_gt)
                    if l <= 0:
                        continue
                    pred_xz = pred_traj[:, :l, [0, 2]]
                    gt_xz = traj[i, :l, [0, 2]].unsqueeze(0).to(pred_xz.device, dtype=pred_xz.dtype)
                    m = traj_mask_arr[i, :l].unsqueeze(0).to(pred_xz.device, dtype=pred_xz.dtype)
                    sq = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
                    total_loss += float((m * sq).sum().item())
                    total_n += float(m.sum().item())

            mean_loss = total_loss / total_n if total_n > 0 else float("nan")
            rows.append({"cfg": cfg, "steps": steps, "loss": mean_loss})
            print(f"  cfg_scale_text={cfg:.1f}  steps={steps:3d}  mean_generate_loss={mean_loss:.5f}")

    if has_cfg and orig_cfg is not None:
        model.cfg_scale_text = orig_cfg  # restore

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Exp 5: Residual 强度 × generate loss 相关性
# ──────────────────────────────────────────────────────────────────────────────

def run_exp5(model, vae, loader, device, seed):
    print("\n" + "=" * 65)
    print("实验 5  ControlNet 残差强度 × generate loss 相关性")
    print("=" * 65)

    if not getattr(model, "use_controlnet_traj", False):
        print("  [SKIP] model.use_controlnet_traj=False，无 ControlNet")
        return []

    # Monkey-patch to capture residuals
    captured = {}  # name → list of residual tensors per layer

    _orig_maybe = model._controlnet_forward.__func__

    def _hooked(self, *args, **kwargs):
        res = _orig_maybe(self, *args, **kwargs)
        if res is not None and _current_name[0] is not None:
            n = _current_name[0]
            if n not in captured:
                captured[n] = []
            captured[n].append([r.detach().cpu() for r in res])
        return res

    _current_name = [None]
    model._controlnet_forward = types.MethodType(_hooked, model)

    _set_seed(seed)
    rows = []

    for batch in loader:
        name = batch.get("name", ["?"])[0]
        if "traj" not in batch:
            continue
        _current_name[0] = name
        mb = _build_model_batch(batch, device)

        # Forward pass to capture residuals
        with torch.no_grad():
            model(mb)
        _current_name[0] = None

        # Generate for control loss
        with torch.no_grad():
            gen_out = model.generate(mb)
        if "generated" not in gen_out:
            continue

        # Compute generate control loss
        traj = mb["traj"]
        traj_mask_arr = mb["traj_mask"]
        traj_length_arr = mb["traj_length"]
        gen_sums, gen_ns = 0.0, 0.0
        for i, gen_lat in enumerate(gen_out["generated"]):
            with torch.no_grad():
                dec = vae.decode(gen_lat.unsqueeze(0))[0].float()
            pred_traj = extract_root_trajectory_263_torch(dec.unsqueeze(0))
            l_gt = min(int(traj_length_arr[i].item()), traj.shape[1])
            l = min(pred_traj.size(1), l_gt)
            if l <= 0:
                continue
            pred_xz = pred_traj[:, :l, [0, 2]]
            gt_xz = traj[i, :l, [0, 2]].unsqueeze(0).to(pred_xz.device, dtype=pred_xz.dtype)
            m = traj_mask_arr[i, :l].unsqueeze(0).to(pred_xz.device, dtype=pred_xz.dtype)
            sq = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
            gen_sums += float((m * sq).sum().item())
            gen_ns += float(m.sum().item())
        loss_gen = gen_sums / gen_ns if gen_ns > 0 else float("nan")

        # Residual norm for this sample
        if name in captured and captured[name]:
            call_res = captured[name][0]  # first forward call
            layer_norms = [r.norm(dim=-1).mean().item() for r in call_res]
            mean_res_norm = float(np.mean(layer_norms))
            max_res_norm = float(np.max(layer_norms))
        else:
            mean_res_norm = float("nan")
            max_res_norm = float("nan")

        rows.append({"name": name, "res_mean": mean_res_norm,
                     "res_max": max_res_norm, "gen_loss": loss_gen})

    print(f"\n  {'sample':<10} {'res_mean_L2':>14} {'res_max_L2':>14} {'generate_loss':>14}")
    print("  " + "-" * 56)
    for r in rows:
        print(f"  {r['name']:<10} {r['res_mean']:>14.2f} {r['res_max']:>14.2f} {r['gen_loss']:>14.5f}")

    valid = [r for r in rows if all(r[k] == r[k] for k in ["res_mean", "gen_loss"])]
    if len(valid) >= 3:
        from scipy import stats as sp_stats
        res_v = [r["res_mean"] for r in valid]
        gen_v = [r["gen_loss"] for r in valid]
        try:
            pearson_r, pearson_p = sp_stats.pearsonr(res_v, gen_v)
            spearman_r, spearman_p = sp_stats.spearmanr(res_v, gen_v)
            print(f"\n  Pearson  r={pearson_r:.3f}  p={pearson_p:.4f}")
            print(f"  Spearman r={spearman_r:.3f}  p={spearman_p:.4f}")
        except Exception as e:
            print(f"\n  [相关性计算失败] {e}")
    else:
        print("\n  [样本数不足，跳过相关性计算]")

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = Config(args.config).config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or OmegaConf.select(cfg, "test_ckpt") or OmegaConf.select(cfg, "resume_ckpt")
    if not ckpt_path:
        raise ValueError("No checkpoint. Set --ckpt or config test_ckpt/resume_ckpt.")

    _set_seed(args.seeds[0])

    print(f"{'='*65}")
    print(f"FloodNet 差距根因实验")
    print(f"  ckpt:  {ckpt_path}")
    print(f"  seeds: {args.seeds}")
    print(f"  exps:  {args.exps}")
    print(f"{'='*65}")

    model = _load_model(cfg, ckpt_path, device, use_ema=not args.no_ema)
    vae = _load_vae(cfg, device)
    chunk_size = getattr(model, "chunk_size", 5)

    collate_fn = get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn") else None
    dataset = instantiate(cfg.data.get("test_target", cfg.data.target), cfg=cfg, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        drop_last=False, num_workers=0, collate_fn=collate_fn)

    print(f"  test samples: {len(dataset)}  chunk_size: {chunk_size}")

    if 1 in args.exps:
        run_exp1(model, vae, loader, device, args.seeds, args.seg_size)
    if 2 in args.exps:
        run_exp2(model, vae, loader, device, args.seeds[0], args.exp2_n_fwd, chunk_size)
    if 3 in args.exps:
        run_exp3(model, vae, loader, device, args.seeds[0], args.exp3_n_fwd, chunk_size)
    if 4 in args.exps:
        run_exp4(model, vae, loader, device, args.seeds[0], args.cfg_scales, args.denoise_steps_list)
    if 5 in args.exps:
        run_exp5(model, vae, loader, device, args.seeds[0])

    print("\n" + "=" * 65)
    print("实验完成")
    print("=" * 65)


if __name__ == "__main__":
    main()
