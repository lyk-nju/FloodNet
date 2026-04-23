"""
eval_generation_metrics.py
==========================
Two-pass evaluation:

  Pass 1 — test_meta_paths (small set, e.g. test_min.txt)
    • Autoregressive generation + video rendering
    • Trajectory control metrics: ADE, FDE, MSE (masked XZ)
    • Segment MSE & prefix MSE (error accumulation over time)
    • [--forward_control_loss]  training-equivalent active-window XZ loss
    • [--traj_ablation]         generate WITHOUT traj conditioning → ablation ADE/FDE/MSE
    • [--viz_traj]              save 2D XZ trajectory comparison plots (PNG)
    • [--topk N]                print N hardest samples by ADE

  Pass 2 — val_meta_paths (large set, e.g. val.txt)  [only when --t2m_metric]
    • Autoregressive generation (no video)
    • T2M FID / R-Precision / Diversity

Usage:
    python tools/eval_generation_metrics.py --config configs/ldf.yaml
    python tools/eval_generation_metrics.py --config configs/ldf.yaml --t2m_metric
    python tools/eval_generation_metrics.py --config configs/ldf.yaml \\
        --forward_control_loss --traj_ablation --viz_traj --topk 3
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from metrics.t2m import T2MMetrics
from utils.initialize import Config, get_function, instantiate, load_config, compare_statedict_and_parameters
from utils.motion_process import extract_root_trajectory_263_torch
from utils.traj_batch import root_to_traj_feats
from utils.visualize import make_composite_compare_videos, render_video


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="FloodNet generation evaluation: video + traj metrics + optional T2M FID."
    )
    parser.add_argument("--config", type=str, default="configs/ldf.yaml")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path. Falls back to cfg.test_ckpt / cfg.resume_ckpt.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override test batch size.")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_ema", action="store_true", help="Skip EMA weight application.")
    parser.add_argument("--seed", type=int, default=1234, help="Global random seed.")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", default=[],
                        help="OmegaConf dot-path overrides, e.g. --set model.params.cfg_scale_text=3.0")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="If >0, limit test pass to first N batches.")
    # T2M flag
    parser.add_argument("--t2m_metric", action="store_true",
                        help="Run T2M FID/R-Precision on val_meta_paths (slow, needs large val set).")
    # Segment MSE
    parser.add_argument("--seg_size", type=int, default=20,
                        help="Frame window size for segment / prefix MSE (default 20).")
    # Forward control loss (training-equivalent active-window XZ loss)
    parser.add_argument("--forward_control_loss", action="store_true",
                        help="Run model() forward pass → active-window XZ control loss (train-equivalent).")
    # Trajectory ablation (generate WITHOUT traj conditioning)
    parser.add_argument("--traj_ablation", action="store_true",
                        help="Generate without traj conditioning to measure ControlNet effectiveness.")
    # XZ trajectory visualization plots
    parser.add_argument("--viz_traj", action="store_true",
                        help="Save 2D XZ trajectory comparison PNG per sample.")
    # Top-k hardest samples
    parser.add_argument("--topk", type=int, default=0,
                        help="If >0, print the N hardest samples by ADE at the end.")
    # Multiple generation runs for stable metrics
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of generation runs per sample; metrics are averaged across runs.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Root directory for eval outputs (default: eval/ next to this script).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _to_device(obj: Any, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _copy_traj_fields(batch: Dict, model_batch: Dict):
    for key in ("traj", "traj_length", "traj_mask", "token_mask", "traj_features"):
        if key in batch:
            model_batch[key] = batch[key]


def _remove_traj_fields(batch: Dict) -> Dict:
    """Return a shallow copy with all traj conditioning fields removed (for ablation)."""
    traj_keys = {"traj", "traj_length", "traj_mask", "traj_features",
                 "traj_features_length", "token_mask"}
    return {k: v for k, v in batch.items() if k not in traj_keys}


def _load_model(cfg, ckpt_path: str, device: torch.device, use_ema: bool):
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_keys = set(checkpoint["state_dict"].keys())
    has_controlnet = any(k.startswith("controlnet.") for k in ckpt_keys)
    strict = has_controlnet  # base FloodDiffusion ckpt has no controlnet.* keys → strict=False
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=strict)

    if not strict:
        if load_result.missing_keys:
            print(f"[load] strict=False  missing keys (new modules): {len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"[load] strict=False  unexpected keys ignored: {len(load_result.unexpected_keys)}")
        if (getattr(model, "controlnet", None) is not None
                and bool(getattr(model, "controlnet_init_from_backbone", True))
                and load_result.missing_keys):
            model.controlnet.init_from_backbone(model.model)
            print("[load] Re-initialized ControlNet from loaded backbone weights.")

    if use_ema and "ema_state" in checkpoint:
        ema = ExponentialMovingAverage(
            [p for p in model.parameters() if p.requires_grad],
            decay=cfg.model.ema_decay,
        )
        try:
            ema.load_state_dict(checkpoint["ema_state"])
            ema.copy_to([p for p in model.parameters() if p.requires_grad])
            print("[load] Applied EMA weights.")
        except ValueError as e:
            print(f"[load] EMA incompatible, skip. Detail: {e}")

    model.to(device).eval()
    return model


def _load_vae(cfg, device: torch.device):
    vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False, **cfg.test_vae.params)
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
    if "ema_state" in vae_ckpt:
        vae_ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
    vae.to(device).eval()
    return vae


def _build_model_batch(batch: Dict, device: torch.device) -> Dict:
    mb = batch.copy()
    mb["feature"] = batch["token"]
    mb["feature_length"] = batch["token_length"]
    if "token_text_end" in batch:
        mb["feature_text_end"] = batch["token_text_end"]
    _copy_traj_fields(batch, mb)
    return _to_device(mb, device)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_traj_metrics(
    decoded_generated: torch.Tensor,   # (T_pred, 263)
    batch: Dict,
    sample_idx: int,
    seg_size: int,
) -> Dict:
    """Return dict with ADE, FDE, MSE, seg_mse list, prefix_mse list."""
    with torch.no_grad():
        pred_traj_xyz = extract_root_trajectory_263_torch(
            decoded_generated[None, :]
        )[0]                                                    # (T_pred, 3)
        pred_xz = pred_traj_xyz[:, [0, 2]].cpu()               # (T_pred, 2)

    traj_len = (int(batch["traj_length"][sample_idx].item())
                if "traj_length" in batch else batch["traj"][sample_idx].shape[0])
    gt_xz  = batch["traj"][sample_idx][:traj_len, [0, 2]].float().cpu()   # (T_gt, 2)
    mask   = batch["traj_mask"][sample_idx][:traj_len].float().cpu()       # (T_gt,)

    T = min(pred_xz.shape[0], gt_xz.shape[0])
    pred_xz, gt_xz, mask = pred_xz[:T], gt_xz[:T], mask[:T]
    n_masked = mask.sum().item()

    result: Dict = {"T": T, "masked_ratio": n_masked / max(T, 1)}

    if n_masked > 0:
        diff   = pred_xz - gt_xz                            # (T, 2)
        l2_t   = diff.norm(dim=-1)                          # (T,)
        sq_t   = (diff ** 2).sum(dim=-1)                    # (T,)
        result["ade"] = float((mask * l2_t).sum().item() / n_masked)
        result["mse"] = float((mask * sq_t).sum().item() / n_masked)
        last_idx = mask.nonzero(as_tuple=False)[-1].item()
        result["fde"] = float(l2_t[last_idx].item())
    else:
        result["ade"] = result["fde"] = result["mse"] = float("nan")

    # ── Segment MSE: non-overlapping windows of seg_size frames ──────────────
    seg_mse: List[Optional[float]] = []
    n_segs = (T + seg_size - 1) // seg_size
    for s in range(n_segs):
        sf, ef = s * seg_size, min((s + 1) * seg_size, T)
        m_s = mask[sf:ef]
        n_s = m_s.sum().item()
        if n_s > 0:
            sq = ((pred_xz[sf:ef] - gt_xz[sf:ef]) ** 2).sum(dim=-1)
            seg_mse.append(float((m_s * sq).sum().item() / n_s))
        else:
            seg_mse.append(None)
    result["seg_mse"] = seg_mse

    # ── Trajectory smoothness: mean squared acceleration (jitter) ─────────────
    # accel[t] = xz[t+1] - 2*xz[t] + xz[t-1]; lower = smoother
    if pred_xz.shape[0] >= 3:
        accel = pred_xz[2:] - 2 * pred_xz[1:-1] + pred_xz[:-2]  # (T-2, 2)
        result["traj_jitter"] = float(accel.pow(2).sum(dim=-1).mean().item())
    else:
        result["traj_jitter"] = float("nan")

    # ── Prefix MSE: cumulative [0:ef] at ef = seg_size, 2*seg_size, ... ──────
    prefix_mse: List[Optional[float]] = []
    for ef in range(seg_size, T + 1, seg_size):
        m_p = mask[:ef]
        n_p = m_p.sum().item()
        if n_p > 0:
            sq = ((pred_xz[:ef] - gt_xz[:ef]) ** 2).sum(dim=-1)
            prefix_mse.append(float((m_p * sq).sum().item() / n_p))
        else:
            prefix_mse.append(None)
    result["prefix_mse"] = prefix_mse

    return result


def _average_traj_metrics(run_metrics: List[Dict]) -> Dict:
    """Average _compute_traj_metrics dicts across multiple runs."""
    if len(run_metrics) == 1:
        return run_metrics[0].copy()
    result: Dict = {}
    if "T" in run_metrics[0]:
        result["T"] = run_metrics[0]["T"]
    if "masked_ratio" in run_metrics[0]:
        result["masked_ratio"] = run_metrics[0]["masked_ratio"]
    for key in ("ade", "fde", "mse", "traj_jitter"):
        vals = [r[key] for r in run_metrics if key in r and r[key] == r[key]]
        if vals:
            result[key]             = float(np.mean(vals))
            result[f"{key}_std"]    = float(np.std(vals))
    for list_key in ("seg_mse", "prefix_mse"):
        if list_key not in run_metrics[0]:
            continue
        n = max(len(r.get(list_key, [])) for r in run_metrics)
        avg = []
        for s in range(n):
            vals = [r[list_key][s] for r in run_metrics
                    if s < len(r.get(list_key, [])) and r[list_key][s] is not None]
            avg.append(float(np.mean(vals)) if vals else None)
        result[list_key] = avg
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Forward control loss (training-equivalent active-window XZ loss)
# From eval_control_loss.py — runs model() forward pass, not generate()
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fwd_ctrl_loss_per_sample(
    pred_list: List[torch.Tensor],
    traj: torch.Tensor,
    traj_mask: torch.Tensor,
    traj_length: torch.Tensor,
    vae,
    device: torch.device,
    chunk_size_tokens: Optional[int] = None,
    token_to_frame: int = 4,
) -> List[Dict]:
    """Active-window XZ control loss per sample, matching train_ldf forward logic."""
    out = []
    for i in range(len(pred_list)):
        pred_latent_full = pred_list[i].to(device)
        t_tok = pred_latent_full.size(0)

        if chunk_size_tokens is not None and t_tok > chunk_size_tokens:
            start_tok = t_tok - chunk_size_tokens
            start_f = 0 if start_tok == 0 else 4 * start_tok - 3
            end_f = t_tok * token_to_frame
        else:
            start_f = 0
            end_f = None

        decoded = vae.decode(pred_latent_full.unsqueeze(0))[0].float()
        l_motion = decoded.size(0)
        l_gt_total = min(int(traj_length[i].item()), traj.shape[1])

        if end_f is None:
            pred_sl = slice(0, l_motion)
            gt_sl   = slice(0, l_gt_total)
        else:
            pred_sl = slice(min(start_f, l_motion), min(end_f, l_motion))
            gt_sl   = slice(min(start_f, l_gt_total), min(end_f, l_gt_total))

        l = min(pred_sl.stop - pred_sl.start, gt_sl.stop - gt_sl.start)
        if l <= 0:
            out.append({"loss": float("nan"), "n_valid": 0, "window_len": 0})
            continue

        pred_traj_full = extract_root_trajectory_263_torch(decoded.unsqueeze(0))
        pred_traj = pred_traj_full[:, pred_sl, :][:, :l, :]
        gt_traj   = traj[i, gt_sl, :][:l].unsqueeze(0).to(pred_traj.device, dtype=pred_traj.dtype)
        mask      = traj_mask[i, gt_sl][:l].unsqueeze(0).to(pred_traj.device, dtype=pred_traj.dtype)

        pred_xz = pred_traj[..., [0, 2]]
        gt_xz   = gt_traj[..., [0, 2]]
        sq_err  = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
        n_valid = float(mask.sum().item())
        loss_val = float((mask * sq_err).sum().item() / n_valid) if n_valid > 0 else float("nan")

        out.append({"loss": loss_val, "n_valid": n_valid, "window_len": l})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# XZ trajectory visualization  (from viz_traj.py)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_traj_xz(
    name: str,
    gt_xz: np.ndarray,
    pred_xz: np.ndarray,
    pred_no_traj_xz: Optional[np.ndarray],
    traj_mask: Optional[np.ndarray],
    output_path: str,
    title_extra: str = "",
):
    """Save 2-panel XZ trajectory comparison PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(gt_xz[:, 0], gt_xz[:, 1], color="green", lw=2, label="GT")
    ax.plot(pred_xz[:, 0], pred_xz[:, 1], color="steelblue", lw=1.5, label="Pred (w/ traj)")
    if pred_no_traj_xz is not None:
        ax.plot(pred_no_traj_xz[:, 0], pred_no_traj_xz[:, 1],
                color="darkorange", lw=1.5, ls="--", label="Pred (no traj)")
    ax.scatter([gt_xz[0, 0]], [gt_xz[0, 1]], color="green", s=60, zorder=5)
    ax.scatter([pred_xz[0, 0]], [pred_xz[0, 1]], color="steelblue", s=60, zorder=5)
    if traj_mask is not None:
        mask_bool = np.asarray(traj_mask, dtype=bool)
        T = min(len(mask_bool), len(gt_xz))
        constrained = np.where(mask_bool[:T])[0]
        if len(constrained) > 0:
            ax.scatter(gt_xz[constrained, 0], gt_xz[constrained, 1],
                       color="lime", s=20, zorder=4, label="GT constrained")
    ax.set_title(f"{name}\nXZ trajectory")
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.legend(fontsize=8); ax.axis("equal"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    T = min(len(gt_xz), len(pred_xz))
    t = np.arange(T)
    ax.plot(t, gt_xz[:T, 0], color="green", lw=1.5, label="GT X")
    ax.plot(t, pred_xz[:T, 0], color="steelblue", lw=1.2, label="Pred X")
    if pred_no_traj_xz is not None:
        Tn = min(T, len(pred_no_traj_xz))
        ax.plot(t[:Tn], pred_no_traj_xz[:Tn, 0], color="darkorange",
                lw=1.2, ls="--", label="No-traj X")
    ax2 = ax.twinx()
    ax2.plot(t, gt_xz[:T, 1], color="green", lw=1.5, alpha=0.5, ls=":")
    ax2.plot(t, pred_xz[:T, 1], color="steelblue", lw=1.2, alpha=0.5, ls=":")
    ax2.set_ylabel("Z (dotted, right axis)", color="gray", fontsize=8)
    ax.set_title("X(t) time series  [Z dotted on right axis]")
    ax.set_xlabel("Frame"); ax.set_ylabel("X")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"{name}{title_extra}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass config_path directly to avoid load_config()'s internal argparse
    # conflicting with this script's argparse.
    override_args = {}
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}")
        key, value = item.split("=", 1)
        override_args[key.strip()] = value.strip()
    cfg = load_config(config_path=args.config, override_args=override_args if override_args else None)

    if args.batch_size is not None:
        cfg.config.data.test_bs = args.batch_size
    if args.num_workers is not None:
        cfg.config.data.num_workers = args.num_workers

    # Resolve whether to run T2M (CLI flag OR yaml key)
    run_t2m = args.t2m_metric or bool(cfg.config.get("t2m_metric", False))

    ckpt_path = args.ckpt or getattr(cfg, "test_ckpt", None) or getattr(cfg, "resume_ckpt", None)
    if ckpt_path is None:
        raise ValueError("No checkpoint provided via --ckpt / cfg.test_ckpt / cfg.resume_ckpt")

    save_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent
    run_name = f"eval_{cfg.exp_name}_seed{args.seed}"
    out_root  = save_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[eval] ckpt     : {ckpt_path}")
    print(f"[eval] out_root : {out_root}")
    print(f"[eval] seed     : {args.seed}")
    print(f"[eval] t2m_metric: {run_t2m}")

    model = _load_model(cfg, ckpt_path, device=device, use_ema=not args.no_ema)
    vae   = _load_vae(cfg, device=device)
    chunk_size_tokens = getattr(model, "chunk_size", None)

    collate_fn = get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn") else None

    # ── Test dataset (video gen + traj metrics) ───────────────────────────────
    test_dataset = instantiate(
        cfg.data.get("test_target", cfg.data.target), cfg=cfg.config, split="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.test_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        collate_fn=collate_fn,
    )

    # ── Val dataset (T2M FID, only when run_t2m) ──────────────────────────────
    if run_t2m:
        t2m_metrics = T2MMetrics(cfg.metrics.t2m)
        val_dataset = instantiate(
            cfg.data.get("val_target", cfg.data.target), cfg=cfg.config, split="val"
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.val_bs,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.data.num_workers,
            persistent_workers=False,
            collate_fn=collate_fn,
        )
        print(f"[eval] val samples (for T2M FID): {len(val_dataset)}")

    print(f"[eval] test samples (video + traj): {len(test_dataset)}")
    print(f"[eval] num_runs   : {args.num_runs}")
    if args.forward_control_loss:
        print(f"[eval] forward_control_loss: ON  (chunk_size={chunk_size_tokens})")
    if args.traj_ablation:
        print(f"[eval] traj_ablation: ON")
    if args.viz_traj:
        print(f"[eval] viz_traj: ON  (PNG per sample)")

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1: test set — video generation + trajectory metrics
    # ══════════════════════════════════════════════════════════════════════════
    traj_records: List[Dict] = []

    for bidx, batch in enumerate(test_loader):
        if args.max_batches > 0 and bidx >= args.max_batches:
            break

        batch = _to_device(batch, device)
        batch_size_actual = len(batch["name"])

        # Accumulate per-sample metrics across runs
        sample_run_metrics: List[List[Dict]] = [[] for _ in range(batch_size_actual)]
        text_cache: Optional[List[str]] = None
        # Saved on run 0 for files/rendering/viz
        decoded_run0:  List[Optional[torch.Tensor]]  = [None] * batch_size_actual
        pred_xz_run0:  List[Optional[np.ndarray]]    = [None] * batch_size_actual
        abl_xz_run0:   List[Optional[np.ndarray]]    = [None] * batch_size_actual
        fwd_stats_batch: Optional[List[Dict]] = None
        ablation_generated_run0: Optional[List] = None

        # Resolve directories once (outside run loop)
        sample_root_cache: Dict[int, Any] = {}
        for i in range(batch_size_actual):
            dataset_id = batch["dataset"][i]
            sample_root = out_root / dataset_id
            dirs = {
                "text":         sample_root / "text",
                "token":        sample_root / "token",
                "feature":      sample_root / "feature",
                "cond_traj":    sample_root / "traj_xz",
                "pred_traj":    sample_root / "pred_traj_xz",
                "traj_mask":    sample_root / "traj_mask",
                "frames":       sample_root / "frames",
                "video":        sample_root / "video",
                "composite":    sample_root / "composite",
                "viz":          sample_root / "traj_viz",
            }
            for k, d in dirs.items():
                if k != "viz":
                    d.mkdir(parents=True, exist_ok=True)
            sample_root_cache[i] = dirs

        for run_idx in range(args.num_runs):
            _set_seed(args.seed + run_idx * 100000 + bidx)

            with torch.no_grad():
                model_batch = _build_model_batch(batch, device)
                output = model.generate(model_batch)

            generated = output["generated"]
            if text_cache is None:
                text_cache = output.get("text", [""] * len(generated))

            # fwd_ctrl_loss and ablation: run once on run 0
            if run_idx == 0:
                if args.forward_control_loss and "traj" in batch:
                    try:
                        with torch.no_grad():
                            fwd_out = model(model_batch)
                        if "control_aux" in fwd_out:
                            pred_list = fwd_out["control_aux"]["pred_x0_latent_list"]
                            fwd_stats_batch = _compute_fwd_ctrl_loss_per_sample(
                                pred_list=pred_list,
                                traj=model_batch["traj"],
                                traj_mask=model_batch["traj_mask"],
                                traj_length=model_batch["traj_length"],
                                vae=vae,
                                device=device,
                                chunk_size_tokens=chunk_size_tokens,
                            )
                    except Exception as e:
                        print(f"[fwd_ctrl_loss] bidx={bidx:05d} failed: {e}")

                if args.traj_ablation:
                    try:
                        with torch.no_grad():
                            mb_no_traj = _build_model_batch(_remove_traj_fields(batch), device)
                            out_no = model.generate(mb_no_traj)
                        ablation_generated_run0 = out_no["generated"]
                    except Exception as e:
                        print(f"[traj_ablation] bidx={bidx:05d} failed: {e}")

            for i in range(len(generated)):
                sample_name = batch["name"][i]
                dirs = sample_root_cache[i]

                single_generated  = generated[i].detach()
                decoded_generated = vae.decode(single_generated[None, :].to(device))[0].float().detach().to(device)

                # Files: save only on run 0
                if run_idx == 0:
                    decoded_run0[i] = decoded_generated
                    sample_text = text_cache[i] if text_cache and i < len(text_cache) else ""
                    (dirs["text"] / f"{sample_name}.txt").write_text(sample_text)
                    np.save(dirs["token"]   / f"{sample_name}.npy", single_generated.float().cpu().numpy())
                    np.save(dirs["feature"] / f"{sample_name}.npy", decoded_generated.float().cpu().numpy())

                    if "traj_features" in batch:
                        cond = batch["traj_features"][i]
                        if torch.is_tensor(cond): cond = cond.detach().cpu().numpy()
                        cond = np.asarray(cond)
                        if cond.ndim == 2 and cond.shape[1] >= 2:
                            np.save(dirs["cond_traj"] / f"{sample_name}.npy", cond[:, :2].astype(np.float32))
                    elif "traj" in batch:
                        tr = batch["traj"][i]
                        if torch.is_tensor(tr): tr = tr.detach().cpu().numpy()
                        tr = np.asarray(tr)
                        if tr.ndim == 2 and tr.shape[1] >= 3:
                            np.save(dirs["cond_traj"] / f"{sample_name}.npy",
                                    root_to_traj_feats(tr)[:, :2].astype(np.float32))
                    if "traj_mask" in batch:
                        m = batch["traj_mask"][i]
                        if torch.is_tensor(m): m = m.detach().cpu().numpy()
                        np.save(dirs["traj_mask"] / f"{sample_name}.npy", np.asarray(m).reshape(-1))
                    if "feature_text_end" in batch:
                        frames = batch["feature_text_end"][i]
                        if torch.is_tensor(frames): frames = frames.detach().cpu().numpy()
                        np.save(dirs["frames"] / f"{sample_name}.npy", np.asarray(frames))

                    with torch.no_grad():
                        pred_xz_np = extract_root_trajectory_263_torch(
                            decoded_generated[None, :]
                        )[0, :, [0, 2]].cpu().numpy().astype(np.float32)
                    pred_xz_run0[i] = pred_xz_np
                    np.save(dirs["pred_traj"] / f"{sample_name}.npy", pred_xz_np)

                # Traj metrics: every run
                if "traj" in batch and "traj_mask" in batch:
                    rec = _compute_traj_metrics(decoded_generated, batch, i, seg_size=args.seg_size)
                    sample_run_metrics[i].append(rec)

        # ── Ablation XZ (run 0) ───────────────────────────────────────────────
        if ablation_generated_run0 is not None:
            for i in range(len(ablation_generated_run0)):
                abl_single  = ablation_generated_run0[i].detach()
                abl_decoded = vae.decode(abl_single[None, :].to(device))[0].float().detach().to(device)
                with torch.no_grad():
                    abl_xz_run0[i] = extract_root_trajectory_263_torch(
                        abl_decoded[None, :]
                    )[0, :, [0, 2]].cpu().numpy().astype(np.float32)

        # ── Build final per-sample record (averaged across runs) ──────────────
        feature_dir = sample_root_cache[0]["feature"] if batch_size_actual > 0 else None
        video_dir   = sample_root_cache[0]["video"]   if batch_size_actual > 0 else None

        for i in range(batch_size_actual):
            sample_name = batch["name"][i]
            dirs = sample_root_cache[i]
            feature_dir = dirs["feature"]
            video_dir   = dirs["video"]
            composite_dir = dirs["composite"]

            rec: Dict = {"name": sample_name, "num_runs": args.num_runs}
            if sample_run_metrics[i]:
                rec.update(_average_traj_metrics(sample_run_metrics[i]))

                # Console: per-sample summary
                seg_str = "  ".join(
                    f"[{s*args.seg_size}:{(s+1)*args.seg_size}]={v:.4f}"
                    if v is not None else f"[{s*args.seg_size}:-]=nan"
                    for s, v in enumerate(rec.get("seg_mse", []))
                )
                ade_std_str = (f"±{rec['ade_std']:.4f}" if "ade_std" in rec else "")
                print(f"  {sample_name}  ADE={rec.get('ade', float('nan')):.4f}{ade_std_str}"
                      f"  FDE={rec.get('fde', float('nan')):.4f}"
                      f"  MSE={rec.get('mse', float('nan')):.4f}  T={rec.get('T', 0)}"
                      f"  (runs={args.num_runs})")
                if seg_str:
                    print(f"    seg_mse : {seg_str}")
                pfx = [f"{(s+1)*args.seg_size}f={v:.4f}" if v is not None else f"{(s+1)*args.seg_size}f=nan"
                       for s, v in enumerate(rec.get("prefix_mse", []))]
                if pfx:
                    print(f"    pfx_mse : {'  '.join(pfx)}")

            # fwd_ctrl_loss (run 0)
            if fwd_stats_batch is not None and i < len(fwd_stats_batch):
                fwd = fwd_stats_batch[i]
                rec["fwd_ctrl_loss"] = fwd["loss"]
                rec["fwd_n_valid"]   = fwd["n_valid"]
                rec["fwd_win_len"]   = fwd["window_len"]
                print(f"    fwd_ctrl_loss={fwd['loss']:.6f}  n_valid={fwd['n_valid']:.0f}"
                      f"  win={fwd['window_len']}")

            # ablation (run 0)
            if ablation_generated_run0 is not None and i < len(ablation_generated_run0):
                abl_single  = ablation_generated_run0[i].detach()
                abl_decoded = vae.decode(abl_single[None, :].to(device))[0].float().detach().to(device)
                if "traj" in batch and "traj_mask" in batch:
                    abl_rec = _compute_traj_metrics(abl_decoded, batch, i, seg_size=args.seg_size)
                    rec["ablation_ade"] = abl_rec.get("ade", float("nan"))
                    rec["ablation_fde"] = abl_rec.get("fde", float("nan"))
                    rec["ablation_mse"] = abl_rec.get("mse", float("nan"))
                    print(f"    ablation: ADE={rec['ablation_ade']:.4f}"
                          f"  FDE={rec['ablation_fde']:.4f}"
                          f"  [ControlNet Δ ADE={rec.get('ade', float('nan')) - rec['ablation_ade']:+.4f}]")

            # viz_traj (uses run 0 decoded)
            if args.viz_traj and "traj" in batch and pred_xz_run0[i] is not None:
                dirs["viz"].mkdir(parents=True, exist_ok=True)
                traj_len_v = (int(batch["traj_length"][i].item())
                              if "traj_length" in batch else batch["traj"][i].shape[0])
                gt_xz_plot = batch["traj"][i][:traj_len_v, [0, 2]].float().cpu().numpy()
                msk_plot   = batch["traj_mask"][i].cpu().numpy() if "traj_mask" in batch else None
                title_extra = f"  ADE={rec.get('ade', float('nan')):.4f}"
                if "ade_std" in rec:
                    title_extra += f"±{rec['ade_std']:.4f}"
                if "ablation_ade" in rec:
                    title_extra += f"  abl_ADE={rec['ablation_ade']:.4f}"
                _plot_traj_xz(
                    name=sample_name,
                    gt_xz=gt_xz_plot,
                    pred_xz=pred_xz_run0[i],
                    pred_no_traj_xz=abl_xz_run0[i],
                    traj_mask=msk_plot,
                    output_path=str(dirs["viz"] / f"{sample_name}.png"),
                    title_extra=title_extra,
                )

            traj_records.append(rec)

        # Render videos for this batch (uses files from run 0)
        if cfg.test_setting.render:
            try:
                render_video(
                    motion_dir=str(feature_dir),
                    save_dir=str(video_dir),
                    render_setting=cfg.test_setting,
                    frames_dir=str(dirs["frames"]),
                    traj_mask_dir=str(dirs["traj_mask"]),
                    cond_traj_dir=str(dirs["cond_traj"]),
                )
                make_composite_compare_videos(
                    result_folder=str(video_dir),
                    compare_folders=(cfg.test_setting.get(test_dataset[0]["dataset"], {})
                                     .get("compare_folders", None)
                                     if hasattr(test_dataset[0], "__getitem__") else None),
                    compare_names=(cfg.test_setting.get(test_dataset[0]["dataset"], {})
                                   .get("compare_names", None)
                                   if hasattr(test_dataset[0], "__getitem__") else None),
                    text_folder=str(text_dir),
                    save_dir=str(composite_dir),
                )
            except Exception as e:
                print(f"[render] bidx={bidx:05d} render failed (skipping): {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2: val set — T2M FID / R-Precision / Diversity (optional)
    # ══════════════════════════════════════════════════════════════════════════
    t2m_results: Dict = {}
    if run_t2m:
        print(f"\n[eval] Running T2M metrics on val set ({len(val_dataset)} samples)...")
        _set_seed(args.seed)  # reset seed for reproducibility of val pass

        for bidx, batch in enumerate(val_loader):
            _set_seed(args.seed + 10000 + bidx)  # distinct seed space from test pass
            batch = _to_device(batch, device)

            with torch.no_grad():
                model_batch = _build_model_batch(batch, device)
                output = model.generate(model_batch)

            generated = output["generated"]

            for i in range(len(generated)):
                single_generated  = generated[i].detach()
                decoded_generated = vae.decode(single_generated[None, :].to(device))[0].float().detach().to(device)

                gt_token   = batch["token"][i][: batch["token_length"][i]]
                gt_decoded = vae.decode(gt_token[None, :].to(device))[0].float().detach().to(device)
                gt_feature = batch["feature"][i][: batch["feature_length"][i]].float().to(device)

                text_tokens_single = batch["text_tokens"][i]
                if cfg.metrics.t2m.fid_target == "vae":
                    t2m_metrics.update(
                        feats_rst=decoded_generated[None, ...],
                        feats_ref=gt_decoded[None, ...],
                        lengths_rst=[int(decoded_generated.shape[0])],
                        lengths_ref=[int(gt_decoded.shape[0])],
                        text_tokens=[text_tokens_single],
                    )
                else:
                    t2m_metrics.update(
                        feats_rst=decoded_generated[None, ...],
                        feats_ref=gt_feature[None, ...],
                        lengths_rst=[int(decoded_generated.shape[0])],
                        lengths_ref=[int(gt_feature.shape[0])],
                        text_tokens=[text_tokens_single],
                    )

        t2m_results = t2m_metrics.compute(sanity_flag=False)
        t2m_results = {k: (v.item() if hasattr(v, "item") else v)
                       for k, v in t2m_results.items()}

    # ══════════════════════════════════════════════════════════════════════════
    # Aggregate & save
    # ══════════════════════════════════════════════════════════════════════════
    final_metrics: Dict = {}
    final_metrics.update(t2m_results)

    valid_traj = [r for r in traj_records if "ade" in r and r["ade"] == r["ade"]]

    if valid_traj:
        ades = [r["ade"] for r in valid_traj]
        fdes = [r["fde"] for r in valid_traj]
        mses = [r["mse"] for r in valid_traj]
        final_metrics.update({
            "traj/ADE_mean":  float(np.mean(ades)),
            "traj/ADE_std":   float(np.std(ades)),
            "traj/FDE_mean":  float(np.mean(fdes)),
            "traj/FDE_std":   float(np.std(fdes)),
            "traj/MSE_mean":  float(np.mean(mses)),
            "traj/MSE_std":   float(np.std(mses)),
            "traj/n_samples": len(valid_traj),
        })

        # Segment MSE aggregation: mean per segment slot across samples
        max_segs = max(len(r.get("seg_mse", [])) for r in valid_traj)
        seg_means = []
        for s in range(max_segs):
            vals = [r["seg_mse"][s] for r in valid_traj
                    if s < len(r.get("seg_mse", [])) and r["seg_mse"][s] is not None]
            seg_means.append(float(np.mean(vals)) if vals else None)
        final_metrics["traj/seg_mse_per_slot"] = seg_means

        # Prefix MSE aggregation
        max_pfx = max(len(r.get("prefix_mse", [])) for r in valid_traj)
        pfx_means = []
        for s in range(max_pfx):
            vals = [r["prefix_mse"][s] for r in valid_traj
                    if s < len(r.get("prefix_mse", [])) and r["prefix_mse"][s] is not None]
            pfx_means.append(float(np.mean(vals)) if vals else None)
        final_metrics["traj/prefix_mse_per_slot"] = pfx_means

        # Trajectory smoothness aggregation
        jitter_vals = [r["traj_jitter"] for r in valid_traj
                       if "traj_jitter" in r and r["traj_jitter"] == r["traj_jitter"]]
        if jitter_vals:
            final_metrics["traj/jitter_mean"] = float(np.mean(jitter_vals))
            final_metrics["traj/jitter_std"]  = float(np.std(jitter_vals))

        # Forward control loss aggregation
        fwd_vals = [r["fwd_ctrl_loss"] for r in valid_traj
                    if "fwd_ctrl_loss" in r and r["fwd_ctrl_loss"] == r["fwd_ctrl_loss"]]
        if fwd_vals:
            final_metrics["traj/fwd_ctrl_loss_mean"] = float(np.mean(fwd_vals))
            final_metrics["traj/fwd_ctrl_loss_std"]  = float(np.std(fwd_vals))

        # Ablation aggregation
        abl_ades = [r["ablation_ade"] for r in valid_traj
                    if "ablation_ade" in r and r["ablation_ade"] == r["ablation_ade"]]
        if abl_ades:
            abl_fdes = [r["ablation_fde"] for r in valid_traj if "ablation_fde" in r]
            abl_mses = [r["ablation_mse"] for r in valid_traj if "ablation_mse" in r]
            final_metrics["traj/ablation_ADE_mean"] = float(np.mean(abl_ades))
            final_metrics["traj/ablation_FDE_mean"] = float(np.mean(abl_fdes))
            final_metrics["traj/ablation_MSE_mean"] = float(np.mean(abl_mses))
            # Controllability: ADE improvement due to traj conditioning
            delta_ades = [r["ade"] - r["ablation_ade"] for r in valid_traj
                          if "ablation_ade" in r and r["ade"] == r["ade"]]
            final_metrics["traj/ctrl_delta_ADE_mean"] = float(np.mean(delta_ades))

    # Build output: per-sample entries first, then aggregate summary
    output_dict: Dict = {}

    # Per-sample section
    for r in traj_records:
        name = r["name"]
        entry: Dict = {"T": r.get("T", 0)}
        for k in ("ade", "fde", "mse", "traj_jitter",
                  "fwd_ctrl_loss", "ablation_ade", "ablation_fde", "ablation_mse"):
            if k in r:
                entry[k] = r[k]
        if "seg_mse" in r:
            entry["seg_mse"] = {
                f"[{s*args.seg_size}:{(s+1)*args.seg_size}]": v
                for s, v in enumerate(r["seg_mse"])
            }
        if "prefix_mse" in r:
            entry["prefix_mse"] = {
                f"[0:{(s+1)*args.seg_size}]": v
                for s, v in enumerate(r["prefix_mse"])
            }
        output_dict[name] = entry

    # Aggregate summary at the end
    output_dict["summary"] = final_metrics

    metrics_path = out_root / "metrics.json"
    metrics_path.write_text(json.dumps(output_dict, indent=2))

    # Keep traj_per_sample.json for backward compatibility (full raw records)
    per_sample_path = out_root / "traj_per_sample.json"
    per_sample_path.write_text(json.dumps(traj_records, indent=2))

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  seed          : {args.seed}")
    print(f"  test samples  : {len(test_dataset)}")
    if run_t2m:
        print(f"  val samples   : {len(val_dataset)}")
    print(f"  metrics saved : {metrics_path}")
    print()
    for k, v in final_metrics.items():
        if isinstance(v, list):
            vals_str = "  ".join(
                f"[{i*args.seg_size}:{(i+1)*args.seg_size}]={x:.4f}"
                if x is not None else f"[{i*args.seg_size}:-]=nan"
                for i, x in enumerate(v)
            )
            print(f"  {k}: {vals_str}")
        else:
            print(f"  {k}: {v}")

    # ── Top-k hardest samples by ADE ─────────────────────────────────────────
    if args.topk > 0 and valid_traj:
        sorted_recs = sorted(valid_traj, key=lambda r: r["ade"], reverse=True)
        top_n = sorted_recs[: args.topk]
        print(f"\n  Top-{args.topk} hardest samples (by ADE):")
        for r in top_n:
            fwd_str = f"  fwd_loss={r['fwd_ctrl_loss']:.6f}" if "fwd_ctrl_loss" in r else ""
            abl_str = f"  abl_ADE={r['ablation_ade']:.4f}" if "ablation_ade" in r else ""
            print(f"    {r['name']}  ADE={r['ade']:.4f}  FDE={r['fde']:.4f}{fwd_str}{abl_str}")


if __name__ == "__main__":
    main()
