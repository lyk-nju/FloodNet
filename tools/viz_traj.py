"""
Trajectory visualization — predicted XZ vs ground-truth XZ.

For each sample generates motion:
  (A) WITH traj conditioning   (solid blue line)
  (B) WITHOUT traj conditioning (dashed orange line, ablation)
  (C) Ground-truth trajectory  (solid green line)

Saves one PNG per sample to --output_dir.

Usage:
    python tools/viz_traj.py --config configs/ldf.yaml --num_samples 6
    python tools/viz_traj.py --config configs/ldf.yaml --output_dir viz_out --no_ablation
"""

import argparse
import os
import sys

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/ldf.yaml")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--output_dir", type=str, default="tools/viz_output")
    p.add_argument("--no_ablation", action="store_true",
                   help="Skip the no-conditioning ablation run (faster)")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(cfg, ckpt_path, device, use_ema):
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    strict = not (
        cfg.model.params.get("use_traj_cond", False)
        or cfg.model.params.get("use_controlnet_traj", False)
    )
    model.load_state_dict(ckpt["state_dict"], strict=strict)
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
    ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(ckpt["state_dict"], strict=True)
    if "ema_state" in ckpt:
        vae_ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        vae_ema.load_state_dict(ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
    vae.to(device).eval()
    return vae


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _decode_latent(latent_tok, vae, device):
    """(T_token, z_dim) → (T_frame, 263)"""
    with torch.no_grad():
        decoded = vae.decode(latent_tok.unsqueeze(0).to(device))[0].float()
    return decoded  # (T_frame, 263)


def _extract_xz(motion_263, device):
    """(T, 263) → (T, 2) xz trajectory"""
    traj = extract_root_trajectory_263_torch(
        motion_263.unsqueeze(0).to(device)
    )  # (1, T, 3)
    return traj[0, :, [0, 2]].cpu().numpy()  # (T, 2)


def _remove_traj_fields(batch):
    """Return a copy with all traj conditioning fields removed."""
    no_traj = {}
    traj_keys = {"traj", "traj_length", "traj_mask", "traj_features",
                 "traj_features_length", "token_mask"}
    for k, v in batch.items():
        if k not in traj_keys:
            no_traj[k] = v
    return no_traj


def _plot_sample(
    name, gt_xz, pred_xz, pred_no_traj_xz, traj_mask,
    output_path, title_extra=""
):
    """Save a 2D XZ trajectory comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping visualization.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: XZ path overlay ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(gt_xz[:, 0], gt_xz[:, 1], color="green", lw=2, label="GT")
    ax.plot(pred_xz[:, 0], pred_xz[:, 1], color="steelblue", lw=1.5, label="Pred (w/ traj)")
    if pred_no_traj_xz is not None:
        ax.plot(pred_no_traj_xz[:, 0], pred_no_traj_xz[:, 1],
                color="darkorange", lw=1.5, ls="--", label="Pred (no traj)")

    # Mark start / end
    ax.scatter([gt_xz[0, 0]], [gt_xz[0, 1]], color="green", s=60, zorder=5)
    ax.scatter([pred_xz[0, 0]], [pred_xz[0, 1]], color="steelblue", s=60, zorder=5)

    # Highlight constrained frames (traj_mask == 1)
    if traj_mask is not None:
        mask = np.asarray(traj_mask, dtype=bool)
        T = min(len(mask), len(gt_xz))
        constrained = np.where(mask[:T])[0]
        if len(constrained) > 0:
            ax.scatter(gt_xz[constrained, 0], gt_xz[constrained, 1],
                       color="lime", s=20, zorder=4, label="GT constrained")

    ax.set_title(f"{name}\nXZ trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend(fontsize=8)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # ── Right: per-axis time series ──────────────────────────────────────────
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

    ax.set_title(f"X(t) time series  [Z dotted on right axis]")
    ax.set_xlabel("Frame")
    ax.set_ylabel("X")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{name}{title_extra}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    _set_seed(args.seed)
    cfg = Config(args.config).config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or OmegaConf.select(cfg, "test_ckpt") or OmegaConf.select(cfg, "resume_ckpt")
    if not ckpt_path:
        raise ValueError("No checkpoint. Set --ckpt or config test_ckpt/resume_ckpt.")

    os.makedirs(args.output_dir, exist_ok=True)

    model = _load_model(cfg, ckpt_path, device, use_ema=not args.no_ema)
    vae = _load_vae(cfg, device)

    collate_fn = get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn") else None
    dataset = instantiate(cfg.data.get("test_target", cfg.data.target), cfg=cfg, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        drop_last=False, num_workers=0, collate_fn=collate_fn)

    print(f"===== Trajectory Visualization =====")
    print(f"ckpt:       {ckpt_path}")
    print(f"output_dir: {args.output_dir}")
    print(f"samples:    {args.num_samples}")
    print()

    done = 0
    with torch.no_grad():
        for batch in loader:
            if done >= args.num_samples:
                break
            name = batch.get("name", [f"sample_{done:04d}"])[0]
            if "traj" not in batch:
                print(f"[SKIP] {name} — no traj in batch")
                continue

            # Build model batch
            def _build_mb(b):
                mb = b.copy()
                mb["feature"] = b["token"]
                mb["feature_length"] = b["token_length"]
                if "token_text_end" in b:
                    mb["feature_text_end"] = b["token_text_end"]
                return _to_device(mb, device)

            mb_with = _build_mb(batch)

            # ── Generate WITH traj conditioning ──────────────────────────────
            out_with = model.generate(mb_with)
            latent_with = out_with["generated"][0]  # (T_token, z_dim)
            motion_with = _decode_latent(latent_with, vae, device)
            pred_xz = _extract_xz(motion_with, device)

            # ── Generate WITHOUT traj conditioning (ablation) ─────────────────
            pred_no_traj_xz = None
            if not args.no_ablation:
                mb_no_traj = _build_mb(_remove_traj_fields(batch))
                out_no = model.generate(mb_no_traj)
                latent_no = out_no["generated"][0]
                motion_no = _decode_latent(latent_no, vae, device)
                pred_no_traj_xz = _extract_xz(motion_no, device)

            # ── Ground-truth trajectory ───────────────────────────────────────
            traj_gt = batch["traj"][0].numpy()          # (T, 3) xyz
            gt_xz = traj_gt[:, [0, 2]]                  # (T, 2)
            traj_mask = batch.get("traj_mask", [None])[0]
            if traj_mask is not None and torch.is_tensor(traj_mask):
                traj_mask = traj_mask.numpy()

            # ── Compute MSE for the title ─────────────────────────────────────
            T = min(len(gt_xz), len(pred_xz))
            mse = float(np.mean((gt_xz[:T] - pred_xz[:T]) ** 2))
            title_extra = f"  |  xz_MSE={mse:.4f}"
            if pred_no_traj_xz is not None:
                T2 = min(T, len(pred_no_traj_xz))
                mse_no = float(np.mean((gt_xz[:T2] - pred_no_traj_xz[:T2]) ** 2))
                title_extra += f"  no-traj_MSE={mse_no:.4f}"

            # ── Save plot ─────────────────────────────────────────────────────
            out_path = os.path.join(args.output_dir, f"{name}.png")
            _plot_sample(
                name, gt_xz, pred_xz, pred_no_traj_xz, traj_mask,
                out_path, title_extra=title_extra
            )
            print(f"  [{done+1}/{args.num_samples}] {name}  xz_MSE={mse:.4f}  → {out_path}")
            done += 1

    print()
    print(f"Done. Saved {done} plots to {args.output_dir}/")
    print()
    print("How to interpret:")
    print("  - Blue  (solid)  = predicted WITH traj conditioning")
    print("  - Orange (dashed)= predicted WITHOUT traj conditioning (ablation)")
    print("  - Green (solid)  = ground-truth trajectory")
    print("  - Lime dots      = constrained frames (traj_mask=1)")
    print()
    print("  If blue ≈ orange : ControlNet is NOT influencing the output")
    print("  If blue ≈ green  : ControlNet is working correctly")


if __name__ == "__main__":
    main()
