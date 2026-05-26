"""
eval_generation_metrics.py
==========================
Two-pass evaluation:

  Pass 1 — test_meta_paths (small set, e.g. test_min.txt)
    • Autoregressive generation + video rendering
    • Trajectory control metrics: ADE, FDE, MSE (masked XZ)
    • OmniControl / MotionLCM-compatible control metrics:
      Control L2 dist, Skating Ratio, traj_fail/kps_fail
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
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from metrics.t2m import T2MMetrics
from metrics.traj import (
    _average_control_metrics,
    _average_traj_metrics,
    _build_model_batch,
    _compute_deterministic_fwd_ctrl_loss_sample,
    _compute_omni_control_metrics,
    _compute_traj_metrics,
    _get_metric_statistics,
    _slice_single_sample_batch,
    _stable_eval_seed,
    _to_device,
)
from utils.initialize import get_function, instantiate, load_config
from utils.motion_process import extract_root_trajectory_263_torch
from utils.traj_batch import root_to_traj_feats
from utils.training.ckpt_compat import strip_legacy_traj_encoder_weights
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
    # Parallel T2M sharding
    parser.add_argument("--skip_test_pass", action="store_true",
                        help="Skip Pass 1 (test set video/traj) and only run the T2M val pass.")
    parser.add_argument("--val_num_shards", type=int, default=1,
                        help="Total number of shards for parallel T2M evaluation.")
    parser.add_argument("--val_shard_idx", type=int, default=0,
                        help="Index of this shard (0 .. val_num_shards-1).")
    parser.add_argument("--t2m_shards_save_dir", type=str, default=None,
                        help="Save per-shard embeddings (.npz) here instead of computing FID.")
    parser.add_argument("--t2m_merge_dir", type=str, default=None,
                        help="Load t2m_shard_*.npz files from this dir, merge, compute FID, and exit.")
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
    parser.add_argument("--probe_tag", type=str, default=None,
                        help="Optional probe label used in output path / summaries, e.g. train or test.")
    parser.add_argument("--meta_paths", nargs="+", default=None,
                        help="Override cfg.data.test_meta_paths for this evaluation run.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    torch.random.set_rng_state(gen.get_state())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _infer_meta_tag(meta_paths) -> str:
    if not meta_paths:
        return "meta"
    first = str(meta_paths[0])
    stem = Path(first).stem
    if stem.endswith("_min"):
        stem = stem[:-4]
    return stem or "meta"


def _remove_traj_fields(batch: Dict) -> Dict:
    """Return a shallow copy with ALL traj conditioning fields removed.

    Must include both legacy 4D keys (traj / traj_features / traj_mask) and the
    7D pipeline keys (traj_cond / traj_cond_7d / traj_cond_mask /
    traj_loss_mask) — otherwise prepare_model_input would still route the 7D
    cond into model_batch and the ablation wouldn't actually disable
    ControlNet.
    """
    traj_keys = {
        "traj", "traj_length", "traj_mask", "traj_features",
        "traj_features_length", "token_mask",
        # 7D pipeline keys consumed by utils.training.model_batch:
        "traj_cond", "traj_cond_7d", "traj_cond_mask", "traj_loss_mask",
    }
    return {k: v for k, v in batch.items() if k not in traj_keys}


def _load_model(cfg, ckpt_path: str, device: torch.device, use_ema: bool):
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 7D traj-encoder rewrite (matches train_ldf.py:244): legacy 4D ckpts must
    # have their traj-encoder / traj_in_proj weights stripped before load_state_dict
    # — otherwise either (a) shape mismatch on local_traj_encoder.conv1
    # (4-channel→7-channel input), or (b) `traj_in_proj.bias` shape happens
    # to match (1024,) and gets restored, breaking the zero-init contract.
    n_traj_exp = strip_legacy_traj_encoder_weights(
        checkpoint["state_dict"], model.state_dict()
    )
    if n_traj_exp:
        print(f"[load] stripped {n_traj_exp} legacy traj-encoder weights "
              "(7D encoder rewrite — eval will use new traj encoder from scratch)")

    ckpt_keys = set(checkpoint["state_dict"].keys())
    has_controlnet = any(k.startswith("controlnet.") for k in ckpt_keys)
    # strict=False when EITHER (a) the ckpt has no ControlNet (base pretrain
    # warm-start path) OR (b) we just stripped legacy traj-encoder weights.
    strict = has_controlnet and n_traj_exp == 0
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=strict)

    if not strict:
        if load_result.missing_keys:
            print(f"[load] strict=False  missing keys (new modules): {len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"[load] strict=False  unexpected keys ignored: {len(load_result.unexpected_keys)}")
        # Re-init ControlNet from backbone ONLY when the ckpt had no ControlNet
        # at all (base pretrain warm-start). Gating on "any controlnet.* key is
        # missing" is WRONG: stripping legacy traj weights puts
        # controlnet.traj_in_proj.* into missing_keys even for a fully-trained
        # ControlNet, which would clobber it with a raw backbone copy. Mirror
        # train_ldf.py's `controlnet_missing` gate.
        controlnet_missing = not has_controlnet
        if (getattr(model, "controlnet", None) is not None
                and bool(getattr(model, "controlnet_init_from_backbone", True))
                and controlnet_missing
                and any("controlnet." in k for k in load_result.missing_keys)):
            model.controlnet.init_from_backbone(model.model)
            print("[load] Re-initialized ControlNet from loaded backbone weights.")

    # EMA: with stripped traj weights or fresh ControlNet, the saved EMA shadow
    # has wrong param count / shape — skip rather than crash.
    if use_ema and "ema_state" in checkpoint and n_traj_exp == 0:
        ema = ExponentialMovingAverage(
            [p for p in model.parameters() if p.requires_grad],
            decay=cfg.model.ema_decay,
        )
        try:
            ema.load_state_dict(checkpoint["ema_state"])
            ema.copy_to([p for p in model.parameters() if p.requires_grad])
            print("[load] Applied EMA weights.")
        except (ValueError, RuntimeError) as e:
            print(f"[load] EMA incompatible, skip. Detail: {e}")
    elif use_ema and n_traj_exp > 0:
        print("[load] Skipping EMA: traj-encoder weights stripped (legacy 4D), "
              "saved EMA shadow has wrong param count for new 7D traj params.")

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


# ─────────────────────────────────────────────────────────────────────────────
# XZ trajectory visualization
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

def _save_t2m_shard(t2m_metrics, args) -> None:
    shard_dir = Path(args.t2m_shards_save_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"t2m_shard_{args.val_shard_idx}_of_{args.val_num_shards}.npz"

    rec = np.concatenate([e.numpy() for e in t2m_metrics.recmotion_embeddings], axis=0)
    gt  = np.concatenate([e.numpy() for e in t2m_metrics.gtmotion_embeddings],  axis=0)
    txt_list = t2m_metrics.text_embeddings
    txt = (np.concatenate([e.numpy() for e in txt_list], axis=0)
           if txt_list else np.zeros((0, rec.shape[-1]), dtype=np.float32))

    np.savez(shard_path, recmotion=rec, gtmotion=gt, text=txt)
    print(f"[eval] Saved shard {args.val_shard_idx}/{args.val_num_shards} "
          f"({len(rec)} samples) → {shard_path}")


def _run_t2m_merge(args, cfg) -> None:
    from pathlib import Path as _Path
    merge_dir = _Path(args.t2m_merge_dir)
    shard_files = sorted(merge_dir.glob("t2m_shard_*_of_*.npz"))
    if not shard_files:
        raise ValueError(f"No t2m_shard_*.npz files found in {merge_dir}")

    print(f"[merge] Found {len(shard_files)} shard files in {merge_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t2m_metrics = T2MMetrics(cfg.metrics.t2m)
    t2m_metrics.to(device)

    for sf in shard_files:
        data = np.load(sf)
        rec_arr = data["recmotion"]   # (N, D)
        gt_arr  = data["gtmotion"]
        txt_arr = data["text"]        # (N, D) or (0, D)
        n = rec_arr.shape[0]
        for i in range(n):
            t2m_metrics.recmotion_embeddings.append(torch.from_numpy(rec_arr[i : i + 1]))
            t2m_metrics.gtmotion_embeddings.append(torch.from_numpy(gt_arr[i : i + 1]))
        if t2m_metrics.evaluate_text and txt_arr.shape[0] > 0:
            for i in range(txt_arr.shape[0]):
                t2m_metrics.text_embeddings.append(torch.from_numpy(txt_arr[i : i + 1]))
        print(f"  loaded {sf.name}: {n} samples")

    total = len(t2m_metrics.recmotion_embeddings)
    print(f"[merge] Total samples: {total}")

    t2m_results = t2m_metrics.compute(sanity_flag=False)
    t2m_results = {k: (v.item() if hasattr(v, "item") else v) for k, v in t2m_results.items()}

    results_path = merge_dir / "t2m_results.json"
    results_path.write_text(json.dumps(t2m_results, indent=2))

    print("\n" + "=" * 60)
    print("T2M Metrics (merged from shards)")
    print("=" * 60)
    for k, v in t2m_results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved → {results_path}")


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

    if args.meta_paths is not None:
        cfg.config.data.test_meta_paths = list(args.meta_paths)

    if args.batch_size is not None:
        cfg.config.data.test_bs = args.batch_size
    if args.num_workers is not None:
        cfg.config.data.num_workers = args.num_workers

    # Resolve whether to run T2M (CLI flag OR yaml key)
    run_t2m = args.t2m_metric or bool(cfg.config.get("t2m_metric", False))

    # ── Merge mode: load shard embeddings → compute FID, then exit ────────────
    if args.t2m_merge_dir:
        _run_t2m_merge(args, cfg)
        return

    ckpt_path = args.ckpt or getattr(cfg, "test_ckpt", None) or getattr(cfg, "resume_ckpt", None)
    if ckpt_path is None:
        raise ValueError("No checkpoint provided via --ckpt / cfg.test_ckpt / cfg.resume_ckpt")

    save_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent
    test_meta_paths = cfg.data.get("test_meta_paths", [])
    probe_tag = args.probe_tag or _infer_meta_tag(test_meta_paths)
    meta_tag = _infer_meta_tag(test_meta_paths)
    run_name = f"eval_{cfg.exp_name}_{probe_tag}_{meta_tag}_seed{args.seed}"
    out_root  = save_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[eval] ckpt     : {ckpt_path}")
    print(f"[eval] out_root : {out_root}")
    print(f"[eval] seed     : {args.seed}")
    print(f"[eval] probe_tag : {probe_tag}")
    print(f"[eval] meta_tag : {meta_tag}")
    print(f"[eval] t2m_metric: {run_t2m}")

    model = _load_model(cfg, ckpt_path, device=device, use_ema=not args.no_ema)
    vae   = _load_vae(cfg, device=device)
    chunk_size_tokens = getattr(model, "chunk_size", None)
    control_loss_train_mode = int(cfg.get("control_loss_train_mode", 3))
    val_cfg = cfg.get("validation", {})
    fwd_ctrl_window_mode = str(val_cfg.get("eval_forward_control_loss_window_mode", "mean_chunk_windows"))

    collate_fn = get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn") else None

    # ── Test dataset (video gen + traj metrics) ───────────────────────────────
    if not args.skip_test_pass:
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
        t2m_metrics = T2MMetrics(cfg.metrics.t2m).to(device)
        val_dataset = instantiate(
            cfg.data.get("val_target", cfg.data.target), cfg=cfg.config, split="val"
        )
        if args.val_num_shards > 1:
            indices = list(range(args.val_shard_idx, len(val_dataset), args.val_num_shards))
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
            print(f"[eval] val shard {args.val_shard_idx}/{args.val_num_shards}: {len(val_dataset)} samples")
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

    if not args.skip_test_pass:
        print(f"[eval] test samples (video + traj): {len(test_dataset)}")
    print(f"[eval] num_runs   : {args.num_runs}")
    if args.forward_control_loss:
        print(
            f"[eval] forward_control_loss: ON  "
            f"(chunk_size={chunk_size_tokens}, mode={control_loss_train_mode}, window_mode={fwd_ctrl_window_mode})"
        )
    if args.traj_ablation:
        print(f"[eval] traj_ablation: ON")
    if args.viz_traj:
        print(f"[eval] viz_traj: ON  (PNG per sample)")

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1: test set — video generation + trajectory metrics
    # ══════════════════════════════════════════════════════════════════════════
    traj_records: List[Dict] = []
    control_run_records: List[List[Dict]] = [[] for _ in range(args.num_runs)]
    dataset_ids_seen = set()

    if args.skip_test_pass:
        print("[eval] Skipping Pass 1 (test set) as requested.")
    for bidx, batch in enumerate([] if args.skip_test_pass else test_loader):
        if args.max_batches > 0 and bidx >= args.max_batches:
            break

        batch_size_actual = len(batch["name"])
        for i in range(batch_size_actual):
            sample_batch = _slice_single_sample_batch(batch, i)
            sample_name = sample_batch["name"][0]
            dataset_id = sample_batch["dataset"][0]
            dataset_ids_seen.add(dataset_id)
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
            for key, out_dir in dirs.items():
                if key != "viz":
                    out_dir.mkdir(parents=True, exist_ok=True)

            sample_run_metrics: List[Dict] = []
            sample_control_run_metrics: List[Dict] = []
            sample_ablation_run_metrics: List[Dict] = []
            fwd_stat: Optional[Dict] = None
            pred_xz_run0: Optional[np.ndarray] = None
            abl_xz_run0: Optional[np.ndarray] = None

            if args.forward_control_loss and "traj" in sample_batch:
                try:
                    fwd_stat = _compute_deterministic_fwd_ctrl_loss_sample(
                        model=model,
                        sample_batch=sample_batch,
                        vae=vae,
                        device=device,
                        train_mode=control_loss_train_mode,
                        chunk_size_tokens=chunk_size_tokens,
                        window_mode=fwd_ctrl_window_mode,
                    )
                except Exception as e:
                    print(f"[fwd_ctrl_loss] sample={sample_name} deterministic eval failed: {e}")

            for run_idx in range(args.num_runs):
                _set_seed(_stable_eval_seed(args.seed, probe_tag, sample_name, run_idx))

                with torch.no_grad():
                    model_batch = _build_model_batch(sample_batch, device)
                    output = model.generate(model_batch)

                single_generated = output["generated"][0].detach()
                decoded_generated = vae.decode(single_generated[None, :].to(device))[0].float().detach().to(device)

                if run_idx == 0:
                    sample_text = output.get("text", [""])[0]
                    (dirs["text"] / f"{sample_name}.txt").write_text(sample_text)
                    np.save(dirs["token"] / f"{sample_name}.npy", single_generated.float().cpu().numpy())
                    np.save(dirs["feature"] / f"{sample_name}.npy", decoded_generated.float().cpu().numpy())

                    if "traj_features" in sample_batch:
                        cond = sample_batch["traj_features"][0]
                        if torch.is_tensor(cond):
                            cond = cond.detach().cpu().numpy()
                        cond = np.asarray(cond)
                        if cond.ndim == 2 and cond.shape[1] >= 2:
                            np.save(dirs["cond_traj"] / f"{sample_name}.npy", cond[:, :2].astype(np.float32))
                    elif "traj" in sample_batch:
                        tr = sample_batch["traj"][0]
                        if torch.is_tensor(tr):
                            tr = tr.detach().cpu().numpy()
                        tr = np.asarray(tr)
                        if tr.ndim == 2 and tr.shape[1] >= 3:
                            np.save(
                                dirs["cond_traj"] / f"{sample_name}.npy",
                                root_to_traj_feats(tr)[:, :2].astype(np.float32),
                            )
                    if "traj_mask" in sample_batch:
                        m = sample_batch["traj_mask"][0]
                        if torch.is_tensor(m):
                            m = m.detach().cpu().numpy()
                        np.save(dirs["traj_mask"] / f"{sample_name}.npy", np.asarray(m).reshape(-1))
                    if "feature_text_end" in sample_batch:
                        frames = sample_batch["feature_text_end"][0]
                        if torch.is_tensor(frames):
                            frames = frames.detach().cpu().numpy()
                        np.save(dirs["frames"] / f"{sample_name}.npy", np.asarray(frames))

                    with torch.no_grad():
                        pred_xz_run0 = extract_root_trajectory_263_torch(
                            decoded_generated[None, :]
                        )[0, :, [0, 2]].cpu().numpy().astype(np.float32)
                    np.save(dirs["pred_traj"] / f"{sample_name}.npy", pred_xz_run0)

                if args.traj_ablation:
                    try:
                        with torch.no_grad():
                            mb_no_traj = _build_model_batch(_remove_traj_fields(sample_batch), device)
                            out_no = model.generate(mb_no_traj)
                        abl_single = out_no["generated"][0].detach()
                        abl_decoded = vae.decode(abl_single[None, :].to(device))[0].float().detach().to(device)

                        if run_idx == 0:
                            with torch.no_grad():
                                abl_xz_run0 = extract_root_trajectory_263_torch(
                                    abl_decoded[None, :]
                                )[0, :, [0, 2]].cpu().numpy().astype(np.float32)

                        if "traj" in sample_batch and "traj_mask" in sample_batch:
                            sample_ablation_run_metrics.append(
                                _compute_traj_metrics(
                                    abl_decoded, sample_batch, 0, seg_size=args.seg_size
                                )
                            )
                    except Exception as e:
                        print(f"[traj_ablation] sample={sample_name} run={run_idx} failed: {e}")

                if "traj" in sample_batch and "traj_mask" in sample_batch:
                    rec = _compute_traj_metrics(decoded_generated, sample_batch, 0, seg_size=args.seg_size)
                    sample_run_metrics.append(rec)
                    control_rec = _compute_omni_control_metrics(decoded_generated, sample_batch, 0)
                    sample_control_run_metrics.append(control_rec)
                    control_run_records[run_idx].append(control_rec)

            rec: Dict = {"name": sample_name, "num_runs": args.num_runs}
            if sample_run_metrics:
                rec.update(_average_traj_metrics(sample_run_metrics))

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

            if sample_control_run_metrics:
                rec.update(_average_control_metrics(sample_control_run_metrics))
                print(
                    "    control : "
                    f"L2={rec.get('control_l2_dist', float('nan')):.4f}  "
                    f"skate={rec.get('skating_ratio', float('nan')):.4f}  "
                    f"traj_fail20={rec.get('traj_fail_20cm', float('nan')):.4f}  "
                    f"traj_fail50={rec.get('traj_fail_50cm', float('nan')):.4f}"
                )

            # fwd_ctrl_loss averaged across runs
            if fwd_stat is not None:
                rec["fwd_ctrl_loss"] = fwd_stat.get("loss", float("nan"))
                rec["fwd_ctrl_loss_std"] = fwd_stat.get("loss_std", float("nan"))
                rec["fwd_n_valid"] = fwd_stat.get("n_valid", float("nan"))
                rec["fwd_win_len"] = fwd_stat.get("window_len", float("nan"))
                rec["fwd_num_windows"] = fwd_stat.get("num_windows", 0)
                print(f"    fwd_ctrl_loss={rec.get('fwd_ctrl_loss', float('nan')):.6f}"
                      f" ±{rec.get('fwd_ctrl_loss_std', float('nan')):.6f}"
                      f"  n_valid={rec.get('fwd_n_valid', float('nan')):.0f}"
                      f"  win={rec.get('fwd_win_len', float('nan')):.0f}"
                      f"  windows={rec.get('fwd_num_windows', 0)}")

            if sample_ablation_run_metrics:
                abl_avg = _average_traj_metrics(sample_ablation_run_metrics)
                rec["ablation_ade"] = abl_avg.get("ade", float("nan"))
                rec["ablation_fde"] = abl_avg.get("fde", float("nan"))
                rec["ablation_mse"] = abl_avg.get("mse", float("nan"))
                if "ade_std" in abl_avg:
                    rec["ablation_ade_std"] = abl_avg["ade_std"]
                if "fde_std" in abl_avg:
                    rec["ablation_fde_std"] = abl_avg["fde_std"]
                if "mse_std" in abl_avg:
                    rec["ablation_mse_std"] = abl_avg["mse_std"]
                abl_ade_std_str = (
                    f"±{rec['ablation_ade_std']:.4f}" if "ablation_ade_std" in rec else ""
                )
                print(f"    ablation: ADE={rec['ablation_ade']:.4f}{abl_ade_std_str}"
                      f"  FDE={rec['ablation_fde']:.4f}"
                      f"  [ControlNet Δ ADE={rec.get('ade', float('nan')) - rec['ablation_ade']:+.4f}]")

            # viz_traj (uses run 0 decoded)
            if args.viz_traj and "traj" in sample_batch and pred_xz_run0 is not None:
                dirs["viz"].mkdir(parents=True, exist_ok=True)
                traj_len_v = (int(sample_batch["traj_length"][0].item())
                              if "traj_length" in sample_batch else sample_batch["traj"][0].shape[0])
                gt_xz_plot = sample_batch["traj"][0][:traj_len_v, [0, 2]].float().cpu().numpy()
                msk_plot   = sample_batch["traj_mask"][0].cpu().numpy() if "traj_mask" in sample_batch else None
                title_extra = f"  ADE={rec.get('ade', float('nan')):.4f}"
                if "ade_std" in rec:
                    title_extra += f"±{rec['ade_std']:.4f}"
                if "ablation_ade" in rec:
                    title_extra += f"  abl_ADE={rec['ablation_ade']:.4f}"
                _plot_traj_xz(
                    name=sample_name,
                    gt_xz=gt_xz_plot,
                    pred_xz=pred_xz_run0,
                    pred_no_traj_xz=abl_xz_run0,
                    traj_mask=msk_plot,
                    output_path=str(dirs["viz"] / f"{sample_name}.png"),
                    title_extra=title_extra,
                )

            traj_records.append(rec)

    # Render videos once per dataset after all sample files are written.
    if cfg.test_setting.render:
        for dataset_id in sorted(dataset_ids_seen):
            sample_root = out_root / dataset_id
            dirs = {
                "text":         sample_root / "text",
                "feature":      sample_root / "feature",
                "cond_traj":    sample_root / "traj_xz",
                "traj_mask":    sample_root / "traj_mask",
                "video":        sample_root / "video",
                "composite":    sample_root / "composite",
                "frames":       sample_root / "frames",
            }
            if not dirs["feature"].exists():
                continue
            try:
                render_video(
                    motion_dir=str(dirs["feature"]),
                    save_dir=str(dirs["video"]),
                    render_setting=cfg.test_setting,
                    frames_dir=str(dirs["frames"]),
                    traj_mask_dir=str(dirs["traj_mask"]),
                    cond_traj_dir=str(dirs["cond_traj"]),
                )
                make_composite_compare_videos(
                    result_folder=str(dirs["video"]),
                    compare_folders=cfg.test_setting.get(dataset_id, {}).get("compare_folders", None),
                    compare_names=cfg.test_setting.get(dataset_id, {}).get("compare_names", None),
                    text_folder=str(dirs["text"]),
                    save_dir=str(dirs["composite"]),
                )
            except Exception as e:
                print(f"[render] dataset={dataset_id} failed (skipping): {e}")

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

        if args.t2m_shards_save_dir:
            _save_t2m_shard(t2m_metrics, args)
        else:
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

        path_arc_ades = [r["path_arc_ade"] for r in valid_traj
                         if "path_arc_ade" in r and r["path_arc_ade"] == r["path_arc_ade"]]
        if path_arc_ades:
            final_metrics["path/arc_ADE_mean"] = float(np.mean(path_arc_ades))
            final_metrics["path/arc_ADE_std"] = float(np.std(path_arc_ades))
        path_chamfers = [r["path_chamfer"] for r in valid_traj
                         if "path_chamfer" in r and r["path_chamfer"] == r["path_chamfer"]]
        if path_chamfers:
            final_metrics["path/chamfer_mean"] = float(np.mean(path_chamfers))
            final_metrics["path/chamfer_std"] = float(np.std(path_chamfers))

        # Forward control loss aggregation
        fwd_vals = [r["fwd_ctrl_loss"] for r in valid_traj
                    if "fwd_ctrl_loss" in r and r["fwd_ctrl_loss"] == r["fwd_ctrl_loss"]]
        if fwd_vals:
            final_metrics["traj/fwd_ctrl_loss_mean"] = float(np.mean(fwd_vals))
            final_metrics["traj/fwd_ctrl_loss_std"]  = float(np.std(fwd_vals))
            fwd_run_std_vals = [r["fwd_ctrl_loss_std"] for r in valid_traj
                                if "fwd_ctrl_loss_std" in r and r["fwd_ctrl_loss_std"] == r["fwd_ctrl_loss_std"]]
            if fwd_run_std_vals:
                final_metrics["traj/fwd_ctrl_loss_run_std_mean"] = float(np.mean(fwd_run_std_vals))

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

    control_metric_keys = (
        "control_l2_dist",
        "skating_ratio",
        "traj_fail_20cm",
        "traj_fail_50cm",
        "kps_fail_20cm",
        "kps_fail_50cm",
        "kps_mean_err_m",
    )
    control_name_map = {
        "control_l2_dist": "Control_L2_dist",
        "skating_ratio": "Skating_Ratio",
        "traj_fail_20cm": "traj_fail_20cm",
        "traj_fail_50cm": "traj_fail_50cm",
        "kps_fail_20cm": "kps_fail_20cm",
        "kps_fail_50cm": "kps_fail_50cm",
        "kps_mean_err_m": "kps_mean_err_m",
    }
    for key in control_metric_keys:
        per_run_vals = []
        for run_recs in control_run_records:
            vals = [r[key] for r in run_recs if key in r and r[key] == r[key]]
            if vals:
                per_run_vals.append(float(np.mean(vals)))
        if per_run_vals:
            mean, std, conf = _get_metric_statistics(np.asarray(per_run_vals, dtype=np.float64), len(per_run_vals))
            out_key = control_name_map[key]
            final_metrics[f"control/{out_key}_mean"] = float(mean)
            final_metrics[f"control/{out_key}_std"] = float(std)
            final_metrics[f"control/{out_key}_conf_interval"] = float(conf)
            final_metrics[f"control/{out_key}_num_runs"] = int(len(per_run_vals))

    # Build output: per-sample entries first, then aggregate summary
    output_dict: Dict = {}

    # Per-sample section
    for r in traj_records:
        name = r["name"]
        entry: Dict = {"T": r.get("T", 0)}
        for k, v in r.items():
            if k in {"name", "T", "seg_mse", "prefix_mse"} or k.startswith("_"):
                continue
            if isinstance(v, dict) or isinstance(v, list):
                continue
            entry[k] = v
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
    output_dict["probe_tag"] = probe_tag

    metrics_path = out_root / "metrics.json"
    metrics_path.write_text(json.dumps(output_dict, indent=2))

    # Keep traj_per_sample.json for backward compatibility (full raw records)
    per_sample_path = out_root / "traj_per_sample.json"
    per_sample_path.write_text(json.dumps(traj_records, indent=2))

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  seed          : {args.seed}")
    if not args.skip_test_pass:
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
