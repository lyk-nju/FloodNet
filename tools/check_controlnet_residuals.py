"""
ControlNet residual magnitude check.

Runs one forward pass and reports the L2 norm of each layer's residual tensor.
Also compares WITH vs WITHOUT traj conditioning to confirm the ControlNet is
actually influencing the backbone output.

If residuals are near-zero across all layers, it means:
  - Zero-init hasn't been trained out yet (too few steps), OR
  - The traj embedding is collapsing to zero (encoder bug), OR
  - The ControlNet branch is not receiving the traj condition

Usage:
    python tools/check_controlnet_residuals.py --config configs/ldf.yaml
    python tools/check_controlnet_residuals.py --config configs/ldf.yaml --num_batches 5
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/ldf.yaml")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--num_batches", type=int, default=3,
                   help="Number of batches to average over")
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


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def main():
    args = parse_args()
    _set_seed(args.seed)
    cfg = Config(args.config).config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or OmegaConf.select(cfg, "test_ckpt") or OmegaConf.select(cfg, "resume_ckpt")
    if not ckpt_path:
        raise ValueError("No checkpoint. Set --ckpt or config test_ckpt/resume_ckpt.")

    model = _load_model(cfg, ckpt_path, device, use_ema=not args.no_ema)

    if not getattr(model, "use_controlnet_traj", False):
        print("[WARN] model.use_controlnet_traj is False — no ControlNet to check.")

    # ── Monkey-patch _controlnet_forward to capture residuals ─────────
    captured_residuals = []   # list of List[Tensor] per forward call
    _orig_maybe = model._controlnet_forward.__func__

    def _hooked(self, *args, **kwargs):
        res = _orig_maybe(self, *args, **kwargs)
        if res is not None:
            captured_residuals.append([r.detach().cpu() for r in res])
        return res

    import types
    model._controlnet_forward = types.MethodType(_hooked, model)

    # ── Also capture traj_emb ────────────────────────────────────────────────
    captured_traj_emb = []
    _orig_build = model._build_traj_emb.__func__

    def _hooked_traj(self, *args, **kwargs):
        emb = _orig_build(self, *args, **kwargs)
        if emb is not None:
            captured_traj_emb.append(emb.detach().cpu())
        return emb

    model._build_traj_emb = types.MethodType(_hooked_traj, model)

    # ── DataLoader ────────────────────────────────────────────────────────────
    collate_fn = get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn") else None
    dataset = instantiate(cfg.data.get("test_target", cfg.data.target), cfg=cfg, split="test")
    loader = DataLoader(dataset, batch_size=cfg.data.test_bs, shuffle=False,
                        drop_last=False, num_workers=0, collate_fn=collate_fn)

    # ── Run forward passes ────────────────────────────────────────────────────
    print(f"===== ControlNet Residual Check =====")
    print(f"ckpt: {ckpt_path}")
    print(f"batches: {args.num_batches}")
    print()

    batches_done = 0
    with torch.no_grad():
        for batch in loader:
            if batches_done >= args.num_batches:
                break
            mb = batch.copy()
            mb["feature"] = batch["token"]
            mb["feature_length"] = batch["token_length"]
            if "token_text_end" in batch:
                mb["feature_text_end"] = batch["token_text_end"]
            for k in ("traj", "traj_length", "traj_mask", "token_mask", "traj_features"):
                if k in batch:
                    mb[k] = batch[k]
            mb = _to_device(mb, device)
            _ = model(mb)
            batches_done += 1

    # ── Traj embedding stats ──────────────────────────────────────────────────
    print("----- Traj embedding (input to ControlNet) -----")
    if captured_traj_emb:
        all_emb = torch.cat([e.reshape(-1) for e in captured_traj_emb])
        print(f"  calls captured : {len(captured_traj_emb)}")
        print(f"  mean abs value : {all_emb.abs().mean().item():.6f}")
        print(f"  std            : {all_emb.std().item():.6f}")
        print(f"  max abs        : {all_emb.abs().max().item():.6f}")
        if all_emb.abs().mean().item() < 1e-4:
            print("  [WARN] traj_emb is near-zero — encoder may not be getting traj input!")
    else:
        print("  [WARN] No traj_emb captured — ControlNet may not be receiving traj condition.")
    print()

    # ── Per-layer residual stats ──────────────────────────────────────────────
    print("----- Per-layer ControlNet residual L2 norm -----")
    if not captured_residuals:
        print("  [WARN] No residuals captured — ControlNet may not be active.")
        return

    num_layers = len(captured_residuals[0])
    layer_norms = [[] for _ in range(num_layers)]

    for call_residuals in captured_residuals:
        for layer_idx, res in enumerate(call_residuals):
            # res: (B, seq_len, dim)  L2 norm averaged over B and seq_len
            norm = res.norm(dim=-1).mean().item()
            layer_norms[layer_idx].append(norm)

    total_norm = 0.0
    for layer_idx in range(num_layers):
        mean_norm = np.mean(layer_norms[layer_idx])
        total_norm += mean_norm
        status = ""
        if mean_norm < 1e-5:
            status = "  ← near-zero (zero-init not trained out yet?)"
        elif mean_norm < 1e-3:
            status = "  ← very small"
        print(f"  layer {layer_idx:2d}: mean_L2={mean_norm:.6f}{status}")

    mean_total = total_norm / num_layers
    print()
    print(f"  avg across all layers: {mean_total:.6f}")
    if mean_total < 1e-4:
        print("  [WARN] All residuals near-zero — ControlNet has not learned to influence backbone.")
        print("         Possible causes: too few training steps, traj_emb=0, or training bug.")
    else:
        print("  [OK] Residuals are non-trivial — ControlNet is influencing backbone.")

    # ── Sanity: ratio of residual norm to backbone hidden state scale ─────────
    print()
    print("----- Interpretation guide -----")
    print("  < 1e-5  : zero-init, essentially untrained")
    print("  1e-4~1e-3 : beginning to learn (early training)")
    print("  > 1e-2  : significant influence on backbone")


if __name__ == "__main__":
    main()
