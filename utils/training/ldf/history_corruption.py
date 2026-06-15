"""History-token corruption for streaming body training (T_B_03).

References:
- docs/design.md §2.1.3 (MotionBricks-style context corruption), §2.1.4
  (apply_prob / curriculum gate), §2.1.5 (SF rollout interaction).
- docs/TODO.md §T_B_03.

Core logic lives here as pure functions so it is fully unit-testable without
constructing the heavy Wan model. The model / SelfForcingTrainer call these:
  - `should_apply_corruption(global_step, total_steps, hc_cfg)` — top-level gate.
  - `apply_history_corruption(clean_feature, end_indices, ...)` — corrupts only
    the history region (left of the active window), returning a NEW tensor.

Corruption (per design §2.1.3): within the history region [0, ctx_end) of each
sample (ctx_end = active-window-right - chunk_size), pick a focus fraction
focus_ratio = cos(π/2 · u), u~U(0,1) (mean 2/π). Of the focus tokens,
alpha_mask are replaced by the learned `mask_emb`, alpha_noisy get additive
N(0, (noise_sigma_factor·z_std)²); the rest are left clean. The active window
(β>0 region) is never touched.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def should_apply_corruption(global_step: int, total_steps: int, hc_cfg: dict) -> bool:
    """Top-level gate for history corruption.

    - disabled → False.
    - apply_prob not None → Bernoulli(apply_prob) (overrides curriculum).
    - else curriculum: early/mid/late prob by training progress
      (global_step / total_steps split into thirds). Explicit
      curriculum.enabled=false disables it.
    """
    if not hc_cfg.get("enabled", False):
        return False
    apply_prob = hc_cfg.get("apply_prob", None)
    if apply_prob is not None:
        return float(torch.rand(())) < float(apply_prob)
    cur = hc_cfg.get("curriculum", {}) or {}
    if cur.get("enabled", True) is False:
        return False
    progress = (global_step / total_steps) if total_steps else 1.0
    if progress < 1.0 / 3.0:
        p = cur.get("early_prob", 0.2)
    elif progress < 2.0 / 3.0:
        p = cur.get("mid_prob", 0.5)
    else:
        p = cur.get("late_prob", 0.8)
    return float(torch.rand(())) < float(p)


def sample_focus_ratio(generator: torch.Generator | None = None) -> float:
    """focus_ratio = cos(π/2 · u), u ~ U(0, 1). E[focus_ratio] = 2/π ≈ 0.637."""
    u = torch.rand((), generator=generator)
    return float(torch.cos(0.5 * math.pi * u))


def apply_history_corruption(
    clean_feature: Tensor,        # [B, T, D]
    end_indices,                  # [B] active-window right boundary (exclusive-ish)
    *,
    mask_emb: Tensor,             # [D] learned replacement vector
    z_std: Tensor,                # [D] VAE latent per-channel std
    chunk_size: int,
    alpha_mask: float = 0.3,
    alpha_noisy: float = 0.3,
    noise_sigma_factor: float = 0.05,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Corrupt the history region of `clean_feature`; return a NEW tensor.

    Only `[0, ctx_end)` per sample is touched (ctx_end = end_indices[b] -
    chunk_size); the active window `[ctx_end, T)` is left identical. If
    ctx_end <= 0 the sample is untouched.
    """
    if clean_feature.dim() != 3:
        raise ValueError(f"clean_feature must be [B, T, D], got {tuple(clean_feature.shape)}")
    B, T, D = clean_feature.shape
    corrupted = clean_feature.clone()
    device = clean_feature.device
    dtype = clean_feature.dtype
    sigma = noise_sigma_factor * z_std.to(device=device, dtype=dtype)         # [D]
    mask_vec = mask_emb.to(device=device, dtype=dtype)                        # [D]

    for b in range(B):
        ctx_end = int(end_indices[b]) - chunk_size
        if ctx_end <= 0:
            continue
        N = ctx_end
        focus_ratio = sample_focus_ratio(generator=generator)
        n_focus = int(N * focus_ratio)
        n_mask = int(n_focus * alpha_mask)
        n_noisy = int(n_focus * alpha_noisy)
        if n_mask + n_noisy == 0:
            continue
        perm = torch.randperm(N, generator=generator, device=device)
        mask_idx = perm[:n_mask]
        noisy_idx = perm[n_mask : n_mask + n_noisy]
        if n_mask > 0:
            corrupted[b, mask_idx, :] = mask_vec
        if n_noisy > 0:
            noise = torch.randn(n_noisy, D, generator=generator, device=device) * sigma
            corrupted[b, noisy_idx, :] = clean_feature[b, noisy_idx, :] + noise

    return corrupted


__all__ = [
    "should_apply_corruption",
    "sample_focus_ratio",
    "apply_history_corruption",
]
