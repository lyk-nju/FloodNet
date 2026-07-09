"""History-token corruption helpers for self-forcing training."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def should_apply_corruption(global_step: int, total_steps: int, hc_cfg: dict) -> bool:
    """Return whether history corruption should be applied this step."""
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
        probability = cur.get("early_prob", 0.2)
    elif progress < 2.0 / 3.0:
        probability = cur.get("mid_prob", 0.5)
    else:
        probability = cur.get("late_prob", 0.8)
    return float(torch.rand(())) < float(probability)


def sample_focus_ratio(generator: torch.Generator | None = None) -> float:
    """Sample `cos(pi / 2 * u)` with `u ~ U(0, 1)`."""
    u = torch.rand((), generator=generator)
    return float(torch.cos(0.5 * math.pi * u))


def apply_history_corruption(
    clean_feature: Tensor,
    end_indices,
    *,
    mask_emb: Tensor,
    z_std: Tensor,
    chunk_size: int,
    alpha_mask: float = 0.3,
    alpha_noisy: float = 0.3,
    noise_sigma_factor: float = 0.05,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Corrupt the history region of `clean_feature` and return a new tensor."""
    if clean_feature.dim() != 3:
        raise ValueError(
            f"clean_feature must be [B, T, D], got {tuple(clean_feature.shape)}"
        )
    batch_size, _, latent_dim = clean_feature.shape
    corrupted = clean_feature.clone()
    device = clean_feature.device
    dtype = clean_feature.dtype
    sigma = noise_sigma_factor * z_std.to(device=device, dtype=dtype)
    mask_vector = mask_emb.to(device=device, dtype=dtype)

    for batch_idx in range(batch_size):
        history_end = int(end_indices[batch_idx]) - chunk_size
        if history_end <= 0:
            continue
        focus_ratio = sample_focus_ratio(generator=generator)
        focus_count = int(history_end * focus_ratio)
        mask_count = int(focus_count * alpha_mask)
        noisy_count = int(focus_count * alpha_noisy)
        if mask_count + noisy_count == 0:
            continue
        permutation = torch.randperm(history_end, generator=generator, device=device)
        mask_indices = permutation[:mask_count]
        noisy_indices = permutation[mask_count:mask_count + noisy_count]
        if mask_count > 0:
            corrupted[batch_idx, mask_indices, :] = mask_vector
        if noisy_count > 0:
            noise = (
                torch.randn(
                    noisy_count,
                    latent_dim,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                * sigma
            )
            corrupted[batch_idx, noisy_indices, :] = (
                clean_feature[batch_idx, noisy_indices, :] + noise
            ).to(dtype=dtype)

    return corrupted


__all__ = [
    "should_apply_corruption",
    "sample_focus_ratio",
    "apply_history_corruption",
]
