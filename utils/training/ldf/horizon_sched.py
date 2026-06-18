"""Horizon-mask sampling helpers for LDF training."""

from __future__ import annotations

import random as _random

from utils.token_frame import token_start_frame


def sample_random_horizon_tokens(
    global_step: int,
    total_steps: int,
    clip_tokens: int,
    cfg: dict,
    *,
    rng: _random.Random | None = None,
) -> int:
    """Sample a token horizon; integer bounds are inclusive."""
    rng_obj = rng or _random
    inference_horizon = int(cfg.get("inference_horizon_tokens", 20))
    p_exact = float(cfg.get("p_exact_inference_horizon", 0.5))
    warmup = float(cfg.get("warmup_ratio", 0.5))
    progress = (global_step / total_steps) if total_steps else 1.0

    if progress < warmup:
        lower = max(
            int(cfg.get("early_min_horizon_tokens", inference_horizon)),
            clip_tokens // 2,
        )
        upper = int(clip_tokens * float(cfg.get("early_max_horizon_ratio", 1.0)))
        upper = max(upper, lower)
        return rng_obj.randint(lower, upper)

    if rng_obj.random() < p_exact:
        return inference_horizon
    lower = int(cfg.get("late_min_horizon_tokens", 10))
    upper = int(cfg.get("late_max_horizon_tokens", inference_horizon))
    upper = max(upper, lower)
    return rng_obj.randint(lower, upper)


def apply_horizon_mask_tokens(
    traj_mask_frame,
    active_end_token,
    horizon_tokens: int,
    frames_per_token: int = 4,
):
    """Zero `traj_mask_frame` at/after the horizon cutoff frame in place."""
    import torch

    num_frames = traj_mask_frame.shape[-1]
    if torch.is_tensor(active_end_token) and active_end_token.dim() > 0:
        for batch_idx in range(active_end_token.shape[0]):
            cutoff = token_start_frame(
                int(active_end_token[batch_idx]) + horizon_tokens,
                frames_per_token,
            )
            if cutoff < num_frames:
                traj_mask_frame[batch_idx, cutoff:] = 0
        return traj_mask_frame
    cutoff_frame = token_start_frame(
        int(active_end_token) + horizon_tokens,
        frames_per_token,
    )
    if cutoff_frame < num_frames:
        traj_mask_frame[..., cutoff_frame:] = 0
    return traj_mask_frame


__all__ = [
    "sample_random_horizon_tokens",
    "apply_horizon_mask_tokens",
]
