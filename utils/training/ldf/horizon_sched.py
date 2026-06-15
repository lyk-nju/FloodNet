"""Random horizon simulation for streaming body training (T_B_04).

References:
- docs/design.md §2.2.2 (sampling), §2.2.3 (config), §2.2.4 (interaction).
- docs/TODO.md §T_B_04.

Training sees (near-)full-clip traj_cond; inference uses a fixed
`horizon_tokens ≈ 20`. To close that gap we randomly truncate how many future
tokens the traj_cond mask exposes during training, ramping toward the exact
inference horizon late in training.

Pure functions (fully unit-testable; the model/training caller supplies
global_step / total_steps / clip_tokens / active_end_token):
  - `sample_random_horizon_tokens(...)` — schedule (early wide, late exact-ish).
  - `apply_horizon_mask_tokens(...)` — zero the frame mask at/after the cutoff
    frame `token_start_frame(active_end_token + horizon_tokens)` (NOT (E+H)*4,
    which would wrongly keep the first 3 frames of the cutoff token).

⚠ Units are token-level throughout (causal VAE frames_per_token=4).
"""

from __future__ import annotations

import random as _random

from utils.token_frame import token_start_frame


def sample_random_horizon_tokens(global_step: int,
                                 total_steps: int,
                                 clip_tokens: int,
                                 cfg: dict,
                                 *,
                                 rng: _random.Random | None = None) -> int:
    """Sample a horizon (in tokens) per design §2.2.2.

    Early (progress < warmup_ratio): U[max(early_min, clip_tokens//2),
        clip_tokens * early_max_horizon_ratio]  (see close to a full plan).
    Late: with p_exact_inference_horizon return inference_horizon_tokens
        exactly; else U[late_min, late_max].

    Bounds are inclusive (matches np.random.randint(lo, hi+1)). cfg keys default
    to the §2.2.3 values.
    """
    r = rng or _random
    inf_h = int(cfg.get("inference_horizon_tokens", 20))
    p_exact = float(cfg.get("p_exact_inference_horizon", 0.5))
    warmup = float(cfg.get("warmup_ratio", 0.5))
    progress = (global_step / total_steps) if total_steps else 1.0

    if progress < warmup:
        lower = max(int(cfg.get("early_min_horizon_tokens", inf_h)), clip_tokens // 2)
        upper = int(clip_tokens * float(cfg.get("early_max_horizon_ratio", 1.0)))
        upper = max(upper, lower)
        return r.randint(lower, upper)

    if r.random() < p_exact:
        return inf_h
    lo = int(cfg.get("late_min_horizon_tokens", 10))
    hi = int(cfg.get("late_max_horizon_tokens", inf_h))
    hi = max(hi, lo)
    return r.randint(lo, hi)


def apply_horizon_mask_tokens(traj_mask_frame,
                              active_end_token,
                              horizon_tokens: int,
                              frames_per_token: int = 4):
    """Zero `traj_mask_frame` at/after the horizon cutoff frame, IN PLACE.

    `traj_mask_frame`: frame-level mask, shape [..., T_frame].
    cutoff_frame = token_start_frame(active_end_token + horizon_tokens) — the
    first frame of the (E+H) token; everything from there on is masked out.
    Using token_start_frame (not (E+H)*frames_per_token = token_end_frame) is
    what keeps the 77 vs 80 boundary correct (see §2.2.2). If the cutoff is at
    or beyond the mask length the mask is unchanged (e.g. horizon ≥ clip).

    `active_end_token` may be an int (one cutoff for the whole batch) OR a
    per-sample tensor [B] (B-P0-2: each rollout sample's own active-window token
    position, NOT a hardcoded 0 = clip start). Returns the (mutated) mask.
    """
    import torch

    T_frame = traj_mask_frame.shape[-1]
    if torch.is_tensor(active_end_token) and active_end_token.dim() > 0:
        # per-sample: traj_mask_frame must be [B, T_frame]
        for b in range(active_end_token.shape[0]):
            cutoff = token_start_frame(int(active_end_token[b]) + horizon_tokens,
                                       frames_per_token)
            if cutoff < T_frame:
                traj_mask_frame[b, cutoff:] = 0
        return traj_mask_frame
    cutoff_frame = token_start_frame(int(active_end_token) + horizon_tokens, frames_per_token)
    if cutoff_frame < T_frame:
        traj_mask_frame[..., cutoff_frame:] = 0
    return traj_mask_frame


__all__ = [
    "sample_random_horizon_tokens",
    "apply_horizon_mask_tokens",
]
