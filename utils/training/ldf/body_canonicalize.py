"""Body-window canonicalize for the 7D traj condition (T_B_05, design §2.3).

Training-time traj_cond is in absolute world coordinates (x/z can be metres);
inference (TrajStreamBuffer) re-anchors every streaming step to the *body window
leftmost frame* (= history0, `start_t = max(0, end_index - seq_len)`). To match
that distribution we canonicalize the world-frame 7D traj_cond into the
body-window-local frame using the GT root pose at history0.

⚠ Anchor = `token_body_window_left_frame(end_token, body_window_tokens)`
(history0), NOT `token_active_window_left_frame` (active-window left = history/
active boundary). The two differ by `body_window_tokens - chunk_size` tokens;
using the wrong one desyncs train/inference distributions.

Invalid samples (anchor frame falls in padding): still canonicalized to a
fallback anchor (last valid frame) — never leave world-frame data in a
local-frame batch — but `sample_loss_mask = 0` so downstream traj/heading/
control loss (T_B_06) ignores them.

Pure function (fully unit-testable; the SF/training caller supplies end_indices,
GT pose, body_window_tokens). Reuses `canonicalize_7d` (no hand-written matrices).
"""

from __future__ import annotations

import torch
from torch import Tensor

from utils.local_frame import canonicalize_7d
from utils.token_frame import token_body_window_left_frame


def apply_body_window_canonicalize(
    traj_cond_7d: Tensor,          # [B, T, 7] world-frame
    end_indices,                   # [B] active-window right boundary (token-level)
    gt_root_xyz: Tensor,           # [B, T, 3] GT root xyz (world, incl. y)
    gt_root_yaw: Tensor,           # [B, T] GT root physical yaw (world)
    gt_root_valid_len,             # [B] valid frame count per sample
    body_window_tokens: int,
    frames_per_token: int = 4,
) -> tuple[Tensor, Tensor]:
    """Canonicalize world-frame 7D traj_cond to the body-window-local frame.

    Returns `(canonicalized [B, T, 7], sample_loss_mask [B])`. Only xz + heading
    (channels 0-4) are anchored; y (channel 1) and fwd/yaw_delta (5, 6) are
    rigid-invariant and left untouched by `canonicalize_7d`.
    """
    if traj_cond_7d.dim() != 3 or traj_cond_7d.shape[-1] != 7:
        raise ValueError(f"traj_cond_7d must be [B, T, 7], got {tuple(traj_cond_7d.shape)}")
    B, T, _ = traj_cond_7d.shape
    device = traj_cond_7d.device

    sample_loss_mask = torch.ones(B, device=device, dtype=traj_cond_7d.dtype)
    anchor_xz = traj_cond_7d.new_zeros(B, 2)
    anchor_yaw = traj_cond_7d.new_zeros(B)

    for b in range(B):
        bwl = token_body_window_left_frame(
            int(end_indices[b]), body_window_tokens, frames_per_token
        )
        bwl = min(bwl, T - 1)
        valid_len = int(gt_root_valid_len[b])
        if bwl >= valid_len:
            # anchor in padding → mask loss, fall back to last valid frame so the
            # sample is still canonicalized (never leave world-frame in the batch).
            sample_loss_mask[b] = 0.0
            anchor_idx = max(0, valid_len - 1)
        else:
            anchor_idx = bwl
        anchor_xz[b] = gt_root_xyz[b, anchor_idx][[0, 2]]
        anchor_yaw[b] = gt_root_yaw[b, anchor_idx]

    out = canonicalize_7d(traj_cond_7d, anchor_xz, anchor_yaw)
    return out, sample_loss_mask


__all__ = ["apply_body_window_canonicalize"]
