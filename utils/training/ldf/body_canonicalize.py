"""Body-window canonicalization for 7D trajectory conditions."""

from __future__ import annotations

import torch
from torch import Tensor

from utils.local_frame import canonicalize_7d
from utils.token_frame import token_body_window_left_frame


def apply_body_window_canonicalize(
    traj_cond_7d: Tensor,
    end_indices,
    gt_root_xyz: Tensor,
    gt_root_yaw: Tensor,
    gt_root_valid_len,
    body_window_tokens: int,
    frames_per_token: int = 4,
) -> tuple[Tensor, Tensor]:
    """Canonicalize world-frame 7D traj_cond to the body-window-local frame."""
    if traj_cond_7d.dim() != 3 or traj_cond_7d.shape[-1] != 7:
        raise ValueError(
            f"traj_cond_7d must be [B, T, 7], got {tuple(traj_cond_7d.shape)}"
        )
    batch_size, num_frames, _ = traj_cond_7d.shape
    device = traj_cond_7d.device

    sample_loss_mask = torch.ones(batch_size, device=device, dtype=traj_cond_7d.dtype)
    anchor_xz = traj_cond_7d.new_zeros(batch_size, 2)
    anchor_yaw = traj_cond_7d.new_zeros(batch_size)

    for batch_idx in range(batch_size):
        anchor_frame = token_body_window_left_frame(
            int(end_indices[batch_idx]),
            body_window_tokens,
            frames_per_token,
        )
        anchor_frame = min(anchor_frame, num_frames - 1)
        valid_len = int(gt_root_valid_len[batch_idx])
        if anchor_frame >= valid_len:
            sample_loss_mask[batch_idx] = 0.0
            anchor_idx = max(0, valid_len - 1)
        else:
            anchor_idx = anchor_frame
        anchor_xz[batch_idx] = gt_root_xyz[batch_idx, anchor_idx][[0, 2]]
        anchor_yaw[batch_idx] = gt_root_yaw[batch_idx, anchor_idx]

    canonicalized = canonicalize_7d(traj_cond_7d, anchor_xz, anchor_yaw)
    return canonicalized, sample_loss_mask


__all__ = ["apply_body_window_canonicalize"]
