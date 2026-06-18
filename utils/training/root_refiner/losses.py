"""RootRefiner masked loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.training.root_refiner.path_condition import (
    map_path_control_mask_to_frame_mask,
)


def smooth_l1_masked(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """SmoothL1 over valid frames. pred/gt: [B, T, C]; mask: [B, T]."""
    mask_float = mask.unsqueeze(-1).to(pred.dtype)
    denominator = mask_float.sum() * pred.shape[-1]
    if denominator <= 0:
        return pred.new_zeros(())
    diff = F.smooth_l1_loss(pred, gt, reduction="none") * mask_float
    return diff.sum() / denominator


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of `values` [B, T] over valid positions."""
    mask_float = mask.to(values.dtype)
    denominator = mask_float.sum()
    if denominator <= 0:
        return values.new_zeros(())
    return (values * mask_float).sum() / denominator


def second_order_diff_l2(values: Tensor, mask: Tensor) -> Tensor:
    """L2 on second-order frame differences of `values` [B, T, C]."""
    if values.shape[1] < 3:
        return values.new_zeros(())
    diff = values[:, 2:] - 2 * values[:, 1:-1] + values[:, :-2]
    valid = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    valid_float = valid.unsqueeze(-1).to(values.dtype)
    denominator = valid_float.sum() * values.shape[-1]
    if denominator <= 0:
        return values.new_zeros(())
    return ((diff ** 2) * valid_float).sum() / denominator


def _interpolate_path_at_frame_progress(path: Tensor, num_frames: int) -> Tensor:
    _, num_path_points, _ = path.shape
    device, dtype = path.device, path.dtype
    frame_t = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype)
    path_pos = frame_t * float(num_path_points - 1)
    left_index = path_pos.floor().long().clamp(0, num_path_points - 1)
    right_index = (left_index + 1).clamp(max=num_path_points - 1)
    alpha = (path_pos - left_index.to(dtype))[None, :, None]
    left_path = path[:, left_index]
    right_path = path[:, right_index]
    return (1.0 - alpha) * left_path + alpha * right_path


def dense_path_control_loss(
    pred_waypoints: Tensor,
    path: Tensor,
    path_supervision_mask: Tensor,
) -> Tensor:
    pred_xz = pred_waypoints[..., [0, 2]]
    total = pred_waypoints.new_zeros(())
    denominator = 0
    for batch_idx in range(pred_waypoints.shape[0]):
        frame_indices = torch.nonzero(
            path_supervision_mask[batch_idx],
            as_tuple=False,
        ).flatten()
        if frame_indices.numel() == 0:
            continue
        start = int(frame_indices[0].item())
        end = int(frame_indices[-1].item())
        local_count = max(1, end - start + 1)
        target = _interpolate_path_at_frame_progress(
            path[batch_idx:batch_idx + 1],
            local_count,
        )[0]
        local_indices = frame_indices - start
        diff = F.smooth_l1_loss(
            pred_xz[batch_idx, frame_indices],
            target[local_indices],
            reduction="none",
        )
        total = total + diff.sum()
        denominator += int(diff.numel())
    if denominator <= 0:
        return pred_waypoints.new_zeros(())
    return total / denominator


def sparse_path_control_loss(
    pred_waypoints: Tensor,
    path: Tensor,
    path_control_mask: Tensor,
    path_supervision_mask: Tensor,
    offset_start_frames: Tensor,
) -> Tensor:
    losses = []
    batch_size, max_frames, _ = pred_waypoints.shape
    num_path_points = path.shape[1]
    pred_xz = pred_waypoints[..., [0, 2]]
    for batch_idx in range(batch_size):
        valid_count = int(path_supervision_mask[batch_idx].sum().item()) + int(
            offset_start_frames[batch_idx].item()
        )
        valid_count = max(1, min(max_frames, valid_count))
        frame_mask = map_path_control_mask_to_frame_mask(
            path_control_mask[batch_idx],
            n_path=num_path_points,
            max_frames=max_frames,
            valid_frame_count=valid_count,
            offset_start_frames=int(offset_start_frames[batch_idx].item()),
        )
        frame_mask = (
            frame_mask.to(path_supervision_mask.device)
            & path_supervision_mask[batch_idx]
        )
        frame_indices = torch.nonzero(frame_mask, as_tuple=False).flatten()
        control_indices = torch.nonzero(
            path_control_mask[batch_idx],
            as_tuple=False,
        ).flatten()
        if frame_indices.numel() == 0 or control_indices.numel() == 0:
            continue
        count = min(frame_indices.numel(), control_indices.numel())
        losses.append(
            F.smooth_l1_loss(
                pred_xz[batch_idx, frame_indices[:count]],
                path[batch_idx, control_indices[:count]],
            )
        )
    if not losses:
        return pred_waypoints.new_zeros(())
    return torch.stack(losses).mean()


def goal_point_control_loss(
    pred_waypoints: Tensor,
    waypoints_mask: Tensor,
    path: Tensor,
    path_control_mask: Tensor,
) -> Tensor:
    pred_xz = pred_waypoints[..., [0, 2]]
    losses = []
    for batch_idx in range(pred_waypoints.shape[0]):
        valid_indices = torch.nonzero(
            waypoints_mask[batch_idx],
            as_tuple=False,
        ).flatten()
        control_indices = torch.nonzero(
            path_control_mask[batch_idx],
            as_tuple=False,
        ).flatten()
        if valid_indices.numel() == 0 or control_indices.numel() == 0:
            continue
        losses.append(
            F.smooth_l1_loss(
                pred_xz[batch_idx, valid_indices[-1]],
                path[batch_idx, control_indices[-1]],
            )
        )
    if not losses:
        return pred_waypoints.new_zeros(())
    return torch.stack(losses).mean()


__all__ = [
    "dense_path_control_loss",
    "goal_point_control_loss",
    "masked_mean",
    "second_order_diff_l2",
    "smooth_l1_masked",
    "sparse_path_control_loss",
]
