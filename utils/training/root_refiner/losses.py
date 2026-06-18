"""RootRefiner masked loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.training.root_refiner.path_condition import map_path_control_mask_to_frame_mask


def smooth_l1_masked(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """SmoothL1 over valid frames. pred/gt: [B, T, C]; mask: [B, T]."""
    mask_f = mask.unsqueeze(-1).to(pred.dtype)
    denom = mask_f.sum() * pred.shape[-1]
    if denom <= 0:
        return pred.new_zeros(())
    diff = F.smooth_l1_loss(pred, gt, reduction="none") * mask_f
    return diff.sum() / denom


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of `values` [B, T] over valid positions."""
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum()
    if denom <= 0:
        return values.new_zeros(())
    return (values * mask_f).sum() / denom


def second_order_diff_l2(values: Tensor, mask: Tensor) -> Tensor:
    """L2 on second-order frame differences of `values` [B, T, C]."""
    if values.shape[1] < 3:
        return values.new_zeros(())
    diff = values[:, 2:] - 2 * values[:, 1:-1] + values[:, :-2]
    valid = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    valid_f = valid.unsqueeze(-1).to(values.dtype)
    denom = valid_f.sum() * values.shape[-1]
    if denom <= 0:
        return values.new_zeros(())
    return ((diff ** 2) * valid_f).sum() / denom


def _interpolate_path_at_frame_progress(path: Tensor, num_frames: int) -> Tensor:
    B, n_path, _ = path.shape
    device, dtype = path.device, path.dtype
    frame_t = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype)
    path_pos = frame_t * float(n_path - 1)
    idx0 = path_pos.floor().long().clamp(0, n_path - 1)
    idx1 = (idx0 + 1).clamp(max=n_path - 1)
    alpha = (path_pos - idx0.to(dtype))[None, :, None]
    p0 = path[:, idx0]
    p1 = path[:, idx1]
    return (1.0 - alpha) * p0 + alpha * p1


def dense_path_control_loss(pred_waypoints: Tensor, path: Tensor, path_supervision_mask: Tensor) -> Tensor:
    pred_xz = pred_waypoints[..., [0, 2]]
    total = pred_waypoints.new_zeros(())
    denom = 0
    for b in range(pred_waypoints.shape[0]):
        frame_idx = torch.nonzero(path_supervision_mask[b], as_tuple=False).flatten()
        if frame_idx.numel() == 0:
            continue
        start = int(frame_idx[0].item())
        end = int(frame_idx[-1].item())
        local_count = max(1, end - start + 1)
        target = _interpolate_path_at_frame_progress(path[b:b + 1], local_count)[0]
        local_idx = frame_idx - start
        diff = F.smooth_l1_loss(
            pred_xz[b, frame_idx],
            target[local_idx],
            reduction="none",
        )
        total = total + diff.sum()
        denom += int(diff.numel())
    if denom <= 0:
        return pred_waypoints.new_zeros(())
    return total / denom


def sparse_path_control_loss(
    pred_waypoints: Tensor,
    path: Tensor,
    path_control_mask: Tensor,
    path_supervision_mask: Tensor,
    offset_start_frames: Tensor,
) -> Tensor:
    losses = []
    B, T, _ = pred_waypoints.shape
    n_path = path.shape[1]
    pred_xz = pred_waypoints[..., [0, 2]]
    for b in range(B):
        valid_count = int(path_supervision_mask[b].sum().item()) + int(offset_start_frames[b].item())
        valid_count = max(1, min(T, valid_count))
        frame_mask = map_path_control_mask_to_frame_mask(
            path_control_mask[b],
            n_path=n_path,
            max_frames=T,
            valid_frame_count=valid_count,
            offset_start_frames=int(offset_start_frames[b].item()),
        )
        frame_mask = frame_mask.to(path_supervision_mask.device) & path_supervision_mask[b]
        frame_idx = torch.nonzero(frame_mask, as_tuple=False).flatten()
        control_idx = torch.nonzero(path_control_mask[b], as_tuple=False).flatten()
        if frame_idx.numel() == 0 or control_idx.numel() == 0:
            continue
        count = min(frame_idx.numel(), control_idx.numel())
        losses.append(
            F.smooth_l1_loss(
                pred_xz[b, frame_idx[:count]],
                path[b, control_idx[:count]],
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
    for b in range(pred_waypoints.shape[0]):
        valid_idx = torch.nonzero(waypoints_mask[b], as_tuple=False).flatten()
        control_idx = torch.nonzero(path_control_mask[b], as_tuple=False).flatten()
        if valid_idx.numel() == 0 or control_idx.numel() == 0:
            continue
        losses.append(
            F.smooth_l1_loss(pred_xz[b, valid_idx[-1]], path[b, control_idx[-1]])
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
