"""RootRefiner masked loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.refiner.path_condition import map_path_control_mask_to_frame_mask


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


def soft_ordinal_targets(target_class: Tensor, num_classes: int, sigma: float = 1.0) -> Tensor:
    """Gaussian soft labels for ordinal token classes."""
    if not sigma > 0:
        raise ValueError(f"ordinal sigma must be > 0, got {sigma}")
    target_class = target_class.to(dtype=torch.float32)
    class_idx = torch.arange(num_classes, device=target_class.device, dtype=torch.float32)
    dist = class_idx[None, :] - target_class[:, None]
    weights = torch.exp(-0.5 * (dist / float(sigma)) ** 2)
    return weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def ordinal_duration_loss(
    logits: Tensor,
    target_num_tokens: Tensor,
    *,
    min_tokens: int,
    sigma: float = 1.0,
) -> dict[str, Tensor]:
    target_class = target_num_tokens.to(dtype=torch.long, device=logits.device) - int(min_tokens)
    # Validate the target lands on the logit grid. The old hard F.cross_entropy
    # raised loudly on out-of-range targets; the soft-label path would instead
    # silently leak Gaussian mass onto the wrong in-range classes (or produce an
    # all-zero row that kills the CE gradient), so we keep an explicit guard. An
    # out-of-range num_tokens means model/data disagree on [min_tokens, max_tokens].
    n_classes = logits.shape[-1]
    if int(target_class.min()) < 0 or int(target_class.max()) >= n_classes:
        raise ValueError(
            f"num_tokens out of range: target_class ∈ "
            f"[{int(target_class.min())}, {int(target_class.max())}] not within "
            f"[0, {n_classes}); check model/dataset min_tokens/max_tokens agreement"
        )
    soft = soft_ordinal_targets(target_class, n_classes, sigma=sigma)
    logp = F.log_softmax(logits, dim=-1)
    ordinal_ce = -(soft * logp).sum(dim=-1).mean()
    probs = logits.softmax(dim=-1)
    class_idx = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    expected_num_tokens = (probs * class_idx[None, :]).sum(dim=-1) + float(min_tokens)
    target_phys = target_num_tokens.to(dtype=logits.dtype, device=logits.device)
    expected = F.smooth_l1_loss(expected_num_tokens, target_phys)
    return {
        "ordinal_ce": ordinal_ce,
        "expected": expected,
        "expected_num_tokens": expected_num_tokens,
    }


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
    "ordinal_duration_loss",
    "second_order_diff_l2",
    "smooth_l1_masked",
    "soft_ordinal_targets",
    "sparse_path_control_loss",
]
