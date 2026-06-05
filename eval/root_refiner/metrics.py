"""RootRefiner benchmark metric helpers."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from utils.local_frame import wrap_angle

_RAD2DEG = 180.0 / math.pi


def _heading_to_yaw(cos_sin: Tensor) -> Tensor:
    """[..., 2] (cos, sin) -> yaw angle via atan2(sin, cos)."""
    return torch.atan2(cos_sin[..., 1], cos_sin[..., 0])


def _lateral_component(xyz: Tensor, yaw: Tensor) -> Tensor:
    """Per-frame lateral displacement magnitude relative to heading."""
    if xyz.shape[0] < 2:
        return xyz.new_zeros(0)
    delta_xz = xyz[1:, [0, 2]] - xyz[:-1, [0, 2]]
    yaw_prev = yaw[:-1]
    perp = torch.stack([torch.cos(yaw_prev), -torch.sin(yaw_prev)], dim=-1)
    return (delta_xz * perp).sum(-1).abs()


def compute_sample_metrics(pred_wp: Tensor, gt_wp: Tensor, mask: Tensor) -> dict:
    """Metrics for a single sample over valid frames."""
    valid = mask.bool()
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {
            key: float("nan")
            for key in (
                "xyz_ADE",
                "xyz_FDE",
                "heading_error_deg",
                "fwd_speed_MAE",
                "yaw_rate_MAE",
                "lateral_speed_MAE",
                "smoothness",
            )
        }

    pv = pred_wp[valid]
    gv = gt_wp[valid]

    xyz_err = (pv[:, 0:3] - gv[:, 0:3]).norm(dim=-1)
    ade = xyz_err.mean().item()
    fde = xyz_err[-1].item()

    pred_yaw = _heading_to_yaw(pv[:, 3:5])
    gt_yaw = _heading_to_yaw(gv[:, 3:5])
    head_err = wrap_angle(pred_yaw - gt_yaw).abs() * _RAD2DEG
    heading_error_deg = head_err.median().item()

    fwd_mae = (pv[:, 5] - gv[:, 5]).abs().mean().item()
    yaw_mae = (pv[:, 6] - gv[:, 6]).abs().mean().item()

    pred_lat = _lateral_component(pv[:, 0:3], pred_yaw)
    gt_lat = _lateral_component(gv[:, 0:3], gt_yaw)
    if pred_lat.numel() > 0:
        lateral_mae = (pred_lat - gt_lat).abs().mean().item()
    else:
        lateral_mae = float("nan")

    if pv.shape[0] >= 3:
        dyn = pv[:, 5:7]
        diff2 = dyn[2:] - 2 * dyn[1:-1] + dyn[:-2]
        smoothness = (diff2 ** 2).sum(-1).mean().item()
    else:
        smoothness = 0.0

    return {
        "xyz_ADE": ade,
        "xyz_FDE": fde,
        "heading_error_deg": heading_error_deg,
        "fwd_speed_MAE": fwd_mae,
        "yaw_rate_MAE": yaw_mae,
        "lateral_speed_MAE": lateral_mae,
        "smoothness": smoothness,
    }


__all__ = ["_heading_to_yaw", "_lateral_component", "compute_sample_metrics"]
