"""Losses for online residual frontier-noise training."""

from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over nonzero mask entries, with zero-safe denom."""

    weights = mask.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp(min=1.0)


def delta_zT_l2_regularization(delta_zT: torch.Tensor) -> torch.Tensor:
    """Residual-size penalty for deterministic z_T correction."""

    return delta_zT.pow(2).mean()


def anchored_root_xz_loss(
    pred_xz: torch.Tensor,
    target_xz: torch.Tensor,
    mask: torch.Tensor,
    *,
    history_frames: int,
    lambda_vel: float = 0.0,
    anchor_mode: str = "target_anchor_abs",
    generated_anchor_xz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Root XZ loss aligned to a stable absolute anchor.

    This mirrors the B-frontier oracle loss: history frames are used only for
    anchoring/context, and optimized frames contribute trajectory and velocity
    error.
    """

    n = min(int(pred_xz.shape[0]), int(target_xz.shape[0]), int(mask.shape[0]))
    if n <= 0:
        zero = pred_xz.sum() * 0.0
        return zero, {
            "traj_loss": 0.0,
            "vel_loss": 0.0,
            "history_frames": 0,
            "optimized_frames": 0,
            "anchor_mode": str(anchor_mode),
        }

    history_frames = max(0, min(int(history_frames), n))
    if history_frames >= n:
        zero = pred_xz[:n].sum() * 0.0
        return zero, {
            "traj_loss": 0.0,
            "vel_loss": 0.0,
            "history_frames": int(history_frames),
            "optimized_frames": 0,
            "anchor_mode": str(anchor_mode),
        }

    pred = pred_xz[:n]
    target = target_xz[:n].to(device=pred.device, dtype=pred.dtype)
    m = mask[:n].to(device=pred.device, dtype=pred.dtype)

    pred_anchor = pred[:1].detach()
    anchor_mode = str(anchor_mode)
    if anchor_mode == "absolute":
        pred_world = pred
    elif anchor_mode == "target_anchor_abs":
        world_anchor = target[:1].detach()
    elif anchor_mode == "generated_anchor_abs":
        if generated_anchor_xz is None:
            world_anchor = pred_anchor
        else:
            world_anchor = (
                generated_anchor_xz.to(device=pred.device, dtype=pred.dtype)
                .view(1, 2)
                .detach()
            )
    else:
        raise ValueError(f"unknown anchor_mode: {anchor_mode!r}")
    if anchor_mode != "absolute":
        pred_world = pred - pred_anchor + world_anchor
    opt = slice(history_frames, n)
    opt_mask = m[opt]
    traj_error = (pred_world[opt] - target[opt]).pow(2).sum(dim=-1)
    traj_loss = (traj_error * opt_mask).sum() / opt_mask.sum().clamp(min=1.0)

    if n - history_frames >= 2:
        pred_v = pred_world[history_frames + 1 : n] - pred_world[history_frames : n - 1]
        target_v = target[history_frames + 1 : n] - target[history_frames : n - 1]
        vel_mask = (m[history_frames + 1 : n] * m[history_frames : n - 1]).float()
        vel_error = (pred_v - target_v).pow(2).sum(dim=-1)
        vel_loss = (vel_error * vel_mask).sum() / vel_mask.sum().clamp(min=1.0)
    else:
        vel_loss = traj_loss.new_zeros(())

    loss = traj_loss + float(lambda_vel) * vel_loss
    return loss, {
        "traj_loss": float(traj_loss.detach().cpu().item()),
        "vel_loss": float(vel_loss.detach().cpu().item()),
        "history_frames": int(history_frames),
        "optimized_frames": int(n - history_frames),
        "anchor_mode": anchor_mode,
    }


__all__ = [
    "anchored_root_xz_loss",
    "delta_zT_l2_regularization",
    "masked_mean",
]
