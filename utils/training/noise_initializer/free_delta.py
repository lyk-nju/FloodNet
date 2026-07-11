"""Direct fixed-snapshot optimization of frontier ``delta_zT``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from utils.training.noise_initializer.losses import delta_zT_l2_regularization
from utils.training.noise_initializer.shadow_rollout import clip_delta_to_base_norm


SnapshotLossFn = Callable[[torch.Tensor], torch.Tensor]
ProgressFn = Callable[[dict[str, float | int]], None]


@dataclass(frozen=True)
class FreeDeltaOptimizationResult:
    raw_delta_zT: torch.Tensor
    delta_zT: torch.Tensor
    delta_scale: torch.Tensor
    frontier_zT: torch.Tensor
    initial_task_loss: float
    final_task_loss: float
    final_total_loss: float
    clip_saturation_ratio: float
    loss_curve: list[dict[str, float | int]]


def optimize_free_delta(
    *,
    base_zT: torch.Tensor,
    loss_fn: SnapshotLossFn,
    steps: int,
    lr: float,
    lambda_delta: float = 0.0,
    max_delta_norm_ratio: float | None = None,
    log_every: int = 50,
    progress_fn: ProgressFn | None = None,
) -> FreeDeltaOptimizationResult:
    """Optimize one free residual tensor under the supplied snapshot loss."""

    steps = int(steps)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if float(lr) <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    log_every = max(1, int(log_every))

    base = base_zT.detach()
    raw_delta = torch.nn.Parameter(torch.zeros_like(base))
    optimizer = torch.optim.Adam([raw_delta], lr=float(lr))
    loss_curve: list[dict[str, float | int]] = []
    saturated_steps = 0
    initial_task_loss: float | None = None

    def evaluate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clipped, scale = clip_delta_to_base_norm(
            raw_delta,
            base,
            max_delta_norm_ratio=max_delta_norm_ratio,
        )
        task_loss = loss_fn(base + clipped)
        delta_reg = delta_zT_l2_regularization(raw_delta)
        total_loss = task_loss + float(lambda_delta) * delta_reg
        return total_loss, task_loss, clipped, scale

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss, task_loss, clipped, scale = evaluate()
        if initial_task_loss is None:
            initial_task_loss = float(task_loss.detach().cpu().item())
        is_saturated = bool((scale.detach() < 0.999999).any().cpu().item())
        saturated_steps += int(is_saturated)
        if step == 0 or step % log_every == 0:
            row = {
                "step": int(step),
                "task_loss": float(task_loss.detach().cpu().item()),
                "total_loss": float(total_loss.detach().cpu().item()),
                "raw_delta_norm": float(raw_delta.detach().float().norm().cpu().item()),
                "clipped_delta_norm": float(clipped.detach().float().norm().cpu().item()),
                "delta_scale": float(scale.detach().float().mean().cpu().item()),
            }
            loss_curve.append(row)
            if progress_fn is not None:
                progress_fn(dict(row))
        total_loss.backward()
        optimizer.step()

    final_total, final_task, final_clipped, final_scale = evaluate()
    final_row = {
        "step": int(steps),
        "task_loss": float(final_task.detach().cpu().item()),
        "total_loss": float(final_total.detach().cpu().item()),
        "raw_delta_norm": float(raw_delta.detach().float().norm().cpu().item()),
        "clipped_delta_norm": float(final_clipped.detach().float().norm().cpu().item()),
        "delta_scale": float(final_scale.detach().float().mean().cpu().item()),
    }
    loss_curve.append(final_row)
    if progress_fn is not None:
        progress_fn(dict(final_row))
    return FreeDeltaOptimizationResult(
        raw_delta_zT=raw_delta.detach().clone(),
        delta_zT=final_clipped.detach().clone(),
        delta_scale=final_scale.detach().clone(),
        frontier_zT=(base + final_clipped).detach().clone(),
        initial_task_loss=float(initial_task_loss),
        final_task_loss=float(final_task.detach().cpu().item()),
        final_total_loss=float(final_total.detach().cpu().item()),
        clip_saturation_ratio=float(saturated_steps / steps),
        loss_curve=loss_curve,
    )


__all__ = ["FreeDeltaOptimizationResult", "optimize_free_delta"]
