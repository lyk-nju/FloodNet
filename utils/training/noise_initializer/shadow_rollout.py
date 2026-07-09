"""Differentiable shadow rollout for online residual z_T training."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from utils.training.noise_initializer.context_builder import NoiseInitializerContext


RolloutFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class ResidualShadowRolloutResult:
    delta_zT: torch.Tensor
    frontier_zT: torch.Tensor
    injected_generated: torch.Tensor
    shadow_latents: torch.Tensor


def _clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def snapshot_stream_state(model) -> dict:
    """Snapshot mutable stream state touched by shadow rollout."""

    return {
        "generated": getattr(model, "generated").detach().clone(),
        "commit_index": int(getattr(model, "commit_index", 0)),
        "current_step": int(getattr(model, "current_step", 0)),
        "text_condition_list": _clone_value(getattr(model, "text_condition_list", [])),
    }


def restore_stream_state(model, state: dict) -> None:
    model.generated = state["generated"].detach().clone()
    model.commit_index = int(state["commit_index"])
    model.current_step = int(state["current_step"])
    if hasattr(model, "text_condition_list"):
        model.text_condition_list = _clone_value(state.get("text_condition_list", []))


def snapshot_vae_cache(vae):
    if hasattr(vae, "snapshot_cache"):
        return vae.snapshot_cache()
    if hasattr(vae, "_cache"):
        return _clone_value(getattr(vae, "_cache"))
    return None


def restore_vae_cache(vae, state) -> None:
    if hasattr(vae, "restore_cache"):
        vae.restore_cache(state)
    elif hasattr(vae, "_cache"):
        setattr(vae, "_cache", _clone_value(state))


def freeze_module(module: nn.Module | None) -> None:
    if module is None:
        return
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def inject_context_frontier_zT(
    generated_preprocessed: torch.Tensor,
    context: NoiseInitializerContext,
    frontier_zT: torch.Tensor,
) -> torch.Tensor:
    """Return generated cache copy with context frontier ids replaced."""

    frontier_ids = context.frontier_ids.to(
        device=generated_preprocessed.device,
        dtype=torch.long,
    )
    expected = (
        int(generated_preprocessed.shape[0]),
        int(frontier_ids.numel()),
        int(generated_preprocessed.shape[1]),
    )
    if tuple(frontier_zT.shape) != expected:
        raise ValueError(
            "frontier_zT must have shape [B,M,C] matching context.frontier_ids; "
            f"expected {expected}, got {tuple(frontier_zT.shape)}"
        )
    injected = generated_preprocessed.clone()
    if frontier_ids.numel() > 0:
        injected[:, :, frontier_ids, 0, 0] = frontier_zT.permute(0, 2, 1)
    return injected


def run_residual_shadow_rollout(
    *,
    model: nn.Module,
    vae: nn.Module | None,
    initializer: nn.Module,
    context: NoiseInitializerContext,
    rollout_fn: RolloutFn,
    alpha: float = 1.0,
    rollout_tokens: int = 1,
    first_chunk: bool = True,
    freeze_frozen_modules: bool = True,
) -> ResidualShadowRolloutResult:
    """Predict residual frontier z_T and run a differentiable short rollout."""

    if int(rollout_tokens) <= 0:
        raise ValueError(f"rollout_tokens must be positive, got {rollout_tokens}")
    if freeze_frozen_modules:
        freeze_module(model)
        freeze_module(vae)

    stream_state = snapshot_stream_state(model)
    vae_state = snapshot_vae_cache(vae) if vae is not None else None
    try:
        context_kwargs = context.as_model_kwargs()
        signature = inspect.signature(initializer.forward)
        if (
            "frontier_base_zT" in signature.parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        ):
            context_kwargs["frontier_base_zT"] = context.frontier_base_zT
        delta_zT = initializer(**context_kwargs)
        frontier_zT = context.frontier_base_zT.to(delta_zT.device, delta_zT.dtype) + (
            float(alpha) * delta_zT
        )
        injected = inject_context_frontier_zT(
            getattr(model, "generated").to(device=frontier_zT.device),
            context,
            frontier_zT,
        )
        model.generated = injected
        shadow_latents = rollout_fn(
            model,
            rollout_tokens=int(rollout_tokens),
            first_chunk=bool(first_chunk),
        )
    finally:
        restore_stream_state(model, stream_state)
        if vae is not None:
            restore_vae_cache(vae, vae_state)

    return ResidualShadowRolloutResult(
        delta_zT=delta_zT,
        frontier_zT=frontier_zT,
        injected_generated=injected,
        shadow_latents=shadow_latents,
    )


__all__ = [
    "ResidualShadowRolloutResult",
    "freeze_module",
    "inject_context_frontier_zT",
    "restore_stream_state",
    "restore_vae_cache",
    "run_residual_shadow_rollout",
    "snapshot_stream_state",
    "snapshot_vae_cache",
]
