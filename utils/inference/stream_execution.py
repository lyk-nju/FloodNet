"""Atomic streaming execution contracts and state helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from utils.motion_process import (
    recover_root_rot_pos,
    replace_root_channels_263_window_from_7d,
)
from utils.token_frame import token_start_frame


def _clone_state(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    return copy.deepcopy(value)


@dataclass(frozen=True)
class RootFeedbackConfig:
    enabled: bool = False
    xz_blend_alpha: float = 0.5

    def __post_init__(self) -> None:
        alpha = float(self.xz_blend_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"xz_blend_alpha must be in [0, 1], got {self.xz_blend_alpha}"
            )


@dataclass(frozen=True)
class StreamCommitEvent:
    local_commit_before: int
    absolute_commit_before: int
    absolute_commit_after: int
    latent_token: torch.Tensor
    decoded_motion_chunk: torch.Tensor
    joint_frames: np.ndarray
    generated_root_traj7: torch.Tensor
    timeline_state: Any
    traj_payload: dict | None
    root_feedback_applied: bool
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RootFeedbackResult:
    latent_token: torch.Tensor
    decoded_motion_chunk: torch.Tensor
    applied: bool
    debug: dict[str, Any] = field(default_factory=dict)


def snapshot_ldf_stream_state(model) -> dict[str, Any]:
    names = (
        "generated",
        "commit_index",
        "current_step",
        "text_condition_list",
    )
    return {
        name: _clone_state(getattr(model, name))
        for name in names
        if hasattr(model, name)
    }


def restore_ldf_stream_state(model, state: dict[str, Any]) -> None:
    for name, value in state.items():
        setattr(model, name, _clone_state(value))


_VAE_CACHE_NAMES = (
    "_conv_num",
    "_conv_idx",
    "_feat_map",
    "_enc_conv_num",
    "_enc_conv_idx",
    "_enc_feat_map",
)


def snapshot_vae_stream_state(vae) -> dict[str, Any] | None:
    if vae is None:
        return None
    if hasattr(vae, "snapshot_cache"):
        return {"custom": _clone_state(vae.snapshot_cache())}
    model = getattr(vae, "model", None)
    if model is None:
        return None
    return {
        "model": {
            name: _clone_state(getattr(model, name))
            for name in _VAE_CACHE_NAMES
            if hasattr(model, name)
        }
    }


def restore_vae_stream_state(vae, state: dict[str, Any] | None) -> None:
    if vae is None or state is None:
        return
    if "custom" in state and hasattr(vae, "restore_cache"):
        vae.restore_cache(_clone_state(state["custom"]))
        return
    model = getattr(vae, "model", None)
    if model is None:
        return
    for name, value in state.get("model", {}).items():
        setattr(model, name, _clone_state(value))


def snapshot_recovery_state(recovery) -> dict[str, Any] | None:
    if recovery is None:
        return None
    return _clone_state(vars(recovery))


def restore_recovery_state(recovery, state: dict[str, Any] | None) -> None:
    if recovery is None or state is None:
        return
    recovery.__dict__.clear()
    recovery.__dict__.update(_clone_state(state))


def _decode_latent_token(vae, latent_token, *, first_chunk: bool, device) -> torch.Tensor:
    latent = latent_token.detach().to(device=device)
    return vae.stream_decode(
        latent.unsqueeze(0),
        first_chunk=bool(first_chunk),
    )[0].float().detach().cpu()


def _root_feedback_target(
    traj_payload: dict | None,
    decoded_chunk: torch.Tensor,
    *,
    generated_frame_count: int,
    xz_blend_alpha: float,
) -> torch.Tensor | None:
    if not isinstance(traj_payload, dict):
        return None
    condition = traj_payload.get("traj_cond_7d_frame")
    if condition is None:
        return None
    traj = condition[0] if torch.is_tensor(condition) and condition.dim() == 3 else condition
    traj = torch.as_tensor(traj, device=decoded_chunk.device, dtype=decoded_chunk.dtype)
    if traj.dim() != 2 or traj.shape[-1] < 5 or int(traj.shape[0]) == 0:
        return None

    absolute_start_token = int(
        traj_payload.get(
            "traj_abs_start_token",
            traj_payload.get("body_anchor_abs_token", 0),
        )
    )
    absolute_start_frame = token_start_frame(absolute_start_token)
    local_start = max(0, int(generated_frame_count) - absolute_start_frame)
    needed = local_start + int(decoded_chunk.shape[0]) + 1
    if int(traj.shape[0]) < needed:
        traj = torch.cat(
            [traj, traj[-1:].expand(needed - int(traj.shape[0]), -1)],
            dim=0,
        )
    target = traj[local_start:needed].clone()
    if int(target.shape[0]) < 2:
        return None

    alpha = float(xz_blend_alpha)
    if alpha < 1.0:
        dummy_tail = decoded_chunk.new_zeros((1, decoded_chunk.shape[-1]))
        generated_prefix = torch.cat([decoded_chunk, dummy_tail], dim=0)
        _, generated_xyz = recover_root_rot_pos(generated_prefix.unsqueeze(0))
        generated_xyz = generated_xyz[0, : int(target.shape[0])]
        target_xz = target[:, [0, 2]] - target[:1, [0, 2]]
        target_xz = target_xz + generated_xyz[:1, [0, 2]]
        target[:, [0, 2]] = (
            (1.0 - alpha) * generated_xyz[:, [0, 2]] + alpha * target_xz
        )
    return target


def _committed_latent_index_after_step(model, local_commit_index: int) -> int:
    current_commit = int(getattr(model, "commit_index", local_commit_index + 1))
    if current_commit == local_commit_index + 1:
        return int(local_commit_index)
    seq_len = int(getattr(model, "seq_len", 0))
    if seq_len > 0 and current_commit == local_commit_index + 1 - seq_len:
        return int(local_commit_index) - seq_len
    return max(0, current_commit - 1)


def _write_committed_latent(model, latent_token, local_commit_index: int) -> None:
    generated = getattr(model, "generated", None)
    if generated is None:
        return
    latent = latent_token.detach().to(device=generated.device, dtype=generated.dtype)
    if latent.dim() == 1:
        latent = latent.view(1, 1, -1)
    elif latent.dim() == 2:
        latent = latent.unsqueeze(0)
    if hasattr(model, "preprocess"):
        latent_pre = model.preprocess(latent)
    else:
        latent_pre = latent.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    write_index = _committed_latent_index_after_step(model, int(local_commit_index))
    if 0 <= write_index < int(generated.shape[2]):
        generated[
            : latent_pre.shape[0],
            :,
            write_index : write_index + 1,
            ...,
        ] = latent_pre


def decode_token_with_root_feedback(
    *,
    model,
    vae,
    latent_token: torch.Tensor,
    traj_payload: dict | None,
    generated_frame_count: int,
    local_commit_index: int,
    first_chunk: bool,
    config: RootFeedbackConfig,
    device,
) -> RootFeedbackResult:
    """Formally decode one token, optionally replacing and re-encoding its root."""

    if not bool(config.enabled):
        decoded = _decode_latent_token(
            vae,
            latent_token,
            first_chunk=first_chunk,
            device=device,
        )
        return RootFeedbackResult(
            latent_token=latent_token.detach().cpu(),
            decoded_motion_chunk=decoded,
            applied=False,
        )

    cache = snapshot_vae_stream_state(vae)
    decoded_raw = _decode_latent_token(
        vae,
        latent_token,
        first_chunk=first_chunk,
        device=device,
    )
    restore_vae_stream_state(vae, cache)
    target = _root_feedback_target(
        traj_payload,
        decoded_raw,
        generated_frame_count=int(generated_frame_count),
        xz_blend_alpha=float(config.xz_blend_alpha),
    )
    if target is None:
        decoded = _decode_latent_token(
            vae,
            latent_token,
            first_chunk=first_chunk,
            device=device,
        )
        return RootFeedbackResult(
            latent_token=latent_token.detach().cpu(),
            decoded_motion_chunk=decoded,
            applied=False,
            debug={"reason": "missing_target"},
        )

    corrected = replace_root_channels_263_window_from_7d(
        decoded_raw,
        target,
        start_frame=0,
    )
    encoded = vae.stream_encode(
        corrected.to(device=device).unsqueeze(0),
        first_chunk=bool(first_chunk),
    )[0].detach()
    corrected_latent = encoded[-1:].detach().cpu()
    _decode_latent_token(
        vae,
        corrected_latent,
        first_chunk=first_chunk,
        device=device,
    )
    _write_committed_latent(model, corrected_latent, int(local_commit_index))
    return RootFeedbackResult(
        latent_token=corrected_latent,
        decoded_motion_chunk=corrected.detach().cpu(),
        applied=True,
        debug={"reason": "applied"},
    )


__all__ = [
    "RootFeedbackConfig",
    "RootFeedbackResult",
    "StreamCommitEvent",
    "decode_token_with_root_feedback",
    "restore_ldf_stream_state",
    "restore_recovery_state",
    "restore_vae_stream_state",
    "snapshot_ldf_stream_state",
    "snapshot_recovery_state",
    "snapshot_vae_stream_state",
]
