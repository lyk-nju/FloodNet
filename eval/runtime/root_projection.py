"""Late-denoise root projection helpers for runtime eval experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from utils.motion_process import (
    recover_root_rot_pos,
    replace_root_channels_263_window_from_7d,
)


@dataclass(frozen=True)
class LateDenoiseRootProjectionConfig:
    enabled: bool = False
    alpha: float = 0.0
    projection_start_step: int = 6
    max_delta_per_frame: float = 0.03
    max_delta_per_chunk: float = 0.10
    latent_blend_gain: float = 0.5
    final_step_gain: float = 1.0
    frames_per_token: int = 4
    frame_ramp: bool = True
    debug: bool = False


@dataclass(frozen=True)
class TimeConsistentRootGuidanceConfig:
    enabled: bool = False
    projection_step: int | tuple[int, ...] = 8
    alpha: float = 0.2
    guidance_strength: float = 1.0
    max_delta_per_frame: float = 0.03
    max_delta_per_chunk: float = 0.10
    max_latent_delta: float = 0.25
    frames_per_token: int = 4
    projection_target_mode: str = "relative_shape"
    mixed_global_weight: float = 0.3
    mixed_local_weight: float = 1.0
    debug: bool = False


@dataclass(frozen=True)
class CurrentTokenTarget:
    traj_7d: torch.Tensor
    start_frame: int
    chunk_frames: int


@dataclass(frozen=True)
class LateDenoiseProjectionResult:
    applied: bool
    reason: str
    strength: float = 0.0
    latent_blend: float = 0.0
    token_noise: float | None = None
    target_start_frame: int | None = None


@dataclass(frozen=True)
class TimeConsistentGuidanceResult:
    applied: bool
    reason: str
    beta_before: float | None = None
    beta_after: float | None = None
    local_step: int | None = None
    latent_delta_norm: float = 0.0
    write_delta_norm: float = 0.0
    target_start_frame: int | None = None


def _clone_state(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _snapshot_vae_stream_cache(vae) -> dict[str, Any] | None:
    model = getattr(vae, "model", None)
    if model is None:
        return None
    names = (
        "_conv_num",
        "_conv_idx",
        "_feat_map",
        "_enc_conv_num",
        "_enc_conv_idx",
        "_enc_feat_map",
    )
    return {
        name: _clone_state(getattr(model, name))
        for name in names
        if hasattr(model, name)
    }


def _restore_vae_stream_cache(vae, cache: dict[str, Any] | None) -> None:
    if cache is None:
        return
    model = getattr(vae, "model", None)
    if model is None:
        return
    for name, value in cache.items():
        setattr(model, name, _clone_state(value))


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _clamp_vectors(vectors: torch.Tensor, max_norm: float) -> torch.Tensor:
    max_norm = float(max_norm)
    if max_norm <= 0.0 or vectors.numel() == 0:
        return vectors
    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (max_norm / norm).clamp(max=1.0)
    return vectors * scale


def _clamp_tensor_norm(value: torch.Tensor, max_norm: float) -> torch.Tensor:
    max_norm = float(max_norm)
    if max_norm <= 0.0 or value.numel() == 0:
        return value
    norm = torch.linalg.norm(value.reshape(-1)).clamp_min(1e-8)
    return value * (max_norm / norm).clamp(max=1.0)


def _flat_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.norm(value.reshape(-1)).detach().cpu().item())


def _flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float().cpu()
    b_flat = b.reshape(-1).float().cpu()
    denom = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    if float(denom.item()) <= 1e-12:
        return 1.0 if torch.allclose(a_flat, b_flat) else 0.0
    return float(torch.dot(a_flat, b_flat).div(denom).clamp(-1.0, 1.0).item())


def _mean_xz_error(pred_xz: torch.Tensor, target_xz: torch.Tensor) -> float:
    valid = min(int(pred_xz.shape[0]), int(target_xz.shape[0]))
    if valid <= 0:
        return 0.0
    err = torch.linalg.norm(
        pred_xz[:valid].float().cpu() - target_xz[:valid].float().cpu(),
        dim=-1,
    )
    return float(err.mean().item())


def _mean_vector_norm(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(value.float().cpu(), dim=-1).mean().item())


def _decoded_root_xz_with_tail(decoded_chunk: torch.Tensor, valid: int) -> torch.Tensor:
    valid = max(0, int(valid))
    if valid <= 0:
        return decoded_chunk.new_zeros((0, 2))
    dummy_tail = decoded_chunk.new_zeros((1, decoded_chunk.shape[-1]))
    pred_feature = torch.cat([decoded_chunk.float(), dummy_tail], dim=0)
    _, pred_xyz = recover_root_rot_pos(pred_feature.unsqueeze(0))
    return pred_xyz[0, :valid, [0, 2]].to(
        device=decoded_chunk.device,
        dtype=decoded_chunk.dtype,
    )


def _compute_time_consistent_projection_residual(
    pred_xz: torch.Tensor,
    target_xz: torch.Tensor,
    *,
    projection_target_mode: str,
    mixed_global_weight: float,
    mixed_local_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mode = str(projection_target_mode)
    global_offset = target_xz[:1] - pred_xz[:1]
    pred_rel = pred_xz - pred_xz[:1]
    target_rel = target_xz - target_xz[:1]
    local_shape_residual = target_rel - pred_rel
    if mode == "relative_shape":
        residual = local_shape_residual
    elif mode == "absolute":
        residual = target_xz - pred_xz
    elif mode == "mixed":
        residual = (
            float(mixed_global_weight) * global_offset
            + float(mixed_local_weight) * local_shape_residual
        )
    else:
        raise ValueError(
            "projection_target_mode must be one of "
            "'relative_shape', 'absolute', or 'mixed'; got "
            f"{projection_target_mode!r}"
        )
    return residual, {
        "global_offset_norm": float(
            torch.linalg.norm(global_offset.reshape(-1)).detach().cpu().item()
        ),
        "local_shape_error": _mean_vector_norm(local_shape_residual),
    }


def _as_scalar_tensor(value, *, device, dtype) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device, dtype=dtype)
    return value.to(device=device, dtype=dtype).reshape(-1)[0]


def _ensure_token_state(value: torch.Tensor) -> torch.Tensor:
    """Normalize a single token latent to [C, 1, 1, 1]."""

    if value.dim() == 4:
        return value
    if value.dim() == 3:
        return value.unsqueeze(1)
    raise ValueError(f"expected token latent [C,1,1] or [C,1,1,1], got {tuple(value.shape)}")


def _estimate_clean_latent_from_velocity(
    x_beta: torch.Tensor,
    predicted_vel: torch.Tensor,
    beta: torch.Tensor | float,
) -> torch.Tensor:
    beta_t = _as_scalar_tensor(beta, device=x_beta.device, dtype=x_beta.dtype)
    return x_beta + beta_t * predicted_vel


def _map_clean_delta_to_flow_state(
    x_after_velocity_update: torch.Tensor,
    delta_clean: torch.Tensor,
    beta_after: torch.Tensor | float,
    *,
    guidance_strength: float,
) -> torch.Tensor:
    beta_t = _as_scalar_tensor(
        beta_after,
        device=x_after_velocity_update.device,
        dtype=x_after_velocity_update.dtype,
    )
    return x_after_velocity_update + float(guidance_strength) * (1.0 - beta_t) * delta_clean


def _projection_strength(
    *,
    model,
    noise_level_full: torch.Tensor,
    commit_index: int,
    config: LateDenoiseRootProjectionConfig,
) -> tuple[float, float | None]:
    if noise_level_full is None or int(commit_index) >= int(noise_level_full.shape[-1]):
        return 0.0, None
    token_noise = float(noise_level_full[0, int(commit_index)].detach().float().cpu().item())
    num_steps = max(1, int(getattr(model, "num_denoise_steps", 10)))
    start_step = min(max(0, int(config.projection_start_step)), num_steps)
    start_noise = max(0.0, 1.0 - float(start_step) / float(num_steps))
    if token_noise > start_noise:
        return 0.0, token_noise
    if start_noise <= 1e-8:
        progress = 1.0 if token_noise <= 1e-8 else 0.0
    else:
        progress = (start_noise - token_noise) / start_noise
    progress_t = torch.tensor(progress, dtype=torch.float32)
    return float(float(config.alpha) * _smoothstep(progress_t).item()), token_noise


def _as_traj_tensor(value) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    if value.dim() == 3:
        value = value[0]
    if value.dim() != 2 or value.shape[-1] < 7:
        return None
    return value.float()


def _as_mask_tensor(value) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.dim() == 2:
        value = value[0]
    if value.dim() != 1:
        return None
    return value.bool()


def select_current_token_target(
    step_input: dict,
    *,
    commit_index: int,
    frames_per_token: int,
    chunk_frames: int,
) -> CurrentTokenTarget | None:
    """Select the full 7D payload and local frame start for the commit token."""

    traj = _as_traj_tensor(step_input.get("traj_cond_7d_frame"))
    if traj is None:
        return None
    frames_per_token = max(1, int(frames_per_token))
    chunk_frames = max(1, int(chunk_frames))
    payload_start_token = int(step_input.get("traj_start_token", 0))
    rel_token = int(commit_index) - payload_start_token
    if rel_token < 0:
        return None
    start_frame = rel_token * frames_per_token
    if start_frame >= int(traj.shape[0]):
        return None

    mask = _as_mask_tensor(
        step_input.get("traj_cond_frame_mask", step_input.get("traj_cond_mask"))
    )
    if mask is not None:
        end_frame = start_frame + chunk_frames
        if end_frame > int(mask.shape[0]):
            return None
        if not bool(mask[start_frame:end_frame].all().item()):
            return None

    return CurrentTokenTarget(
        traj_7d=traj,
        start_frame=int(start_frame),
        chunk_frames=int(chunk_frames),
    )


def _decode_current_token(model, vae, *, token_idx: int, first_chunk: bool) -> torch.Tensor:
    latent = model.postprocess(model.generated[:, :, token_idx:token_idx + 1, ...])
    return vae.stream_decode(latent, first_chunk=first_chunk)[0].float()


def _encode_chunk_token(model, vae, corrected: torch.Tensor, *, first_chunk: bool) -> torch.Tensor:
    del model, first_chunk
    # Late projection is an in-denoise temporary correction. It must not advance
    # the VAE stream encoder cache; otherwise later stream chunks depend on how
    # often projection ran. Offline encode produces the current token latent from
    # the corrected local motion window and the caller restores VAE stream caches.
    encoded = vae.encode(corrected.unsqueeze(0))[0]
    return encoded[-1:].detach()


def _decode_clean_token(model, vae, clean_token: torch.Tensor, *, first_chunk: bool) -> torch.Tensor:
    latent = model.postprocess(_ensure_token_state(clean_token).unsqueeze(0))
    return vae.stream_decode(latent, first_chunk=first_chunk)[0].float()


def _project_decoded_root(
    decoded_chunk: torch.Tensor,
    target: CurrentTokenTarget,
    *,
    strength: float,
    config: LateDenoiseRootProjectionConfig,
) -> torch.Tensor:
    decoded = decoded_chunk.float()
    device = decoded.device
    dtype = decoded.dtype
    target_traj = target.traj_7d.to(device=device, dtype=dtype).clone()
    start = int(target.start_frame)
    valid = min(int(decoded.shape[0]) + 1, int(target_traj.shape[0]) - start)
    if valid <= 1:
        return replace_root_channels_263_window_from_7d(
            decoded,
            target_traj,
            start_frame=start,
        )

    dummy_tail = decoded.new_zeros((1, decoded.shape[-1]))
    pred_feature = torch.cat([decoded, dummy_tail], dim=0)
    _, pred_xyz = recover_root_rot_pos(pred_feature.unsqueeze(0))
    pred_xz = pred_xyz[0, :valid, [0, 2]].to(device=device, dtype=dtype)
    target_xz = target_traj[start:start + valid, [0, 2]]

    pred_rel = pred_xz - pred_xz[:1]
    target_rel = target_xz - target_xz[:1]
    residual = target_rel - pred_rel
    residual = _clamp_vectors(residual, float(config.max_delta_per_frame))
    max_chunk = float(config.max_delta_per_chunk)
    if max_chunk > 0.0:
        max_residual = torch.linalg.norm(residual, dim=-1).max().clamp_min(1e-8)
        residual = residual * (max_chunk / max_residual).clamp(max=1.0)
    if bool(config.frame_ramp):
        frame_u = torch.linspace(0.0, 1.0, valid, device=device, dtype=dtype)
        frame_weight = 0.2 + 0.8 * _smoothstep(frame_u)
        residual = residual * frame_weight[:, None]

    corrected_xz = target_xz[:1] + pred_rel + float(strength) * residual
    target_traj[start:start + valid, [0, 2]] = corrected_xz
    return replace_root_channels_263_window_from_7d(
        decoded,
        target_traj,
        start_frame=start,
    )


def _project_decoded_root_time_consistent(
    decoded_chunk: torch.Tensor,
    target: CurrentTokenTarget,
    *,
    alpha: float,
    max_delta_per_frame: float,
    max_delta_per_chunk: float,
    projection_target_mode: str,
    mixed_global_weight: float,
    mixed_local_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    decoded = decoded_chunk.float()
    device = decoded.device
    dtype = decoded.dtype
    target_traj = target.traj_7d.to(device=device, dtype=dtype).clone()
    start = int(target.start_frame)
    valid = min(int(decoded.shape[0]) + 1, int(target_traj.shape[0]) - start)
    if valid <= 1:
        corrected = replace_root_channels_263_window_from_7d(
            decoded,
            target_traj,
            start_frame=start,
        )
        return corrected, {"global_offset_norm": 0.0, "local_shape_error": 0.0}

    pred_xz = _decoded_root_xz_with_tail(decoded, valid)
    target_xz = target_traj[start:start + valid, [0, 2]]
    residual_raw, projection_metrics = _compute_time_consistent_projection_residual(
        pred_xz,
        target_xz,
        projection_target_mode=projection_target_mode,
        mixed_global_weight=float(mixed_global_weight),
        mixed_local_weight=float(mixed_local_weight),
    )
    residual = _clamp_vectors(residual_raw, float(max_delta_per_frame))
    max_chunk = float(max_delta_per_chunk)
    if max_chunk > 0.0:
        max_residual = torch.linalg.norm(residual, dim=-1).max().clamp_min(1e-8)
        residual = residual * (max_chunk / max_residual).clamp(max=1.0)

    target_traj[start:start + valid, [0, 2]] = pred_xz + float(alpha) * residual
    corrected = replace_root_channels_263_window_from_7d(
        decoded,
        target_traj,
        start_frame=start,
    )
    return corrected, projection_metrics


def apply_late_denoise_root_projection(
    *,
    model,
    vae,
    step_input: dict,
    first_chunk: bool,
    commit_index: int,
    current_step: int,
    start_index: int,
    end_index: int,
    noise_level_full: torch.Tensor,
    config: LateDenoiseRootProjectionConfig,
) -> LateDenoiseProjectionResult:
    del current_step
    if not bool(config.enabled):
        return LateDenoiseProjectionResult(False, "disabled")
    if float(config.alpha) <= 0.0:
        return LateDenoiseProjectionResult(False, "alpha_zero")
    token_idx = int(commit_index)
    if not (int(start_index) <= token_idx < int(end_index)):
        return LateDenoiseProjectionResult(False, "commit_token_not_in_update_window")

    strength, token_noise = _projection_strength(
        model=model,
        noise_level_full=noise_level_full,
        commit_index=token_idx,
        config=config,
    )
    if strength <= 0.0:
        return LateDenoiseProjectionResult(
            False,
            "before_projection_ramp",
            token_noise=token_noise,
        )

    target = select_current_token_target(
        step_input,
        commit_index=token_idx,
        frames_per_token=int(config.frames_per_token),
        chunk_frames=int(config.frames_per_token),
    )
    if target is None:
        return LateDenoiseProjectionResult(False, "missing_target", token_noise=token_noise)

    cache = _snapshot_vae_stream_cache(vae)
    try:
        decoded = _decode_current_token(
            model,
            vae,
            token_idx=token_idx,
            first_chunk=first_chunk,
        )
        corrected = _project_decoded_root(
            decoded,
            target,
            strength=strength,
            config=config,
        )
        projected_latent = _encode_chunk_token(
            model,
            vae,
            corrected,
            first_chunk=first_chunk,
        ).to(device=model.generated.device, dtype=model.generated.dtype)
    finally:
        _restore_vae_stream_cache(vae, cache)

    latent_blend = max(0.0, min(1.0, float(config.latent_blend_gain) * strength))
    if token_noise is not None and token_noise <= float(getattr(model, "dt", 0.0)) + 1e-6:
        latent_blend = max(
            latent_blend,
            max(0.0, min(1.0, float(config.final_step_gain) * strength)),
        )
    if latent_blend <= 0.0:
        return LateDenoiseProjectionResult(
            False,
            "latent_blend_zero",
            strength=float(strength),
            token_noise=token_noise,
            target_start_frame=int(target.start_frame),
        )

    projected_pre = model.preprocess(projected_latent.unsqueeze(0))
    current_pre = model.generated[:, :, token_idx:token_idx + 1, ...]
    model.generated[:, :, token_idx:token_idx + 1, ...] = (
        (1.0 - latent_blend) * current_pre + latent_blend * projected_pre
    )
    return LateDenoiseProjectionResult(
        True,
        "applied",
        strength=float(strength),
        latent_blend=float(latent_blend),
        token_noise=token_noise,
        target_start_frame=int(target.start_frame),
    )


def _token_local_step_from_beta(model, beta_before: torch.Tensor | float) -> int:
    beta = float(_as_scalar_tensor(beta_before, device=torch.device("cpu"), dtype=torch.float32).item())
    num_steps = max(1, int(getattr(model, "num_denoise_steps", 10)))
    return int(round((1.0 - beta) * float(num_steps)))


def _projection_step_tuple(value: int | tuple[int, ...] | list[int] | str) -> tuple[int, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(int(item) for item in items)
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    return (int(value),)


def apply_time_consistent_root_guidance(
    *,
    model,
    vae,
    step_input: dict,
    first_chunk: bool,
    commit_index: int,
    current_step: int,
    start_index: int,
    end_index: int,
    x_beta_before_update: torch.Tensor | None,
    predicted_vel: torch.Tensor | None,
    beta_before: torch.Tensor | float | None,
    beta_after: torch.Tensor | float | None,
    x_after_velocity_update: torch.Tensor | None,
    config: TimeConsistentRootGuidanceConfig,
    debug_pending: list[dict[str, Any]] | None = None,
) -> TimeConsistentGuidanceResult:
    del current_step
    if not bool(config.enabled):
        return TimeConsistentGuidanceResult(False, "disabled")
    if float(config.alpha) <= 0.0:
        return TimeConsistentGuidanceResult(False, "alpha_zero")
    if float(config.guidance_strength) <= 0.0:
        return TimeConsistentGuidanceResult(False, "guidance_strength_zero")
    if getattr(model, "prediction_type", "vel") != "vel":
        return TimeConsistentGuidanceResult(False, "unsupported_prediction_type")
    if (
        x_beta_before_update is None
        or predicted_vel is None
        or beta_before is None
        or beta_after is None
        or x_after_velocity_update is None
    ):
        return TimeConsistentGuidanceResult(False, "missing_flow_state")

    token_idx = int(commit_index)
    if not (int(start_index) <= token_idx < int(end_index)):
        return TimeConsistentGuidanceResult(False, "commit_token_not_in_update_window")

    x_before = _ensure_token_state(x_beta_before_update).to(
        device=model.generated.device,
        dtype=model.generated.dtype,
    )
    vel = _ensure_token_state(predicted_vel).to(
        device=model.generated.device,
        dtype=model.generated.dtype,
    )
    x_after = _ensure_token_state(x_after_velocity_update).to(
        device=model.generated.device,
        dtype=model.generated.dtype,
    )
    beta_before_t = _as_scalar_tensor(
        beta_before,
        device=model.generated.device,
        dtype=model.generated.dtype,
    )
    beta_after_t = _as_scalar_tensor(
        beta_after,
        device=model.generated.device,
        dtype=model.generated.dtype,
    )
    local_step = _token_local_step_from_beta(model, beta_before_t)
    projection_steps = _projection_step_tuple(config.projection_step)
    if local_step not in projection_steps:
        return TimeConsistentGuidanceResult(
            False,
            "not_projection_step",
            beta_before=float(beta_before_t.detach().cpu().item()),
            beta_after=float(beta_after_t.detach().cpu().item()),
            local_step=int(local_step),
        )

    target = select_current_token_target(
        step_input,
        commit_index=token_idx,
        frames_per_token=int(config.frames_per_token),
        chunk_frames=int(config.frames_per_token),
    )
    if target is None:
        return TimeConsistentGuidanceResult(
            False,
            "missing_target",
            beta_before=float(beta_before_t.detach().cpu().item()),
            beta_after=float(beta_after_t.detach().cpu().item()),
            local_step=int(local_step),
        )

    z_hat = _estimate_clean_latent_from_velocity(x_before, vel, beta_before_t)
    cache = _snapshot_vae_stream_cache(vae)
    try:
        decoded = _decode_clean_token(model, vae, z_hat, first_chunk=first_chunk)
        corrected, projection_metrics = _project_decoded_root_time_consistent(
            decoded,
            target,
            alpha=float(config.alpha),
            max_delta_per_frame=float(config.max_delta_per_frame),
            max_delta_per_chunk=float(config.max_delta_per_chunk),
            projection_target_mode=str(config.projection_target_mode),
            mixed_global_weight=float(config.mixed_global_weight),
            mixed_local_weight=float(config.mixed_local_weight),
        )
        z_projected = _encode_chunk_token(
            model,
            vae,
            corrected,
            first_chunk=first_chunk,
        ).to(device=model.generated.device, dtype=model.generated.dtype)
        z_projected_pre = _ensure_token_state(model.preprocess(z_projected.unsqueeze(0))[0])
    finally:
        _restore_vae_stream_cache(vae, cache)

    delta_clean_raw = z_projected_pre - z_hat
    delta_clean = _clamp_tensor_norm(
        delta_clean_raw,
        float(config.max_latent_delta),
    )
    z_hat_after = _estimate_clean_latent_from_velocity(x_after, vel, beta_after_t)
    x_guided = _map_clean_delta_to_flow_state(
        x_after,
        delta_clean,
        beta_after_t,
        guidance_strength=float(config.guidance_strength),
    )
    model.generated[0, :, token_idx:token_idx + 1, ...] = x_guided
    write_delta = x_guided - x_after
    latent_delta_raw_norm = _flat_norm(delta_clean_raw)
    latent_delta_norm = _flat_norm(delta_clean)
    write_delta_norm = _flat_norm(write_delta)
    if debug_pending is not None:
        dt = float(getattr(model, "dt", 0.0))
        normal_update = vel * dt
        normal_update_norm = _flat_norm(normal_update)
        start = int(target.start_frame)
        valid = min(int(decoded.shape[0]) + 1, int(target.traj_7d.shape[0]) - start)
        root_abs_error_before = root_abs_error_after = 0.0
        root_rel_error_before = root_rel_error_after = 0.0
        if valid > 0:
            target_abs_xz = target.traj_7d[
                start:start + valid,
                [0, 2],
            ].to(device=decoded.device, dtype=decoded.dtype)
            decoded_xz = _decoded_root_xz_with_tail(decoded, valid)
            corrected_xz = _decoded_root_xz_with_tail(corrected, valid)
            target_rel_xz = target_abs_xz - target_abs_xz[:1] + decoded_xz[:1]
            root_abs_error_before = _mean_xz_error(decoded_xz, target_abs_xz)
            root_abs_error_after = _mean_xz_error(corrected_xz, target_abs_xz)
            root_rel_error_before = _mean_xz_error(decoded_xz, target_rel_xz)
            root_rel_error_after = _mean_xz_error(corrected_xz, target_rel_xz)
        one_minus_beta_after = float((1.0 - beta_after_t).detach().cpu().item())
        debug_pending.append(
            {
                "applied": True,
                "commit_index": int(token_idx),
                "local_step": int(local_step),
                "projection_steps": [int(item) for item in projection_steps],
                "beta_before": float(beta_before_t.detach().cpu().item()),
                "beta_after": float(beta_after_t.detach().cpu().item()),
                "one_minus_beta_after": one_minus_beta_after,
                "normal_update_norm": normal_update_norm,
                "delta_clean_raw_norm": latent_delta_raw_norm,
                "delta_clean_after_clamp_norm": latent_delta_norm,
                "clamp_ratio": (
                    latent_delta_norm / latent_delta_raw_norm
                    if latent_delta_raw_norm > 1e-12
                    else 1.0
                ),
                "guidance_update_norm": write_delta_norm,
                "guidance_to_denoise_ratio": (
                    write_delta_norm / normal_update_norm
                    if normal_update_norm > 1e-12
                    else 0.0
                ),
                "root_abs_error_before_projection": root_abs_error_before,
                "root_abs_error_after_projection": root_abs_error_after,
                "root_relative_error_before_projection": root_rel_error_before,
                "root_relative_error_after_projection": root_rel_error_after,
                "global_offset_norm": float(
                    projection_metrics.get("global_offset_norm", 0.0)
                ),
                "local_shape_error": float(
                    projection_metrics.get("local_shape_error", 0.0)
                ),
                "projection_target_mode": str(config.projection_target_mode),
                "mixed_global_weight": float(config.mixed_global_weight),
                "mixed_local_weight": float(config.mixed_local_weight),
                "latent_cos_delta_clean_predicted_vel": _flat_cos(delta_clean, vel),
                "latent_delta_norm": latent_delta_norm,
                "write_delta_norm": write_delta_norm,
                "target_start_frame": int(target.start_frame),
                "first_chunk": bool(first_chunk),
                "z_hat": z_hat.detach().cpu(),
                "z_hat_after": z_hat_after.detach().cpu(),
            }
        )
    return TimeConsistentGuidanceResult(
        True,
        "applied",
        beta_before=float(beta_before_t.detach().cpu().item()),
        beta_after=float(beta_after_t.detach().cpu().item()),
        local_step=int(local_step),
        latent_delta_norm=latent_delta_norm,
        write_delta_norm=write_delta_norm,
        target_start_frame=int(target.start_frame),
    )


def build_late_denoise_root_projection_callback(
    *,
    vae,
    config: LateDenoiseRootProjectionConfig,
):
    if not bool(config.enabled) or float(config.alpha) <= 0.0:
        return None

    def _callback(**kwargs):
        return apply_late_denoise_root_projection(
            vae=vae,
            config=config,
            model=kwargs["model"],
            step_input=kwargs["step_input"],
            first_chunk=kwargs["first_chunk"],
            commit_index=kwargs["commit_index"],
            current_step=kwargs["current_step"],
            start_index=kwargs["start_index"],
            end_index=kwargs["end_index"],
            noise_level_full=kwargs["noise_level_full"],
        )

    return _callback


def build_time_consistent_root_guidance_callback(
    *,
    vae,
    config: TimeConsistentRootGuidanceConfig,
):
    if (
        not bool(config.enabled)
        or float(config.alpha) <= 0.0
        or float(config.guidance_strength) <= 0.0
    ):
        return None

    debug_pending: list[dict[str, Any]] = []
    if bool(config.debug):
        debug_records = getattr(
            vae,
            "_time_consistent_guidance_debug_records",
            None,
        )
        if debug_records is None:
            debug_records = []
            setattr(vae, "_time_consistent_guidance_debug_records", debug_records)
    else:
        debug_records: list[dict[str, Any]] = []

    def _callback(**kwargs):
        return apply_time_consistent_root_guidance(
            vae=vae,
            config=config,
            model=kwargs["model"],
            step_input=kwargs["step_input"],
            first_chunk=kwargs["first_chunk"],
            commit_index=kwargs["commit_index"],
            current_step=kwargs["current_step"],
            start_index=kwargs["start_index"],
            end_index=kwargs["end_index"],
            x_beta_before_update=kwargs.get("x_beta_before_update"),
            predicted_vel=kwargs.get("predicted_vel"),
            beta_before=kwargs.get("beta_before"),
            beta_after=kwargs.get("beta_after"),
            x_after_velocity_update=kwargs.get("x_after_velocity_update"),
            debug_pending=debug_pending if bool(config.debug) else None,
        )

    def _finalize_commit_debug(*, model, commit_index: int) -> None:
        if not bool(config.debug):
            return
        token_idx = int(commit_index)
        if token_idx >= int(model.generated.shape[2]):
            return
        z_final = _ensure_token_state(
            model.generated[0, :, token_idx:token_idx + 1, ...].detach().cpu()
        )
        remaining: list[dict[str, Any]] = []
        for pending in debug_pending:
            if int(pending.get("commit_index", -1)) != token_idx:
                remaining.append(pending)
                continue
            z_hat = pending["z_hat"]
            z_hat_after = pending["z_hat_after"]
            delta = z_hat.reshape(-1).float() - z_final.reshape(-1).float()
            record = {
                key: value
                for key, value in pending.items()
                if key not in {"z_hat", "z_hat_after"}
            }
            record["z_hat_l2_to_z_final"] = float(torch.linalg.norm(delta).item())
            record["z_hat_cos_to_z_final"] = _flat_cos(z_hat, z_final)
            record["z_hat_after_l2_to_z_final"] = _flat_norm(z_hat_after - z_final)
            record["z_hat_after_cos_to_z_final"] = _flat_cos(z_hat_after, z_final)
            cache = _snapshot_vae_stream_cache(vae)
            try:
                z_hat_dev = z_hat.to(
                    device=model.generated.device,
                    dtype=model.generated.dtype,
                )
                z_final_dev = z_final.to(
                    device=model.generated.device,
                    dtype=model.generated.dtype,
                )
                decoded_hat = _decode_clean_token(
                    model,
                    vae,
                    z_hat_dev,
                    first_chunk=bool(pending.get("first_chunk", False)),
                ).detach()
                _restore_vae_stream_cache(vae, cache)
                decoded_final = _decode_clean_token(
                    model,
                    vae,
                    z_final_dev,
                    first_chunk=bool(pending.get("first_chunk", False)),
                ).detach()
            finally:
                _restore_vae_stream_cache(vae, cache)
            valid = min(int(decoded_hat.shape[0]) + 1, int(decoded_final.shape[0]) + 1)
            hat_xz = _decoded_root_xz_with_tail(decoded_hat, valid)
            final_xz = _decoded_root_xz_with_tail(decoded_final, valid)
            record["decode_z_hat_root_xz_l2_to_z_final"] = _mean_xz_error(
                hat_xz,
                final_xz,
            )
            n = min(int(decoded_hat.shape[0]), int(decoded_final.shape[0]))
            if n > 0:
                record["decode_z_hat_motion_l2_to_z_final"] = _flat_norm(
                    decoded_hat[:n].float().cpu() - decoded_final[:n].float().cpu()
                ) / float(n)
            else:
                record["decode_z_hat_motion_l2_to_z_final"] = 0.0
            debug_records.append(record)
        debug_pending[:] = remaining

    _callback.debug_records = debug_records
    _callback.finalize_commit_debug = _finalize_commit_debug
    return _callback


__all__ = [
    "CurrentTokenTarget",
    "LateDenoiseProjectionResult",
    "LateDenoiseRootProjectionConfig",
    "TimeConsistentGuidanceResult",
    "TimeConsistentRootGuidanceConfig",
    "_estimate_clean_latent_from_velocity",
    "_map_clean_delta_to_flow_state",
    "apply_late_denoise_root_projection",
    "apply_time_consistent_root_guidance",
    "build_late_denoise_root_projection_callback",
    "build_time_consistent_root_guidance_callback",
    "select_current_token_target",
]
