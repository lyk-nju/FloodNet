"""Condition-source builders for LDF-only stream-update experiments.

These helpers deliberately live under ``eval``. They define experimental 7D
conditions for probing LDF behavior and should not become training/runtime
model utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

from utils.motion_process import append_traj_deltas_5d_to_7d


_AUTO_SMOOTH_PROBE_FRAMES = 48
_AUTO_SMOOTH_MIN_FRAMES = 12
_AUTO_SMOOTH_MAX_FRAMES = 80


@dataclass(frozen=True)
class ConditionScenario:
    """Unified input produced by any external 7D condition source."""

    name: str
    condition_traj7: torch.Tensor
    update_frames: list[int] = field(default_factory=list)
    visual_mask: torch.Tensor | None = None
    base_sample_name: str | None = None
    caption_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _require_traj7(traj7: torch.Tensor, *, name: str = "traj7") -> torch.Tensor:
    if not torch.is_tensor(traj7):
        raise TypeError(f"{name} must be a torch.Tensor")
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"{name} must be [T,>=5], got {tuple(traj7.shape)}")
    if int(traj7.shape[0]) <= 0:
        raise ValueError(f"{name} must contain at least one frame")
    return traj7


def _yaw_from_7d(traj7: torch.Tensor) -> torch.Tensor:
    return torch.atan2(traj7[:, 4], traj7[:, 3])


def _path_yaw_from_xz(xz: torch.Tensor, *, min_speed: float = 1e-3) -> torch.Tensor:
    if xz.dim() != 2 or xz.shape[-1] != 2:
        raise ValueError(f"Expected xz [T,2], got {tuple(xz.shape)}")
    count = int(xz.shape[0])
    if count <= 1:
        return torch.zeros((count,), device=xz.device, dtype=xz.dtype)
    delta = torch.zeros_like(xz)
    delta[:-1] = xz[1:] - xz[:-1]
    delta[-1] = delta[-2]
    yaw = torch.atan2(delta[:, 0], delta[:, 1])
    speed = torch.linalg.norm(delta, dim=-1)
    out = yaw.clone()
    last = torch.zeros((), device=xz.device, dtype=xz.dtype)
    min_speed_t = torch.as_tensor(float(min_speed), device=xz.device, dtype=xz.dtype)
    for idx in range(count):
        if bool(speed[idx] > min_speed_t):
            last = yaw[idx]
        out[idx] = last
    return out


def _resample_transition_interval(
    values: torch.Tensor,
    *,
    transition_frames: int,
    transition_output_frames: int,
) -> torch.Tensor:
    source_steps = int(transition_frames)
    target_steps = int(transition_output_frames)
    if (
        source_steps <= 0
        or target_steps <= 0
        or source_steps == target_steps
        or int(values.shape[0]) <= 1
    ):
        return values
    source_steps = min(source_steps, int(values.shape[0]) - 1)
    positions = torch.linspace(
        0.0,
        float(source_steps),
        target_steps + 1,
        device=values.device,
        dtype=values.dtype,
    )
    left = positions.floor().long().clamp(0, source_steps)
    right = (left + 1).clamp(max=source_steps)
    alpha = (positions - left.to(values.dtype)).unsqueeze(-1)
    segment = (1.0 - alpha) * values[left] + alpha * values[right]
    return torch.cat([segment, values[source_steps + 1:]], dim=0)


def _smoothstep(progress: torch.Tensor) -> torch.Tensor:
    progress = progress.clamp(0.0, 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def _rotate_xz_delta(delta_xz: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    x = delta_xz[:, 0]
    z = delta_xz[:, 1]
    return torch.stack([cos_a * x + sin_a * z, -sin_a * x + cos_a * z], dim=-1)


def _stable_prefix_motion(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    speed_window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate stable pre-update speed, yaw, and y delta."""
    update = int(update_frame)
    window = max(1, int(speed_window))
    start = max(0, update - window)
    deltas = traj7[start + 1:update + 1, :3] - traj7[start:update, :3]
    if int(deltas.shape[0]) <= 0:
        deltas = traj7[update:update + 1, :3] - traj7[update - 1:update, :3]
    xz_delta = deltas[:, [0, 2]]
    speed = torch.linalg.norm(xz_delta, dim=-1).mean()
    mean_xz = xz_delta.mean(dim=0)
    if bool(torch.linalg.norm(mean_xz) > torch.as_tensor(1e-6, device=traj7.device)):
        yaw = torch.atan2(mean_xz[0], mean_xz[1])
    else:
        yaw = _yaw_from_7d(traj7[update - 1:update])[0]
    y_delta = deltas[:, 1].mean()
    return speed, yaw, y_delta


def _trim_leading_slow_deltas(
    deltas: torch.Tensor,
    *,
    min_speed: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Drop leading near-stationary XZ deltas while keeping at least one step."""
    if int(deltas.shape[0]) <= 1:
        return deltas, 0
    speed = torch.linalg.norm(deltas[:, [0, 2]], dim=-1)
    keep_index = 0
    threshold = min_speed.to(device=deltas.device, dtype=deltas.dtype)
    for idx in range(int(speed.shape[0]) - 1):
        if bool(speed[idx] >= threshold):
            keep_index = idx
            break
        keep_index = idx + 1
    return deltas[keep_index:], keep_index


def _mean_delta_yaw(
    deltas: torch.Tensor,
    *,
    fallback_yaw: torch.Tensor,
    window: int,
) -> torch.Tensor:
    count = max(1, min(int(window), int(deltas.shape[0])))
    mean_xz = deltas[:count, [0, 2]].mean(dim=0)
    if bool(torch.linalg.norm(mean_xz) > torch.as_tensor(1e-6, device=deltas.device)):
        return torch.atan2(mean_xz[0], mean_xz[1])
    return fallback_yaw


def _mean_delta_vector(
    deltas: torch.Tensor,
    *,
    window: int,
    from_end: bool = False,
) -> torch.Tensor:
    count = max(1, min(int(window), int(deltas.shape[0])))
    selected = deltas[-count:] if bool(from_end) else deltas[:count]
    return selected.mean(dim=0)


def _circular_mean_angle(angles: torch.Tensor) -> torch.Tensor:
    if int(angles.numel()) <= 0:
        return torch.zeros((), device=angles.device, dtype=angles.dtype)
    return torch.atan2(torch.sin(angles).mean(), torch.cos(angles).mean())


def _stable_prefix_heading(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    window: int,
) -> torch.Tensor:
    update = int(update_frame)
    count = max(1, int(window))
    start = max(0, update - count + 1)
    heading = _yaw_from_7d(traj7[start:update + 1])
    return _circular_mean_angle(heading)


def _reference_heading_tangent_residuals(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    window: int,
) -> torch.Tensor:
    """Return stable reference residuals ``heading - path_tangent``.

    Low-speed frames make path tangent numerically unstable, so residual samples
    are drawn from the faster part of the recent reference history.
    """
    update = int(update_frame)
    count = max(2, int(window))
    start = max(0, update - count)
    xz_delta = traj7[start + 1:update + 1, [0, 2]] - traj7[start:update, [0, 2]]
    if int(xz_delta.shape[0]) <= 0:
        return torch.zeros((1,), device=traj7.device, dtype=traj7.dtype)
    tangent = torch.atan2(xz_delta[:, 0], xz_delta[:, 1])
    heading = _yaw_from_7d(traj7[start:update])
    residual = torch.atan2(torch.sin(heading - tangent), torch.cos(heading - tangent))
    speed = torch.linalg.norm(xz_delta, dim=-1)
    max_speed = speed.max()
    eps = torch.as_tensor(1e-6, device=traj7.device, dtype=traj7.dtype)
    if bool(max_speed > eps):
        keep = speed >= max_speed * 0.5
        if bool(keep.any()):
            residual = residual[keep]
    if int(residual.numel()) <= 0:
        return torch.zeros((1,), device=traj7.device, dtype=traj7.dtype)
    return residual


def _project_deltas_to_yaw(
    deltas: torch.Tensor,
    *,
    source_yaw: torch.Tensor,
    target_yaw: torch.Tensor,
    lateral_scale: float = 0.0,
) -> torch.Tensor:
    """Express deltas in source frame, then rebuild them in target frame."""
    source_forward = torch.stack([torch.sin(source_yaw), torch.cos(source_yaw)])
    source_right = torch.stack([torch.cos(source_yaw), -torch.sin(source_yaw)])
    target_forward = torch.stack([torch.sin(target_yaw), torch.cos(target_yaw)])
    target_right = torch.stack([torch.cos(target_yaw), -torch.sin(target_yaw)])
    xz = deltas[:, [0, 2]]
    forward = (xz * source_forward[None, :]).sum(dim=-1)
    lateral = (xz * source_right[None, :]).sum(dim=-1) * float(lateral_scale)
    projected_xz = (
        forward[:, None] * target_forward[None, :]
        + lateral[:, None] * target_right[None, :]
    )
    out = deltas.clone()
    out[:, 0] = projected_xz[:, 0]
    out[:, 2] = projected_xz[:, 1]
    return out


def _finite_difference_hermite_points(
    start: torch.Tensor,
    end: torch.Tensor,
    start_delta: torch.Tensor,
    end_delta: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Cubic bridge with exact first/last finite-difference deltas."""
    n_steps = int(steps)
    if n_steps <= 0:
        return start[None, :]
    if n_steps == 1:
        return torch.stack([start, end], dim=0)
    start = start.to(dtype=end.dtype, device=end.device)
    start_delta = start_delta.to(dtype=end.dtype, device=end.device)
    end_delta = end_delta.to(dtype=end.dtype, device=end.device)
    n = torch.as_tensor(float(n_steps), dtype=end.dtype, device=end.device)
    one = torch.ones((), dtype=end.dtype, device=end.device)
    b1 = start_delta
    b2 = end - start - n * start_delta
    b3 = end_delta - start_delta
    matrix = torch.stack(
        [
            torch.stack([n**3 - n, n**2 - n], dim=0),
            torch.stack([3.0 * n**2 - 3.0 * n, 2.0 * n - 2.0 * one], dim=0),
        ],
        dim=0,
    )
    rhs = torch.stack(
        [
            b2,
            b3,
        ],
        dim=0,
    )
    coeff = torch.linalg.solve(matrix, rhs)
    a = coeff[0]
    b = coeff[1]
    c = b1 - a - b
    k = torch.arange(
        n_steps + 1,
        dtype=end.dtype,
        device=end.device,
    )[:, None]
    return a[None, :] * k**3 + b[None, :] * k**2 + c[None, :] * k + start[None, :]


def _resample_xyz_by_xz_step_lengths(
    xyz: torch.Tensor,
    *,
    step_lengths: torch.Tensor,
) -> torch.Tensor:
    """Resample xyz points with per-step XZ arc-length weights."""
    weights = step_lengths.to(device=xyz.device, dtype=xyz.dtype).view(-1)
    target_count = int(weights.numel()) + 1
    count = int(xyz.shape[0])
    if target_count <= 1 or count <= 1:
        return xyz
    xz = xyz[:, [0, 2]]
    seg_lengths = torch.linalg.norm(xz[1:] - xz[:-1], dim=-1)
    total_length = seg_lengths.sum()
    eps = torch.as_tensor(1e-8, device=xyz.device, dtype=xyz.dtype)
    if bool(total_length <= eps):
        return xyz

    cumulative = torch.cat(
        [
            torch.zeros((1,), device=xyz.device, dtype=xyz.dtype),
            torch.cumsum(seg_lengths, dim=0),
        ],
        dim=0,
    )
    weights = weights.clamp_min(0.0)
    weight_sum = weights.sum()
    if bool(weight_sum <= eps):
        weights = torch.ones_like(weights)
        weight_sum = weights.sum()
    step_distances = weights / weight_sum.clamp_min(eps) * total_length
    targets = torch.cat(
        [
            torch.zeros((1,), device=xyz.device, dtype=xyz.dtype),
            torch.cumsum(step_distances, dim=0),
        ],
        dim=0,
    )
    targets[-1] = total_length
    right = torch.searchsorted(cumulative, targets, right=True)
    right = right.clamp(min=1, max=count - 1)
    left = right - 1
    left_s = cumulative[left]
    right_s = cumulative[right]
    alpha = ((targets - left_s) / (right_s - left_s).clamp_min(eps)).unsqueeze(-1)
    out = (1.0 - alpha) * xyz[left] + alpha * xyz[right]
    out[0] = xyz[0]
    out[-1] = xyz[-1]
    return out


def _build_route_derived_7d_from_xyz(xyz: torch.Tensor) -> torch.Tensor:
    if xyz.dim() != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be [T,3], got {tuple(xyz.shape)}")
    yaw = _path_yaw_from_xz(xyz[:, [0, 2]])
    traj5 = torch.cat(
        [xyz, torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]],
        dim=-1,
    )
    return append_traj_deltas_5d_to_7d(traj5)


def _sanitize_update_frames(frames: Iterable[int] | None, total_frames: int) -> list[int]:
    if frames is None:
        return []
    total = max(0, int(total_frames))
    out = sorted({max(0, min(int(frame), max(0, total - 1))) for frame in frames})
    return out


def compose_center_symmetric_updated_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    source_start_frame: int = 0,
    derive_heading_from_path: bool = True,
    transition_frames: int = 0,
    transition_output_frames: int = 0,
) -> torch.Tensor:
    """Build a sample-derived S-like continuation by reflecting history.

    The prefix keeps the sample's original heading channels. When path-derived
    heading is enabled, it applies only to the reflected suffix; this preserves
    the original pre-update condition exactly.
    """
    traj7 = _require_traj7(traj7)
    total = int(traj7.shape[0])
    if total <= 1:
        raise ValueError("traj7 must contain at least two frames")
    update = max(1, min(int(update_frame), total - 1))
    start = max(0, min(int(source_start_frame), update))
    history = traj7[start:update + 1]
    if suffix_frames is not None:
        suffix_cap = max(1, int(suffix_frames))
        if int(history.shape[0]) > suffix_cap:
            history = history[-suffix_cap:]

    anchor = traj7[update]
    reflected_history = torch.flip(history, dims=[0])
    suffix_xyz = reflected_history[:, :3].clone()
    suffix_xyz[:, 0] = 2.0 * anchor[0] - reflected_history[:, 0]
    suffix_xyz[:, 2] = 2.0 * anchor[2] - reflected_history[:, 2]
    suffix_xyz = _resample_transition_interval(
        suffix_xyz,
        transition_frames=int(transition_frames),
        transition_output_frames=int(transition_output_frames),
    )

    if bool(derive_heading_from_path):
        suffix_7d = _build_route_derived_7d_from_xyz(suffix_xyz)
    else:
        suffix_yaw = _path_yaw_from_xz(suffix_xyz[:, [0, 2]])
        suffix_5d = torch.cat(
            [
                suffix_xyz,
                torch.cos(suffix_yaw)[:, None],
                torch.sin(suffix_yaw)[:, None],
            ],
            dim=-1,
        )
        suffix_7d = append_traj_deltas_5d_to_7d(suffix_5d)
    combined_5d = torch.cat([traj7[:update, :5], suffix_7d[:, :5]], dim=0)
    return append_traj_deltas_5d_to_7d(combined_5d)


def compose_rotated_suffix_updated_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    source_start_frame: int = 0,
    suffix_rotation_deg: float = 0.0,
    turn_transition_frames: int = 0,
) -> torch.Tensor:
    """Append a sample-derived suffix that gradually turns after the update.

    The suffix is built from source deltas, rotated in the update anchor's local
    frame, and then integrated. The pre-update prefix keeps the sample's
    original heading channels; heading and delta channels are regenerated only
    for the new suffix so position and facing stay coupled after the update.
    """
    traj7 = _require_traj7(traj7)
    total = int(traj7.shape[0])
    if total <= 1:
        raise ValueError("traj7 must contain at least two frames")
    update = max(1, min(int(update_frame), total - 1))
    start = max(0, min(int(source_start_frame), update - 1))
    source = traj7[start:update + 1, :3]
    if suffix_frames is not None:
        suffix_cap = max(2, int(suffix_frames))
        if int(source.shape[0]) > suffix_cap:
            source = source[-suffix_cap:]
    if int(source.shape[0]) <= 1:
        raise ValueError("rotated_suffix requires at least two source frames")

    source_deltas = source[1:] - source[:-1]
    delta_xz = source_deltas[:, [0, 2]]
    target_angle = math.radians(float(suffix_rotation_deg))
    transition = max(0, int(turn_transition_frames))
    if abs(target_angle) <= 1e-12:
        angles = torch.zeros(
            int(delta_xz.shape[0]),
            device=delta_xz.device,
            dtype=delta_xz.dtype,
        )
    elif transition <= 0:
        angles = torch.full(
            (int(delta_xz.shape[0]),),
            float(target_angle),
            device=delta_xz.device,
            dtype=delta_xz.dtype,
        )
    else:
        step = torch.arange(
            int(delta_xz.shape[0]),
            device=delta_xz.device,
            dtype=delta_xz.dtype,
        )
        progress = step / float(max(transition, 1))
        angles = _smoothstep(progress) * float(target_angle)

    rotated_xz = _rotate_xz_delta(delta_xz, angles)
    rotated_deltas = source_deltas.clone()
    rotated_deltas[:, 0] = rotated_xz[:, 0]
    rotated_deltas[:, 2] = rotated_xz[:, 1]

    anchor = traj7[update, :3]
    suffix_xyz = torch.cat(
        [
            anchor[None, :],
            anchor[None, :] + torch.cumsum(rotated_deltas, dim=0),
        ],
        dim=0,
    )
    suffix_7d = _build_route_derived_7d_from_xyz(suffix_xyz)
    combined_5d = torch.cat([traj7[:update, :5], suffix_7d[:, :5]], dim=0)
    return append_traj_deltas_5d_to_7d(combined_5d)


@dataclass(frozen=True)
class _RotatedRepeatSegment:
    source_5d: torch.Tensor
    deltas: torch.Tensor
    angle_rad: torch.Tensor
    trim_start: int


def _build_rotated_repeat_segment(
    source: torch.Tensor,
    *,
    angle_rad: torch.Tensor,
    min_speed: torch.Tensor,
) -> _RotatedRepeatSegment:
    """Build the locked repeat segment from sample deltas.

    This is the "second segment condition" builder: it may trim unstable
    leading source deltas, then applies one rigid XZ rotation to all remaining
    deltas. It does not know anything about the transition bridge.
    """
    source_5d = source[:, :5]
    source_deltas = source[:, :3][1:] - source[:, :3][:-1]
    source_deltas, trim_start = _trim_leading_slow_deltas(
        source_deltas,
        min_speed=min_speed,
    )
    source_5d = source_5d[trim_start:]

    rotated_xz = _rotate_xz_delta(source_deltas[:, [0, 2]], angle_rad)
    repeat_deltas = source_deltas.clone()
    repeat_deltas[:, 0] = rotated_xz[:, 0]
    repeat_deltas[:, 2] = rotated_xz[:, 1]
    return _RotatedRepeatSegment(
        source_5d=source_5d,
        deltas=repeat_deltas,
        angle_rad=angle_rad,
        trim_start=int(trim_start),
    )


def _integrate_rotated_repeat_5d(
    repeat: _RotatedRepeatSegment,
    *,
    start_xyz: torch.Tensor,
    seam_yaw: torch.Tensor | None = None,
    heading_blend_frames: int = 0,
) -> torch.Tensor:
    """Integrate locked repeat deltas from ``start_xyz`` with rotated heading."""
    repeat_xyz = torch.cat(
        [
            start_xyz[None, :],
            start_xyz[None, :] + torch.cumsum(repeat.deltas, dim=0),
        ],
        dim=0,
    )
    repeat_yaw = _yaw_from_7d(repeat.source_5d) + repeat.angle_rad
    blend_frames = max(0, int(heading_blend_frames))
    if seam_yaw is not None and blend_frames > 0 and int(repeat_yaw.numel()) > 1:
        seam = seam_yaw.to(device=repeat_yaw.device, dtype=repeat_yaw.dtype)
        frame_idx = torch.arange(
            repeat_yaw.shape[0],
            device=repeat_yaw.device,
            dtype=repeat_yaw.dtype,
        )
        # repeat_yaw[0] is duplicate seam state and is not appended downstream.
        # The first visible repeat frame is repeat_yaw[1], so start blending
        # there to remove the repeat_start heading jump.
        progress = ((frame_idx - 1.0) / float(max(blend_frames, 1))).clamp(0.0, 1.0)
        alpha = _smoothstep(progress)
        yaw_delta = torch.atan2(torch.sin(repeat_yaw - seam), torch.cos(repeat_yaw - seam))
        repeat_yaw = seam + alpha * yaw_delta
    return torch.cat(
        [
            repeat_xyz,
            torch.cos(repeat_yaw)[:, None],
            torch.sin(repeat_yaw)[:, None],
        ],
        dim=1,
    )


def _build_smooth_bridge_5d(
    traj7: torch.Tensor,
    repeat: _RotatedRepeatSegment,
    *,
    update_frame: int,
    transition_frames: int,
    speed_window: int,
    prefix_yaw: torch.Tensor,
    target_yaw: torch.Tensor,
    smooth_profile: str = "geometric",
) -> torch.Tensor:
    """Build only the bridge from reference tail into repeat head.

    The bridge endpoint and finite-difference tangents are derived from:
      point 1 = reference tail at ``update_frame``;
      tangent 1 = mean XZ/XYZ delta over the reference tail window;
      tangent 2 = mean XZ/XYZ delta over the repeat head window.

    Heading on the bridge is path-derived because there is no source heading
    ground truth for synthesized transition frames.
    """
    update = int(update_frame)
    transition = max(0, int(transition_frames))
    profile = str(smooth_profile)
    anchor = traj7[update, :3]
    if transition <= 0:
        smooth_xyz = anchor[None, :]
        first_repeat_xz = repeat.deltas[0, [0, 2]]
        if bool(
            torch.linalg.norm(first_repeat_xz)
            > torch.as_tensor(1e-6, device=traj7.device)
        ):
            smooth_yaw = torch.atan2(first_repeat_xz[0], first_repeat_xz[1])[None]
        else:
            smooth_yaw = target_yaw[None]
    else:
        window = max(1, int(speed_window))
        ref_delta_start = max(0, update - window)
        ref_deltas = traj7[ref_delta_start + 1:update + 1, :3] - traj7[
            ref_delta_start:update,
            :3,
        ]
        start_delta = _mean_delta_vector(
            ref_deltas,
            window=window,
            from_end=True,
        )
        end_delta = _mean_delta_vector(
            repeat.deltas,
            window=window,
        )
        if profile == "geometric" and int(repeat.deltas.shape[0]) > 0:
            end_delta = repeat.deltas[0]
        start_xz = start_delta[[0, 2]]
        end_xz = end_delta[[0, 2]]
        eps = torch.as_tensor(1e-6, device=traj7.device, dtype=traj7.dtype)
        if profile == "geometric":
            if bool(torch.linalg.norm(start_xz) > eps):
                start_yaw = torch.atan2(start_xz[0], start_xz[1])
            else:
                start_yaw = prefix_yaw
            if bool(torch.linalg.norm(end_xz) > eps):
                end_yaw = torch.atan2(end_xz[0], end_xz[1])
            else:
                end_yaw = target_yaw

            step = torch.arange(
                transition,
                device=traj7.device,
                dtype=traj7.dtype,
            )
            progress = step / float(max(transition - 1, 1))
            alpha = _smoothstep(progress)
            yaw_delta = torch.atan2(
                torch.sin(end_yaw - start_yaw),
                torch.cos(end_yaw - start_yaw),
            )
            yaw = start_yaw + alpha * yaw_delta
            start_speed = torch.linalg.norm(start_xz)
            end_speed = torch.linalg.norm(end_xz)
            delta_speed = start_speed + alpha * (end_speed - start_speed)
            delta_y = start_delta[1] + alpha * (end_delta[1] - start_delta[1])
            baseline_deltas = torch.stack(
                [
                    torch.sin(yaw) * delta_speed,
                    delta_y,
                    torch.cos(yaw) * delta_speed,
                ],
                dim=-1,
            )
            smooth_end = anchor + baseline_deltas.sum(dim=0)
            smooth_xyz = _finite_difference_hermite_points(
                anchor,
                smooth_end,
                start_delta,
                end_delta,
                steps=transition,
            )
            speed_ramp_alpha = torch.linspace(
                0.0,
                1.0,
                transition,
                device=traj7.device,
                dtype=traj7.dtype,
            )
            step_speeds = start_speed + speed_ramp_alpha * (end_speed - start_speed)
            smooth_xyz = _resample_xyz_by_xz_step_lengths(
                smooth_xyz,
                step_lengths=step_speeds,
            )
            smooth_yaw = _path_yaw_from_xz(smooth_xyz[:, [0, 2]])
        elif profile == "heading_residual":
            start_heading = _stable_prefix_heading(
                traj7,
                update_frame=update,
                window=window,
            )
            repeat_yaw = _yaw_from_7d(repeat.source_5d) + repeat.angle_rad
            end_heading = repeat_yaw[0] if int(repeat_yaw.numel()) > 0 else target_yaw
            frame_step = torch.arange(
                transition + 1,
                device=traj7.device,
                dtype=traj7.dtype,
            )
            frame_alpha = _smoothstep(frame_step / float(max(transition, 1)))
            heading_delta = torch.atan2(
                torch.sin(end_heading - start_heading),
                torch.cos(end_heading - start_heading),
            )
            smooth_yaw = start_heading + frame_alpha * heading_delta

            residual_source = _reference_heading_tangent_residuals(
                traj7,
                update_frame=update,
                window=max(window * 4, transition),
            )
            residual_idx = torch.arange(transition, device=traj7.device) % int(
                residual_source.numel()
            )
            residual = residual_source[residual_idx]
            tangent_yaw = smooth_yaw[:-1] - residual
            start_speed = torch.linalg.norm(start_xz)
            end_speed = torch.linalg.norm(end_xz)
            speed_alpha = torch.linspace(
                0.0,
                1.0,
                transition,
                device=traj7.device,
                dtype=traj7.dtype,
            )
            step_speeds = start_speed + speed_alpha * (end_speed - start_speed)
            delta_y = start_delta[1] + speed_alpha * (end_delta[1] - start_delta[1])
            smooth_deltas = torch.stack(
                [
                    torch.sin(tangent_yaw) * step_speeds,
                    delta_y,
                    torch.cos(tangent_yaw) * step_speeds,
                ],
                dim=-1,
            )
            smooth_xyz = torch.cat(
                [
                    anchor[None, :],
                    anchor[None, :] + torch.cumsum(smooth_deltas, dim=0),
                ],
                dim=0,
            )
        else:
            raise ValueError(
                "arc_smooth_profile must be 'geometric' or "
                f"'heading_residual', got {profile!r}"
            )

    return torch.cat(
        [
            smooth_xyz,
            torch.cos(smooth_yaw)[:, None],
            torch.sin(smooth_yaw)[:, None],
        ],
        dim=-1,
    )


def _mean_tail_xz_speed(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    window: int,
) -> torch.Tensor:
    update = int(update_frame)
    count = max(1, int(window))
    start = max(0, update - count)
    xz = traj7[start:update + 1, [0, 2]]
    if int(xz.shape[0]) <= 1:
        return torch.zeros((), device=traj7.device, dtype=traj7.dtype)
    return torch.linalg.norm(xz[1:] - xz[:-1], dim=-1).mean()


def _mean_repeat_head_xz_speed(
    repeat: _RotatedRepeatSegment,
    *,
    window: int,
) -> torch.Tensor:
    count = max(1, min(int(window), int(repeat.deltas.shape[0])))
    xz_delta = repeat.deltas[:count, [0, 2]]
    if int(xz_delta.shape[0]) <= 0:
        return torch.zeros((), device=repeat.deltas.device, dtype=repeat.deltas.dtype)
    return torch.linalg.norm(xz_delta, dim=-1).mean()


def _smooth_bridge_xz_path_length(smooth_5d: torch.Tensor) -> torch.Tensor:
    if int(smooth_5d.shape[0]) <= 1:
        return torch.zeros((), device=smooth_5d.device, dtype=smooth_5d.dtype)
    xz = smooth_5d[:, [0, 2]]
    return torch.linalg.norm(xz[1:] - xz[:-1], dim=-1).sum()


def _resolve_smooth_transition_frames(
    configured_frames: int,
    traj7: torch.Tensor,
    repeat: _RotatedRepeatSegment,
    *,
    update_frame: int,
    speed_window: int,
    prefix_yaw: torch.Tensor,
    target_yaw: torch.Tensor,
    smooth_profile: str = "geometric",
) -> int:
    configured = int(configured_frames)
    if configured >= 0:
        return configured

    probe_smooth = _build_smooth_bridge_5d(
        traj7,
        repeat,
        update_frame=int(update_frame),
        transition_frames=_AUTO_SMOOTH_PROBE_FRAMES,
        speed_window=int(speed_window),
        prefix_yaw=prefix_yaw,
        target_yaw=target_yaw,
        smooth_profile=str(smooth_profile),
    )
    path_length = _smooth_bridge_xz_path_length(probe_smooth)
    tail_speed = _mean_tail_xz_speed(
        traj7,
        update_frame=int(update_frame),
        window=int(speed_window),
    )
    repeat_head_speed = _mean_repeat_head_xz_speed(
        repeat,
        window=int(speed_window),
    )
    target_speed = 0.5 * (tail_speed + repeat_head_speed)
    eps = torch.as_tensor(1e-6, device=traj7.device, dtype=traj7.dtype)
    frames = int(round(float(path_length / target_speed.clamp_min(eps))))
    return max(_AUTO_SMOOTH_MIN_FRAMES, min(frames, _AUTO_SMOOTH_MAX_FRAMES))


def _compose_reference_smooth_repeat_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    smooth_5d: torch.Tensor,
    repeat_5d: torch.Tensor,
) -> torch.Tensor:
    # The repeat start is already the final smooth point. Keep one copy with the
    # smooth segment so the boundary position stays exact.
    suffix_5d = torch.cat([smooth_5d, repeat_5d[1:]], dim=0)
    combined_5d = torch.cat([traj7[: int(update_frame), :5], suffix_5d], dim=0)
    return append_traj_deltas_5d_to_7d(combined_5d)


def _compose_rotated_suffix_arc_once(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    source_start_frame: int = 0,
    suffix_rotation_deg: float = 0.0,
    turn_transition_frames: int = 0,
    speed_window: int = 8,
    speed_scale: float = 1.0,
    suffix_blend_frames: int = 8,
    suffix_min_speed_factor: float = 0.25,
    suffix_lateral_scale: float = 0.0,
    arc_y_mode: str = "source",
    repeat_heading_blend_frames: int = 0,
    arc_smooth_profile: str = "geometric",
) -> tuple[torch.Tensor, dict[str, Any]]:
    traj7 = _require_traj7(traj7)
    total = int(traj7.shape[0])
    if total <= 1:
        raise ValueError("traj7 must contain at least two frames")
    update = max(1, min(int(update_frame), total - 1))
    start = max(0, min(int(source_start_frame), update - 1))
    source = traj7[start:update + 1]
    if suffix_frames is not None:
        suffix_cap = max(2, int(suffix_frames))
        if int(source.shape[0]) > suffix_cap:
            source = source[-suffix_cap:]
    if int(source.shape[0]) <= 1:
        raise ValueError("rotated_suffix_arc requires at least two source frames")

    target_angle = math.radians(float(suffix_rotation_deg))
    # ``suffix_blend_frames`` and ``suffix_lateral_scale`` are kept as public
    # experiment knobs for CLI/metadata compatibility. The current arc bridge
    # is controlled by endpoint positions and averaged XZ tangents instead.
    _ = suffix_blend_frames, suffix_lateral_scale

    speed, prefix_yaw, _y_delta = _stable_prefix_motion(
        traj7,
        update_frame=update,
        speed_window=int(speed_window),
    )
    speed = speed * torch.as_tensor(
        float(speed_scale),
        device=traj7.device,
        dtype=traj7.dtype,
    )
    target_yaw = prefix_yaw + torch.as_tensor(
        float(target_angle),
        device=traj7.device,
        dtype=traj7.dtype,
    )
    min_suffix_speed = speed * torch.as_tensor(
        float(suffix_min_speed_factor),
        device=traj7.device,
        dtype=traj7.dtype,
    )
    angle_tensor = torch.as_tensor(
        float(target_angle),
        device=traj7.device,
        dtype=traj7.dtype,
    )
    repeat = _build_rotated_repeat_segment(
        source,
        angle_rad=angle_tensor,
        min_speed=min_suffix_speed,
    )
    resolved_transition_frames = _resolve_smooth_transition_frames(
        int(turn_transition_frames),
        traj7,
        repeat,
        update_frame=update,
        speed_window=int(speed_window),
        prefix_yaw=prefix_yaw,
        target_yaw=target_yaw,
        smooth_profile=str(arc_smooth_profile),
    )
    smooth_5d = _build_smooth_bridge_5d(
        traj7,
        repeat,
        update_frame=update,
        transition_frames=resolved_transition_frames,
        speed_window=int(speed_window),
        prefix_yaw=prefix_yaw,
        target_yaw=target_yaw,
        smooth_profile=str(arc_smooth_profile),
    )
    repeat_5d = _integrate_rotated_repeat_5d(
        repeat,
        start_xyz=smooth_5d[-1, :3],
        seam_yaw=_yaw_from_7d(smooth_5d[-1:])[0],
        heading_blend_frames=int(repeat_heading_blend_frames),
    )
    arc_y_mode = str(arc_y_mode)
    if arc_y_mode == "anchor":
        anchor_y = traj7[update, 1]
        smooth_5d = smooth_5d.clone()
        repeat_5d = repeat_5d.clone()
        smooth_5d[:, 1] = anchor_y
        repeat_5d[:, 1] = anchor_y
    elif arc_y_mode != "source":
        raise ValueError(
            f"arc_y_mode must be 'source' or 'anchor', got {arc_y_mode!r}"
        )
    condition = _compose_reference_smooth_repeat_traj7(
        traj7,
        update_frame=update,
        smooth_5d=smooth_5d,
        repeat_5d=repeat_5d,
    )
    metadata = {
        "update_frame": int(update),
        "repeat_start_frame": int(update + resolved_transition_frames),
        "resolved_turn_transition_frames": int(resolved_transition_frames),
        "source_start_frame": int(start),
        "effective_source_start_frame": int(start + repeat.trim_start),
        "stable_skip_deltas": int(repeat.trim_start),
        "repeat_delta_count": int(repeat.deltas.shape[0]),
        "suffix_rotation_deg": float(suffix_rotation_deg),
        "arc_y_mode": arc_y_mode,
        "repeat_heading_blend_frames": int(repeat_heading_blend_frames),
        "arc_smooth_profile": str(arc_smooth_profile),
    }
    return condition, metadata


def compose_rotated_suffix_arc_updated_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    source_start_frame: int = 0,
    suffix_rotation_deg: float = 0.0,
    turn_transition_frames: int = 0,
    speed_window: int = 8,
    speed_scale: float = 1.0,
    suffix_blend_frames: int = 8,
    suffix_min_speed_factor: float = 0.25,
    suffix_lateral_scale: float = 0.0,
    arc_y_mode: str = "source",
    repeat_heading_blend_frames: int = 0,
    arc_smooth_profile: str = "geometric",
) -> torch.Tensor:
    """Append an unchanged rotated suffix after a forward-moving transition.

    The condition is split into three protected regions:
      reference: original frames before ``update_frame``;
      smooth: an inserted transition from the reference tangent into repeat;
      repeat: source frames rigidly transformed by ``suffix_rotation_deg``.

    Only the smooth region is synthesized. The repeat region keeps the source
    XZ shape and heading under one rigid rotation, so later edits cannot
    accidentally straighten the reused trajectory.
    """
    condition, _metadata = _compose_rotated_suffix_arc_once(
        traj7,
        update_frame=update_frame,
        suffix_frames=suffix_frames,
        source_start_frame=source_start_frame,
        suffix_rotation_deg=suffix_rotation_deg,
        turn_transition_frames=turn_transition_frames,
        speed_window=speed_window,
        speed_scale=speed_scale,
        suffix_blend_frames=suffix_blend_frames,
        suffix_min_speed_factor=suffix_min_speed_factor,
        suffix_lateral_scale=suffix_lateral_scale,
        arc_y_mode=str(arc_y_mode),
        repeat_heading_blend_frames=int(repeat_heading_blend_frames),
        arc_smooth_profile=str(arc_smooth_profile),
    )
    return condition


def _coerce_repeat_angle_choices(
    value: Iterable[float] | str | None,
    *,
    fallback_angle: float,
) -> list[float]:
    if value is None:
        return [float(fallback_angle)]
    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in {"", "none", "null"}:
            return [float(fallback_angle)]
        choices = [float(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        choices = [float(item) for item in value]
    if not choices:
        return [float(fallback_angle)]
    return choices


def _sample_repeat_rotation_degrees(
    *,
    repeat_count: int,
    angle_choices: Iterable[float] | str | None,
    fallback_angle: float,
    seed: int,
) -> list[float]:
    count = max(1, int(repeat_count))
    choices = _coerce_repeat_angle_choices(
        angle_choices,
        fallback_angle=float(fallback_angle),
    )
    if len(choices) == 1:
        return [float(choices[0]) for _ in range(count)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    choice_idx = torch.randint(
        0,
        len(choices),
        (count,),
        generator=generator,
    ).tolist()
    return [float(choices[int(idx)]) for idx in choice_idx]


def _compose_rotated_suffix_arc_chain(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    source_start_frame: int = 0,
    suffix_rotation_deg: float = 0.0,
    turn_transition_frames: int = 0,
    speed_window: int = 8,
    speed_scale: float = 1.0,
    suffix_blend_frames: int = 8,
    suffix_min_speed_factor: float = 0.25,
    suffix_lateral_scale: float = 0.0,
    arc_y_mode: str = "source",
    repeat_heading_blend_frames: int = 0,
    arc_smooth_profile: str = "geometric",
    repeat_count: int = 4,
    repeat_angle_choices: Iterable[float] | str | None = None,
    repeat_seed: int = 1234,
) -> tuple[torch.Tensor, dict[str, Any]]:
    current = _require_traj7(traj7)
    rotations = _sample_repeat_rotation_degrees(
        repeat_count=int(repeat_count),
        angle_choices=repeat_angle_choices,
        fallback_angle=float(suffix_rotation_deg),
        seed=int(repeat_seed),
    )
    next_update = int(update_frame)
    next_source_start = int(source_start_frame)
    chain_updates: list[int] = []
    chain_repeat_starts: list[int] = []
    chain_transition_frames: list[int] = []
    chain_effective_source_starts: list[int] = []
    chain_stable_skip_deltas: list[int] = []

    for rotation in rotations:
        current, step_meta = _compose_rotated_suffix_arc_once(
            current,
            update_frame=int(next_update),
            suffix_frames=suffix_frames,
            source_start_frame=int(next_source_start),
            suffix_rotation_deg=float(rotation),
            turn_transition_frames=int(turn_transition_frames),
            speed_window=int(speed_window),
            speed_scale=float(speed_scale),
            suffix_blend_frames=int(suffix_blend_frames),
            suffix_min_speed_factor=float(suffix_min_speed_factor),
            suffix_lateral_scale=float(suffix_lateral_scale),
            arc_y_mode=str(arc_y_mode),
            repeat_heading_blend_frames=int(repeat_heading_blend_frames),
            arc_smooth_profile=str(arc_smooth_profile),
        )
        chain_updates.append(int(step_meta["update_frame"]))
        chain_repeat_starts.append(int(step_meta["repeat_start_frame"]))
        chain_transition_frames.append(
            int(step_meta["resolved_turn_transition_frames"])
        )
        chain_effective_source_starts.append(
            int(step_meta["effective_source_start_frame"])
        )
        chain_stable_skip_deltas.append(int(step_meta["stable_skip_deltas"]))
        next_update = int(current.shape[0]) - 1
        next_source_start = int(step_meta["repeat_start_frame"])

    return current, {
        "chain_update_frames": chain_updates,
        "chain_repeat_start_frames": chain_repeat_starts,
        "chain_transition_frames": chain_transition_frames,
        "chain_effective_source_start_frames": chain_effective_source_starts,
        "chain_stable_skip_deltas": chain_stable_skip_deltas,
        "chain_rotation_degrees": rotations,
        "repeat_count": int(len(rotations)),
        "repeat_angle_choices": _coerce_repeat_angle_choices(
            repeat_angle_choices,
            fallback_angle=float(suffix_rotation_deg),
        ),
        "repeat_seed": int(repeat_seed),
        "arc_y_mode": str(arc_y_mode),
        "repeat_heading_blend_frames": int(repeat_heading_blend_frames),
        "arc_smooth_profile": str(arc_smooth_profile),
    }


def compose_forward_line_traj7(
    base_traj7: torch.Tensor | None = None,
    *,
    num_frames: int,
    step_length: float,
) -> torch.Tensor:
    """Hand-author a straight forward route from a base pose or the origin."""
    frames = max(2, int(num_frames))
    step = float(step_length)
    if step < 0.0:
        raise ValueError(f"step_length must be >= 0, got {step_length}")
    if base_traj7 is None:
        device = torch.device("cpu")
        dtype = torch.float32
        anchor = torch.zeros((5,), device=device, dtype=dtype)
        anchor[3] = 1.0
        yaw = torch.zeros((), device=device, dtype=dtype)
    else:
        base_traj7 = _require_traj7(base_traj7, name="base_traj7")
        anchor = base_traj7[0, :5]
        yaw = _yaw_from_7d(base_traj7[:1])[0]
    frame_idx = torch.arange(frames, device=anchor.device, dtype=anchor.dtype)
    distance = frame_idx * torch.as_tensor(step, device=anchor.device, dtype=anchor.dtype)
    x = anchor[0] + torch.sin(yaw) * distance
    y = anchor[1].expand_as(x)
    z = anchor[2] + torch.cos(yaw) * distance
    heading = yaw.expand_as(x)
    traj5 = torch.stack(
        [x, y, z, torch.cos(heading), torch.sin(heading)],
        dim=-1,
    )
    return append_traj_deltas_5d_to_7d(traj5)


def compose_four_segment_forward_curve_traj7(
    *,
    num_frames: int = 240,
    total_forward: float = 4.2,
    primary_amplitude: float = 0.22,
    secondary_amplitude: float = 0.08,
) -> torch.Tensor:
    """Hand-author a long forward route with four mild-curvature segments."""
    frames = max(2, int(num_frames))
    frame_idx = torch.arange(frames, dtype=torch.float32)
    progress = frame_idx / float(max(frames - 1, 1))
    z = torch.as_tensor(float(total_forward), dtype=torch.float32) * progress
    x = (
        torch.as_tensor(float(primary_amplitude), dtype=torch.float32)
        * torch.sin(2.0 * math.pi * (progress - 0.08))
        + torch.as_tensor(float(secondary_amplitude), dtype=torch.float32)
        * torch.sin(4.0 * math.pi * progress + 0.35)
    )
    x = x - x[:1]
    y = torch.zeros_like(x)
    xyz = torch.stack([x, y, z - z[:1]], dim=-1)
    return _build_route_derived_7d_from_xyz(xyz)


def compose_constant_curvature_arc_traj7(
    *,
    num_frames: int = 240,
    arc_length: float = 4.2,
    turn_degrees: float = 20.0,
) -> torch.Tensor:
    """Hand-author a uniformly sampled circular arc.

    ``arc_length`` is the XZ path length, not just the final z displacement.
    Positive ``turn_degrees`` bends the route toward +x while starting from a
    forward-facing heading along +z.
    """
    frames = max(2, int(num_frames))
    length = max(0.0, float(arc_length))
    turn = math.radians(float(turn_degrees))
    frame_idx = torch.arange(frames, dtype=torch.float32)
    progress = frame_idx / float(max(frames - 1, 1))
    if abs(turn) < 1e-7 or length <= 0.0:
        z = torch.as_tensor(length, dtype=torch.float32) * progress
        x = torch.zeros_like(z)
        yaw = torch.zeros_like(z)
    else:
        phi = torch.as_tensor(turn, dtype=torch.float32) * progress
        curvature = torch.as_tensor(turn / length, dtype=torch.float32)
        x = (1.0 - torch.cos(phi)) / curvature
        z = torch.sin(phi) / curvature
        yaw = phi
    y = torch.zeros_like(x)
    traj5 = torch.stack(
        [x, y, z, torch.cos(yaw), torch.sin(yaw)],
        dim=-1,
    )
    return append_traj_deltas_5d_to_7d(traj5)


def build_sample_condition(
    traj7: torch.Tensor,
    *,
    valid_frames: int | None = None,
    sample_name: str | None = None,
    caption_index: int | None = None,
) -> ConditionScenario:
    """Use the original sample condition without route updates."""
    traj7 = _require_traj7(traj7)
    count = int(traj7.shape[0]) if valid_frames is None else int(valid_frames)
    count = max(1, min(count, int(traj7.shape[0])))
    condition = append_traj_deltas_5d_to_7d(traj7[:count, :5])
    return ConditionScenario(
        name="sample",
        condition_traj7=condition,
        update_frames=[],
        base_sample_name=sample_name,
        caption_index=caption_index,
        metadata={"valid_frames": count},
    )


def build_repeat_splice_condition(
    traj7: torch.Tensor,
    *,
    mode: str = "center_symmetric",
    update_frame: int,
    suffix_frames: int | None = None,
    source_start_frame: int = 0,
    derive_heading_from_path: bool = True,
    transition_frames: int = 0,
    transition_output_frames: int = 0,
    suffix_rotation_deg: float = 0.0,
    turn_transition_frames: int = 0,
    arc_speed_window: int = 8,
    arc_speed_scale: float = 1.0,
    suffix_blend_frames: int = 8,
    suffix_min_speed_factor: float = 0.25,
    suffix_lateral_scale: float = 0.0,
    arc_y_mode: str = "source",
    repeat_heading_blend_frames: int = 0,
    arc_smooth_profile: str = "geometric",
    repeat_count: int = 4,
    repeat_angle_choices: Iterable[float] | str | None = None,
    repeat_seed: int = 1234,
    sample_name: str | None = None,
    caption_index: int | None = None,
) -> ConditionScenario:
    """Build a condition update by reusing and splicing a real sample route."""
    traj7 = _require_traj7(traj7)
    mode = str(mode)
    if mode not in {
        "center_symmetric",
        "reflected_history",
        "rotated_suffix",
        "rotated_suffix_arc",
        "rotated_suffix_arc_chain",
    }:
        raise ValueError(
            "repeat-splice source currently supports only "
            "'center_symmetric'/'reflected_history'/'rotated_suffix'/"
            f"'rotated_suffix_arc'/'rotated_suffix_arc_chain', got {mode!r}"
        )
    update = max(1, min(int(update_frame), int(traj7.shape[0]) - 1))
    chain_metadata: dict[str, Any] = {}
    scenario_update_frames = [update]
    if mode == "rotated_suffix":
        condition = compose_rotated_suffix_updated_traj7(
            traj7,
            update_frame=update,
            suffix_frames=suffix_frames,
            source_start_frame=int(source_start_frame),
            suffix_rotation_deg=float(suffix_rotation_deg),
            turn_transition_frames=int(turn_transition_frames),
        )
    elif mode == "rotated_suffix_arc":
        condition = compose_rotated_suffix_arc_updated_traj7(
            traj7,
            update_frame=update,
            suffix_frames=suffix_frames,
            source_start_frame=int(source_start_frame),
            suffix_rotation_deg=float(suffix_rotation_deg),
            turn_transition_frames=int(turn_transition_frames),
            speed_window=int(arc_speed_window),
            speed_scale=float(arc_speed_scale),
            suffix_blend_frames=int(suffix_blend_frames),
            suffix_min_speed_factor=float(suffix_min_speed_factor),
            suffix_lateral_scale=float(suffix_lateral_scale),
            arc_y_mode=str(arc_y_mode),
            repeat_heading_blend_frames=int(repeat_heading_blend_frames),
            arc_smooth_profile=str(arc_smooth_profile),
        )
    elif mode == "rotated_suffix_arc_chain":
        condition, chain_metadata = _compose_rotated_suffix_arc_chain(
            traj7,
            update_frame=update,
            suffix_frames=suffix_frames,
            source_start_frame=int(source_start_frame),
            suffix_rotation_deg=float(suffix_rotation_deg),
            turn_transition_frames=int(turn_transition_frames),
            speed_window=int(arc_speed_window),
            speed_scale=float(arc_speed_scale),
            suffix_blend_frames=int(suffix_blend_frames),
            suffix_min_speed_factor=float(suffix_min_speed_factor),
            suffix_lateral_scale=float(suffix_lateral_scale),
            arc_y_mode=str(arc_y_mode),
            repeat_heading_blend_frames=int(repeat_heading_blend_frames),
            arc_smooth_profile=str(arc_smooth_profile),
            repeat_count=int(repeat_count),
            repeat_angle_choices=repeat_angle_choices,
            repeat_seed=int(repeat_seed),
        )
        scenario_update_frames = [
            int(frame) for frame in chain_metadata["chain_update_frames"]
        ]
    else:
        condition = compose_center_symmetric_updated_traj7(
            traj7,
            update_frame=update,
            suffix_frames=suffix_frames,
            source_start_frame=int(source_start_frame),
            derive_heading_from_path=bool(derive_heading_from_path),
            transition_frames=int(transition_frames),
            transition_output_frames=int(transition_output_frames),
        )
    return ConditionScenario(
        name=f"repeat_splice:{mode}",
        condition_traj7=condition,
        update_frames=scenario_update_frames,
        base_sample_name=sample_name,
        caption_index=caption_index,
        metadata={
            "mode": mode,
            "suffix_frames": None if suffix_frames is None else int(suffix_frames),
            "source_start_frame": int(source_start_frame),
            "derive_heading_from_path": bool(derive_heading_from_path),
            "transition_frames": int(transition_frames),
            "transition_output_frames": int(transition_output_frames),
            "suffix_rotation_deg": float(suffix_rotation_deg),
            "turn_transition_frames": int(turn_transition_frames),
            "arc_speed_window": int(arc_speed_window),
            "arc_speed_scale": float(arc_speed_scale),
            "suffix_blend_frames": int(suffix_blend_frames),
            "suffix_min_speed_factor": float(suffix_min_speed_factor),
            "suffix_lateral_scale": float(suffix_lateral_scale),
            "arc_y_mode": str(arc_y_mode),
            "repeat_heading_blend_frames": int(repeat_heading_blend_frames),
            "arc_smooth_profile": str(arc_smooth_profile),
            "repeat_count": int(repeat_count),
            "repeat_angle_choices": _coerce_repeat_angle_choices(
                repeat_angle_choices,
                fallback_angle=float(suffix_rotation_deg),
            ),
            "repeat_seed": int(repeat_seed),
            **chain_metadata,
        },
    )


def build_synthetic_condition(
    *,
    preset: str,
    base_traj7: torch.Tensor | None = None,
    num_frames: int = 240,
    update_frames: Iterable[int] | None = None,
    sample_name: str | None = None,
    caption_index: int | None = None,
    forward_step_length: float = 0.015,
    total_forward: float = 4.2,
    arc_turn_degrees: float = 20.0,
) -> ConditionScenario:
    """Build a hand-authored or machine-authored 7D condition."""
    preset = str(preset)
    if preset == "forward_line":
        condition = compose_forward_line_traj7(
            base_traj7,
            num_frames=int(num_frames),
            step_length=float(forward_step_length),
        )
    elif preset == "four_segment_curve":
        condition = compose_four_segment_forward_curve_traj7(
            num_frames=int(num_frames),
            total_forward=float(total_forward),
        )
    elif preset == "constant_arc":
        condition = compose_constant_curvature_arc_traj7(
            num_frames=int(num_frames),
            arc_length=float(total_forward),
            turn_degrees=float(arc_turn_degrees),
        )
    else:
        raise ValueError(
            "synthetic preset must be 'forward_line', 'four_segment_curve', "
            "or 'constant_arc', "
            f"got {preset!r}"
        )
    updates = _sanitize_update_frames(update_frames, int(condition.shape[0]))
    return ConditionScenario(
        name=f"synthetic:{preset}",
        condition_traj7=condition,
        update_frames=updates,
        base_sample_name=sample_name,
        caption_index=caption_index,
        metadata={
            "preset": preset,
            "num_frames": int(num_frames),
            "forward_step_length": float(forward_step_length),
            "total_forward": float(total_forward),
            "arc_turn_degrees": float(arc_turn_degrees),
        },
    )


__all__ = [
    "ConditionScenario",
    "build_repeat_splice_condition",
    "build_sample_condition",
    "build_synthetic_condition",
    "compose_center_symmetric_updated_traj7",
    "compose_constant_curvature_arc_traj7",
    "compose_forward_line_traj7",
    "compose_four_segment_forward_curve_traj7",
    "compose_rotated_suffix_arc_updated_traj7",
    "compose_rotated_suffix_updated_traj7",
]
