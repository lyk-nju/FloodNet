"""World-frame active condition composition for streaming runtime updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.inference.runtime_update.route_tracker import RouteProgressTracker
from utils.local_frame import heading_dir_xz, wrap_angle
from utils.motion_process import build_physical_7d_from_5d


@dataclass(frozen=True)
class ActiveWindowSegment:
    segment_traj7: torch.Tensor
    route_index: int
    future_index: int
    current_frame: int
    bridge_frames: int


def _yaw_from_7d(traj7: torch.Tensor) -> torch.Tensor:
    return torch.atan2(traj7[:, 4], traj7[:, 3])


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    return t * t * (3.0 - 2.0 * t)


def _cubic_hermite(
    p0: torch.Tensor,
    p1: torch.Tensor,
    m0: torch.Tensor,
    m1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00[:, None] * p0[None, :]
        + h10[:, None] * m0[None, :]
        + h01[:, None] * p1[None, :]
        + h11[:, None] * m1[None, :]
    )




def _rotate_xz(delta_xz: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate [x,z] vectors by a physical yaw angle.

    Yaw zero points along +z, so a positive yaw maps the forward vector [0, 1]
    to [sin(yaw), cos(yaw)].
    """
    c = torch.cos(angle)
    s = torch.sin(angle)
    x = c * delta_xz[..., 0] + s * delta_xz[..., 1]
    z = -s * delta_xz[..., 0] + c * delta_xz[..., 1]
    return torch.stack([x, z], dim=-1)


def _build_bridge_5d(
    *,
    start_5d: torch.Tensor,
    end_5d: torch.Tensor,
    start_yaw: torch.Tensor,
    end_yaw: torch.Tensor,
    frames: int,
) -> torch.Tensor:
    frames = max(2, int(frames))
    dtype = start_5d.dtype
    t = torch.linspace(0.0, 1.0, frames, dtype=dtype)
    xz0 = start_5d[[0, 2]]
    xz1 = end_5d[[0, 2]]
    chord = torch.linalg.norm(xz1 - xz0).clamp_min(torch.tensor(1e-6, dtype=dtype))
    m0 = heading_dir_xz(start_yaw.to(dtype=dtype)) * chord
    m1 = heading_dir_xz(end_yaw.to(dtype=dtype)) * chord
    xz = _cubic_hermite(xz0, xz1, m0, m1, t)
    s = _smoothstep(t)
    y = start_5d[1] + (end_5d[1] - start_5d[1]) * s
    yaw = start_yaw.to(dtype=dtype) + wrap_angle(end_yaw.to(dtype=dtype) - start_yaw.to(dtype=dtype)) * s
    return torch.stack(
        [xz[:, 0], y, xz[:, 1], torch.cos(yaw), torch.sin(yaw)],
        dim=-1,
    )


def compose_active_window_segment(
    route_traj7: torch.Tensor,
    generated_traj7: torch.Tensor,
    *,
    current_frame: int,
    route_frame_local: int | None = None,
    current_yaw: torch.Tensor | float | None = None,
    target_end_frame: int | None = None,
    lookahead_m: float = 0.25,
    bridge_frames: int = 12,
    min_route_index: int | None = None,
    tracker: RouteProgressTracker | None = None,
) -> ActiveWindowSegment:
    """Compose a continuous world-frame update segment from generated root to route.

    The returned segment starts exactly at ``generated_traj7[current_frame]``.
    ``current_frame`` belongs to the absolute generated timeline, while
    ``route_frame_local`` belongs to the proposal-local route timeline. The
    latter defaults to ``current_frame`` for legacy full-timeline routes.
    """
    route = route_traj7.detach().cpu().float()
    generated = generated_traj7.detach().cpu().float()
    if route.dim() != 2 or route.shape[-1] < 5:
        raise ValueError(f"route_traj7 must be [T,>=5], got {tuple(route.shape)}")
    if generated.dim() != 2 or generated.shape[-1] < 5:
        raise ValueError(
            f"generated_traj7 must be [T,>=5], got {tuple(generated.shape)}"
        )
    cur = int(current_frame)
    if cur < 0 or cur >= int(generated.shape[0]):
        raise ValueError(
            f"current_frame {cur} is outside generated trajectory range "
            f"[0, {int(generated.shape[0]) - 1}]"
        )
    route_cur = cur if route_frame_local is None else int(route_frame_local)
    if route_cur < 0 or route_cur >= int(route.shape[0]):
        raise ValueError(
            f"route_frame_local {route_cur} is outside route range "
            f"[0, {int(route.shape[0]) - 1}]"
        )
    route_end = int(route.shape[0]) - 1 if target_end_frame is None else int(target_end_frame)
    route_end = max(1, min(route_end, int(route.shape[0]) - 1))
    desired_frame_count = max(1, int(route_end) - int(route_cur) + 1)
    if current_yaw is None:
        current_yaw_t = _yaw_from_7d(generated[cur : cur + 1])[0]
    else:
        current_yaw_t = torch.as_tensor(current_yaw, dtype=route.dtype).reshape(())
    active_tracker = tracker or RouteProgressTracker(route, lookahead_m=lookahead_m)
    progress = active_tracker.project(
        current_xz=generated[cur, [0, 2]],
        current_yaw=current_yaw_t,
        min_index=(route_cur if min_route_index is None else int(min_route_index)),
    )
    route_idx = min(int(progress.route_index), int(route.shape[0]) - 2)
    route_end = min(int(route.shape[0]) - 1, route_idx + int(desired_frame_count) - 1)
    future_idx = min(max(progress.future_index, route_idx + 1), route_end)
    route_slice = route[route_idx : route_end + 1, :5]
    if int(route_slice.shape[0]) <= 0:
        route_slice = route[route_idx : route_idx + 1, :5]

    start_5d = generated[cur, :5].clone()
    start_5d[3] = torch.cos(current_yaw_t)
    start_5d[4] = torch.sin(current_yaw_t)

    route_yaw = _yaw_from_7d(route[route_idx : route_end + 1])
    route_anchor_yaw = route_yaw[0]
    yaw_delta = current_yaw_t.to(dtype=route.dtype) - route_anchor_yaw
    delta_xz = route_slice[:, [0, 2]] - route_slice[0, [0, 2]]
    rebased_xz = start_5d[[0, 2]][None, :] + _rotate_xz(delta_xz, yaw_delta)
    rebased_y = start_5d[1] + (route_slice[:, 1] - route_slice[0, 1])
    rebased_yaw = current_yaw_t.to(dtype=route.dtype) + wrap_angle(route_yaw - route_anchor_yaw)
    rebased_5d = torch.stack(
        [
            rebased_xz[:, 0],
            rebased_y,
            rebased_xz[:, 1],
            torch.cos(rebased_yaw),
            torch.sin(rebased_yaw),
        ],
        dim=-1,
    )

    future_offset = max(1, min(int(future_idx - route_idx), int(rebased_5d.shape[0]) - 1))
    end_yaw = rebased_yaw[future_offset]
    timeline_bridge_frames = max(2, int(bridge_frames), int(future_offset) + 1)
    bridge_5d = _build_bridge_5d(
        start_5d=start_5d,
        end_5d=rebased_5d[future_offset],
        start_yaw=current_yaw_t,
        end_yaw=end_yaw,
        frames=timeline_bridge_frames,
    )
    if future_offset + 1 < int(rebased_5d.shape[0]):
        out_5d = torch.cat([bridge_5d, rebased_5d[future_offset + 1 :]], dim=0)
    else:
        out_5d = bridge_5d
    if int(out_5d.shape[0]) < int(desired_frame_count):
        pad = out_5d[-1:].expand(int(desired_frame_count) - int(out_5d.shape[0]), -1)
        out_5d = torch.cat([out_5d, pad], dim=0)
    elif int(out_5d.shape[0]) > int(desired_frame_count):
        out_5d = out_5d[: int(desired_frame_count)]
    return ActiveWindowSegment(
        segment_traj7=build_physical_7d_from_5d(out_5d),
        route_index=int(progress.route_index),
        future_index=int(future_idx),
        current_frame=int(cur),
        bridge_frames=int(bridge_5d.shape[0]),
    )


def compose_active_window_world_condition(
    route_traj7: torch.Tensor,
    generated_history_traj7: torch.Tensor,
    segment: ActiveWindowSegment,
    *,
    current_frame: int | None = None,
    route_start_frame_abs: int = 0,
) -> torch.Tensor:
    """Build a continuous world condition for active-window display/feedback.

    Active-window updates are local payloads anchored at the current generated
    root. A single global condition can only be used for rendering or feedback
    if its past is also the generated root history; otherwise patching a rebased
    segment into the original route creates a discontinuity at the switch frame.
    """
    route = route_traj7.detach().clone().cpu().float()
    history = generated_history_traj7.detach().cpu().float()
    seg = segment.segment_traj7.detach().cpu().float()
    if route.dim() != 2 or route.shape[-1] != 7:
        raise ValueError(f"route_traj7 must be [T,7], got {tuple(route.shape)}")
    if history.dim() != 2 or history.shape[-1] != 7:
        raise ValueError(
            f"generated_history_traj7 must be [T,7], got {tuple(history.shape)}"
        )
    if seg.dim() != 2 or seg.shape[-1] != 7:
        raise ValueError(f"segment.segment_traj7 must be [T,7], got {tuple(seg.shape)}")
    if int(route.shape[0]) == 0:
        return route
    if int(history.shape[0]) == 0 or int(seg.shape[0]) == 0:
        return route
    route_start = int(route_start_frame_abs)
    if route_start < 0:
        raise ValueError(
            f"route_start_frame_abs must be >= 0, got {route_start_frame_abs}"
        )
    frame = int(segment.current_frame if current_frame is None else current_frame)
    if frame < 0 or frame >= int(history.shape[0]):
        raise ValueError(
            f"current_frame {frame} is outside generated history range "
            f"[0, {int(history.shape[0]) - 1}]"
        )
    route_stop = route_start + int(route.shape[0])
    total_frames = max(
        route_stop,
        int(history.shape[0]),
        frame + int(seg.shape[0]),
    )
    out_5d = route[:1, :5].expand(total_frames, -1).clone()
    out_5d[route_start:route_stop] = route[:, :5]
    if route_stop < total_frames:
        out_5d[route_stop:] = route[-1, :5]
    out_5d[: frame + 1] = history[: frame + 1, :5]
    patch_end = min(total_frames, frame + int(seg.shape[0]))
    if patch_end > frame:
        out_5d[frame:patch_end] = seg[: patch_end - frame, :5].to(dtype=out_5d.dtype)
    return build_physical_7d_from_5d(out_5d)


__all__ = [
    "ActiveWindowSegment",
    "compose_active_window_segment",
    "compose_active_window_world_condition",
]
