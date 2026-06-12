"""Coordinate and 7D root transforms for runtime debug evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from utils.inference_glue import InferenceGlueState
from utils.local_frame import (
    canonicalize_7d,
    uncanonicalize_7d,
)
from utils.motion_process import append_traj_deltas_5d_to_7d
from utils.root_plan import RootPlan
from utils.runtime_timeline import recovery_root_state_to_world
from utils.token_frame import num_tokens_for_frame_len, token_start_frame


def _infer_physical_yaw_from_points(points_xyz: torch.Tensor) -> torch.Tensor:
    yaw_values = []
    last_yaw = points_xyz.new_tensor(0.0)
    n = int(points_xyz.shape[0])
    for i in range(n):
        if i < n - 1:
            delta = points_xyz[i + 1, [0, 2]] - points_xyz[i, [0, 2]]
        elif i > 0:
            delta = points_xyz[i, [0, 2]] - points_xyz[i - 1, [0, 2]]
        else:
            delta = points_xyz.new_zeros(2)
        if torch.linalg.norm(delta) > 1e-6:
            last_yaw = torch.atan2(delta[0], delta[1])
        yaw_values.append(last_yaw)
    return torch.stack(yaw_values) if yaw_values else points_xyz.new_zeros(0)


def build_eval_root_plan_from_points(
    points_xyz: Any,
    *,
    anchor_state: InferenceGlueState,
    token_dt: float,
    frames_per_token: int = 4,
    source: str = "eval_route",
) -> RootPlan:
    """Convert a world-space route sampled at frame cadence into a 7D RootPlan."""

    points = torch.as_tensor(
        points_xyz,
        device=anchor_state.world_xz.device,
        dtype=torch.float32,
    )
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points_xyz must be [T,3], got {tuple(points.shape)}")
    yaw = _infer_physical_yaw_from_points(points)
    traj_5d_world = torch.cat(
        [points, torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)],
        dim=-1,
    )
    traj_7d_world = append_traj_deltas_5d_to_7d(
        traj_5d_world,
        physical_yaw=yaw,
    )
    anchor_xz = anchor_state.world_xz.to(device=points.device, dtype=torch.float32)
    anchor_yaw = anchor_state.world_yaw.to(device=points.device, dtype=torch.float32)
    traj_7d_local = canonicalize_7d(traj_7d_world, anchor_xz, anchor_yaw)
    valid_frames = int(traj_7d_local.shape[0])
    return RootPlan(
        num_tokens_pred=num_tokens_for_frame_len(valid_frames, frames_per_token),
        valid_frames=valid_frames,
        waypoints_local_7d=traj_7d_local,
        frame_dt=float(token_dt) / float(frames_per_token),
        frames_per_token=int(frames_per_token),
        anchor_commit_idx=int(anchor_state.commit_idx),
        anchor_world_xz=anchor_xz,
        anchor_world_yaw=anchor_yaw,
        source=str(source),
    )


def rotate_world_7d_about_anchor(
    traj_7d_world: Any,
    *,
    anchor_xyz: Any,
    degrees: float,
) -> torch.Tensor:
    """Rotate world-space 7D root features around an XZ anchor.

    Channels 0/2 and heading cos/sin are rotated together. Y, fwd_delta, and
    yaw_delta stay unchanged because they are rigid-transform invariant.
    """

    traj = torch.as_tensor(traj_7d_world, dtype=torch.float32).clone()
    if traj.ndim != 2 or traj.shape[-1] != 7:
        raise ValueError(f"traj_7d_world must be [T,7], got {tuple(traj.shape)}")
    anchor = torch.as_tensor(
        anchor_xyz,
        device=traj.device,
        dtype=torch.float32,
    ).reshape(-1)
    if anchor.numel() < 3:
        raise ValueError("anchor_xyz must contain at least x/y/z")

    rad = torch.as_tensor(
        np.deg2rad(float(degrees)),
        device=traj.device,
        dtype=torch.float32,
    )
    c = torch.cos(rad)
    s = torch.sin(rad)

    rel_x = traj[:, 0] - anchor[0]
    rel_z = traj[:, 2] - anchor[2]
    traj[:, 0] = anchor[0] + c * rel_x - s * rel_z
    traj[:, 2] = anchor[2] + s * rel_x + c * rel_z

    cos_h = traj[:, 3].clone()
    sin_h = traj[:, 4].clone()
    # Heading direction is [sin(yaw), cos(yaw)] in XZ. Applying the same XZ
    # rotation matrix corresponds to yaw -= degrees in this representation.
    traj[:, 3] = c * cos_h + s * sin_h
    traj[:, 4] = c * sin_h - s * cos_h
    return traj


def rotate_xz_points(points: Any, anchor: Any, degrees: float) -> np.ndarray:
    """Rotate world-space XYZ points around an XZ anchor."""

    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[-1] < 3:
        raise ValueError(f"points must be [T,>=3], got {pts.shape}")
    anc = np.asarray(anchor, dtype=np.float32).reshape(-1)
    if anc.size < 3:
        raise ValueError("anchor must contain at least x/y/z")
    c, s = float(np.cos(np.deg2rad(degrees))), float(np.sin(np.deg2rad(degrees)))
    rel = pts[:, [0, 2]] - anc[[0, 2]][None, :]
    pts[:, 0] = anc[0] + c * rel[:, 0] - s * rel[:, 1]
    pts[:, 2] = anc[2] + s * rel[:, 0] + c * rel[:, 1]
    return pts


def build_eval_root_plan_from_world_7d(
    traj_7d_world: Any,
    *,
    anchor_state: InferenceGlueState,
    token_dt: float,
    frames_per_token: int = 4,
    source: str = "eval_gt_motion_7d",
) -> RootPlan:
    """Convert already-formed world-space GT/root-refiner 7D into a RootPlan."""

    traj_world = torch.as_tensor(
        traj_7d_world,
        device=anchor_state.world_xz.device,
        dtype=torch.float32,
    )
    if traj_world.ndim != 2 or traj_world.shape[-1] != 7:
        raise ValueError(f"traj_7d_world must be [T,7], got {tuple(traj_world.shape)}")
    anchor_xz = anchor_state.world_xz.to(device=traj_world.device, dtype=torch.float32)
    anchor_yaw = anchor_state.world_yaw.to(device=traj_world.device, dtype=torch.float32)
    traj_7d_local = canonicalize_7d(traj_world, anchor_xz, anchor_yaw)
    valid_frames = int(traj_7d_local.shape[0])
    return RootPlan(
        num_tokens_pred=num_tokens_for_frame_len(valid_frames, frames_per_token),
        valid_frames=valid_frames,
        waypoints_local_7d=traj_7d_local,
        frame_dt=float(token_dt) / float(frames_per_token),
        frames_per_token=int(frames_per_token),
        anchor_commit_idx=int(anchor_state.commit_idx),
        anchor_world_xz=anchor_xz,
        anchor_world_yaw=anchor_yaw,
        source=str(source),
    )


def root_plan_to_world_7d(root_plan: RootPlan) -> torch.Tensor:
    """Convert a RootPlan's valid plan-local 7D prefix back to world 7D."""

    valid_frames = int(root_plan.valid_frames)
    traj_local = root_plan.waypoints_local_7d[:valid_frames]
    return uncanonicalize_7d(
        traj_local,
        root_plan.anchor_world_xz,
        root_plan.anchor_world_yaw,
    )


def _hold_last_to_length(values: torch.Tensor, length: int) -> torch.Tensor:
    if int(values.shape[0]) >= int(length):
        return values[:length]
    if int(values.shape[0]) <= 0:
        raise ValueError("cannot extend an empty RootPlan")
    pad = values[-1:].expand(int(length) - int(values.shape[0]), -1)
    return torch.cat([values, pad], dim=0)


def compose_turn_root_plan(
    old_plan: RootPlan,
    new_plan: RootPlan,
    *,
    switch_commit: int,
    blend_tokens: int = 0,
    source: str = "bench_composed",
) -> RootPlan:
    """Compose old/new RootPlans into one session-anchored turn RootPlan.

    The result keeps the old plan before ``switch_commit`` and uses the new plan
    after the optional blend. Delta channels are rebuilt from the composed 5D
    path so the 7D condition is internally consistent.
    """

    old_world = root_plan_to_world_7d(old_plan)
    new_world = root_plan_to_world_7d(new_plan).to(
        device=old_world.device,
        dtype=old_world.dtype,
    )
    valid_frames = max(int(old_world.shape[0]), int(new_world.shape[0]))
    old_world = _hold_last_to_length(old_world, valid_frames)
    new_world = _hold_last_to_length(new_world, valid_frames)

    frames_per_token = int(old_plan.frames_per_token)
    switch_frame = token_start_frame(int(switch_commit), frames_per_token)
    blend_tokens = max(0, int(blend_tokens))
    blend_end = (
        token_start_frame(int(switch_commit) + blend_tokens, frames_per_token)
        if blend_tokens > 0
        else switch_frame
    )

    idx = torch.arange(valid_frames, device=old_world.device, dtype=old_world.dtype)
    weight = torch.zeros(valid_frames, device=old_world.device, dtype=old_world.dtype)
    if blend_tokens > 0 and blend_end > switch_frame:
        raw = ((idx - float(switch_frame)) / float(blend_end - switch_frame)).clamp(0.0, 1.0)
        weight = raw * raw * (3.0 - 2.0 * raw)
        weight = torch.where(idx >= float(blend_end), torch.ones_like(weight), weight)
    else:
        weight = torch.where(idx >= float(switch_frame), torch.ones_like(weight), weight)

    xyz = old_world[:, :3] * (1.0 - weight[:, None]) + new_world[:, :3] * weight[:, None]
    old_yaw = torch.atan2(old_world[:, 4], old_world[:, 3])
    new_yaw = torch.atan2(new_world[:, 4], new_world[:, 3])
    yaw_delta = torch.atan2(torch.sin(new_yaw - old_yaw), torch.cos(new_yaw - old_yaw))
    yaw = old_yaw + yaw_delta * weight
    traj_5d_world = torch.cat(
        [xyz, torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)],
        dim=-1,
    )
    traj_7d_world = append_traj_deltas_5d_to_7d(traj_5d_world, physical_yaw=yaw)
    anchor_xz = old_plan.anchor_world_xz.to(device=old_world.device, dtype=old_world.dtype)
    anchor_yaw = old_plan.anchor_world_yaw.to(device=old_world.device, dtype=old_world.dtype)
    traj_7d_local = canonicalize_7d(traj_7d_world, anchor_xz, anchor_yaw)
    return RootPlan(
        num_tokens_pred=num_tokens_for_frame_len(valid_frames, frames_per_token),
        valid_frames=valid_frames,
        waypoints_local_7d=traj_7d_local,
        frame_dt=float(old_plan.frame_dt),
        frames_per_token=frames_per_token,
        anchor_commit_idx=int(old_plan.anchor_commit_idx),
        anchor_world_xz=anchor_xz,
        anchor_world_yaw=anchor_yaw,
        source=str(source),
    )


__all__ = [
    "build_eval_root_plan_from_points",
    "build_eval_root_plan_from_world_7d",
    "compose_turn_root_plan",
    "recovery_root_state_to_world",
    "root_plan_to_world_7d",
    "rotate_xz_points",
    "rotate_world_7d_about_anchor",
]
