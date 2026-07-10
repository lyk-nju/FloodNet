"""Compatibility adapters for legacy runtime-update payload callers."""

from __future__ import annotations

import torch

from utils.inference.root_plan import RootPlan, build_root_plan_stream_payload
from utils.inference.stream_runtime.contracts import (
    ComposeResult,
    RouteProgressState,
    RouteStatus,
    SegmentLabel,
)
from utils.inference.stream_runtime.payload_builder import PayloadBuilder
from utils.inference.timeline import RootFrameState
from utils.local_frame import canonicalize_7d
from utils.motion_process import build_physical_7d_from_5d
from utils.token_frame import num_tokens_for_frame_len


def build_active_window_root_plan(
    world_segment_traj7: torch.Tensor,
    *,
    anchor_state: RootFrameState,
    anchor_commit_idx: int,
    token_dt: float,
    frames_per_token: int = 4,
    source: str = "active_window",
) -> RootPlan:
    """Build the legacy/diagnostic RootPlan representation."""
    segment = world_segment_traj7.detach().to(
        device=anchor_state.world_xz.device,
        dtype=torch.float32,
    )
    if segment.dim() != 2 or segment.shape[-1] != 7:
        raise ValueError(
            f"world_segment_traj7 must be [T,7], got {tuple(segment.shape)}"
        )
    anchor_xz = anchor_state.world_xz.detach().to(
        device=segment.device,
        dtype=segment.dtype,
    )
    anchor_yaw = anchor_state.world_yaw.detach().to(
        device=segment.device,
        dtype=segment.dtype,
    )
    local = canonicalize_7d(
        segment.unsqueeze(0),
        anchor_xz.unsqueeze(0),
        anchor_yaw.reshape(1),
    )[0]
    return RootPlan(
        num_tokens_pred=num_tokens_for_frame_len(
            int(segment.shape[0]),
            int(frames_per_token),
        ),
        valid_frames=int(segment.shape[0]),
        waypoints_local_7d=local,
        frame_dt=float(token_dt) / float(frames_per_token),
        frames_per_token=int(frames_per_token),
        anchor_commit_idx=int(anchor_commit_idx),
        anchor_world_xz=anchor_xz,
        anchor_world_yaw=anchor_yaw,
        source=str(source),
    )


def _legacy_compose_result(
    world_condition_traj7: torch.Tensor,
    generated_history_traj7: torch.Tensor | None,
) -> ComposeResult:
    """Adapt the old absolute-origin tensors into the new pure contract."""
    world = world_condition_traj7.detach().float()
    if world.dim() != 2 or world.shape[-1] != 7:
        raise ValueError(
            f"world_condition_traj7 must be [T,7], got {tuple(world.shape)}"
        )
    if generated_history_traj7 is not None:
        generated = generated_history_traj7.detach().to(
            device=world.device,
            dtype=world.dtype,
        )
        if generated.dim() != 2 or generated.shape[-1] != 7:
            raise ValueError(
                f"generated_history_traj7 must be [T,7], got {tuple(generated.shape)}"
            )
        generated_count = min(int(world.shape[0]), int(generated.shape[0]))
        if generated_count:
            world = world.clone()
            world[:generated_count] = generated[:generated_count]
    else:
        generated_count = 0

    world = build_physical_7d_from_5d(world[:, :5])
    labels = torch.full(
        (int(world.shape[0]),),
        SegmentLabel.ROUTE.value,
        dtype=torch.int64,
        device=world.device,
    )
    if generated_count:
        labels[:generated_count] = SegmentLabel.GENERATED_HISTORY.value
    return ComposeResult(
        frame_start_abs=0,
        world_condition_7d=world,
        frame_mask=torch.ones(int(world.shape[0]), dtype=torch.bool, device=world.device),
        segment_labels=labels,
        proposed_route_progress=RouteProgressState.initial(),
        route_status=RouteStatus.ACTIVE,
        diagnostics={"compatibility_adapter": "runtime_update.payload_builder"},
    )


def build_world_condition_stream_payload(
    world_condition_traj7: torch.Tensor,
    timeline,
    *,
    local_commit_index: int,
    absolute_commit_index: int,
    chunk_size: int,
    history_length: int,
    traj_horizon_tokens: int,
    frames_per_token: int = 4,
    generated_history_traj7: torch.Tensor | None = None,
) -> dict | None:
    """Adapt the legacy tensor API to :class:`PayloadBuilder`."""
    return PayloadBuilder(frames_per_token=int(frames_per_token)).build(
        _legacy_compose_result(world_condition_traj7, generated_history_traj7),
        timeline,
        local_commit_before=int(local_commit_index),
        absolute_commit_before=int(absolute_commit_index),
        chunk_size=int(chunk_size),
        history_tokens=int(history_length),
        horizon_tokens=int(traj_horizon_tokens),
    )


def build_active_window_stream_payload(
    root_plan: RootPlan,
    timeline,
    *,
    local_commit_index: int,
    absolute_commit_index: int,
    chunk_size: int,
    history_length: int,
    traj_horizon_tokens: int,
) -> dict | None:
    """Compatibility adapter for legacy RootPlan payload generation."""
    return build_root_plan_stream_payload(
        root_plan,
        timeline,
        local_commit_index=int(local_commit_index),
        absolute_commit_index=int(absolute_commit_index),
        chunk_size=int(chunk_size),
        history_length=int(history_length),
        traj_horizon_tokens=int(traj_horizon_tokens),
    )


__all__ = [
    "PayloadBuilder",
    "build_active_window_root_plan",
    "build_active_window_stream_payload",
    "build_world_condition_stream_payload",
]
