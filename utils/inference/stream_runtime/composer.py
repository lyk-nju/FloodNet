"""Pure world-condition composition for activated root sources."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from utils.inference.timeline import RootFrameState
from utils.local_frame import (
    heading_dir_xz,
    transform_xz_local_delta_to_world,
    wrap_angle,
)
from utils.motion_process import build_physical_7d_from_5d

from .contracts import (
    ActivatedRootSource,
    ComposeResult,
    RootSourceProposal,
    RouteProgressState,
    RouteStatus,
    SegmentLabel,
    SpaceContract,
)
from .history import GeneratedRootHistory
from .progress import (
    RelativeRouteProgressPolicy,
    RouteProjection,
    WorldRouteProgressPolicy,
)


def _valid_prefix_length(mask: Tensor) -> int:
    """Return the valid route prefix before terminal proposal padding."""
    invalid = torch.nonzero(~mask, as_tuple=False)
    if int(invalid.shape[0]) == 0:
        return int(mask.shape[0])
    return int(invalid[0, 0].item())


def _cubic_hermite(
    start: Tensor,
    end: Tensor,
    start_tangent: Tensor,
    end_tangent: Tensor,
    t: Tensor,
) -> Tensor:
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00[:, None] * start[None, :]
        + h10[:, None] * start_tangent[None, :]
        + h01[:, None] * end[None, :]
        + h11[:, None] * end_tangent[None, :]
    )


def _smoothstep(t: Tensor) -> Tensor:
    return t * t * (3.0 - 2.0 * t)


def _build_bridge_5d(start_5d: Tensor, end_5d: Tensor, frames: int) -> Tensor:
    """Sample a Hermite bridge at exactly ``frames`` values in ``(0, 1]``."""
    if frames == 0:
        return start_5d.new_empty((0, 5))
    t = torch.linspace(
        0.0,
        1.0,
        frames + 1,
        dtype=start_5d.dtype,
        device=start_5d.device,
    )[1:]
    start_yaw = torch.atan2(start_5d[4], start_5d[3])
    end_yaw = torch.atan2(end_5d[4], end_5d[3])
    xz_start = start_5d[[0, 2]]
    xz_end = end_5d[[0, 2]]
    chord = torch.linalg.norm(xz_end - xz_start)
    xz = _cubic_hermite(
        xz_start,
        xz_end,
        heading_dir_xz(start_yaw) * chord,
        heading_dir_xz(end_yaw) * chord,
        t,
    )
    blend = _smoothstep(t)
    y = start_5d[1] + (end_5d[1] - start_5d[1]) * blend
    yaw = start_yaw + wrap_angle(end_yaw - start_yaw) * blend
    return torch.stack(
        [xz[:, 0], y, xz[:, 1], torch.cos(yaw), torch.sin(yaw)],
        dim=-1,
    )


def _relative_suffix_5d(
    route_5d: Tensor,
    *,
    route_index: int,
    boundary_5d: Tensor,
) -> Tensor:
    """Rigidly place a remaining authored suffix at the actor boundary."""
    suffix = route_5d[route_index:]
    route_anchor = suffix[0]
    route_yaw = torch.atan2(suffix[:, 4], suffix[:, 3])
    route_anchor_yaw = route_yaw[0]
    boundary_yaw = torch.atan2(boundary_5d[4], boundary_5d[3])
    yaw_offset = wrap_angle(boundary_yaw - route_anchor_yaw)
    relative_xz = suffix[:, [0, 2]] - route_anchor[[0, 2]]
    world_xz = boundary_5d[[0, 2]][None, :] + transform_xz_local_delta_to_world(
        relative_xz,
        yaw_offset,
    )
    world_y = boundary_5d[1] + suffix[:, 1] - route_anchor[1]
    world_yaw = boundary_yaw + wrap_angle(route_yaw - route_anchor_yaw)
    return torch.stack(
        [
            world_xz[:, 0],
            world_y,
            world_xz[:, 1],
            torch.cos(world_yaw),
            torch.sin(world_yaw),
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class ConditionComposer:
    """Statelessly combine generated history with one activated route future."""

    world_progress_policy: WorldRouteProgressPolicy = field(
        default_factory=WorldRouteProgressPolicy
    )
    relative_progress_policy: RelativeRouteProgressPolicy = field(
        default_factory=RelativeRouteProgressPolicy
    )

    def _project(
        self,
        activated: ActivatedRootSource,
        *,
        valid_prefix_length: int,
        boundary_state: RootFrameState,
        first_future_frame_abs: int,
        previous_progress: RouteProgressState,
    ) -> RouteProjection | None:
        if valid_prefix_length == 0:
            return None
        route = activated.proposal.future_traj7[:valid_prefix_length]
        if activated.space_contract is SpaceContract.WORLD_ROUTE:
            return self.world_progress_policy.project(
                route,
                boundary_state.world_xz,
                boundary_state.world_yaw,
                previous_progress,
            )

        valid_proposal = RootSourceProposal(
            future_traj7=route,
            future_frame_mask=torch.ones(valid_prefix_length, dtype=torch.bool),
            source_id=activated.proposal.source_id,
            version=activated.proposal.version,
            metadata=activated.proposal.metadata,
        )
        valid_activated = ActivatedRootSource(
            proposal=valid_proposal,
            requested_activation_commit=activated.requested_activation_commit,
            actual_activation_commit=activated.actual_activation_commit,
            boundary_state=activated.boundary_state,
            first_future_frame_abs=activated.first_future_frame_abs,
            space_contract=activated.space_contract,
            progress=activated.progress,
        )
        return self.relative_progress_policy.project(
            valid_activated,
            current_first_future_frame_abs=first_future_frame_abs,
            previous_progress=previous_progress,
        )

    def compose(
        self,
        activated: ActivatedRootSource,
        history: GeneratedRootHistory,
        boundary_state: RootFrameState,
        first_future_frame_abs: int,
        previous_progress: RouteProgressState,
        horizon_frames: int,
        bridge_frames: int = 8,
    ) -> ComposeResult:
        """Return a pure proposal for condition geometry, validity, and progress."""
        if not isinstance(activated, ActivatedRootSource):
            raise TypeError("activated must be ActivatedRootSource")
        if not isinstance(history, GeneratedRootHistory):
            raise TypeError("history must be GeneratedRootHistory")
        if not isinstance(boundary_state, RootFrameState):
            raise TypeError("boundary_state must be RootFrameState")
        if not isinstance(previous_progress, RouteProgressState):
            raise TypeError("previous_progress must be RouteProgressState")

        future_start = int(first_future_frame_abs)
        horizon = int(horizon_frames)
        bridge_count = int(bridge_frames)
        if future_start < activated.first_future_frame_abs:
            raise ValueError(
                "first_future_frame_abs must not precede the activated source"
            )
        if horizon < 0:
            raise ValueError("horizon_frames must be >= 0")
        if bridge_count < 0:
            raise ValueError("bridge_frames must be >= 0")
        if history.next_frame_abs != future_start:
            raise ValueError(
                "history.next_frame_abs must equal first_future_frame_abs: "
                f"got {history.next_frame_abs} and {future_start}"
            )

        route_7d = activated.proposal.future_traj7
        dtype = route_7d.dtype
        device = route_7d.device
        history_7d = history.slice_abs(history.base_frame_abs, future_start).to(
            device=device,
            dtype=dtype,
        )
        history_5d = history_7d[:, :5]
        valid_prefix = _valid_prefix_length(activated.proposal.future_frame_mask)
        projection = self._project(
            activated,
            valid_prefix_length=valid_prefix,
            boundary_state=boundary_state,
            first_future_frame_abs=future_start,
            previous_progress=previous_progress,
        )

        if projection is None:
            route_index = None
            future_index = None
            proposed_progress = previous_progress
            route_exhausted = True
            fallback_y = route_7d[0, 1]
        else:
            route_index = projection.route_index
            future_index = projection.future_index
            proposed_progress = projection.proposed_progress
            route_exhausted = route_index >= valid_prefix - 1
            fallback_y = route_7d[route_index, 1]

        boundary_xz = torch.as_tensor(
            boundary_state.world_xz,
            dtype=dtype,
            device=device,
        ).reshape(-1)
        if int(boundary_xz.numel()) != 2:
            raise ValueError("boundary_state.world_xz must contain two coordinates")
        boundary_yaw = torch.as_tensor(
            boundary_state.world_yaw,
            dtype=dtype,
            device=device,
        ).reshape(())
        boundary_y = history_5d[-1, 1] if int(history_5d.shape[0]) else fallback_y
        boundary_5d = torch.stack(
            [
                boundary_xz[0],
                boundary_y,
                boundary_xz[1],
                torch.cos(boundary_yaw),
                torch.sin(boundary_yaw),
            ]
        )

        if route_exhausted:
            future_5d = boundary_5d[None, :].expand(horizon, -1).clone()
            future_mask = torch.zeros(horizon, dtype=torch.bool, device=device)
            future_labels = torch.full(
                (horizon,),
                SegmentLabel.PADDING.value,
                dtype=torch.int64,
                device=device,
            )
            emitted_bridge_frames = 0
            emitted_route_frames = 0
        else:
            assert route_index is not None and future_index is not None
            route_5d = route_7d[:valid_prefix, :5]
            if activated.space_contract is SpaceContract.RELATIVE_ROUTE:
                materialized = _relative_suffix_5d(
                    route_5d,
                    route_index=route_index,
                    boundary_5d=boundary_5d,
                )
                target_local = future_index - route_index
            else:
                materialized = route_5d
                target_local = future_index

            target_5d = materialized[target_local]
            bridge_5d = _build_bridge_5d(boundary_5d, target_5d, bridge_count)
            if bridge_count:
                route_tail = materialized[target_local + 1 :]
            else:
                route_tail = materialized[target_local:]
            valid_future_5d = torch.cat([bridge_5d, route_tail], dim=0)
            valid_labels = torch.cat(
                [
                    torch.full(
                        (bridge_count,),
                        SegmentLabel.BRIDGE.value,
                        dtype=torch.int64,
                        device=device,
                    ),
                    torch.full(
                        (int(route_tail.shape[0]),),
                        SegmentLabel.ROUTE.value,
                        dtype=torch.int64,
                        device=device,
                    ),
                ]
            )
            valid_count = min(horizon, int(valid_future_5d.shape[0]))
            future_5d = valid_future_5d[:valid_count]
            future_labels = valid_labels[:valid_count]
            future_mask = torch.ones(valid_count, dtype=torch.bool, device=device)
            padding_count = horizon - valid_count
            if padding_count:
                hold_5d = future_5d[-1] if valid_count else boundary_5d
                future_5d = torch.cat(
                    [future_5d, hold_5d[None, :].expand(padding_count, -1)],
                    dim=0,
                )
                future_mask = torch.cat(
                    [
                        future_mask,
                        torch.zeros(padding_count, dtype=torch.bool, device=device),
                    ]
                )
                future_labels = torch.cat(
                    [
                        future_labels,
                        torch.full(
                            (padding_count,),
                            SegmentLabel.PADDING.value,
                            dtype=torch.int64,
                            device=device,
                        ),
                    ]
                )
            emitted_bridge_frames = int(
                future_labels.eq(SegmentLabel.BRIDGE.value).sum().item()
            )
            emitted_route_frames = int(
                future_labels.eq(SegmentLabel.ROUTE.value).sum().item()
            )

        history_count = int(history_5d.shape[0])
        all_5d = torch.cat([history_5d, future_5d], dim=0)
        frame_mask = torch.cat(
            [
                torch.ones(history_count, dtype=torch.bool, device=device),
                future_mask,
            ]
        )
        segment_labels = torch.cat(
            [
                torch.full(
                    (history_count,),
                    SegmentLabel.GENERATED_HISTORY.value,
                    dtype=torch.int64,
                    device=device,
                ),
                future_labels,
            ]
        )
        world_7d = build_physical_7d_from_5d(all_5d)
        route_status = (
            RouteStatus.EXHAUSTED if route_exhausted else RouteStatus.ACTIVE
        )
        padding_frames = int(
            segment_labels.eq(SegmentLabel.PADDING.value).sum().item()
        )

        return ComposeResult(
            frame_start_abs=history.base_frame_abs,
            world_condition_7d=world_7d,
            frame_mask=frame_mask,
            segment_labels=segment_labels,
            proposed_route_progress=proposed_progress,
            route_status=route_status,
            diagnostics={
                "source_id": activated.proposal.source_id,
                "source_version": activated.proposal.version,
                "first_future_frame_abs": future_start,
                "selected_route_index": route_index,
                "selected_future_index": future_index,
                "projection_distance": (
                    None if projection is None else projection.distance
                ),
                "heading_dot": None if projection is None else projection.heading_dot,
                "bridge_frames_requested": bridge_count,
                "bridge_frames_emitted": emitted_bridge_frames,
                "route_frames_emitted": emitted_route_frames,
                "padding_frames": padding_frames,
                "valid_route_frames": valid_prefix,
            },
        )


__all__ = ["ConditionComposer"]
