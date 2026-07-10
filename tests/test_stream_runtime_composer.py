"""Tests for stateless world-condition composition."""

from __future__ import annotations

import math

import pytest
import torch

from utils.inference.stream_runtime import (
    ActivatedRootSource,
    ConditionComposer,
    GeneratedRootHistory,
    RootSourceProposal,
    RouteProgressState,
    RouteStatus,
    SegmentLabel,
    SpaceContract,
)
from utils.inference.timeline import RootFrameState
from utils.motion_process import build_physical_7d_from_5d
from utils.token_frame import first_future_frame_abs


def _traj7(
    points: list[tuple[float, float]],
    *,
    y: float | list[float] = 0.0,
    yaw: float | list[float] = 0.0,
) -> torch.Tensor:
    count = len(points)
    y_values = [y] * count if isinstance(y, (int, float)) else y
    yaw_values = [yaw] * count if isinstance(yaw, (int, float)) else yaw
    traj5 = torch.tensor(
        [
            [
                point[0],
                y_value,
                point[1],
                math.cos(yaw_value),
                math.sin(yaw_value),
            ]
            for point, y_value, yaw_value in zip(points, y_values, yaw_values)
        ],
        dtype=torch.float32,
    )
    return build_physical_7d_from_5d(traj5)


def _activated(
    route: torch.Tensor,
    *,
    space_contract: SpaceContract,
    activation_commit: int = 0,
    boundary_xz: tuple[float, float] = (0.0, 0.0),
    boundary_yaw: float = 0.0,
    mask: torch.Tensor | None = None,
) -> ActivatedRootSource:
    proposal = RootSourceProposal(
        future_traj7=route,
        future_frame_mask=(
            torch.ones(route.shape[0], dtype=torch.bool) if mask is None else mask
        ),
        source_id="route-a",
        version=4,
        metadata={"adapter": "test"},
    )
    boundary = RootFrameState(
        commit_idx=activation_commit,
        world_xz=torch.tensor(boundary_xz, dtype=route.dtype),
        world_yaw=torch.tensor(boundary_yaw, dtype=route.dtype),
        source="activation",
    )
    return ActivatedRootSource(
        proposal=proposal,
        requested_activation_commit=activation_commit,
        actual_activation_commit=activation_commit,
        boundary_state=boundary,
        first_future_frame_abs=first_future_frame_abs(activation_commit),
        space_contract=space_contract,
        progress=RouteProgressState.initial(),
    )


def _empty_history(frame_abs: int = 0) -> GeneratedRootHistory:
    return GeneratedRootHistory.empty(frame_abs, dtype=torch.float32)


def test_cold_start_uses_virtual_boundary_without_emitting_duplicate_anchor():
    route = _traj7([(0.0, 0.1 * (index + 1)) for index in range(32)])
    activated = _activated(route, space_contract=SpaceContract.WORLD_ROUTE)

    result = ConditionComposer().compose(
        activated,
        _empty_history(),
        activated.boundary_state,
        0,
        RouteProgressState.initial(),
        16,
    )

    assert result.frame_start_abs == 0
    assert result.world_condition_7d.shape == (16, 7)
    assert not torch.equal(result.world_condition_7d[0, :5], route[0, :5])
    assert result.segment_labels[0] == SegmentLabel.BRIDGE.value
    assert result.segment_labels.eq(SegmentLabel.BOUNDARY.value).sum() == 0


def test_history_uses_checked_absolute_slice_and_sets_output_frame_base():
    activation_commit = 10
    route = _traj7([(1.0, 0.4 + 0.1 * index) for index in range(24)])
    activated = _activated(
        route,
        space_contract=SpaceContract.WORLD_ROUTE,
        activation_commit=activation_commit,
        boundary_xz=(1.0, 0.3),
    )
    history_frames = _traj7([(1.0, 0.1 * index) for index in range(4)])
    history_frames[:, 5:7] = 99.0
    history = GeneratedRootHistory(base_frame_abs=33, frames_7d=history_frames)

    result = ConditionComposer().compose(
        activated,
        history,
        activated.boundary_state,
        37,
        RouteProgressState.initial(),
        12,
        bridge_frames=4,
    )

    assert result.frame_start_abs == 33
    assert torch.equal(result.world_condition_7d[:4, :5], history_frames[:, :5])
    assert result.segment_labels[:4].eq(SegmentLabel.GENERATED_HISTORY.value).all()
    assert result.frame_mask[:4].all()
    assert result.segment_labels.eq(SegmentLabel.BOUNDARY.value).sum() == 0
    expected = build_physical_7d_from_5d(result.world_condition_7d[:, :5])
    assert torch.allclose(result.world_condition_7d, expected, atol=1e-6)


def test_history_must_end_exactly_where_future_composition_starts():
    route = _traj7([(0.0, float(index)) for index in range(8)])
    activated = _activated(route, space_contract=SpaceContract.WORLD_ROUTE)
    history = GeneratedRootHistory(base_frame_abs=5, frames_7d=route[:3].clone())

    with pytest.raises(ValueError, match="history.next_frame_abs"):
        ConditionComposer().compose(
            activated,
            history,
            RootFrameState(
                commit_idx=2,
                world_xz=route[2, [0, 2]],
                world_yaw=torch.tensor(0.0),
                source="commit",
            ),
            9,
            RouteProgressState.initial(),
            8,
        )


def test_world_and_relative_contracts_diverge_after_generated_drift():
    route = _traj7([(0.0, 0.5 * index) for index in range(7)])
    boundary = RootFrameState(
        commit_idx=0,
        world_xz=torch.tensor([2.0, 0.0]),
        world_yaw=torch.tensor(0.0),
        source="drifted",
    )
    world_source = _activated(
        route,
        space_contract=SpaceContract.WORLD_ROUTE,
        boundary_xz=(2.0, 0.0),
    )
    relative_source = _activated(
        route,
        space_contract=SpaceContract.RELATIVE_ROUTE,
        boundary_xz=(2.0, 0.0),
    )
    composer = ConditionComposer()

    world = composer.compose(
        world_source,
        _empty_history(),
        boundary,
        0,
        RouteProgressState.initial(),
        8,
        bridge_frames=2,
    )
    relative = composer.compose(
        relative_source,
        _empty_history(),
        boundary,
        0,
        RouteProgressState.initial(),
        8,
        bridge_frames=2,
    )

    target_xz = route[-1, [0, 2]]
    world_error = torch.linalg.norm(world.world_condition_7d[-1, [0, 2]] - target_xz)
    relative_error = torch.linalg.norm(relative.world_condition_7d[-1, [0, 2]] - target_xz)
    assert world_error < relative_error
    assert torch.equal(world.world_condition_7d[2, :5], route[2, :5])
    assert relative.world_condition_7d[2, 0] == pytest.approx(2.0)


def test_relative_route_rotates_remaining_shape_to_actor_boundary_pose():
    route = _traj7([(0.0, float(index)) for index in range(6)])
    activated = _activated(
        route,
        space_contract=SpaceContract.RELATIVE_ROUTE,
        boundary_xz=(3.0, 4.0),
        boundary_yaw=float(torch.pi / 2),
    )

    result = ConditionComposer().compose(
        activated,
        _empty_history(),
        activated.boundary_state,
        0,
        RouteProgressState.initial(),
        6,
        bridge_frames=0,
    )

    assert result.segment_labels[0] == SegmentLabel.ROUTE.value
    assert result.world_condition_7d[0, [0, 2]].tolist() == pytest.approx([4.0, 4.0])
    assert torch.atan2(
        result.world_condition_7d[0, 4], result.world_condition_7d[0, 3]
    ).item() == pytest.approx(torch.pi / 2)


def test_bridge_frames_controls_exact_emitted_bridge_span():
    route = _traj7([(0.0, 0.1 * (index + 1)) for index in range(80)])
    activated = _activated(route, space_contract=SpaceContract.WORLD_ROUTE)
    composer = ConditionComposer()

    short = composer.compose(
        activated,
        _empty_history(),
        activated.boundary_state,
        0,
        RouteProgressState.initial(),
        40,
        bridge_frames=4,
    )
    long = composer.compose(
        activated,
        _empty_history(),
        activated.boundary_state,
        0,
        RouteProgressState.initial(),
        40,
        bridge_frames=12,
    )

    assert short.segment_labels.eq(SegmentLabel.BRIDGE.value).sum() == 4
    assert long.segment_labels.eq(SegmentLabel.BRIDGE.value).sum() == 12
    assert not torch.equal(short.world_condition_7d[:12], long.world_condition_7d[:12])
    assert not torch.allclose(
        short.world_condition_7d[0, :5],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
    )


def test_zero_bridge_starts_directly_at_selected_future_route_frame():
    route = _traj7([(0.0, float(index)) for index in range(8)])
    activated = _activated(route, space_contract=SpaceContract.WORLD_ROUTE)

    result = ConditionComposer().compose(
        activated,
        _empty_history(),
        activated.boundary_state,
        0,
        RouteProgressState.initial(),
        5,
        bridge_frames=0,
    )

    selected = result.diagnostics["selected_future_index"]
    assert result.segment_labels.eq(SegmentLabel.BRIDGE.value).sum() == 0
    assert result.segment_labels[0] == SegmentLabel.ROUTE.value
    assert torch.equal(result.world_condition_7d[0, :5], route[selected, :5])


def test_negative_bridge_frames_are_rejected():
    route = _traj7([(0.0, float(index)) for index in range(8)])
    activated = _activated(route, space_contract=SpaceContract.WORLD_ROUTE)

    with pytest.raises(ValueError, match="bridge_frames"):
        ConditionComposer().compose(
            activated,
            _empty_history(),
            activated.boundary_state,
            0,
            RouteProgressState.initial(),
            8,
            bridge_frames=-1,
        )


def test_relative_route_preserves_generated_boundary_y_and_recomputes_deltas():
    route = _traj7([(0.0, 0.25 * index) for index in range(12)], y=10.0)
    activated = _activated(route, space_contract=SpaceContract.RELATIVE_ROUTE)
    history_frame = _traj7([(0.0, 0.0)], y=1.75)
    history_frame[:, 5:7] = -123.0
    history = GeneratedRootHistory(base_frame_abs=0, frames_7d=history_frame)
    boundary = RootFrameState(
        commit_idx=1,
        world_xz=torch.tensor([0.0, 0.0]),
        world_yaw=torch.tensor(0.0),
        source="commit",
    )

    result = ConditionComposer().compose(
        activated,
        history,
        boundary,
        1,
        RouteProgressState.initial(),
        10,
        bridge_frames=4,
    )

    assert torch.allclose(result.world_condition_7d[:, 1], torch.full((11,), 1.75))
    expected = build_physical_7d_from_5d(result.world_condition_7d[:, :5])
    assert torch.allclose(result.world_condition_7d, expected, atol=1e-6)


def test_one_frame_route_emits_terminal_frame_before_exhaustion():
    route = _traj7([(4.0, 9.0)])
    activated = _activated(
        route,
        space_contract=SpaceContract.WORLD_ROUTE,
        boundary_xz=(-3.0, -5.0),
    )
    composer = ConditionComposer()

    first = composer.compose(
        activated,
        _empty_history(),
        activated.boundary_state,
        0,
        RouteProgressState.initial(),
        4,
        bridge_frames=0,
    )

    assert torch.equal(first.world_condition_7d[0, :5], route[0, :5])
    assert first.frame_mask[0]
    assert first.segment_labels[0] == SegmentLabel.ROUTE.value
    assert first.route_status is RouteStatus.ACTIVE


def test_exhaustion_holds_last_composed_history_pose_not_current_boundary():
    route = _traj7([(4.0, 9.0)])
    activated = _activated(
        route,
        space_contract=SpaceContract.WORLD_ROUTE,
        boundary_xz=(-3.0, -5.0),
    )
    first = ConditionComposer().compose(
        activated,
        _empty_history(),
        activated.boundary_state,
        0,
        RouteProgressState.initial(),
        1,
        bridge_frames=0,
    )
    history = GeneratedRootHistory(
        base_frame_abs=0,
        frames_7d=first.world_condition_7d.clone(),
    )

    result = ConditionComposer().compose(
        activated,
        history,
        RootFrameState(
            commit_idx=1,
            world_xz=torch.tensor([-20.0, 30.0]),
            world_yaw=torch.tensor(torch.pi / 2),
            source="drifted-boundary",
        ),
        1,
        first.proposed_route_progress,
        4,
        bridge_frames=0,
    )

    assert torch.equal(
        result.world_condition_7d[1:],
        first.world_condition_7d[0].expand(4, -1),
    )
    assert not result.frame_mask[1:].any()
    assert result.segment_labels[1:].eq(SegmentLabel.PADDING.value).all()
    assert result.route_status is RouteStatus.EXHAUSTED
    assert activated.progress == RouteProgressState.initial()
    assert "lifecycle_events" not in result.diagnostics


def test_terminal_proposal_mask_holds_last_valid_pose_as_padding():
    route = _traj7([(0.0, float(index)) for index in range(5)])
    activated = _activated(
        route,
        space_contract=SpaceContract.WORLD_ROUTE,
        mask=torch.tensor([True, True, False, False, False]),
    )

    result = ConditionComposer().compose(
        activated,
        _empty_history(),
        activated.boundary_state,
        0,
        RouteProgressState.initial(),
        6,
        bridge_frames=2,
    )

    assert result.frame_mask[:2].all()
    assert not result.frame_mask[2:].any()
    assert result.segment_labels[:2].eq(SegmentLabel.BRIDGE.value).all()
    assert result.segment_labels[2:].eq(SegmentLabel.PADDING.value).all()
    assert torch.equal(result.world_condition_7d[-1, :5], route[1, :5])


def test_active_bridge_requires_horizon_to_cover_its_exact_span():
    route = _traj7([(0.0, float(index)) for index in range(3)])
    activated = _activated(route, space_contract=SpaceContract.WORLD_ROUTE)

    with pytest.raises(ValueError, match="horizon_frames.*bridge_frames"):
        ConditionComposer().compose(
            activated,
            _empty_history(),
            activated.boundary_state,
            0,
            RouteProgressState.initial(),
            1,
            bridge_frames=2,
        )
