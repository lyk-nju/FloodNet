"""Tests for pure route-progress policies in the authoritative runtime."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from utils.inference.stream_runtime import (
    ActivatedRootSource,
    RelativeRouteProgressPolicy,
    RootSourceProposal,
    RouteProgressState,
    SpaceContract,
    WorldRouteProgressPolicy,
)
from utils.inference.timeline import RootFrameState
from utils.token_frame import first_future_frame_abs


def _route(points: list[tuple[float, float]], yaws: list[float] | None = None) -> torch.Tensor:
    route = torch.zeros(len(points), 7)
    route[:, 0] = torch.tensor([point[0] for point in points])
    route[:, 2] = torch.tensor([point[1] for point in points])
    yaw = torch.tensor(yaws if yaws is not None else [0.0] * len(points))
    route[:, 3] = torch.cos(yaw)
    route[:, 4] = torch.sin(yaw)
    return route


def _activated(
    route: torch.Tensor,
    *,
    activation_commit: int = 0,
) -> ActivatedRootSource:
    proposal = RootSourceProposal(
        future_traj7=route,
        future_frame_mask=torch.ones(route.shape[0], dtype=torch.bool),
        source_id="route-a",
        version=1,
        metadata={},
    )
    return ActivatedRootSource(
        proposal=proposal,
        requested_activation_commit=activation_commit,
        actual_activation_commit=activation_commit,
        boundary_state=RootFrameState(
            commit_idx=activation_commit,
            world_xz=torch.zeros(2),
            world_yaw=torch.zeros(()),
            source="activation",
        ),
        first_future_frame_abs=first_future_frame_abs(activation_commit),
        space_contract=SpaceContract.RELATIVE_ROUTE,
        progress=RouteProgressState.initial(),
    )


def test_world_policy_is_monotonic_and_heading_aware():
    route = _route([(0.0, float(index)) for index in range(31)])
    policy = WorldRouteProgressPolicy(lookahead_m=0.25)

    first = policy.project(
        route,
        torch.tensor([0.0, 20.0]),
        torch.tensor(0.0),
        RouteProgressState.initial(),
    )
    second = policy.project(
        route,
        torch.tensor([0.0, 5.0]),
        torch.tensor(0.0),
        first.proposed_progress,
    )

    assert first.route_index == 20
    assert second.route_index >= first.route_index
    assert second.distance == pytest.approx(15.0)
    assert second.proposed_progress.route_index == second.route_index

    heading_route = _route([(0.0, 0.0), (0.0, 0.0)], [torch.pi, 0.0])
    heading_projection = policy.project(
        heading_route,
        torch.tensor([0.0, 0.0]),
        torch.tensor(0.0),
        RouteProgressState.initial(),
    )
    assert heading_projection.route_index == 1
    assert heading_projection.heading_dot == pytest.approx(1.0)


def test_relative_policy_advances_by_absolute_future_phase_not_world_projection():
    activated = _activated(_route([(100.0, float(index)) for index in range(12)]))
    policy = RelativeRouteProgressPolicy(lookahead_m=0.25)

    projection = policy.project(
        activated,
        current_first_future_frame_abs=activated.first_future_frame_abs + 8,
        previous_progress=RouteProgressState.initial(),
    )

    assert projection.route_index == 8
    assert projection.proposed_progress.route_index == 8
    assert projection.distance == 0.0
    assert projection.heading_dot == 1.0

    later = policy.project(
        activated,
        current_first_future_frame_abs=activated.first_future_frame_abs + 3,
        previous_progress=projection.proposed_progress,
    )
    assert later.route_index == 8
    assert not hasattr(policy, "_last_index")
    with pytest.raises(FrozenInstanceError):
        policy.lookahead_m = 1.0


def test_relative_policy_subtracts_nonzero_activation_frame_base():
    activated = _activated(
        _route([(100.0, float(index)) for index in range(12)]),
        activation_commit=10,
    )

    projection = RelativeRouteProgressPolicy(lookahead_m=0.25).project(
        activated,
        current_first_future_frame_abs=45,
        previous_progress=RouteProgressState.initial(),
    )

    assert activated.first_future_frame_abs == 37
    assert projection.route_index == 8
    assert projection.proposed_progress.route_index == 8


@pytest.mark.parametrize(
    ("route_length", "current_first_future_frame_abs", "expected_index"),
    [
        (3, 3, 2),
        (3, 30, 2),
        (1, 0, 0),
        (1, 1, 0),
    ],
)
def test_relative_policy_clamps_single_frame_exact_end_and_past_end(
    route_length: int,
    current_first_future_frame_abs: int,
    expected_index: int,
):
    activated = _activated(_route([(0.0, float(index)) for index in range(route_length)]))

    projection = RelativeRouteProgressPolicy().project(
        activated,
        current_first_future_frame_abs=current_first_future_frame_abs,
        previous_progress=RouteProgressState.initial(),
    )

    assert projection.route_index == expected_index
    assert projection.future_index == expected_index
    assert projection.proposed_progress.route_index == expected_index


def test_world_policy_clamps_a_single_frame_route():
    projection = WorldRouteProgressPolicy().project(
        _route([(4.0, 6.0)]),
        torch.tensor([4.0, 6.0]),
        torch.tensor(0.0),
        RouteProgressState.initial(),
    )

    assert projection.route_index == 0
    assert projection.future_index == 0
    assert projection.proposed_progress == RouteProgressState.initial()
