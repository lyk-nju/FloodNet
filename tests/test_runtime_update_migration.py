import pytest
import torch

from utils.token_frame import commit_boundary_frame
from utils.motion_process import build_physical_7d_from_5d


def test_runtime_update_public_api_is_importable_from_utils():
    from utils.inference.runtime_update import (
        ActiveWindowSegment,
        RootSourceProposal,
        RouteProgressTracker,
        build_world_condition_stream_payload,
        compose_active_window_segment,
    )

    assert ActiveWindowSegment is not None
    assert RootSourceProposal is not None
    assert RouteProgressTracker is not None
    assert build_world_condition_stream_payload is not None
    assert compose_active_window_segment is not None


def test_eval_runtime_update_wrappers_reexport_utils_implementations():
    from eval.ldf.runtime_update.active_condition import (
        compose_active_window_segment as eval_compose_segment,
    )
    from eval.ldf.runtime_update.payload_builder import (
        build_world_condition_stream_payload as eval_build_payload,
    )
    from eval.ldf.runtime_update.route_tracker import (
        RouteProgressTracker as EvalRouteProgressTracker,
    )
    from eval.ldf.runtime_update.root_source import RootSourceProposal as EvalRootSource
    from utils.inference.runtime_update.active_condition import (
        compose_active_window_segment as runtime_compose_segment,
    )
    from utils.inference.runtime_update.payload_builder import (
        build_world_condition_stream_payload as runtime_build_payload,
    )
    from utils.inference.runtime_update.route_tracker import (
        RouteProgressTracker as RuntimeRouteProgressTracker,
    )
    from utils.inference.runtime_update.root_source import (
        RootSourceProposal as RuntimeRootSource,
    )

    assert eval_compose_segment is runtime_compose_segment
    assert eval_build_payload is runtime_build_payload
    assert EvalRouteProgressTracker is RuntimeRouteProgressTracker
    assert EvalRootSource is RuntimeRootSource


def test_root_source_proposal_keeps_world_frame_route_contract():
    from utils.inference.runtime_update import RootSourceProposal

    traj7 = torch.zeros(8, 7)
    traj7[:, 3] = 1.0
    proposal = RootSourceProposal(
        name="unit_route",
        proposal_traj7=traj7,
        source_kind="synthetic",
        update_frames=[2, 4, 6],
        metadata={"purpose": "test"},
    )

    assert proposal.num_frames == 8
    assert proposal.update_frames == [2, 4, 6]
    assert proposal.metadata["purpose"] == "test"
    assert proposal.start_frame_abs == 0
    assert proposal.start_commit_abs == 0
    assert proposal.timeline_mode == "absolute_timeline"


def test_root_source_proposal_can_be_built_from_root_plan_world_route():
    from utils.inference.root_plan import RootPlan
    from utils.inference.runtime_update import RootSourceProposal

    local = torch.zeros(6, 7)
    local[:, 2] = torch.arange(6, dtype=torch.float32)
    local[:, 3] = 1.0
    plan = RootPlan(
        num_tokens_pred=2,
        valid_frames=6,
        waypoints_local_7d=local,
        frame_dt=0.05,
        frames_per_token=4,
        anchor_commit_idx=3,
        anchor_world_xz=torch.tensor([10.0, 20.0]),
        anchor_world_yaw=torch.tensor(0.0),
        source="root_refiner",
    )

    proposal = RootSourceProposal.from_root_plan(
        plan,
        name="refined_route",
        source_kind="root_refiner",
        update_frames=[12],
    )

    assert proposal.name == "refined_route"
    assert proposal.source_kind == "root_refiner"
    assert proposal.update_frames == [12]
    assert torch.allclose(proposal.proposal_traj7[:, 0], torch.full((6,), 10.0))
    assert torch.allclose(
        proposal.proposal_traj7[:, 2],
        torch.arange(6, dtype=torch.float32) + 20.0,
    )
    assert proposal.metadata["root_plan_source"] == "root_refiner"
    assert proposal.start_commit_abs == 3
    assert proposal.start_frame_abs == commit_boundary_frame(3, 4)
    assert proposal.timeline_mode == "anchor_relative"


def test_root_source_proposal_converts_absolute_and_local_frames_strictly():
    from utils.inference.runtime_update import RootSourceProposal

    traj7 = torch.zeros(6, 7)
    traj7[:, 3] = 1.0
    proposal = RootSourceProposal(
        name="anchored_route",
        proposal_traj7=traj7,
        source_kind="root_refiner",
        start_frame_abs=commit_boundary_frame(3, 4),
        start_commit_abs=3,
        timeline_mode="anchor_relative",
    )

    assert proposal.absolute_to_local_frame(commit_boundary_frame(3, 4)) == 0
    assert proposal.absolute_to_local_frame(commit_boundary_frame(3, 4) + 5) == 5
    assert proposal.local_to_absolute_frame(4) == commit_boundary_frame(3, 4) + 4
    with pytest.raises(ValueError, match="before proposal origin"):
        proposal.absolute_to_local_frame(commit_boundary_frame(3, 4) - 1)
    with pytest.raises(ValueError, match="outside proposal"):
        proposal.absolute_to_local_frame(commit_boundary_frame(3, 4) + 6)
    with pytest.raises(ValueError, match="outside proposal"):
        proposal.local_to_absolute_frame(6)


def test_root_source_proposal_materializes_route_at_absolute_origin():
    from utils.inference.runtime_update import RootSourceProposal

    traj7 = torch.zeros(4, 7)
    traj7[:, 0] = torch.arange(4, dtype=torch.float32)
    traj7[:, 1] = 1.0
    traj7[:, 3] = 1.0
    start_commit = 2
    start_frame = commit_boundary_frame(start_commit, 4)
    proposal = RootSourceProposal(
        name="anchored_route",
        proposal_traj7=traj7,
        source_kind="root_refiner",
        start_frame_abs=start_frame,
        start_commit_abs=start_commit,
        timeline_mode="anchor_relative",
    )

    materialized = proposal.to_absolute_timeline()

    assert materialized.shape == (start_frame + traj7.shape[0], 7)
    assert torch.allclose(materialized[:start_frame, :5], traj7[:1, :5])
    assert torch.allclose(materialized[start_frame:, :5], traj7[:, :5])
    recomputed = build_physical_7d_from_5d(materialized[:, :5])
    assert torch.allclose(materialized[:, 5:7], recomputed[:, 5:7])
