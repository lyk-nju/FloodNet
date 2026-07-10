import torch


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


def test_rootplan_adapter_removes_current_anchor_from_future():
    from utils.inference.root_plan import RootPlan
    from utils.inference.runtime_update import root_plan_to_proposal
    from utils.local_frame import uncanonicalize_7d

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

    world = uncanonicalize_7d(
        local.unsqueeze(0),
        plan.anchor_world_xz.unsqueeze(0),
        plan.anchor_world_yaw.reshape(1),
    )[0]
    proposal = root_plan_to_proposal(
        plan,
        source_id="refined_route",
        version=4,
        source_kind="root_refiner",
        update_frames=[12],
    )

    assert proposal.source_id == "refined_route"
    assert proposal.version == 4
    assert proposal.future_traj7.shape[0] == plan.valid_frames - 1
    assert torch.equal(proposal.future_traj7[:, :5], world[1:, :5])
    assert proposal.future_frame_mask.all()
    assert proposal.metadata["source_kind"] == "root_refiner"
    assert proposal.metadata["update_frames"] == (12,)
    assert proposal.metadata["root_plan_source"] == "root_refiner"
    assert proposal.metadata["anchor_commit_abs"] == 3


def test_eval_root_source_is_authoritative_identity_reexport():
    from eval.ldf.runtime_update.root_source import RootSourceProposal as EvalProposal
    from utils.inference.stream_runtime import RootSourceProposal as RuntimeProposal

    assert EvalProposal is RuntimeProposal
