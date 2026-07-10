"""Tests for immutable authoritative stream-runtime DTOs."""

from __future__ import annotations

import pytest
import torch

from utils.inference.stream_runtime import (
    ActivatedRootSource,
    ComposeResult,
    KernelStepResult,
    RootSourceCommand,
    RootSourceProposal,
    RouteProgressState,
    RouteStatus,
    RuntimeStepConfig,
    SegmentLabel,
    SessionResetEvent,
    SpaceContract,
    StreamCommitEvent,
)
from utils.inference.timeline import RootFrameState
from utils.token_frame import first_future_frame_abs


def _future(count: int = 4) -> torch.Tensor:
    future = torch.zeros(count, 7)
    future[:, 3] = 1.0
    return future


def _root_state(commit_idx: int) -> RootFrameState:
    return RootFrameState(
        commit_idx=commit_idx,
        world_xz=torch.zeros(2),
        world_yaw=torch.zeros(()),
        source="commit",
    )


def _commit_event_kwargs(absolute_commit_before: int) -> dict:
    absolute_commit_after = absolute_commit_before + 1
    root_frames_start_abs = first_future_frame_abs(absolute_commit_before)
    frame_count = (
        first_future_frame_abs(absolute_commit_after) - root_frames_start_abs
    )
    return {
        "absolute_commit_before": absolute_commit_before,
        "absolute_commit_after": absolute_commit_after,
        "local_commit_before": 3,
        "local_commit_after": 4,
        "latent_buffer_start_commit_abs": 7,
        "latent_buffer_epoch": 2,
        "committed_latent": torch.ones(1, 4),
        "decoded_chunk": torch.zeros(frame_count, 263),
        "joint_frames": torch.zeros(frame_count, 22, 3),
        "root_frames_start_abs": root_frames_start_abs,
        "root_frames": torch.zeros(frame_count, 7),
        "timeline_state": _root_state(absolute_commit_after),
        "actual_payload": None,
        "source_id": None,
        "source_version": None,
        "actual_activation_commit": None,
        "lifecycle_events": (),
    }


def test_proposal_contains_future_only_and_clones_mutable_inputs():
    future = _future()
    metadata = {"origin": {"kind": "route"}}
    proposal = RootSourceProposal(
        future_traj7=future,
        future_frame_mask=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        source_id="route-a",
        version=3,
        metadata=metadata,
    )

    future[0, 0] = 99
    metadata["origin"]["kind"] = "changed"

    assert proposal.future_traj7[0, 0] == 0
    assert proposal.metadata["origin"]["kind"] == "route"
    assert proposal.future_traj7.device.type == "cpu"


@pytest.mark.parametrize(
    ("future", "mask", "error"),
    [
        (torch.zeros(0, 7), torch.zeros(0, dtype=torch.bool), "at least one"),
        (torch.zeros(2, 6), torch.ones(2, dtype=torch.bool), "shape"),
        (torch.zeros(2, 7), torch.ones(3, dtype=torch.bool), "same length"),
        (torch.zeros(2, 7), torch.ones(2, dtype=torch.int64), "dtype"),
    ],
)
def test_proposal_rejects_ambiguous_future_or_mask(future, mask, error):
    with pytest.raises((TypeError, ValueError), match=error):
        RootSourceProposal(
            future_traj7=future,
            future_frame_mask=mask,
            source_id="route-a",
            version=3,
            metadata={},
        )


def test_root_source_command_enforces_replace_and_clear_contracts():
    proposal = RootSourceProposal(
        future_traj7=_future(),
        future_frame_mask=torch.ones(4, dtype=torch.bool),
        source_id="route-a",
        version=3,
        metadata={},
    )

    command = RootSourceCommand.replace(
        proposal=proposal,
        command_version=4,
        requested_activation_commit=2,
        space_contract=SpaceContract.WORLD_ROUTE,
    )
    clear = RootSourceCommand.clear(command_version=5, requested_activation_commit=2)

    assert command.proposal is proposal
    assert clear.proposal is None
    assert clear.space_contract is None

    with pytest.raises(ValueError, match="clear"):
        RootSourceCommand(
            proposal=proposal,
            command_version=6,
            requested_activation_commit=2,
            space_contract=None,
            kind="clear",
        )
    with pytest.raises(ValueError, match="replace"):
        RootSourceCommand(
            proposal=None,
            command_version=7,
            requested_activation_commit=2,
            space_contract=None,
            kind="replace",
        )


def test_activated_source_records_actual_activation_and_checked_future_indices():
    proposal = RootSourceProposal(
        future_traj7=_future(),
        future_frame_mask=torch.ones(4, dtype=torch.bool),
        source_id="route-a",
        version=3,
        metadata={},
    )
    activated = ActivatedRootSource(
        proposal=proposal,
        requested_activation_commit=7,
        actual_activation_commit=10,
        boundary_state=_root_state(10),
        first_future_frame_abs=37,
        space_contract=SpaceContract.RELATIVE_ROUTE,
        progress=RouteProgressState.initial(),
    )

    assert activated.actual_activation_commit == 10
    assert activated.first_future_frame_abs == 37
    assert activated.future_local_index(39) == 2
    assert activated.future_frame_abs(3) == 40
    with pytest.raises(ValueError, match="before"):
        activated.future_local_index(36)
    with pytest.raises(ValueError, match="outside"):
        activated.future_frame_abs(4)


def test_event_clones_mutable_payloads_and_records_rolling_buffer_fields():
    latent = torch.ones(1, 4)
    payload = {"trajectory": torch.ones(1, 2, 7)}
    kwargs = _commit_event_kwargs(10)
    kwargs.update(
        committed_latent=latent,
        actual_payload=payload,
        source_id="route-a",
        source_version=3,
        actual_activation_commit=10,
        lifecycle_events=("route_active",),
    )
    event = StreamCommitEvent(**kwargs)

    latent.zero_()
    payload["trajectory"].zero_()

    assert event.committed_latent.eq(1).all()
    assert event.actual_payload["trajectory"].eq(1).all()
    assert event.absolute_commit_after == event.absolute_commit_before + 1
    assert event.local_commit_after == event.local_commit_before + 1
    assert event.latent_buffer_start_commit_abs == 7
    assert event.latent_buffer_epoch == 2
    assert event.actual_activation_commit == 10


def test_kernel_result_retains_the_exact_precommit_payload_object():
    payload = {"trajectory": torch.ones(1, 2, 7)}
    result = KernelStepResult(
        raw_latent=torch.ones(1, 4),
        actual_payload=payload,
        local_commit_before=3,
        local_commit_after=4,
        latent_buffer_start_commit_abs=7,
        latent_buffer_epoch=2,
    )

    assert result.actual_payload is payload
    assert result.actual_payload["trajectory"] is payload["trajectory"]
    assert result.actual_payload["trajectory"].device == payload["trajectory"].device
    assert result.actual_payload["trajectory"].eq(1).all()


@pytest.mark.parametrize("absolute_commit_before", [0, 10])
def test_event_accepts_exact_causal_root_span(absolute_commit_before):
    kwargs = _commit_event_kwargs(absolute_commit_before)
    event = StreamCommitEvent(**kwargs)

    expected_start = first_future_frame_abs(absolute_commit_before)
    expected_count = first_future_frame_abs(absolute_commit_before + 1) - expected_start
    assert event.root_frames_start_abs == expected_start
    assert event.root_frames.shape[0] == expected_count
    assert event.joint_frames.shape[0] == expected_count


@pytest.mark.parametrize("absolute_commit_before", [0, 10])
def test_event_rejects_wrong_causal_root_start_or_span(absolute_commit_before):
    kwargs = _commit_event_kwargs(absolute_commit_before)
    kwargs["root_frames_start_abs"] += 1
    with pytest.raises(ValueError, match="root_frames_start_abs"):
        StreamCommitEvent(**kwargs)

    kwargs = _commit_event_kwargs(absolute_commit_before)
    expected_count = kwargs["root_frames"].shape[0]
    kwargs["joint_frames"] = torch.zeros(expected_count + 1, 22, 3)
    kwargs["root_frames"] = torch.zeros(expected_count + 1, 7)
    with pytest.raises(ValueError, match="causal frame span"):
        StreamCommitEvent(**kwargs)


@pytest.mark.parametrize("lifecycle_events", ["route_active", ["route_active"]])
def test_event_requires_tuple_lifecycle_events(lifecycle_events):
    kwargs = _commit_event_kwargs(0)
    kwargs["lifecycle_events"] = lifecycle_events
    with pytest.raises(TypeError, match="lifecycle_events"):
        StreamCommitEvent(**kwargs)


def test_compose_result_clones_payloads_and_validates_frame_mask():
    world = _future(2)
    diagnostics = {"route_error": torch.ones(1)}
    result = ComposeResult(
        frame_start_abs=5,
        world_condition_7d=world,
        frame_mask=torch.ones(2, dtype=torch.bool),
        segment_labels=torch.full((2,), SegmentLabel.ROUTE.value),
        proposed_route_progress=RouteProgressState.initial(),
        route_status=RouteStatus.ACTIVE,
        diagnostics=diagnostics,
    )

    world.zero_()
    diagnostics["route_error"].zero_()

    assert result.world_condition_7d[:, 3].eq(1).all()
    assert result.diagnostics["route_error"].eq(1).all()
    with pytest.raises(TypeError, match="frame_mask dtype"):
        ComposeResult(
            frame_start_abs=5,
            world_condition_7d=_future(2),
            frame_mask=torch.ones(2),
            segment_labels=torch.full((2,), SegmentLabel.ROUTE.value),
            proposed_route_progress=RouteProgressState.initial(),
            route_status=RouteStatus.ACTIVE,
            diagnostics={},
        )


def test_runtime_step_config_coerces_controls_and_rejects_invalid_values():
    config = RuntimeStepConfig(
        text=123,
        text_guidance_scale="1.5",
        trajectory_guidance_scale="2.5",
        root_feedback_enabled=1,
        root_feedback_xz_blend_alpha="0.25",
        history_tokens="4",
        horizon_tokens="8",
        num_denoise_steps="10",
    )

    assert config.text == "123"
    assert config.text_guidance_scale == 1.5
    assert config.trajectory_guidance_scale == 2.5
    assert config.root_feedback_enabled is True
    assert config.num_denoise_steps == 10
    with pytest.raises(ValueError, match="root_feedback_xz_blend_alpha"):
        RuntimeStepConfig(root_feedback_xz_blend_alpha=1.1)


def test_kernel_result_validates_one_local_commit_step():
    with pytest.raises(ValueError, match="local_commit_after"):
        KernelStepResult(
            raw_latent=torch.ones(1, 4),
            actual_payload=None,
            local_commit_before=3,
            local_commit_after=5,
            latent_buffer_start_commit_abs=7,
            latent_buffer_epoch=2,
        )


def test_session_reset_event_validates_the_next_epoch():
    event = SessionResetEvent(
        previous_session_epoch=2,
        session_epoch=3,
        applied_command_version=10,
    )
    assert event.session_epoch == 3
    with pytest.raises(ValueError, match="session_epoch"):
        SessionResetEvent(
            previous_session_epoch=2,
            session_epoch=4,
            applied_command_version=10,
        )
