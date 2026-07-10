"""Tests for immutable authoritative stream-runtime DTOs."""

from __future__ import annotations

import pytest
import torch

from utils.inference.stream_runtime import (
    ActivatedRootSource,
    RootSourceCommand,
    RootSourceProposal,
    RouteProgressState,
    SpaceContract,
    StreamCommitEvent,
)
from utils.inference.timeline import RootFrameState


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
    event = StreamCommitEvent(
        absolute_commit_before=10,
        absolute_commit_after=11,
        local_commit_before=3,
        local_commit_after=4,
        latent_buffer_start_commit_abs=7,
        latent_buffer_epoch=2,
        committed_latent=latent,
        decoded_chunk=torch.zeros(1, 263),
        joint_frames=torch.zeros(1, 22, 3),
        root_frames_start_abs=37,
        root_frames=torch.zeros(1, 7),
        timeline_state=_root_state(11),
        actual_payload=payload,
        source_id="route-a",
        source_version=3,
        actual_activation_commit=10,
        lifecycle_events=("route_active",),
    )

    latent.zero_()
    payload["trajectory"].zero_()

    assert event.committed_latent.eq(1).all()
    assert event.actual_payload["trajectory"].eq(1).all()
    assert event.absolute_commit_after == event.absolute_commit_before + 1
    assert event.local_commit_after == event.local_commit_before + 1
    assert event.latent_buffer_start_commit_abs == 7
    assert event.latent_buffer_epoch == 2
    assert event.actual_activation_commit == 10
