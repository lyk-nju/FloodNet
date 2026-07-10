"""Tests for boundary-applied stream-runtime commands."""

from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError

import pytest
import torch

from utils.inference.stream_runtime import (
    ClearRootSource,
    PreparedCommandBatch,
    PreparedRuntimeTransition,
    ResetSession,
    RootSourceProposal,
    RuntimeCommand,
    RuntimeCommandQueue,
    RuntimeStepConfig,
    SetGuidance,
    SetRootFeedback,
    SetRootSource,
    SetRuntimeControls,
    SetText,
    SpaceContract,
    reduce_commands,
)
from utils.inference.timeline import RootFrameState


def _proposal(source_id: str, version: int) -> RootSourceProposal:
    future = torch.zeros(2, 7)
    future[:, 3] = 1.0
    return RootSourceProposal(
        future_traj7=future,
        future_frame_mask=torch.ones(2, dtype=torch.bool),
        source_id=source_id,
        version=version,
        metadata={},
    )


def test_prepare_is_non_destructive_and_ack_removes_exact_versions():
    queue = RuntimeCommandQueue()
    queue.submit(SetText(version=1, requested_commit_abs=0, text="walk"))
    batch = queue.prepare_due(0)
    queue.submit(SetText(version=2, requested_commit_abs=0, text="run"))

    assert [command.version for command in queue.prepare_due(0).commands] == [1, 2]

    queue.ack(batch)

    assert [command.version for command in queue.prepare_due(0).commands] == [2]


def test_release_discards_prepared_handle_without_removing_pending_commands():
    queue = RuntimeCommandQueue()
    queue.submit(SetText(version=1, requested_commit_abs=0, text="walk"))
    batch = queue.prepare_due(0)

    queue.release(batch)

    assert queue.pending_versions == (1,)
    with pytest.raises(ValueError, match="no longer valid"):
        queue.ack(batch)


def test_ack_rejects_a_caller_constructed_batch_without_removing_future_commands():
    queue = RuntimeCommandQueue()
    queue.submit(SetText(version=1, requested_commit_abs=5, text="future"))
    forged = PreparedCommandBatch(
        commands=(SetText(version=1, requested_commit_abs=0, text="forged"),)
    )

    with pytest.raises(ValueError, match="issued"):
        queue.ack(forged)

    assert [command.version for command in queue.snapshot()] == [1]


def test_ack_rejects_a_batch_issued_by_a_different_queue():
    source = RuntimeCommandQueue()
    other = RuntimeCommandQueue()
    source.submit(SetText(version=1, requested_commit_abs=0, text="walk"))
    batch = source.prepare_due(0)

    with pytest.raises(ValueError, match="issued"):
        other.ack(batch)

    assert source.pending_versions == (1,)


def test_ack_rejects_stale_and_double_acknowledgements():
    queue = RuntimeCommandQueue()
    queue.submit(SetText(version=1, requested_commit_abs=0, text="walk"))
    first = queue.prepare_due(0)
    overlapping = queue.prepare_due(0)

    queue.ack(first)

    with pytest.raises(ValueError, match="issued"):
        queue.ack(first)
    with pytest.raises(ValueError, match="issued"):
        queue.ack(overlapping)
    assert queue.pending_versions == ()


def test_submit_between_prepare_and_ack_remains_pending_without_timing_sleep():
    queue = RuntimeCommandQueue()
    queue.submit(SetText(version=1, requested_commit_abs=0, text="walk"))
    batch = queue.prepare_due(0)
    submitted = threading.Barrier(2)

    def submit_later() -> None:
        submitted.wait()
        queue.submit(SetText(version=2, requested_commit_abs=0, text="run"))

    thread = threading.Thread(target=submit_later)
    thread.start()
    submitted.wait()
    thread.join()

    queue.ack(batch)

    assert [command.version for command in queue.snapshot()] == [2]


def test_queue_requires_strictly_increasing_global_versions():
    queue = RuntimeCommandQueue()
    queue.submit(SetText(version=4, requested_commit_abs=0, text="walk"))

    with pytest.raises(ValueError, match="strictly increasing"):
        queue.submit(SetGuidance(version=4, requested_commit_abs=0, text_guidance_scale=2.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        queue.submit(SetText(version=3, requested_commit_abs=0, text="run"))


def test_queue_and_batches_reject_base_and_unknown_runtime_commands():
    class UnknownRuntimeCommand(RuntimeCommand):
        pass

    queue = RuntimeCommandQueue()
    base = RuntimeCommand(version=1, requested_commit_abs=0)
    unknown = UnknownRuntimeCommand(version=2, requested_commit_abs=0)

    for command in (base, unknown):
        with pytest.raises(TypeError, match="supported runtime command"):
            queue.submit(command)
        with pytest.raises(TypeError, match="supported runtime command"):
            PreparedCommandBatch(commands=(command,))


def test_replace_clear_replace_reduces_by_global_version():
    boundary = RootFrameState.initial(dtype=torch.float32)
    batch = PreparedCommandBatch(
        commands=(
            SetRootSource(
                version=10,
                requested_commit_abs=0,
                proposal=_proposal("a", version=10),
                space_contract=SpaceContract.WORLD_ROUTE,
            ),
            ClearRootSource(version=11, requested_commit_abs=0),
            SetRootSource(
                version=12,
                requested_commit_abs=0,
                proposal=_proposal("b", version=12),
                space_contract=SpaceContract.WORLD_ROUTE,
            ),
        )
    )

    transition = reduce_commands(RuntimeStepConfig.default(), batch, boundary)

    assert transition.root_source_command is not None
    assert transition.root_source_command.proposal is not None
    assert transition.root_source_command.proposal.source_id == "b"
    assert transition.root_source_command.command_version == 12
    assert transition.root_source_command.requested_activation_commit == 0
    assert transition.superseded_versions == (10, 11)


def test_reduction_is_pure_and_uses_last_write_per_field():
    boundary = RootFrameState.initial(dtype=torch.float32)
    base = RuntimeStepConfig(text="idle", text_guidance_scale=1.0)
    batch = PreparedCommandBatch(
        commands=(
            SetText(version=1, requested_commit_abs=0, text="walk"),
            SetGuidance(version=2, requested_commit_abs=0, text_guidance_scale=2.0),
            SetRootFeedback(version=3, requested_commit_abs=0, enabled=True),
            SetRuntimeControls(version=4, requested_commit_abs=0, horizon_tokens=30),
            SetText(version=5, requested_commit_abs=0, text="run"),
            SetGuidance(
                version=6,
                requested_commit_abs=0,
                trajectory_guidance_scale=3.0,
            ),
        )
    )

    transition = reduce_commands(base, batch, boundary)

    assert base.text == "idle"
    assert transition.proposed_config == RuntimeStepConfig(
        text="run",
        text_guidance_scale=2.0,
        trajectory_guidance_scale=3.0,
        root_feedback_enabled=True,
        root_feedback_xz_blend_alpha=0.5,
        history_tokens=30,
        horizon_tokens=30,
        num_denoise_steps=None,
    )
    assert transition.superseded_versions == (1,)
    with pytest.raises(FrozenInstanceError):
        transition.proposed_config.text = "mutate"  # type: ignore[misc]


def test_final_reset_is_exclusive_but_later_commands_apply_in_the_new_epoch():
    boundary = RootFrameState.initial(dtype=torch.float32)
    base = RuntimeStepConfig(text="idle", text_guidance_scale=7.0)
    final_reset = reduce_commands(
        base,
        PreparedCommandBatch(
            commands=(
                SetText(version=1, requested_commit_abs=0, text="walk"),
                ResetSession(version=2, requested_commit_abs=0),
            )
        ),
        boundary,
    )
    later_text = reduce_commands(
        base,
        PreparedCommandBatch(
            commands=(
                SetText(version=1, requested_commit_abs=0, text="walk"),
                ResetSession(version=2, requested_commit_abs=0),
                SetText(version=3, requested_commit_abs=0, text="run"),
            )
        ),
        boundary,
    )

    assert final_reset.reset_intent is not None
    assert final_reset.reset_intent.version == 2
    assert final_reset.proposed_config == RuntimeStepConfig.default()
    assert final_reset.superseded_versions == (1,)
    assert later_text.reset_intent is not None
    assert later_text.proposed_config.text == "run"
    assert later_text.proposed_config.text_guidance_scale == 1.0
    assert later_text.superseded_versions == (1,)


def test_prepared_transition_requires_a_reset_session_for_reset_intent():
    with pytest.raises(TypeError, match="reset_intent must be ResetSession"):
        PreparedRuntimeTransition(
            proposed_config=RuntimeStepConfig.default(),
            root_source_command=None,
            superseded_versions=(),
            diagnostics={},
            reset_intent=SetText(version=1, requested_commit_abs=0, text="walk"),
        )
