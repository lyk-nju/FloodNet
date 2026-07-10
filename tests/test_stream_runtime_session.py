from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from utils.inference.stream_runtime import (
    ConditionComposer,
    GeneratedRootHistory,
    KernelStepResult,
    PayloadBuilder,
    RootSourceManager,
    RootSourceProposal,
    RouteStatus,
    RuntimeCommandQueue,
    RuntimeStepConfig,
    ResetSession,
    SetRootFeedback,
    SetRootSource,
    SpaceContract,
    StreamRuntimeSession,
)
from utils.inference.timeline import RootFrameState, RootTimeline


class _FakeModel:
    def __init__(self):
        self.generated = torch.zeros(1, 2, 8, 1, 1)
        self.commit_index = 0
        self.current_step = 0
        self.cfg_scale_text = 1.0
        self.cfg_scale_traj = 1.0

    def snapshot_stream_state(self):
        return {
            "generated": self.generated.clone(),
            "commit_index": self.commit_index,
            "current_step": self.current_step,
            "cfg_scale_text": self.cfg_scale_text,
            "cfg_scale_traj": self.cfg_scale_traj,
        }

    def restore_stream_state(self, state):
        for key, value in state.items():
            setattr(
                self,
                key,
                value.clone() if torch.is_tensor(value) else copy.deepcopy(value),
            )


class _FakeKernel:
    def __init__(self):
        self.ldf_model = _FakeModel()
        self.device = torch.device("cpu")
        self.chunk_size = 1

    def generate_token(self, text, payload, *, first_chunk, num_denoise_steps=None):
        del text, first_chunk, num_denoise_steps
        before = self.ldf_model.commit_index
        latent = torch.rand(1, 2)
        self.ldf_model.generated[0, :, before, 0, 0] = latent[0]
        self.ldf_model.commit_index += 1
        return KernelStepResult(
            raw_latent=latent,
            actual_payload=payload,
            local_commit_before=before,
            local_commit_after=before + 1,
            latent_buffer_start_commit_abs=0,
            latent_buffer_epoch=0,
        )


class _FakeVae:
    def __init__(self):
        self.decoder_count = 0
        self.encoder_count = 0
        self.calls = []

    def snapshot_stream_state(self):
        return {
            "decoder_count": self.decoder_count,
            "encoder_count": self.encoder_count,
        }

    def restore_stream_state(self, state):
        self.decoder_count = int(state["decoder_count"])
        self.encoder_count = int(state["encoder_count"])

    def stream_decode(self, latent, first_chunk=True):
        del latent
        self.calls.append("decode")
        self.decoder_count += 1
        frames = 1 if first_chunk else 4
        decoded = torch.zeros(1, frames, 263)
        decoded[..., 2] = 0.1
        decoded[..., 3] = 1.0
        return decoded

    def stream_encode(self, motion, first_chunk=True):
        del motion, first_chunk
        self.calls.append("encode")
        self.encoder_count += 1
        return torch.zeros(1, 1, 2)

    def clear_cache(self):
        self.decoder_count = 0
        self.encoder_count = 0


class _FakeRecovery:
    def __init__(self):
        self.calls = 0
        self.fail = False
        self.r_pos_accum = np.zeros(3, dtype=np.float32)
        self.r_rot_ang_accum = 0.0

    def process_frame(self, frame):
        del frame
        if self.fail:
            raise RuntimeError("recovery failed")
        self.calls += 1
        self.r_pos_accum[2] += 0.1
        joints = np.zeros((22, 3), dtype=np.float32)
        joints[:, 2] = self.r_pos_accum[2]
        return joints

    def reset(self):
        self.__init__()


def _session(*, seed=1234):
    torch.manual_seed(seed)
    initial = RootFrameState.initial(dtype=torch.float32)
    return StreamRuntimeSession(
        kernel=_FakeKernel(),
        vae=_FakeVae(),
        recovery=_FakeRecovery(),
        timeline=RootTimeline(initial),
        generated_history=GeneratedRootHistory.empty(dtype=torch.float32),
        command_queue=RuntimeCommandQueue(),
        source_manager=RootSourceManager(),
        composer=ConditionComposer(),
        payload_builder=PayloadBuilder(),
        initial_config=RuntimeStepConfig(history_tokens=4, horizon_tokens=4),
    )


def _proposal():
    future = torch.zeros(32, 7)
    future[:, 2] = torch.arange(1, 33, dtype=torch.float32) * 0.1
    future[:, 3] = 1.0
    return RootSourceProposal(
        future_traj7=future,
        future_frame_mask=torch.ones(32, dtype=torch.bool),
        source_id="route-a",
        version=1,
        metadata={},
    )


def test_two_steps_commit_frame0_then_frame4():
    session = _session()

    first = session.step()
    second = session.step()

    assert first.absolute_commit_after == 1
    assert first.root_frames_start_abs == 0
    assert first.root_frames.shape[0] == 1
    assert second.absolute_commit_after == 2
    assert second.root_frames_start_abs == 1
    assert second.root_frames.shape[0] == 4
    assert session.timeline.head.commit_idx == 2
    assert session.generated_history.next_frame_abs == 5
    assert torch.equal(second.timeline_state.world_xz, second.root_frames[-1, [0, 2]])


def test_progress_and_commands_commit_only_after_success():
    session = _session()
    session.submit(
        SetRootSource(
            version=1,
            requested_commit_abs=0,
            proposal=_proposal(),
            space_contract=SpaceContract.WORLD_ROUTE,
        )
    )
    session.recovery.fail = True

    with pytest.raises(RuntimeError, match="recovery failed"):
        session.step()

    assert session.source_manager.active is None
    assert session.command_queue.pending_versions == (1,)
    assert session.timeline.head.commit_idx == 0
    assert session.generated_history.next_frame_abs == 0


def test_failed_step_retry_matches_fresh_success_including_rng_and_caches():
    failed = _session(seed=1234)
    failed.recovery.fail = True
    with pytest.raises(RuntimeError):
        failed.step()
    failed.recovery.fail = False
    retry = failed.step()

    fresh = _session(seed=1234)
    expected = fresh.step()

    assert torch.equal(retry.committed_latent, expected.committed_latent)
    assert torch.equal(retry.decoded_chunk, expected.decoded_chunk)
    assert torch.equal(retry.root_frames, expected.root_frames)
    assert retry.timeline_state.commit_idx == expected.timeline_state.commit_idx
    assert failed.vae.decoder_count == fresh.vae.decoder_count == 1


def test_route_exhaustion_lifecycle_is_emitted_once_after_commit():
    session = _session()
    base = _proposal()
    proposal = RootSourceProposal(
        future_traj7=base.future_traj7[:1],
        future_frame_mask=torch.ones(1, dtype=torch.bool),
        source_id="short",
        version=2,
        metadata={},
    )
    session.submit(
        SetRootSource(
            version=1,
            requested_commit_abs=0,
            proposal=proposal,
            space_contract=SpaceContract.WORLD_ROUTE,
        )
    )

    first = session.step()
    second = session.step()
    third = session.step()

    assert "route_active" in first.lifecycle_events
    assert first.route_status is RouteStatus.ACTIVE
    assert second.route_status is RouteStatus.EXHAUSTED
    assert second.lifecycle_events.count("route_exhausted") == 1
    assert "route_exhausted" not in third.lifecycle_events


def test_root_feedback_preview_encode_and_formal_decode_cache_order():
    session = _session()
    session.submit(
        SetRootSource(
            version=1,
            requested_commit_abs=0,
            proposal=_proposal(),
            space_contract=SpaceContract.WORLD_ROUTE,
        )
    )
    session.submit(
        SetRootFeedback(
            version=2,
            requested_commit_abs=0,
            enabled=True,
            xz_blend_alpha=1.0,
        )
    )

    event = session.step()

    assert event.root_feedback_diagnostics["reason"] == "applied"
    assert session.vae.calls == ["decode", "encode", "decode"]
    assert session.vae.decoder_count == 1
    assert session.vae.encoder_count == 1


def test_exclusive_reset_command_resets_owned_state_without_generating():
    session = _session()
    session.step()
    session.submit(ResetSession(version=1, requested_commit_abs=1))

    event = session.step()

    assert event.previous_session_epoch == 0
    assert event.session_epoch == 1
    assert session.timeline.head.commit_idx == 0
    assert session.generated_history.next_frame_abs == 0
    assert session.kernel.ldf_model.commit_index == 0
    assert session.command_queue.pending_versions == ()
