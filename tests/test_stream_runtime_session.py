from __future__ import annotations

import copy
import threading

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
    SessionResetEvent,
    SetRootFeedback,
    SetRootSource,
    SetRuntimeControls,
    SetText,
    SpaceContract,
    StreamRuntimeSession,
)
from utils.inference.timeline import RootFrameState, RootTimeline


class _FakeModel:
    def __init__(self):
        self.generated = torch.zeros(1, 2, 256, 1, 1)
        self.commit_index = 0
        self.current_step = 0
        self.cfg_scale_text = 1.0
        self.cfg_scale_traj = 1.0
        self.noise_steps = 10

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
    def __init__(self, model=None):
        self.ldf_model = model or _FakeModel()
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
        self.clear_count = 0

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
        self.clear_count += 1
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


def _session(*, seed=1234, model=None, initial_config=None):
    torch.manual_seed(seed)
    initial = RootFrameState.initial(dtype=torch.float32)
    return StreamRuntimeSession(
        kernel=_FakeKernel(model=model),
        vae=_FakeVae(),
        recovery=_FakeRecovery(),
        timeline=RootTimeline(initial),
        generated_history=GeneratedRootHistory.empty(dtype=torch.float32),
        command_queue=RuntimeCommandQueue(),
        source_manager=RootSourceManager(),
        composer=ConditionComposer(),
        payload_builder=PayloadBuilder(),
        initial_config=(
            initial_config
            if initial_config is not None
            else RuntimeStepConfig(history_tokens=4, horizon_tokens=4)
        ),
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


def test_session_initializes_owned_vae_cache_before_first_step():
    session = _session()

    assert session.vae.clear_count == 1


class _FreshModel:
    def __init__(self):
        self.cfg_scale_text = 1.0
        self.cfg_scale_traj = 1.0
        self.noise_steps = 10
        self.chunk_size = 1
        self.init_calls = []

    def init_generated(self, seq_len, *, batch_size, num_denoise_steps, traj_buffer):
        self.init_calls.append((seq_len, batch_size, num_denoise_steps, traj_buffer))
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.num_denoise_steps = int(num_denoise_steps)
        self.dt = 1.0 / float(num_denoise_steps)
        self.generated = torch.zeros(batch_size, 2, seq_len * 2 + 1, 1, 1)
        self.commit_index = 0
        self.current_step = 0
        self.text_condition_list = [[] for _ in range(batch_size)]
        self.latent_buffer_start_commit_abs = 0
        self.latent_buffer_epoch = 0

    def snapshot_stream_state(self):
        return copy.deepcopy(self.__dict__)

    def restore_stream_state(self, state):
        self.__dict__.update(copy.deepcopy(state))


def test_fresh_model_session_initializes_before_first_step():
    model = _FreshModel()

    session = _session(
        model=model,
        initial_config=RuntimeStepConfig(
            history_tokens=6,
            horizon_tokens=4,
            num_denoise_steps=10,
        ),
    )

    assert model.init_calls == [(6, 1, 10, None)]
    assert model.seq_len == 6
    assert model.commit_index == 0
    assert model.current_step == 0
    assert model.dt == pytest.approx(0.1)
    assert model.text_condition_list == [[]]
    assert model.latent_buffer_start_commit_abs == 0
    event = session.step()
    assert event.absolute_commit_after == 1


def test_session_rejects_model_timeline_absolute_commit_mismatch():
    session = _session()
    session.timeline.append(
        RootFrameState(
            commit_idx=1,
            world_xz=torch.zeros(2),
            world_yaw=torch.tensor(0.0),
        )
    )

    with pytest.raises(RuntimeError, match="LDF/timeline commit mismatch"):
        session.step()

    assert session.model.commit_index == 0
    assert session.timeline.head.commit_idx == 1


def test_runtime_history_and_denoise_changes_require_fresh_epoch():
    session = _session()
    session.step()

    with pytest.raises(RuntimeError, match="require a reset"):
        session.submit(
            SetRuntimeControls(
                version=1,
                requested_commit_abs=1,
                history_tokens=8,
            )
        )
    with pytest.raises(RuntimeError, match="require a reset"):
        session.submit(
            SetRuntimeControls(
                version=2,
                requested_commit_abs=1,
                num_denoise_steps=20,
            )
        )

    assert session.command_queue.pending_versions == ()


def test_future_old_epoch_command_is_discarded_by_reset():
    session = _session()
    session.submit(
        SetRootSource(
            version=1,
            requested_commit_abs=100,
            proposal=_proposal(),
            space_contract=SpaceContract.WORLD_ROUTE,
        )
    )
    session.submit(ResetSession(version=2, requested_commit_abs=0))

    event = session.step()

    assert isinstance(event, SessionResetEvent)
    assert session.command_queue.pending_versions == ()
    assert session.source_manager.active is None


def test_nonexclusive_reset_reduces_from_session_initial_config():
    initial = RuntimeStepConfig(
        text_guidance_scale=1.25,
        trajectory_guidance_scale=3.0,
        root_feedback_enabled=True,
        history_tokens=4,
        horizon_tokens=7,
    )
    session = _session(initial_config=initial)
    session.submit(ResetSession(version=1, requested_commit_abs=0))
    session.submit(SetText(version=2, requested_commit_abs=0, text="walk"))

    session.step()

    assert session.config.text == "walk"
    assert session.config.text_guidance_scale == pytest.approx(1.25)
    assert session.config.trajectory_guidance_scale == pytest.approx(3.0)
    assert session.config.root_feedback_enabled is True
    assert session.config.horizon_tokens == 7


def test_failed_nonexclusive_reset_restores_object_identity():
    session = _session()
    timeline = session.timeline
    history = session.generated_history
    session.submit(ResetSession(version=1, requested_commit_abs=0))
    session.submit(SetText(version=2, requested_commit_abs=0, text="walk"))

    def fail_generation(*args, **kwargs):
        raise RuntimeError("generation failed")

    session.kernel.generate_token = fail_generation

    with pytest.raises(RuntimeError, match="generation failed"):
        session.step()

    assert session.timeline is timeline
    assert session.generated_history is history
    assert session.timeline.head.commit_idx == 0
    assert session.generated_history.next_frame_abs == 0


def test_concurrent_session_steps_are_rejected():
    session = _session()
    entered = threading.Event()
    release = threading.Event()
    original = session.kernel.generate_token
    calls = 0

    def blocking_generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            release.wait(timeout=5.0)
        return original(*args, **kwargs)

    session.kernel.generate_token = blocking_generate
    errors = []

    def run_step():
        try:
            session.step()
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    worker = threading.Thread(target=run_step)
    worker.start()
    assert entered.wait(timeout=2.0)
    try:
        with pytest.raises(RuntimeError, match="concurrent.*step"):
            session.step()
    finally:
        release.set()
        worker.join(timeout=5.0)
    assert not errors


def test_long_rollout_keeps_history_and_timeline_bounded():
    session = _session(
        initial_config=RuntimeStepConfig(history_tokens=4, horizon_tokens=4)
    )

    for _ in range(100):
        session.step()

    assert session.timeline.head.commit_idx == 100
    assert len(session.timeline) <= 5
    assert int(session.generated_history.frames_7d.shape[0]) <= 20
    assert session.generated_history.next_frame_abs == 397


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
