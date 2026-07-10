from __future__ import annotations

import copy
import threading
from types import MappingProxyType

import numpy as np
import torch

from utils.inference.stream_generator import StreamGenerator
from utils.inference.stream_runtime import (
    ConditionComposer,
    GeneratedRootHistory,
    KernelStepResult,
    PayloadBuilder,
    RootSourceManager,
    RootSourceProposal,
    RuntimeCommandQueue,
    RuntimeStepConfig,
    SetRootSource,
    SetText,
    SpaceContract,
    StreamRuntimeSession,
)
from utils.inference.timeline import RootFrameState, RootTimeline
from tools.check_stream_runtime_parity import _equal, _traj_cfg


class _RollingModel:
    def __init__(self, capacity=16):
        self.generated = torch.zeros(1, 2, capacity, 1, 1)
        self.commit_index = 0
        self.current_step = 0
        self.cfg_scale_text = 1.0
        self.cfg_scale_traj = 1.0
        self.latent_buffer_start_commit_abs = 0
        self.latent_buffer_epoch = 0

    def snapshot_stream_state(self):
        return copy.deepcopy(self.__dict__)

    def restore_stream_state(self, state):
        self.__dict__.update(copy.deepcopy(state))


class _RollingKernel(StreamGenerator):
    def __init__(self, *, barrier=None):
        super().__init__(ldf_model=_RollingModel(), device="cpu")
        self.barrier = barrier
        self.calls = []

    def generate_token(self, text, payload, *, first_chunk, num_denoise_steps=None):
        del first_chunk, num_denoise_steps
        model = self.ldf_model
        if model.commit_index == model.generated.shape[2]:
            model.generated.zero_()
            model.commit_index = 0
            model.latent_buffer_start_commit_abs += model.generated.shape[2]
            model.latent_buffer_epoch += 1
        before = model.commit_index
        self.calls.append((str(text), payload))
        if self.barrier is not None:
            entered, release = self.barrier
            entered.set()
            release.wait(timeout=5.0)
        latent = torch.tensor([[float(before), float(model.latent_buffer_epoch)]])
        model.generated[0, :, before, 0, 0] = latent[0]
        model.commit_index += 1
        return KernelStepResult(
            raw_latent=latent,
            actual_payload=payload,
            local_commit_before=before,
            local_commit_after=before + 1,
            latent_buffer_start_commit_abs=model.latent_buffer_start_commit_abs,
            latent_buffer_epoch=model.latent_buffer_epoch,
        )


class _Vae:
    def __init__(self):
        self.count = 0

    def snapshot_stream_state(self):
        return {"count": self.count}

    def restore_stream_state(self, state):
        self.count = int(state["count"])

    def stream_decode(self, latent, first_chunk=True):
        self.count += 1
        frames = 1 if first_chunk else 4
        out = torch.zeros(1, frames, 263)
        out[..., 2] = 0.05 + float(latent[0, 0, 0]) * 0.001
        out[..., 3] = 1.0
        return out


class _Recovery:
    def __init__(self):
        self.r_pos_accum = np.zeros(3, dtype=np.float32)
        self.r_rot_ang_accum = 0.0

    def process_frame(self, frame):
        self.r_pos_accum[2] += float(frame[2])
        joints = np.zeros((22, 3), dtype=np.float32)
        joints[:, 2] = self.r_pos_accum[2]
        return joints

    def reset(self):
        self.__init__()


def _session(*, barrier=None):
    generator = _RollingKernel(barrier=barrier)
    session = StreamRuntimeSession(
        kernel=generator,
        vae=_Vae(),
        recovery=_Recovery(),
        timeline=RootTimeline(RootFrameState.initial()),
        generated_history=GeneratedRootHistory.empty(),
        command_queue=RuntimeCommandQueue(),
        source_manager=RootSourceManager(),
        composer=ConditionComposer(),
        payload_builder=PayloadBuilder(),
        initial_config=RuntimeStepConfig(history_tokens=4, horizon_tokens=4),
    )
    generator.attach_runtime_session(session)
    return generator, session, generator


def _assert_events_equal(left, right):
    scalar_fields = (
        "absolute_commit_before",
        "absolute_commit_after",
        "local_commit_before",
        "local_commit_after",
        "latent_buffer_start_commit_abs",
        "latent_buffer_epoch",
        "root_frames_start_abs",
        "source_id",
        "source_version",
        "actual_activation_commit",
        "route_status",
    )
    for field in scalar_fields:
        assert getattr(left, field) == getattr(right, field)
    for field in ("committed_latent", "decoded_chunk", "root_frames", "joint_frames"):
        assert torch.equal(getattr(left, field), getattr(right, field))
    assert left.timeline_state.commit_idx == right.timeline_state.commit_idx
    assert torch.equal(left.timeline_state.world_xz, right.timeline_state.world_xz)
    assert left.lifecycle_events == right.lifecycle_events


def test_direct_and_compatibility_entry_points_match_across_buffer_rolls():
    direct_generator, direct, _ = _session()
    compat_generator, compat, _ = _session()
    del direct_generator, compat

    direct_events = [direct.step() for _ in range(70)]
    compat_events = [compat_generator.execute_step() for _ in range(70)]

    for left, right in zip(direct_events, compat_events):
        _assert_events_equal(left, right)
    assert direct_events[-1].latent_buffer_epoch >= 4
    assert direct_events[-1].absolute_commit_after == 70


def _proposal():
    future = torch.zeros(24, 7)
    future[:, 2] = torch.arange(1, 25, dtype=torch.float32) * 0.1
    future[:, 3] = 1.0
    return RootSourceProposal(
        future_traj7=future,
        future_frame_mask=torch.ones(24, dtype=torch.bool),
        source_id="route-next",
        version=2,
        metadata={},
    )


def test_mid_step_commands_activate_together_at_next_boundary():
    entered = threading.Event()
    release = threading.Event()
    _, session, kernel = _session(barrier=(entered, release))
    result = []
    thread = threading.Thread(target=lambda: result.append(session.step()))
    thread.start()
    assert entered.wait(timeout=2.0)

    session.submit(SetText(version=1, requested_commit_abs=0, text="turn left"))
    session.submit(
        SetRootSource(
            version=2,
            requested_commit_abs=0,
            proposal=_proposal(),
            space_contract=SpaceContract.WORLD_ROUTE,
        )
    )
    release.set()
    thread.join(timeout=5.0)

    assert result[0].source_id is None
    assert kernel.calls[0][0] == ""
    kernel.barrier = None
    second = session.step()
    assert kernel.calls[1][0] == "turn left"
    assert second.source_id == "route-next"
    assert second.actual_activation_commit == 1


def test_real_parity_comparator_supports_read_only_nested_mappings():
    left = MappingProxyType(
        {"payload": MappingProxyType({"condition": torch.tensor([1.0, 2.0])})}
    )
    right = {"payload": {"condition": torch.tensor([1.0, 2.0])}}

    assert _equal(left, right) is None


def test_real_parity_config_disables_unused_root_refiner(tmp_path):
    config = tmp_path / "stream.yaml"
    config.write_text(
        "traj_mask:\n"
        "  history_length: 30\n"
        "  root_refiner:\n"
        "    enabled: true\n"
        "    checkpoint: /tmp/refiner.ckpt\n"
    )

    traj_cfg = _traj_cfg(str(config))

    assert traj_cfg["history_length"] == 30
    assert traj_cfg["root_refiner"]["enabled"] is False
    assert traj_cfg["root_refiner"]["checkpoint"] == "/tmp/refiner.ckpt"
