from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from utils.inference.stream_runtime import (
    RootSourceProposal,
    SetRootSource,
    SetText,
    SpaceContract,
    StreamCommitEvent,
)
from utils.inference.timeline import RootFrameState, RootTimeline
from web_demo.model_manager import ModelManager
from web_demo.runtime.model_bundle import ModelBundle
from web_demo.runtime.model_loader import build_runtime_session


class _FakeGenerator:
    def __init__(self):
        self.ldf_model = SimpleNamespace(
            cfg_scale_text=1.25,
            cfg_scale_traj=3.0,
            chunk_size=1,
        )
        self.device = torch.device("cpu")
        self.timeline = RootTimeline(RootFrameState.initial())
        self.history_length = 30
        self.traj_horizon_tokens = 20
        self.attached_session = None

    def attach_runtime_session(self, session):
        self.attached_session = session


class _FakeVae:
    pass


def _proposal() -> RootSourceProposal:
    future = torch.zeros(8, 7)
    future[:, 2] = torch.arange(1, 9, dtype=torch.float32) * 0.1
    future[:, 3] = 1.0
    return RootSourceProposal(
        future_traj7=future,
        future_frame_mask=torch.ones(8, dtype=torch.bool),
        source_id="manual:1",
        version=1,
        metadata={"source_kind": "manual"},
    )


def test_model_bundle_requires_one_authoritative_session():
    generator = _FakeGenerator()
    vae = _FakeVae()
    session = build_runtime_session(
        generator,
        vae,
        traj_mask_cfg={"horizon_tokens": 20},
    )

    bundle = ModelBundle(
        vae=vae,
        ldf_model=generator.ldf_model,
        cfg={},
        device="cpu",
        stream_generator=generator,
        runtime_session=session,
    )

    assert bundle.runtime_session is session
    assert session.kernel is generator
    assert session.vae is vae
    assert generator.attached_session is session


def test_text_and_route_updates_only_submit_commands():
    generator = _FakeGenerator()
    session = build_runtime_session(generator, _FakeVae(), traj_mask_cfg={})
    manager = ModelManager.__new__(ModelManager)
    manager.runtime_session = session
    manager.current_text = "walk"
    manager._runtime_command_version = 0

    active_before = session.source_manager.active
    manager.update_text("turn left")
    manager._submit_root_source(
        _proposal(),
        space_contract=SpaceContract.WORLD_ROUTE,
        requested_commit_abs=0,
    )

    commands = session.command_queue.snapshot()
    assert session.source_manager.active is active_before
    assert [type(command) for command in commands] == [SetText, SetRootSource]
    assert commands[0].text == "turn left"
    assert commands[1].proposal.source_id == "manual:1"


class _FrameBuffer:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def size(self):
        return len(self.frames)


def _commit_event() -> StreamCommitEvent:
    roots = torch.zeros(4, 7)
    roots[:, 3] = 1.0
    state = RootFrameState(
        commit_idx=2,
        world_xz=torch.zeros(2),
        world_yaw=torch.tensor(0.0),
    )
    return StreamCommitEvent(
        absolute_commit_before=1,
        absolute_commit_after=2,
        local_commit_before=1,
        local_commit_after=2,
        latent_buffer_start_commit_abs=0,
        latent_buffer_epoch=0,
        committed_latent=torch.zeros(1, 2),
        decoded_chunk=torch.zeros(4, 263),
        joint_frames=torch.arange(4 * 22 * 3, dtype=torch.float32).reshape(4, 22, 3),
        root_frames_start_abs=1,
        root_frames=roots,
        timeline_state=state,
        actual_payload={"payload": True},
        source_id=None,
        source_version=None,
        actual_activation_commit=None,
        lifecycle_events=(),
    )


def test_generate_once_only_consumes_authoritative_session_event():
    manager = ModelManager.__new__(ModelManager)
    event = _commit_event()
    timeline = RootTimeline(RootFrameState.initial())
    timeline.append(
        RootFrameState(
            commit_idx=1,
            world_xz=torch.zeros(2),
            world_yaw=torch.tensor(0.0),
        )
    )
    timeline.append(event.timeline_state)
    manager.runtime_session = SimpleNamespace(
        step=Mock(return_value=event),
        timeline=timeline,
        session_anchor_state=timeline.earliest,
        recovery=object(),
        first_chunk=False,
        generated_history=SimpleNamespace(next_frame_abs=5),
    )
    manager.frame_buffer = _FrameBuffer()
    manager.root_xz_history = []
    manager.root_5d_history = []

    event = manager._generate_once()

    manager.runtime_session.step.assert_called_once_with()
    assert event.absolute_commit_after == 2
    assert manager.frame_buffer.size() == 4
    assert torch.equal(
        torch.as_tensor(manager.frame_buffer.frames[-1]),
        event.joint_frames[-1],
    )


def test_web_manager_has_no_legacy_payload_execution_api():
    assert not hasattr(ModelManager, "_build_stream_traj_input")
    assert not hasattr(ModelManager, "_build_rootplan_stream_traj_input")
    assert not hasattr(ModelManager, "_build_temporary_rootplan_payload")
