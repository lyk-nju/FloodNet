from __future__ import annotations

import threading
import time
from dataclasses import replace
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
from web_demo.runtime.contracts import TrajectoryRuntimeControls
from web_demo.runtime.trajectory_controller import TrajectoryController
from web_demo.runtime.trajectory_diagnostics import TrajectoryDiagnosticsStore
from web_demo.runtime.state import GenerationState


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


def test_web_runtime_session_defaults_root_feedback_to_alpha_one():
    generator = _FakeGenerator()
    session = build_runtime_session(generator, _FakeVae(), traj_mask_cfg={})

    assert session.config.root_feedback_enabled is True
    assert session.config.root_feedback_xz_blend_alpha == 1.0


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
        self.atomic_batches = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_frames_atomic(self, frames):
        batch = list(frames)
        self.frames.extend(batch)
        self.atomic_batches += 1

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
    manager.trajectory_diagnostics = TrajectoryDiagnosticsStore()
    manager.runtime_session.source_manager = SimpleNamespace(active=None)

    event = manager._generate_once()

    manager.runtime_session.step.assert_called_once_with()
    assert event.absolute_commit_after == 2
    assert manager.frame_buffer.size() == 4
    assert manager.frame_buffer.atomic_batches == 1
    assert torch.equal(
        torch.as_tensor(manager.frame_buffer.frames[-1]),
        event.joint_frames[-1],
    )
    diagnostics = manager.get_trajectory_debug()
    assert diagnostics["current"]["payload_commit"] == 1


def test_generate_once_keeps_running_when_diagnostic_extraction_fails():
    manager = ModelManager.__new__(ModelManager)
    event = replace(_commit_event(), actual_payload={"traj_cond_7d_frame": "bad"})
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
        source_manager=SimpleNamespace(active=None),
        session_anchor_state=timeline.earliest,
        recovery=object(),
        first_chunk=False,
        generated_history=SimpleNamespace(next_frame_abs=5),
    )
    manager.frame_buffer = _FrameBuffer()
    manager.root_xz_history = []
    manager.root_5d_history = []
    manager.trajectory_diagnostics = TrajectoryDiagnosticsStore()

    committed = manager._generate_once()

    assert committed is event
    assert manager.get_trajectory_debug()["last_error"]


def test_generation_loop_stops_after_transaction_failure_instead_of_retrying():
    manager = ModelManager.__new__(ModelManager)
    manager.frame_buffer = SimpleNamespace(needs_generation=lambda: True)
    calls = []

    def fail_once():
        calls.append(1)
        raise RuntimeError("generation failed")

    manager._generate_once = fail_once
    manager.generation_state = GenerationState.RUNNING
    manager.is_generating = True

    manager._generation_loop(threading.Event())

    assert len(calls) == 1
    assert manager.generation_state is GenerationState.ERROR
    assert manager.is_generating is False
    assert manager._last_generation_error == "generation failed"


def test_web_runtime_command_version_and_submit_are_serialized():
    manager = ModelManager.__new__(ModelManager)
    manager._runtime_command_version = 0
    manager._runtime_command_lock = threading.RLock()
    manager.runtime_session = SimpleNamespace(
        timeline=RootTimeline(RootFrameState.initial()),
    )
    active_submits = 0
    overlap = False
    submitted = []
    submit_lock = threading.Lock()

    def submit(command):
        nonlocal active_submits, overlap
        with submit_lock:
            active_submits += 1
            overlap |= active_submits > 1
        time.sleep(0.005)
        submitted.append(command)
        with submit_lock:
            active_submits -= 1

    manager.runtime_session.submit = submit
    threads = [
        threading.Thread(
            target=lambda index=index: manager._submit_runtime_command(
                SetText,
                text=f"text-{index}",
            )
        )
        for index in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert overlap is False
    assert sorted(command.version for command in submitted) == list(range(1, 13))


def test_web_trajectory_updates_are_serialized_as_one_route_transaction():
    manager = ModelManager.__new__(ModelManager)
    manager.trajectory_controller = TrajectoryController(
        TrajectoryRuntimeControls(
            route_mode="relative_to_actor",
            horizon_tokens=20,
            delay_enabled=True,
            delay_tokens=5,
            blend_enabled=False,
            blend_tokens=0,
        )
    )
    active = 0
    overlap = False
    guard = threading.Lock()

    def update(*args, **kwargs):
        nonlocal active, overlap
        with guard:
            active += 1
            overlap |= active > 1
        time.sleep(0.01)
        with guard:
            active -= 1

    manager._update_trajectory_locked = update
    workers = [
        threading.Thread(target=manager.update_trajectory, args=([[0.0, 0.0]],))
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2.0)

    assert overlap is False


def test_web_pending_route_promotes_on_route_active_event():
    manager = ModelManager.__new__(ModelManager)
    event = replace(
        _commit_event(),
        source_id="manual:2",
        source_version=2,
        actual_activation_commit=1,
        lifecycle_events=("route_active",),
    )
    timeline = RootTimeline(RootFrameState.initial())
    timeline.append(event.timeline_state)
    manager.runtime_session = SimpleNamespace(
        step=Mock(return_value=event),
        timeline=timeline,
        session_anchor_state=timeline.earliest,
        recovery=object(),
        first_chunk=False,
        generated_history=SimpleNamespace(next_frame_abs=1),
    )
    manager.frame_buffer = _FrameBuffer()
    manager.root_xz_history = []
    manager.root_5d_history = []
    manager.trajectory_controller = TrajectoryController(
        TrajectoryRuntimeControls(
            route_mode="relative_to_actor",
            horizon_tokens=20,
            delay_enabled=True,
            delay_tokens=5,
            blend_enabled=False,
            blend_tokens=0,
        )
    )
    old_route = SimpleNamespace(version=1)
    new_route = SimpleNamespace(version=2)
    manager.trajectory_controller.active_route = old_route
    manager.trajectory_controller.pending_update = SimpleNamespace(
        version=2,
        new_route=new_route,
        effective_commit_index=1,
    )
    manager.trajectory_controller.state = "pending"

    manager._generate_once()

    assert manager.trajectory_controller.active_route is new_route
    assert manager.trajectory_controller.pending_update is None
    assert manager.trajectory_controller.state == "active_7d"
    assert manager._active_source_version == 2
    assert manager._actual_activation_commit == 1


def test_committed_route_clear_clears_compatibility_route_state():
    manager = ModelManager.__new__(ModelManager)
    route_state = SimpleNamespace(route=object(), pending_update=object())

    def clear():
        route_state.route = None
        route_state.pending_update = None

    route_state.clear = clear
    manager.stream_generator = SimpleNamespace(
        condition_manager=SimpleNamespace(route=route_state)
    )
    manager.trajectory_controller = TrajectoryController(
        TrajectoryRuntimeControls(
            route_mode="relative_to_actor",
            horizon_tokens=20,
            delay_enabled=True,
            delay_tokens=5,
            blend_enabled=False,
            blend_tokens=0,
        )
    )
    event = replace(_commit_event(), lifecycle_events=("route_cleared",))

    manager._reconcile_runtime_route_event(event)

    assert route_state.route is None
    assert route_state.pending_update is None


def test_debug_repeat_rotates_reference_and_submits_at_route_exhaustion():
    manager = ModelManager.__new__(ModelManager)
    manager.configure_debug_repeat(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
        {
            "enabled": True,
            "angle_min_degrees": 30.0,
            "angle_max_degrees": 30.0,
            "random_sign": False,
            "seed": 7,
        },
        duration_seconds=2.0,
    )
    submitted = []
    manager.update_trajectory = lambda points, **kwargs: submitted.append(
        (torch.as_tensor(points), kwargs)
    )
    manager.trajectory_controller = TrajectoryController(
        TrajectoryRuntimeControls(
            route_mode="relative_to_actor",
            horizon_tokens=20,
            delay_enabled=True,
            delay_tokens=5,
            blend_enabled=False,
            blend_tokens=0,
        )
    )
    event = replace(
        _commit_event(),
        source_id="debug_preset:1",
        lifecycle_events=("route_exhausted",),
    )

    manager._reconcile_runtime_route_event(event)

    assert len(submitted) == 1
    points, kwargs = submitted[0]
    expected_xz = torch.tensor([[0.0, 0.0], [1.0, 3.0**0.5]])
    assert torch.allclose(points[:, [0, 2]], expected_xz, atol=1e-5)
    assert kwargs["route_mode"] == "relative_to_actor"
    assert kwargs["delay_enabled"] is False
    assert kwargs["delay_tokens"] == 0
    assert kwargs["source"] == "debug_repeat_1"


def test_debug_repeat_uses_incremental_rotation_and_respects_limit():
    manager = ModelManager.__new__(ModelManager)
    manager.configure_debug_repeat(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        {
            "enabled": True,
            "angle_min_degrees": 20.0,
            "angle_max_degrees": 20.0,
            "random_sign": False,
            "seed": 1,
            "max_repeats": 2,
        },
        duration_seconds=1.0,
    )
    submitted = []
    manager.update_trajectory = lambda points, **kwargs: submitted.append(
        torch.as_tensor(points)
    )

    assert manager._submit_next_debug_repeat() is True
    assert manager._submit_next_debug_repeat() is True
    assert manager._submit_next_debug_repeat() is False
    first_angle = torch.atan2(submitted[0][-1, 0], submitted[0][-1, 2])
    second_angle = torch.atan2(submitted[1][-1, 0], submitted[1][-1, 2])
    assert torch.allclose(first_angle, torch.deg2rad(torch.tensor(20.0)), atol=1e-5)
    assert torch.allclose(second_angle, torch.deg2rad(torch.tensor(40.0)), atol=1e-5)


def test_debug_repeat_prefetches_before_terminal_commit_without_control_gap():
    manager = ModelManager.__new__(ModelManager)
    manager.configure_debug_repeat(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        {
            "enabled": True,
            "angle_min_degrees": 20.0,
            "angle_max_degrees": 20.0,
            "random_sign": False,
            "prefetch_tokens": 5,
        },
        duration_seconds=1.0,
    )
    submitted = []
    manager.update_trajectory = lambda points, **kwargs: submitted.append(kwargs)
    manager._get_commit_index = lambda: 2
    proposal = SimpleNamespace(
        version=2,
        future_frame_mask=torch.ones(17, dtype=torch.bool),
    )
    manager.runtime_session = SimpleNamespace(
        source_manager=SimpleNamespace(
            active=SimpleNamespace(
                proposal=proposal,
                first_future_frame_abs=1,
            )
        )
    )
    event = replace(
        _commit_event(),
        absolute_commit_after=2,
        source_id="debug_preset:2",
        source_version=2,
        lifecycle_events=(),
    )

    assert manager._maybe_prefetch_debug_repeat(event) is True
    assert submitted[0]["delay_enabled"] is True
    assert submitted[0]["delay_tokens"] == 4
    assert manager._debug_repeat_scheduled_from_version == 2
    assert manager._maybe_prefetch_debug_repeat(event) is False


def test_web_manager_has_no_legacy_payload_execution_api():
    assert not hasattr(ModelManager, "_build_stream_traj_input")
    assert not hasattr(ModelManager, "_build_rootplan_stream_traj_input")
    assert not hasattr(ModelManager, "_build_temporary_rootplan_payload")
    assert "use_owned_stream_execution" not in ModelManager.__init__.__code__.co_names
