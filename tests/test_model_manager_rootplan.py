from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace

import torch
import numpy as np

from utils.inference_glue import InferenceGlueState, InferenceGlueTimeline
from utils.root_plan import RootPlan
from utils.token_frame import token_range_to_frame_slice, token_start_frame
from utils.traj_stream_buffer import TrajStreamBuffer
from utils.stream_traj import StreamTrajectoryPlan, TrajectoryUpdateEvent
from web_demo.model_manager import ModelManager


class _DummyModel:
    def __init__(self):
        self.commit_index = 10
        self.chunk_size = 5
        self._traj_buf = TrajStreamBuffer(device="cpu", dtype=torch.float32)


def _state(commit_idx: int):
    return InferenceGlueState(
        commit_idx=commit_idx,
        world_xz=torch.zeros(2),
        world_yaw=torch.tensor(0.0),
    )


def _timeline(up_to: int):
    timeline = InferenceGlueTimeline(_state(0))
    for idx in range(1, up_to + 1):
        timeline.append(_state(idx))
    return timeline


def _plan(valid_frames=200, *, source="test", anchor_commit_idx=0):
    wp = torch.zeros(valid_frames, 7)
    wp[:, 0] = torch.arange(valid_frames, dtype=torch.float32)
    wp[:, 3] = 1.0
    return RootPlan(
        num_tokens_pred=30,
        valid_frames=valid_frames,
        waypoints_local_7d=wp,
        frame_dt=0.05,
        frames_per_token=4,
        anchor_commit_idx=anchor_commit_idx,
        anchor_world_xz=torch.zeros(2),
        anchor_world_yaw=torch.tensor(0.0),
        source=source,
    )


class _FakeRootRefinerRuntime:
    def __init__(self):
        self.calls = []

    def build_root_plan(
        self,
        *,
        text,
        plan,
        anchor_state,
        token_dt,
        history_motion_world_5d=None,
    ):
        self.calls.append(
            {
                "text": text,
                "plan": plan,
                "anchor_state": anchor_state,
                "token_dt": token_dt,
                "history_motion_world_5d": history_motion_world_5d,
            }
        )
        return _plan(
            valid_frames=120,
            source="root_refiner",
            anchor_commit_idx=anchor_state.commit_idx,
        )


def _manager():
    mgr = ModelManager.__new__(ModelManager)
    mgr.model = _DummyModel()
    mgr.history_length = 9
    mgr.traj_horizon_tokens = 20
    mgr.token_dt = 0.20
    mgr._glue_timeline = _timeline(10)
    mgr.stream_recovery = SimpleNamespace(r_pos_accum=np.zeros(3, dtype=np.float32))
    mgr.root_5d_history = deque()
    mgr.root_refiner_runtime = None
    mgr.model._traj_buf.set_root_plan(_plan())
    return mgr


def test_rootplan_stream_payload_uses_body_window_left_commit():
    mgr = _manager()

    payload = mgr._build_rootplan_stream_traj_input()

    start_token = 6  # commit 10 + chunk 5, history_length 9 -> left edge 6
    num_tokens = 29
    frame_slice = token_range_to_frame_slice(start_token, num_tokens)
    assert payload["traj_start_token"] == start_token
    assert payload["traj_abs_start_token"] == start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == start_token
    assert payload["body_anchor_abs_token"] == start_token
    assert payload["traj_cond_7d_frame"].shape == (1, frame_slice.stop - frame_slice.start, 7)
    assert payload["traj_cond_frame_mask"].shape == (1, frame_slice.stop - frame_slice.start)
    assert payload["traj_cond_frame_mask"].all()
    assert float(payload["traj_cond_7d_frame"][0, 0, 0]) == float(token_start_frame(start_token))


def test_rootplan_stream_payload_uses_absolute_commit_after_model_roll():
    mgr = _manager()
    mgr.history_length = 9
    mgr.model.commit_index = 30
    mgr._absolute_commit_index = 60
    mgr._glue_timeline = _timeline(100)
    mgr.model._traj_buf.set_root_plan(_plan(valid_frames=400))

    payload = mgr._build_rootplan_stream_traj_input()

    local_start_token = 26
    absolute_start_token = 56
    num_tokens = 29
    frame_slice = token_range_to_frame_slice(absolute_start_token, num_tokens)
    assert payload["traj_start_token"] == local_start_token
    assert payload["traj_abs_start_token"] == absolute_start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == local_start_token
    assert payload["body_anchor_abs_token"] == absolute_start_token
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert float(payload["traj_cond_7d_frame"][0, 0, 0]) == float(
        token_start_frame(absolute_start_token)
    )


def test_rootplan_stream_payload_frame_slice_uses_absolute_start_after_roll_to_local_zero():
    mgr = _manager()
    mgr.history_length = 9
    mgr.model.commit_index = 1
    mgr.model.chunk_size = 1
    mgr._absolute_commit_index = 31
    mgr._glue_timeline = _timeline(100)
    mgr.model._traj_buf.set_root_plan(_plan(valid_frames=400))

    payload = mgr._build_rootplan_stream_traj_input()

    local_start_token = 0
    absolute_start_token = 30
    num_tokens = 2 + mgr.traj_horizon_tokens
    frame_slice = token_range_to_frame_slice(absolute_start_token, num_tokens)
    assert payload["traj_start_token"] == local_start_token
    assert payload["traj_abs_start_token"] == absolute_start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == local_start_token
    assert payload["body_anchor_abs_token"] == absolute_start_token
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert float(payload["traj_cond_7d_frame"][0, 0, 0]) == float(
        token_start_frame(absolute_start_token)
    )


def test_rootplan_stream_payload_requires_exact_body_anchor_state():
    mgr = _manager()
    mgr._glue_timeline.trim_before(8)

    payload = mgr._build_rootplan_stream_traj_input()

    assert payload is None


def test_build_stream_traj_input_prefers_active_rootplan_payload():
    mgr = _manager()
    mgr.traj_state_lock = threading.Lock()
    mgr.active_traj_plan = None
    mgr.pending_update_event = None

    payload = mgr._build_stream_traj_input()

    assert payload is not None
    assert "traj_cond_7d_frame" in payload
    assert "traj" not in payload
    assert payload["traj_start_token"] == 6
    assert payload["body_anchor_token"] == 6
    assert mgr._trajectory_state == "active_7d"


def test_build_stream_traj_input_does_not_fallback_to_legacy_xyz_when_rootplan_unavailable():
    mgr = _manager()
    mgr.traj_state_lock = threading.Lock()
    mgr._display_traj_lock = threading.Lock()
    mgr.active_traj_plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=0,
        version=1,
        source="test",
    )
    mgr.pending_update_event = None
    mgr.current_traj_mode = "replace_future"
    mgr._display_traj = None
    mgr._glue_timeline.trim_before(8)

    payload = mgr._build_stream_traj_input()

    assert payload is None
    assert mgr._trajectory_state == "active_7d_unavailable"


def test_pending_update_delay_uses_existing_rootplan_payload_not_legacy_xyz():
    mgr = _manager()
    mgr.traj_state_lock = threading.Lock()
    mgr._display_traj_lock = threading.Lock()
    mgr.current_traj_mode = "replace_future"
    mgr._display_traj = None
    old_plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=0,
        version=1,
        source="old",
    )
    new_plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        start_commit_index=30,
        version=2,
        source="new",
    )
    mgr.active_traj_plan = old_plan
    mgr.pending_update_event = TrajectoryUpdateEvent(
        old_plan=old_plan,
        new_plan=new_plan,
        edit_commit_index=10,
        effective_commit_index=30,
        delay_tokens=20,
        blend_tokens=4,
        version=2,
    )

    payload = mgr._build_stream_traj_input()

    assert payload is not None
    assert "traj_cond_7d_frame" in payload
    assert "traj" not in payload
    assert payload["trajectory_state"] == "delay"


def test_pending_update_replacement_activates_new_rootplan_payload_not_legacy_xyz():
    mgr = _manager()
    mgr.traj_state_lock = threading.Lock()
    mgr._display_traj_lock = threading.Lock()
    mgr.current_traj_mode = "replace_future"
    mgr._display_traj = None
    old_plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=0,
        version=1,
        source="old",
    )
    new_plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        start_commit_index=0,
        version=2,
        source="new",
    )
    mgr.active_traj_plan = old_plan
    mgr.pending_update_event = TrajectoryUpdateEvent(
        old_plan=old_plan,
        new_plan=new_plan,
        edit_commit_index=0,
        effective_commit_index=0,
        delay_tokens=0,
        blend_tokens=0,
        version=2,
    )

    payload = mgr._build_stream_traj_input()

    assert payload is not None
    assert "traj_cond_7d_frame" in payload
    assert "traj" not in payload
    assert payload["trajectory_state"] == "replaced"
    assert mgr.active_traj_plan is new_plan
    assert mgr.pending_update_event is None


def test_reset_glue_timeline_initializes_commit_zero_state():
    mgr = ModelManager.__new__(ModelManager)
    mgr.device = "cpu"

    mgr._reset_glue_timeline()

    assert mgr._glue_timeline.head.commit_idx == 0
    assert torch.allclose(mgr._glue_timeline.head.world_xz, torch.zeros(2))
    assert torch.allclose(mgr._glue_timeline.head.world_yaw, torch.tensor(0.0))


def test_append_glue_state_from_stream_recovery_records_commit_pose_once():
    mgr = ModelManager.__new__(ModelManager)
    mgr.device = "cpu"
    mgr.model = _DummyModel()
    mgr.model.commit_index = 0
    mgr.model.commit_index = 7
    mgr.stream_recovery = SimpleNamespace(
        r_pos_accum=np.array([1.0, 0.0, 2.0], dtype=np.float32),
        r_rot_ang_accum=0.25,
    )
    mgr._reset_glue_timeline()

    mgr._append_glue_state_from_stream_recovery()
    mgr._append_glue_state_from_stream_recovery()

    assert len(mgr._glue_timeline) == 2
    assert mgr._glue_timeline.head.commit_idx == 7
    assert torch.allclose(mgr._glue_timeline.head.world_xz, torch.tensor([1.0, 2.0]))
    assert torch.allclose(mgr._glue_timeline.head.world_yaw, torch.tensor(-0.5))


def test_stream_plan_to_root_plan_builds_plan_anchor_local_7d():
    mgr = ModelManager.__new__(ModelManager)
    mgr.device = "cpu"
    mgr.token_dt = 0.20
    mgr.traj_horizon_tokens = 20
    mgr.history_length = 9
    mgr.model = _DummyModel()
    plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=0,
        version=1,
        source="test",
    )

    root_plan = mgr._stream_plan_to_root_plan(plan, _state(0))

    assert root_plan.anchor_commit_idx == 0
    assert root_plan.waypoints_local_7d.shape[1] == 7
    assert torch.allclose(root_plan.waypoints_local_7d[0, :3], torch.zeros(3))
    assert torch.allclose(root_plan.waypoints_local_7d[1, 3:5], torch.tensor([1.0, 0.0]))
    assert root_plan.waypoints_local_7d[1, 5] > 0.0
    assert torch.allclose(root_plan.waypoints_local_7d[1, 6], torch.tensor(0.0))


def test_activate_root_plan_from_stream_plan_sets_traj_buffer():
    mgr = ModelManager.__new__(ModelManager)
    mgr.device = "cpu"
    mgr.token_dt = 0.20
    mgr.traj_horizon_tokens = 20
    mgr.history_length = 9
    mgr.model = _DummyModel()
    mgr.root_refiner_runtime = None
    mgr._glue_timeline = _timeline(10)
    plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=0,
        version=1,
        source="test",
    )

    ok = mgr._activate_root_plan_from_stream_plan(plan)

    assert ok is True
    assert mgr.model._traj_buf.has_active_plan()


def test_activate_root_plan_from_stream_plan_uses_root_refiner_runtime():
    mgr = ModelManager.__new__(ModelManager)
    mgr.device = "cpu"
    mgr.token_dt = 0.20
    mgr.traj_horizon_tokens = 20
    mgr.history_length = 9
    mgr.model = _DummyModel()
    mgr.current_text = "walk along the route"
    mgr._glue_timeline = _timeline(10)
    refiner = _FakeRootRefinerRuntime()
    mgr.root_refiner_runtime = refiner
    plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=0,
        version=1,
        source="manual",
    )

    ok = mgr._activate_root_plan_from_stream_plan(plan)

    assert ok is True
    assert len(refiner.calls) == 1
    assert refiner.calls[0]["text"] == "walk along the route"
    assert refiner.calls[0]["plan"] is plan
    assert refiner.calls[0]["anchor_state"].commit_idx == 0
    assert refiner.calls[0]["token_dt"] == 0.20
    assert mgr.model._traj_buf._active_plan.source == "root_refiner"


def test_activate_root_plan_reanchors_plan_points_to_anchor_state_for_refiner():
    mgr = ModelManager.__new__(ModelManager)
    mgr.device = "cpu"
    mgr.token_dt = 0.20
    mgr.traj_horizon_tokens = 20
    mgr.history_length = 9
    mgr.model = _DummyModel()
    mgr.current_text = "turn right"
    timeline = InferenceGlueTimeline(_state(0))
    timeline.append(
        InferenceGlueState(
            commit_idx=5,
            world_xz=torch.tensor([2.0, 3.0]),
            world_yaw=torch.tensor(0.0),
        )
    )
    mgr._glue_timeline = timeline
    refiner = _FakeRootRefinerRuntime()
    mgr.root_refiner_runtime = refiner
    plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=5,
        version=2,
        source="manual",
    )

    ok = mgr._activate_root_plan_from_stream_plan(plan)

    assert ok is True
    refiner_plan = refiner.calls[0]["plan"]
    assert refiner_plan is not plan
    assert np.allclose(refiner_plan.points_xyz[0, [0, 2]], [2.0, 3.0])
    assert np.allclose(refiner_plan.points_xyz[1, [0, 2]], [2.0, 4.0])


def test_activate_root_plan_passes_anchor_aligned_history_to_root_refiner_runtime():
    mgr = ModelManager.__new__(ModelManager)
    mgr.device = "cpu"
    mgr.token_dt = 0.20
    mgr.traj_horizon_tokens = 20
    mgr.history_length = 9
    mgr.model = _DummyModel()
    mgr.current_text = "walk along the route"
    mgr._glue_timeline = _timeline(10)
    mgr.root_5d_history = deque(
        [
            (0, np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
            (1, np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)),
            (5, np.array([0.0, 0.0, 5.0, 1.0, 0.0], dtype=np.float32)),
        ]
    )
    refiner = _FakeRootRefinerRuntime()
    mgr.root_refiner_runtime = refiner
    plan = StreamTrajectoryPlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=1,
        version=1,
        source="manual",
    )

    ok = mgr._activate_root_plan_from_stream_plan(plan)

    assert ok is True
    history = refiner.calls[0]["history_motion_world_5d"]
    assert history.shape == (2, 5)
    assert np.allclose(history[:, 2], np.array([0.0, 1.0], dtype=np.float32))


def test_update_trajectory_clear_clears_active_rootplan():
    mgr = _manager()
    mgr.traj_state_lock = threading.Lock()
    mgr._display_traj_lock = threading.Lock()
    mgr.active_traj_plan = None
    mgr.pending_update_event = None
    mgr.current_traj_waypoints = None
    mgr.current_traj_times = None
    mgr.current_traj_mode = "replace_future"
    mgr._display_traj = None

    mgr.update_trajectory(None)

    assert not mgr.model._traj_buf.has_active_plan()
    assert mgr._trajectory_state == "none"


def test_initial_update_trajectory_activates_rootplan_immediately():
    mgr = ModelManager.__new__(ModelManager)
    mgr.device = "cpu"
    mgr.model = _DummyModel()
    mgr.model.commit_index = 0
    mgr.history_length = 9
    mgr.traj_horizon_tokens = 20
    mgr.token_dt = 0.20
    mgr.waypoint_dt = 0.05
    mgr.manual_duration_seconds = 1.0
    mgr.manual_resample_arclength = False
    mgr.traj_update_delay_tokens = 20
    mgr.traj_update_blend_tokens = 4
    mgr.traj_state_lock = threading.Lock()
    mgr._display_traj_lock = threading.Lock()
    mgr._display_traj = None
    mgr._trajectory_state = "none"
    mgr.active_traj_plan = None
    mgr.pending_update_event = None
    mgr.current_traj_waypoints = None
    mgr.current_traj_times = None
    mgr.current_traj_mode = "replace_future"
    mgr._plan_version_counter = 0
    mgr.stream_recovery = SimpleNamespace(r_pos_accum=np.zeros(3, dtype=np.float32))
    mgr._glue_timeline = _timeline(0)
    refiner = _FakeRootRefinerRuntime()
    mgr.root_refiner_runtime = refiner
    mgr.current_text = "turn right"

    mgr.update_trajectory(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))

    assert mgr.active_traj_plan is not None
    assert mgr.pending_update_event is None
    assert mgr.model._traj_buf.has_active_plan()
    assert len(refiner.calls) == 1
    assert refiner.calls[0]["text"] == "turn right"
    assert mgr.model._traj_buf._active_plan.source == "root_refiner"
