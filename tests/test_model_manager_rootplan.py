from __future__ import annotations

import numpy as np
import torch
from torch import nn

from types import SimpleNamespace
import threading

from utils.inference.root_plan import RootPlan
from utils.inference.route_condition import RoutePlan
from utils.inference.stream_generator import StreamGenerator
from utils.inference.timeline import RootFrameState, RootTimeline
from utils.token_frame import token_range_to_frame_slice, token_start_frame
from web_demo.model_manager import ModelManager


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.commit_index = 10
        self.chunk_size = 5


def _state(commit_idx: int, xz=(0.0, 0.0)):
    return RootFrameState(
        commit_idx=commit_idx,
        world_xz=torch.tensor(xz, dtype=torch.float32),
        world_yaw=torch.tensor(0.0),
    )


def _timeline(up_to: int):
    timeline = RootTimeline(_state(0))
    for idx in range(1, up_to + 1):
        timeline.append(_state(idx))
    return timeline


def _plan(valid_frames=200, *, source="test", anchor_commit_idx=0):
    waypoints = torch.zeros(valid_frames, 7)
    waypoints[:, 0] = torch.arange(valid_frames, dtype=torch.float32)
    waypoints[:, 3] = 1.0
    return RootPlan(
        num_tokens_pred=30,
        valid_frames=valid_frames,
        waypoints_local_7d=waypoints,
        frame_dt=0.05,
        frames_per_token=4,
        anchor_commit_idx=anchor_commit_idx,
        anchor_world_xz=torch.zeros(2),
        anchor_world_yaw=torch.tensor(0.0),
        source=source,
    )


def _manager():
    mgr = ModelManager.__new__(ModelManager)
    mgr.device = "cpu"
    mgr.model = _DummyModel()
    mgr.history_length = 9
    mgr.traj_horizon_tokens = 20
    mgr.token_dt = 0.20
    mgr._root_timeline = _timeline(10)
    mgr.stream_generator = StreamGenerator(
        ldf_model=mgr.model,
        device="cpu",
        history_length=mgr.history_length,
        traj_horizon_tokens=mgr.traj_horizon_tokens,
        token_dt=mgr.token_dt,
    )
    mgr.stream_generator.timeline = mgr._root_timeline
    mgr.stream_generator.active_root_plan = _plan()
    mgr.stream_recovery = SimpleNamespace(r_pos_accum=np.zeros(3, dtype=np.float32))
    return mgr


def _trajectory_manager():
    mgr = _manager()
    mgr.traj_state_lock = threading.Lock()
    mgr.active_traj_plan = None
    mgr.pending_update_event = None
    mgr._trajectory_state = "none"
    mgr._plan_version_counter = 0
    mgr.current_traj_mode = "replace_future"
    mgr.current_traj_waypoints = None
    mgr.current_traj_times = None
    mgr.traj_update_delay_tokens = 2
    mgr.traj_update_blend_tokens = 3
    mgr.manual_duration_seconds = 1.0
    mgr.waypoint_dt = 0.2
    mgr.manual_resample_arclength = False
    mgr._display_traj_lock = threading.Lock()
    mgr._display_traj = None
    mgr._absolute_commit_index = 0
    mgr.route_reference_mode = "relative_to_actor"
    mgr.stream_generator.root_refiner = None
    return mgr


def test_rootplan_stream_payload_uses_body_window_left_commit():
    mgr = _manager()

    payload = mgr._build_rootplan_stream_traj_input()

    start_token = 2
    num_tokens = 33
    frame_slice = token_range_to_frame_slice(start_token, num_tokens)
    assert payload["traj_start_token"] == start_token
    assert payload["traj_abs_start_token"] == start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == start_token
    assert payload["body_anchor_abs_token"] == start_token
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert payload["traj_cond_frame_mask"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
    )
    assert payload["traj_cond_frame_mask"].all()
    assert float(payload["traj_cond_7d_frame"][0, 0, 0]) == float(
        token_start_frame(start_token)
    )


def test_activate_root_plan_from_route_sets_stream_generator_active_plan():
    mgr = _manager()
    mgr.model.commit_index = 0
    mgr.current_text = "turn right"
    mgr._root_timeline = _timeline(0)
    mgr.stream_generator.timeline = mgr._root_timeline
    mgr.stream_generator.root_refiner = None

    route = RoutePlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=0,
        version=1,
        source="manual",
    )

    ok = mgr._activate_root_plan_from_stream_plan(route)

    assert ok is True
    assert mgr.stream_generator.active_root_plan is not None
    assert mgr.stream_generator.active_root_plan.source == "manual"


def test_update_trajectory_second_edit_uses_route_update_contract():
    mgr = _trajectory_manager()
    first = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    second = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    mgr.update_trajectory(first, source="manual", route_mode="relative_to_actor")
    mgr.update_trajectory(second, source="manual", route_mode="relative_to_actor")

    assert mgr.pending_update_event is not None
    assert mgr.pending_update_event.old_route is not None
    assert mgr.pending_update_event.new_route is not None
    assert mgr.stream_generator.condition_manager.route.mode.value == "relative_to_actor"


def test_update_trajectory_sets_absolute_route_mode_without_reanchoring():
    mgr = _trajectory_manager()
    route = np.array([[10.0, 0.0], [10.0, 2.0]], dtype=np.float32)

    mgr.update_trajectory(route, source="manual", route_mode="absolute")

    assert mgr.stream_generator.condition_manager.route.mode.value == "absolute"
    assert mgr.active_traj_plan is not None
    np.testing.assert_allclose(
        mgr.active_traj_plan.points_xyz[:, [0, 2]],
        route,
        atol=1e-6,
    )
