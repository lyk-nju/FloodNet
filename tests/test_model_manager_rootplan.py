from __future__ import annotations

import importlib
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
from web_demo.runtime.model_bundle import ModelBundle


def test_web_demo_layered_runtime_import_contract():
    import web_demo.app
    from web_demo.model_manager import ModelManager, get_model_manager
    from web_demo.runtime.state import GenerationState

    assert web_demo.app is not None
    assert ModelManager is not None
    assert get_model_manager is not None
    assert GenerationState.IDLE.value == "idle"
    route_rules = {str(rule) for rule in web_demo.app.app.url_map.iter_rules()}
    assert {
        "/",
        "/api/start",
        "/api/update_text",
        "/api/update_trajectory",
        "/api/pause",
        "/api/resume",
        "/api/reset",
        "/api/get_frame",
        "/api/status",
    }.issubset(route_rules)
    for module_name in (
        "web_demo.config",
        "web_demo.bootstrap",
        "web_demo.api.routes",
        "web_demo.api.schemas",
        "web_demo.api.responses",
        "web_demo.services.session_service",
        "web_demo.runtime.contracts",
        "web_demo.runtime.model_bundle",
        "web_demo.runtime.model_loader",
        "web_demo.runtime.frame_buffer",
        "web_demo.runtime.web_runtime",
        "web_demo.runtime.trajectory_controller",
        "web_demo.runtime.rootplan_controller",
        "web_demo.runtime.generation_worker",
    ):
        assert importlib.import_module(module_name) is not None


def test_load_model_bundle_builds_one_stream_generator_with_root_modules(monkeypatch):
    from web_demo.runtime import model_loader

    fake_vae = object()
    fake_ldf = _DummyModel()
    fake_cfg = SimpleNamespace(name="cfg")
    fake_refiner = object()
    fake_text_encoder = object()
    calls = []

    def fake_load_ldf_models(config_path, device):
        assert config_path == "cfg.yaml"
        assert device == "cpu"
        return fake_vae, fake_ldf, fake_cfg

    def fake_load_root_refiner_modules(root_cfg):
        calls.append(root_cfg)
        return fake_refiner, fake_text_encoder, (3, 8)

    monkeypatch.setattr(model_loader, "load_ldf_models", fake_load_ldf_models)
    monkeypatch.setattr(
        model_loader,
        "load_root_refiner_modules",
        fake_load_root_refiner_modules,
    )

    bundle = model_loader.load_model_bundle(
        "cfg.yaml",
        traj_mask_cfg={"root_refiner": {"enabled": True}},
        device="cpu",
    )

    assert calls == [{"enabled": True}]
    assert bundle.vae is fake_vae
    assert bundle.ldf_model is fake_ldf
    assert bundle.cfg is fake_cfg
    assert bundle.stream_generator.ldf_model is fake_ldf
    assert bundle.stream_generator.root_refiner is fake_refiner
    assert bundle.root_refiner is fake_refiner
    assert bundle.stream_generator.root_text_encoder is fake_text_encoder
    assert bundle.root_text_encoder is fake_text_encoder


def test_model_manager_init_uses_model_bundle_not_stream_generator_helper(monkeypatch):
    fake_model = _DummyModel()
    fake_stream_generator = StreamGenerator(ldf_model=fake_model, device="cpu")
    fake_bundle = ModelBundle(
        vae=SimpleNamespace(),
        ldf_model=fake_model,
        cfg=SimpleNamespace(name="cfg"),
        device="cpu",
        stream_generator=fake_stream_generator,
    )

    def fake_load_model_bundle(self, config_path, traj_mask_cfg):
        assert config_path == "cfg.yaml"
        assert traj_mask_cfg == {}
        return fake_bundle

    def fail_load_stream_generator(self, config_path, traj_mask_cfg):
        raise AssertionError("ModelManager.__init__ must use _load_model_bundle")

    monkeypatch.setattr(ModelManager, "_load_model_bundle", fake_load_model_bundle)
    monkeypatch.setattr(
        ModelManager,
        "_load_stream_generator",
        fail_load_stream_generator,
    )

    mgr = ModelManager(config_path="cfg.yaml", traj_mask_cfg={})

    assert mgr.vae is fake_bundle.vae
    assert mgr.model is fake_model
    assert mgr.cfg is fake_bundle.cfg
    assert mgr.stream_generator is fake_stream_generator
    assert mgr.rootplan_controller.stream_generator is fake_stream_generator


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.commit_index = 10
        self.chunk_size = 5
        self.init_calls = []

    def init_generated(self, history_length, *, batch_size, num_denoise_steps, traj_buffer=None):
        self.init_calls.append(
            {
                "history_length": history_length,
                "batch_size": batch_size,
                "num_denoise_steps": num_denoise_steps,
                "traj_buffer": traj_buffer,
            }
        )


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


def _route(version=1, *, start_commit_index=0, source="manual", end_x=0.0, end_z=1.0):
    return RoutePlan(
        times=np.array([0.0, 1.0], dtype=np.float32),
        points_xyz=np.array(
            [[0.0, 0.0, 0.0], [end_x, 0.0, end_z]],
            dtype=np.float32,
        ),
        start_commit_index=int(start_commit_index),
        version=int(version),
        source=str(source),
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


class _FakeFrameBuffer:
    target_size = 0

    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True

    def size(self):
        return 0


class _FakeVae:
    def __init__(self):
        self.cleared = False

    def clear_cache(self):
        self.cleared = True


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


def test_update_trajectory_accepts_per_update_horizon_delay_blend_controls():
    mgr = _trajectory_manager()
    first = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    second = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    preview = mgr.update_trajectory(
        first,
        source="manual",
        route_mode="relative_to_actor",
        horizon_tokens=5,
        delay_enabled=True,
        delay_tokens=8,
        blend_enabled=True,
        blend_tokens=2,
    )

    assert preview is not None
    assert len(preview) == 5
    assert mgr.traj_horizon_tokens == 5

    mgr.update_trajectory(
        second,
        source="manual",
        route_mode="relative_to_actor",
        horizon_tokens=7,
        delay_enabled=False,
        delay_tokens=8,
        blend_enabled=True,
        blend_tokens=6,
    )

    assert mgr.traj_horizon_tokens == 7
    assert mgr.traj_update_delay_enabled is False
    assert mgr.traj_update_delay_tokens == 8
    assert mgr.traj_update_blend_enabled is True
    assert mgr.traj_update_blend_tokens == 6
    assert mgr.pending_update_event is not None
    assert mgr.pending_update_event.delay_tokens == 0
    assert mgr.pending_update_event.blend_tokens == 6


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


def test_get_current_root_xyz_prefers_timeline_head_world_xz():
    mgr = _trajectory_manager()
    mgr._root_timeline = RootTimeline(_state(7, xz=(3.5, -2.0)))
    mgr.stream_recovery.r_pos_accum = np.array([99.0, 1.25, 88.0], dtype=np.float32)

    root = mgr._get_current_root_xyz()

    np.testing.assert_allclose(root, np.array([3.5, 1.25, -2.0], dtype=np.float32))


def test_first_update_trajectory_returns_display_preview_immediately():
    mgr = _trajectory_manager()
    mgr._root_timeline = RootTimeline(_state(0, xz=(2.0, 3.0)))
    mgr.stream_generator.timeline = mgr._root_timeline
    route = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    preview = mgr.update_trajectory(route, source="manual", route_mode="relative_to_actor")

    assert preview is not None
    assert preview.shape[1] == 3
    np.testing.assert_allclose(preview[0, [0, 2]], np.array([2.0, 3.0]), atol=1e-6)
    np.testing.assert_allclose(mgr.get_display_traj(), preview)


def test_reset_clears_model_manager_and_stream_generator_route_state():
    mgr = _trajectory_manager()
    mgr.frame_buffer = _FakeFrameBuffer()
    mgr.vae = _FakeVae()
    mgr.first_chunk = False
    mgr.root_xz_history = [np.zeros(2, dtype=np.float32)]
    mgr.root_5d_history = [(0, np.zeros(5, dtype=np.float32))]
    mgr._generated_frame_count = 7
    mgr._absolute_commit_index = 3
    mgr.is_generating = False
    mgr.generation_thread = None
    mgr.reset_pending = False
    mgr.smoothing_alpha = 1.0
    mgr.denoise_steps = 10
    mgr.active_traj_plan = _route(version=1)
    mgr.pending_update_event = object()
    mgr.current_traj_waypoints = np.ones((2, 3), dtype=np.float32)
    mgr.current_traj_times = np.array([0.0, 1.0], dtype=np.float32)
    mgr._display_traj = np.ones((2, 3), dtype=np.float32)
    mgr.stream_generator.active_root_plan = _plan(source="stale")
    mgr.stream_generator.condition_manager.route.route = _route(version=2)
    mgr.stream_generator.condition_manager.route.pending_update = object()

    assert mgr.reset() is True

    assert mgr.active_traj_plan is None
    assert mgr.pending_update_event is None
    assert mgr.get_display_traj() is None
    assert mgr.stream_generator.active_root_plan is None
    assert mgr.stream_generator.condition_manager.route.route is None
    assert mgr.stream_generator.condition_manager.route.pending_update is None


def test_stream_recovery_append_uses_session_anchor_after_timeline_trim():
    mgr = _trajectory_manager()
    mgr._session_anchor_state = _state(0, xz=(0.0, 0.0))
    timeline = RootTimeline(_state(5, xz=(100.0, 0.0)))
    mgr._root_timeline = timeline
    mgr.stream_generator.timeline = timeline
    mgr.stream_recovery = SimpleNamespace(
        r_pos_accum=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        r_rot_ang_accum=0.0,
    )

    assert mgr._append_root_state_from_stream_recovery(frame_idx=token_start_frame(6)) is True

    assert mgr._root_timeline.head.commit_idx == 6
    assert torch.allclose(
        mgr._root_timeline.head.world_xz,
        torch.tensor([1.0, 0.0]),
    )


def test_build_stream_traj_input_retries_activation_after_anchor_state_arrives():
    mgr = _trajectory_manager()
    mgr.route_reference_mode = "absolute"
    mgr.history_length = 1
    mgr.stream_generator.history_length = 1
    mgr._root_timeline = _timeline(2)
    mgr.stream_generator.timeline = mgr._root_timeline
    mgr.stream_generator.active_root_plan = None
    mgr.model.commit_index = 3
    mgr.model.chunk_size = 1
    mgr.active_traj_plan = _route(version=1, start_commit_index=3, end_z=4.0)
    mgr._absolute_commit_index = 3

    assert mgr._activate_root_plan_from_stream_plan(mgr.active_traj_plan) is False
    mgr._root_timeline.append(_state(3))

    payload = mgr._build_stream_traj_input()

    assert payload is not None
    assert mgr.stream_generator.active_root_plan is not None
    assert mgr._trajectory_state == "active_7d"


def test_pending_route_blend_payload_uses_temporary_blended_root_plan():
    mgr = _trajectory_manager()
    mgr.route_reference_mode = "absolute"
    mgr.history_length = 1
    mgr.stream_generator.history_length = 1
    mgr._root_timeline = _timeline(2)
    mgr.stream_generator.timeline = mgr._root_timeline
    mgr.model.commit_index = 2
    mgr.model.chunk_size = 1
    old_active_root_plan = _plan(source="old_active")
    old_active_root_plan.waypoints_local_7d[:, 0] = 100.0
    mgr.stream_generator.active_root_plan = old_active_root_plan
    mgr.active_traj_plan = _route(version=1, start_commit_index=0, end_x=0.0)
    mgr.pending_update_event = SimpleNamespace(
        old_route=mgr.active_traj_plan,
        new_route=_route(version=2, start_commit_index=0, end_x=10.0),
        edit_commit_index=0,
        effective_commit_index=0,
        delay_tokens=0,
        blend_tokens=4,
        version=2,
    )

    payload = mgr._build_stream_traj_input()

    assert payload is not None
    assert payload["trajectory_state"] == "blend"
    assert payload["model_traj_plan_version"] == "blend:1->2"
    assert mgr.stream_generator.active_root_plan is old_active_root_plan
    assert not torch.allclose(
        payload["traj_cond_7d_frame"][0, :, 0],
        torch.full_like(payload["traj_cond_7d_frame"][0, :, 0], 100.0),
    )


def test_load_stream_generator_rejects_normalized_root_refiner_config(tmp_path):
    mgr = _trajectory_manager()
    config_path = tmp_path / "root_refiner_normalized.yaml"
    config_path.write_text(
        "data:\n"
        "  normalize: true\n"
        "model:\n"
        "  target: models.root_refiner.RootRefiner\n"
        "  params: {}\n"
    )

    try:
        mgr._load_stream_generator(
            None,
            {
                "root_refiner": {
                    "enabled": True,
                    "config_path": str(config_path),
                    "ckpt": str(tmp_path / "missing.ckpt"),
                }
            },
        )
    except ValueError as exc:
        assert "physical RootRefiner output" in str(exc)
    else:
        raise AssertionError("normalized RootRefiner config should fail fast")
