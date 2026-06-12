from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from eval.runtime.benchmark import (
    DEFAULT_RUNTIME_OUTPUT_DIR,
    _csv_safe_record,
    _motion_video_overlay_kwargs,
    _transform_joints_to_runtime_world,
    _visual_target_root_from_plan,
    _write_runtime_case_visuals,
    aggregate_runtime_records,
    parse_condition_variants,
    resolve_traj_condition_source,
    write_runtime_report,
    write_stream_summary,
)
from eval.runtime.cases import REAL_CASES, get_cases
from eval.runtime.runners import (
    append_eval_root_history,
    append_eval_timeline_state,
    build_turn_metric_target,
    build_rootplan_stream_step_payload,
    clear_model_traj_state,
    new_eval_timeline,
    root_plan_events_to_diagnostic_arrays,
    run_babel_case as _run_babel,
    run_real_case as _run_real,
    run_step_case as _run_step,
    run_turn_case as _run_turn,
    set_eval_root_plan,
    set_eval_root_plan_from_world_7d,
)
from eval.runtime.transforms import (
    build_eval_root_plan_from_points,
    build_eval_root_plan_from_world_7d,
    compose_turn_root_plan,
    recovery_root_state_to_world,
    root_plan_to_world_7d,
    rotate_xz_points,
    rotate_world_7d_about_anchor,
)
from utils.inference_glue import InferenceGlueState, InferenceGlueTimeline
from utils.motion_process import append_traj_deltas_5d_to_7d
from utils.runtime_rootplan import build_rootplan_stream_payload_from_buffer
from utils.runtime_timeline import append_timeline_state_at_token_start_frame
from utils.stream_traj import StreamTrajectoryPlan
from utils.token_frame import token_range_to_frame_slice, token_start_frame
from utils.traj_stream_buffer import TrajStreamBuffer


def _timeline(num_commits: int) -> InferenceGlueTimeline:
    timeline = InferenceGlueTimeline(
        InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )
    for commit_idx in range(1, num_commits + 1):
        timeline.append(
            InferenceGlueState(
                commit_idx=commit_idx,
                world_xz=torch.zeros(2),
                world_yaw=torch.tensor(0.0),
                source="test",
            )
        )
    return timeline


def _moving_z_timeline(num_commits: int) -> InferenceGlueTimeline:
    timeline = InferenceGlueTimeline(
        InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )
    for commit_idx in range(1, num_commits + 1):
        timeline.append(
            InferenceGlueState(
                commit_idx=commit_idx,
                world_xz=torch.tensor(
                    [0.0, float(token_start_frame(commit_idx))],
                    dtype=torch.float32,
                ),
                world_yaw=torch.tensor(0.0),
                source="test",
            )
        )
    return timeline


def test_runtime_default_output_dir_is_eval_output_eval():
    assert DEFAULT_RUNTIME_OUTPUT_DIR.endswith("/eval/output_eval")


def test_real_suite_uses_route_and_motion7d_gtroot_cases():
    assert [(case.name, case.mode) for case in REAL_CASES] == [
        ("real_route_001168", "real_route"),
        ("real_gtroot_001168", "real_gtroot"),
        ("real_route_rot90_001168", "real_route"),
        ("real_gtroot_rot90_001168", "real_gtroot_rot90"),
    ]
    gtroot_case = REAL_CASES[1]
    oracle_rot_case = REAL_CASES[3]
    assert gtroot_case.mode_kwargs["gt_motion_7d"] is True
    assert gtroot_case.mode_kwargs.get("rotate_plan_deg", 0.0) == 0.0
    assert oracle_rot_case.mode_kwargs["gt_motion_7d"] is True
    assert oracle_rot_case.mode_kwargs["rotate_plan_deg"] == 90.0


def test_real_suite_selector_returns_route_and_oracle_cases():
    assert [(case.name, case.mode) for case in get_cases(suites=["real"])] == [
        ("real_route_001168", "real_route"),
        ("real_gtroot_001168", "real_gtroot"),
        ("real_route_rot90_001168", "real_route"),
        ("real_gtroot_rot90_001168", "real_gtroot_rot90"),
    ]


def test_rotate_world_7d_about_anchor_rotates_xyz_and_heading_only():
    traj_7d = torch.tensor(
        [
            [0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 1.0, 1.0, 0.0, 0.4, 0.1],
        ],
        dtype=torch.float32,
    )
    rotated = rotate_world_7d_about_anchor(
        traj_7d,
        anchor_xyz=torch.tensor([0.0, 0.2, 0.0]),
        degrees=90.0,
    )

    np.testing.assert_allclose(
        rotated.numpy(),
        np.asarray(
            [
                [0.0, 0.2, 0.0, 0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.2, 0.0, 0.0, -1.0, 0.4, 0.1],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_rotate_xz_points_rotates_route_about_anchor():
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    rotated = rotate_xz_points(points, anchor=np.asarray([0.0, 0.0, 0.0]), degrees=90.0)

    np.testing.assert_allclose(
        rotated,
        np.asarray([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_build_eval_root_plan_from_world_7d_preserves_oracle_channels():
    traj_7d = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.4, 0.1],
        ],
        dtype=torch.float32,
    )

    root_plan = build_eval_root_plan_from_world_7d(
        traj_7d,
        anchor_state=InferenceGlueState.initial(
            xz=(0.0, 0.0),
            yaw=0.0,
            dtype=torch.float32,
        ),
        token_dt=0.20,
        frames_per_token=4,
        source="test_gt_motion_7d",
    )

    np.testing.assert_allclose(
        root_plan.waypoints_local_7d.numpy(),
        traj_7d.numpy(),
        atol=1e-6,
    )
    assert root_plan.source == "test_gt_motion_7d"


def test_root_plan_to_world_7d_inverts_plan_anchor_canonicalization():
    traj_7d = torch.tensor(
        [
            [2.0, 0.0, 3.0, 0.0, -1.0, 0.2, 0.1],
            [1.0, 0.0, 3.0, 0.0, -1.0, 0.3, 0.2],
        ],
        dtype=torch.float32,
    )
    root_plan = build_eval_root_plan_from_world_7d(
        traj_7d,
        anchor_state=InferenceGlueState.initial(
            xz=(2.0, 3.0),
            yaw=-np.pi / 2.0,
            dtype=torch.float32,
        ),
        token_dt=0.20,
    )

    world = root_plan_to_world_7d(root_plan)

    np.testing.assert_allclose(world.numpy(), traj_7d.numpy(), atol=1e-6)


def test_compose_turn_root_plan_switches_world_7d_and_recomputes_deltas():
    anchor = InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    old_5d = torch.zeros(13, 5, dtype=torch.float32)
    old_5d[:, 2] = torch.arange(13, dtype=torch.float32)
    old_5d[:, 3] = 1.0
    new_5d = old_5d.clone()
    new_5d[:, 0] = torch.arange(13, dtype=torch.float32)
    new_5d[:, 2] = 10.0
    new_5d[:, 3] = 0.0
    new_5d[:, 4] = 1.0
    old_plan = build_eval_root_plan_from_world_7d(
        append_traj_deltas_5d_to_7d(old_5d),
        anchor_state=anchor,
        token_dt=0.20,
        source="old",
    )
    new_plan = build_eval_root_plan_from_world_7d(
        append_traj_deltas_5d_to_7d(new_5d),
        anchor_state=anchor,
        token_dt=0.20,
        source="new",
    )

    composed = compose_turn_root_plan(
        old_plan,
        new_plan,
        switch_commit=2,
        blend_tokens=0,
        source="composed",
    )
    world = root_plan_to_world_7d(composed)

    np.testing.assert_allclose(world[:5, :3].numpy(), old_5d[:5, :3].numpy(), atol=1e-6)
    np.testing.assert_allclose(world[5:, :3].numpy(), new_5d[5:, :3].numpy(), atol=1e-6)
    np.testing.assert_allclose(world[4, 3:5].numpy(), old_5d[4, 3:5].numpy(), atol=1e-6)
    np.testing.assert_allclose(world[5, 3:5].numpy(), new_5d[5, 3:5].numpy(), atol=1e-6)
    assert np.isfinite(world[:, 5:].numpy()).all()
    assert not np.isclose(float(world[5, 5]), float(new_plan.waypoints_local_7d[5, 5]))
    assert composed.anchor_commit_idx == 0
    assert composed.source == "composed"


def test_compose_turn_root_plan_blends_xyz_and_heading_continuously():
    anchor = InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    old_5d = torch.zeros(21, 5, dtype=torch.float32)
    old_5d[:, 2] = torch.arange(21, dtype=torch.float32)
    old_5d[:, 3] = 1.0
    new_5d = old_5d.clone()
    new_5d[:, 0] = 10.0
    new_5d[:, 3] = 0.0
    new_5d[:, 4] = 1.0
    old_plan = build_eval_root_plan_from_world_7d(
        append_traj_deltas_5d_to_7d(old_5d),
        anchor_state=anchor,
        token_dt=0.20,
    )
    new_plan = build_eval_root_plan_from_world_7d(
        append_traj_deltas_5d_to_7d(new_5d),
        anchor_state=anchor,
        token_dt=0.20,
    )

    composed = compose_turn_root_plan(
        old_plan,
        new_plan,
        switch_commit=2,
        blend_tokens=1,
    )
    world = root_plan_to_world_7d(composed)

    assert 0.0 < float(world[6, 0]) < 10.0
    heading_norm = torch.linalg.norm(world[:, 3:5], dim=-1)
    np.testing.assert_allclose(heading_norm.numpy(), np.ones(len(world)), atol=1e-5)
    yaw = torch.atan2(world[:, 4], world[:, 3])
    assert float(yaw[5]) <= float(yaw[6]) <= float(yaw[8])


def test_composed_turn_root_plan_keeps_body_window_prefix_valid_after_replacement():
    anchor = InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    old_5d = torch.zeros(140, 5, dtype=torch.float32)
    old_5d[:, 2] = torch.arange(140, dtype=torch.float32)
    old_5d[:, 3] = 1.0
    new_5d = old_5d.clone()
    new_5d[:, 0] = 5.0
    old_plan = build_eval_root_plan_from_world_7d(
        append_traj_deltas_5d_to_7d(old_5d),
        anchor_state=anchor,
        token_dt=0.20,
    )
    new_plan = build_eval_root_plan_from_world_7d(
        append_traj_deltas_5d_to_7d(new_5d),
        anchor_state=anchor,
        token_dt=0.20,
    )
    composed = compose_turn_root_plan(old_plan, new_plan, switch_commit=10, blend_tokens=0)
    buf = TrajStreamBuffer(batch_size=1, buf_len=256)
    buf.set_root_plan(composed)
    timeline = _timeline(32)

    payload = build_rootplan_stream_payload_from_buffer(
        buf,
        timeline,
        local_commit_index=20,
        absolute_commit_index=20,
        chunk_size=1,
        history_length=20,
        traj_horizon_tokens=4,
    )

    assert payload is not None
    assert bool(payload["traj_cond_frame_mask"].all())


def test_root_plan_events_to_diagnostic_arrays_uses_last_diagnostic_plan():
    traj_a = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0]],
        dtype=torch.float32,
    )
    traj_b = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0, 0.0, 0.2, 0.1]],
        dtype=torch.float32,
    )
    anchor = InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    plan_a = build_eval_root_plan_from_world_7d(traj_a, anchor_state=anchor, token_dt=0.20)
    plan_b = build_eval_root_plan_from_world_7d(traj_b, anchor_state=anchor, token_dt=0.20)

    root_7d, num_tokens = root_plan_events_to_diagnostic_arrays(
        [
            {"root_plan": plan_a, "diagnostic_plan": True},
            {"root_plan": plan_a, "diagnostic_plan": False},
            {"root_plan": plan_b, "diagnostic_plan": True},
        ]
    )

    np.testing.assert_allclose(root_7d, traj_b.numpy(), atol=1e-6)
    assert num_tokens == int(plan_b.num_tokens_pred)


def test_rotated_gt7d_anchor_canonicalizes_back_to_original_local_plan():
    traj_7d = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.4, 0.1],
        ],
        dtype=torch.float32,
    )
    rotated = rotate_world_7d_about_anchor(
        traj_7d,
        anchor_xyz=traj_7d[0, :3],
        degrees=90.0,
    )

    original_plan = build_eval_root_plan_from_world_7d(
        traj_7d,
        anchor_state=InferenceGlueState.initial(
            xz=(float(traj_7d[0, 0]), float(traj_7d[0, 2])),
            yaw=float(torch.atan2(traj_7d[0, 4], traj_7d[0, 3])),
            dtype=torch.float32,
        ),
        token_dt=0.20,
    )
    rotated_plan = build_eval_root_plan_from_world_7d(
        rotated,
        anchor_state=InferenceGlueState.initial(
            xz=(float(rotated[0, 0]), float(rotated[0, 2])),
            yaw=float(torch.atan2(rotated[0, 4], rotated[0, 3])),
            dtype=torch.float32,
        ),
        token_dt=0.20,
    )

    np.testing.assert_allclose(
        rotated_plan.waypoints_local_7d.numpy(),
        original_plan.waypoints_local_7d.numpy(),
        atol=1e-6,
    )


def test_recovery_root_state_to_world_applies_session_anchor_yaw():
    recovery = SimpleNamespace(
        r_pos_accum=np.asarray([0.0, 0.3, 1.0], dtype=np.float32),
        r_rot_ang_accum=0.0,
    )
    anchor = InferenceGlueState.initial(
        xz=(2.0, 3.0),
        yaw=-np.pi / 2.0,
        dtype=torch.float32,
    )

    root, yaw = recovery_root_state_to_world(recovery, anchor)

    np.testing.assert_allclose(root, np.asarray([1.0, 0.3, 3.0], dtype=np.float32), atol=1e-6)
    assert abs(float(yaw) + np.pi / 2.0) < 1e-6


def test_append_eval_timeline_state_uses_session_anchor_world_frame():
    timeline = new_eval_timeline()
    recovery = SimpleNamespace(
        r_pos_accum=np.asarray([0.0, 0.3, 1.0], dtype=np.float32),
        r_rot_ang_accum=0.0,
    )
    anchor = InferenceGlueState.initial(
        xz=(2.0, 3.0),
        yaw=-np.pi / 2.0,
        dtype=torch.float32,
    )

    append_eval_timeline_state(
        timeline,
        commit_idx=1,
        recovery=recovery,
        session_anchor_state=anchor,
    )

    state = timeline.at_commit(1)
    np.testing.assert_allclose(state.world_xz.numpy(), np.asarray([1.0, 3.0], dtype=np.float32))
    assert abs(float(state.world_yaw) + np.pi / 2.0) < 1e-6


def test_frame_aware_timeline_helper_skips_first_chunk_frame_zero():
    timeline = new_eval_timeline()
    recovery = SimpleNamespace(
        r_pos_accum=np.asarray([0.0, 0.0, 9.0], dtype=np.float32),
        r_rot_ang_accum=0.0,
    )

    appended = append_timeline_state_at_token_start_frame(
        timeline,
        frame_idx=0,
        recovery=recovery,
    )

    assert appended is False
    assert timeline.head.commit_idx == 0
    assert len(timeline) == 1


def test_frame_aware_timeline_helper_appends_only_token_start_frames():
    timeline = new_eval_timeline()
    recovery = SimpleNamespace(
        r_pos_accum=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        r_rot_ang_accum=0.0,
    )

    assert append_timeline_state_at_token_start_frame(
        timeline,
        frame_idx=1,
        recovery=recovery,
    )
    assert timeline.head.commit_idx == 1
    np.testing.assert_allclose(timeline.head.world_xz.numpy(), np.asarray([0.0, 1.0]))

    recovery.r_pos_accum = np.asarray([0.0, 0.0, 4.0], dtype=np.float32)
    assert not append_timeline_state_at_token_start_frame(
        timeline,
        frame_idx=4,
        recovery=recovery,
    )
    assert timeline.head.commit_idx == 1

    recovery.r_pos_accum = np.asarray([0.0, 0.0, 5.0], dtype=np.float32)
    assert append_timeline_state_at_token_start_frame(
        timeline,
        frame_idx=5,
        recovery=recovery,
    )
    assert timeline.head.commit_idx == 2
    np.testing.assert_allclose(timeline.head.world_xz.numpy(), np.asarray([0.0, 5.0]))


def test_append_eval_root_history_records_world_5d_and_next_frame():
    history = []
    recovery = SimpleNamespace(
        r_pos_accum=np.asarray([0.0, 0.3, 1.0], dtype=np.float32),
        r_rot_ang_accum=0.0,
    )
    anchor = InferenceGlueState.initial(
        xz=(2.0, 3.0),
        yaw=-np.pi / 2.0,
        dtype=torch.float32,
    )

    next_frame = append_eval_root_history(
        history,
        5,
        recovery,
        session_anchor_state=anchor,
    )

    assert next_frame == 6
    frame_idx, root5d = history[0]
    assert frame_idx == 5
    np.testing.assert_allclose(root5d[:3], np.asarray([1.0, 0.3, 3.0], dtype=np.float32), atol=1e-6)
    assert abs(float(root5d[3])) < 1e-6
    assert abs(float(root5d[4]) + 1.0) < 1e-6


def test_clear_model_traj_state_calls_available_clear_methods():
    calls = []

    class Buffer:
        def reset(self):
            calls.append("reset")

        def clear(self):
            calls.append("clear")

    clear_model_traj_state(SimpleNamespace(_traj_buf=Buffer()))

    assert calls == ["reset", "clear"]


def test_set_eval_root_plan_from_world_7d_records_root_plan_event():
    events = []
    replan_events = []
    model = SimpleNamespace(
        _traj_buf=SimpleNamespace(set_root_plan=lambda plan: setattr(model, "root_plan", plan))
    )
    timeline = InferenceGlueTimeline(
        InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )
    traj_7d = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0]],
        dtype=torch.float32,
    )

    ok = set_eval_root_plan_from_world_7d(
        model,
        timeline,
        traj_7d,
        start_commit_index=0,
        text="walk",
        token_dt=0.20,
        source="test_gt",
        replan_events=replan_events,
        root_plan_events=events,
    )

    assert ok is True
    assert model.root_plan.source == "test_gt"
    assert events[0]["root_plan"] is model.root_plan
    assert events[0]["source"] == "test_gt"
    assert replan_events[0]["source"] == "test_gt"


def test_set_eval_root_plan_from_route_records_root_plan_event():
    events = []
    model = SimpleNamespace(
        _traj_buf=SimpleNamespace(set_root_plan=lambda plan: setattr(model, "root_plan", plan))
    )
    timeline = InferenceGlueTimeline(
        InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )
    stream_plan = StreamTrajectoryPlan(
        times=np.asarray([0.0, 0.05], dtype=np.float32),
        points_xyz=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        start_commit_index=0,
        version=0,
        source="test_route",
    )

    ok = set_eval_root_plan(
        model,
        timeline,
        stream_plan,
        text="walk",
        token_dt=0.20,
        root_plan_events=events,
    )

    assert ok is True
    assert model.root_plan.source == "test_route"
    assert events[0]["root_plan"] is model.root_plan
    assert events[0]["root_refiner"] is False


def test_motion_video_overlay_kwargs_uses_target_root_xz():
    target_root = np.array(
        [
            [1.0, 0.0, 2.0, 1.0, 0.0],
            [3.0, 0.0, 4.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    kwargs = _motion_video_overlay_kwargs(target_root)

    assert kwargs["render_setting"]["cond_traj_show_full"] is True
    np.testing.assert_allclose(kwargs["traj_xz"], target_root[:, [0, 2]])
    np.testing.assert_allclose(kwargs["traj_mask"], np.ones(2, dtype=np.float32))


def test_transform_joints_to_runtime_world_aligns_video_root_with_metric_root():
    joints = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.5, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.5, 2.0]],
        ],
        dtype=np.float32,
    )
    pred_root = np.asarray(
        [
            [10.0, 0.0, 20.0],
            [9.0, 0.0, 20.0],
        ],
        dtype=np.float32,
    )

    world = _transform_joints_to_runtime_world(
        joints,
        pred_root=pred_root,
        pred_yaw_offset=-np.pi / 2.0,
    )

    np.testing.assert_allclose(world[:, 0, :], pred_root, atol=1e-6)
    np.testing.assert_allclose(world[0, 1, [0, 2]], [9.0, 20.0], atol=1e-6)
    np.testing.assert_allclose(world[1, 1, [0, 2]], [8.0, 20.0], atol=1e-6)


def test_stream_benchmark_builds_7d_rootplan_payload_for_model_step():
    timeline = _timeline(10)
    model = SimpleNamespace(
        commit_index=10,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(80, 3)
    points[:, 2] = torch.arange(80, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=4,
        traj_horizon_tokens=3,
    )

    start_token = 7
    num_tokens = 7
    frame_slice = token_range_to_frame_slice(start_token, num_tokens)
    assert payload["traj_start_token"] == start_token
    assert payload["traj_abs_start_token"] == start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == 7
    assert payload["body_anchor_abs_token"] == 7
    assert "traj_cond_7d_frame" in payload
    assert "traj_cond_frame_mask" in payload
    assert "traj" not in payload
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert float(payload["traj_cond_7d_frame"][0, 0, 2]) == float(
        token_start_frame(start_token)
    )
    assert payload["traj_cond_frame_mask"].all()


def test_rootplan_payload_covers_all_denoise_substep_windows_within_chunk():
    history_length = 30
    horizon_tokens = 20
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=40,
        chunk_size=5,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=128),
    )
    points = torch.zeros(400, 3)
    points[:, 2] = torch.arange(400, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=40,
    )

    start_token = 11  # earliest substep right token 41, history 30 -> left edge 11
    num_tokens = 54  # final right token 45 + horizon 20 - start 11
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
    assert float(payload["traj_cond_7d_frame"][0, 0, 2]) == float(
        token_start_frame(start_token)
    )


def test_rootplan_payload_canonicalizes_each_substep_with_its_body_anchor():
    history_length = 30
    horizon_tokens = 20
    timeline = _moving_z_timeline(100)
    model = SimpleNamespace(
        commit_index=40,
        chunk_size=5,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=128),
    )
    points = torch.zeros(400, 3)
    points[:, 2] = torch.arange(400, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=40,
    )

    subpayloads = payload["traj_substep_payloads"]
    assert [p["traj_start_token"] for p in subpayloads] == [11, 12, 13, 14, 15]
    assert [p["body_anchor_token"] for p in subpayloads] == [11, 12, 13, 14, 15]
    for subpayload in subpayloads:
        assert abs(float(subpayload["traj_cond_7d_frame"][0, 0, 2])) < 1e-6


def test_rootplan_payload_uses_absolute_commit_for_plan_and_local_commit_for_model():
    history_length = 30
    horizon_tokens = 3
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=30,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(300, 3)
    points[:, 2] = torch.arange(300, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=60,
    )

    body_anchor_token = 1
    body_anchor_abs_token = 31
    local_start_token = body_anchor_token
    absolute_start_token = body_anchor_abs_token
    num_tokens = history_length + horizon_tokens
    frame_slice = token_range_to_frame_slice(absolute_start_token, num_tokens)
    assert payload["traj_start_token"] == local_start_token
    assert payload["traj_abs_start_token"] == absolute_start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == body_anchor_token
    assert payload["body_anchor_abs_token"] == body_anchor_abs_token
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert float(payload["traj_cond_7d_frame"][0, 0, 2]) == float(
        token_start_frame(absolute_start_token)
    )


def test_rootplan_payload_frame_slice_uses_absolute_start_when_local_start_is_zero():
    history_length = 30
    horizon_tokens = 3
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=1,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(300, 3)
    points[:, 2] = torch.arange(300, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=60,
    )

    body_anchor_token = 0
    body_anchor_abs_token = 59
    local_start_token = body_anchor_token
    absolute_start_token = body_anchor_abs_token
    num_tokens = 2 + horizon_tokens
    frame_slice = token_range_to_frame_slice(absolute_start_token, num_tokens)
    assert payload["traj_start_token"] == local_start_token
    assert payload["traj_abs_start_token"] == absolute_start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == body_anchor_token
    assert payload["body_anchor_abs_token"] == body_anchor_abs_token
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert float(payload["traj_cond_7d_frame"][0, 0, 2]) == float(
        token_start_frame(absolute_start_token)
    )


def test_rootplan_payload_masks_until_future_plan_anchor_after_model_roll():
    history_length = 30
    horizon_tokens = 10
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=30,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(300, 3)
    points[:, 2] = torch.arange(300, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(65),
        token_dt=0.20,
        frames_per_token=4,
        source="future_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=60,
    )

    local_start_token = 1
    absolute_start_token = 31
    prefix_tokens = 65 - absolute_start_token
    prefix_frames = (
        token_start_frame(absolute_start_token + prefix_tokens)
        - token_start_frame(absolute_start_token)
    )
    mask = payload["traj_cond_frame_mask"][0].bool()
    assert payload["traj_start_token"] == local_start_token
    assert payload["traj_abs_start_token"] == absolute_start_token
    assert prefix_frames > 0
    assert not mask[:prefix_frames].any()
    assert mask[prefix_frames:].all()


def test_rootplan_substep_payloads_partially_mask_until_replan_anchor_inside_window():
    history_length = 30
    horizon_tokens = 20
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=40,
        chunk_size=5,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=128),
    )
    points = torch.zeros(400, 3)
    points[:, 2] = torch.arange(400, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(40),
        token_dt=0.20,
        frames_per_token=4,
        source="mid_window_replan",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=40,
    )

    assert [p["traj_abs_start_token"] for p in payload["traj_substep_payloads"]] == [
        11,
        12,
        13,
        14,
        15,
    ]
    for subpayload in payload["traj_substep_payloads"]:
        absolute_start = int(subpayload["traj_abs_start_token"])
        prefix_frames = token_start_frame(40) - token_start_frame(absolute_start)
        mask = subpayload["traj_cond_frame_mask"][0].bool()
        assert 0 < prefix_frames < mask.numel()
        assert not mask[:prefix_frames].any()
        assert mask[prefix_frames:].all()


def test_eval_rootplan_payload_wrapper_matches_shared_runtime_helper():
    history_length = 4
    horizon_tokens = 2
    timeline = _timeline(10)
    model = SimpleNamespace(
        commit_index=10,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(80, 3)
    points[:, 2] = torch.arange(80, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    wrapped = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
    )
    direct = build_rootplan_stream_payload_from_buffer(
        model._traj_buf,
        timeline,
        local_commit_index=model.commit_index,
        absolute_commit_index=model.commit_index,
        chunk_size=model.chunk_size,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
    )

    assert wrapped["traj_start_token"] == direct["traj_start_token"]
    assert wrapped["traj_abs_start_token"] == direct["traj_abs_start_token"]
    assert torch.equal(wrapped["traj_cond_frame_mask"], direct["traj_cond_frame_mask"])
    assert torch.allclose(wrapped["traj_cond_7d_frame"], direct["traj_cond_7d_frame"])


def test_write_stream_summary_sanitizes_nan_for_strict_json(tmp_path):
    path = tmp_path / "summary.json"
    write_stream_summary(
        path,
        {
            "records": [
                {"ADE": float("nan"), "FDE": float("inf")},
                {"ADE": 1.0, "FDE": -float("inf")},
            ]
        },
    )

    text = path.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(text)
    assert payload["records"][0]["ADE"] is None
    assert payload["records"][0]["FDE"] is None
    assert payload["records"][1]["FDE"] is None


def test_write_stream_summary_sanitizes_numpy_and_torch_scalars(tmp_path):
    path = tmp_path / "summary.json"
    write_stream_summary(
        path,
        {
            "np_bad": np.float32(np.inf),
            "np_ok": np.int64(7),
            "torch_bad": torch.tensor(float("nan")),
            "torch_ok": torch.tensor([1.0, 2.0]),
        },
    )

    payload = json.loads(path.read_text())
    assert payload["np_bad"] is None
    assert payload["np_ok"] == 7
    assert payload["torch_bad"] is None
    assert payload["torch_ok"] == [1.0, 2.0]


def test_write_runtime_report_mirrors_standard_eval_layout(tmp_path):
    payload = {
        "run_id": "step_000500",
        "summary": {"ADE_mean": 0.2},
        "aggregate": {"ADE_mean": 0.2},
        "records": [
            {
                "suite": "humanml3d_control",
                "mode": "gt_traj",
                "sample_id": "000021",
                "ADE": 0.2,
                "FDE": 0.3,
            }
        ],
    }

    write_runtime_report(
        output_dir=tmp_path,
        run_id="step_000500",
        suite_tag="smoke",
        payload=payload,
        records=payload["records"],
    )

    legacy_root = tmp_path / "step_000500"
    metrics_dir = tmp_path / "Runtime" / "metrics" / "smoke" / "step_000500"

    assert (legacy_root / "summary.json").is_file()
    assert (legacy_root / "summary.csv").is_file()
    assert (metrics_dir / "summary.json").is_file()
    assert (metrics_dir / "records.csv").is_file()

    standard_payload = json.loads((metrics_dir / "summary.json").read_text())
    assert standard_payload["run_id"] == "step_000500"
    assert standard_payload["summary"]["ADE_mean"] == 0.2


def test_write_runtime_report_exposes_standard_media_dirs(tmp_path):
    payload = {
        "run_id": "step_000500",
        "summary": {"ADE_mean": 0.2},
        "aggregate": {"ADE_mean": 0.2},
        "records": [
            {
                "suite": "real",
                "mode": "real_route",
                "case_name": "real_route_rot90_001168",
                "condition_variant": "gt_7d_ldf",
                "ADE": 0.2,
                "FDE": 0.3,
            }
        ],
    }

    dirs = write_runtime_report(
        output_dir=tmp_path,
        run_id="step_000500",
        suite_tag="control",
        payload=payload,
        records=payload["records"],
        artifact_kinds=("metrics", "plot", "video"),
    )

    assert dirs["metrics"] == tmp_path / "Runtime/metrics/control/step_000500"
    assert dirs["plot"] == tmp_path / "Runtime/plot/control/step_000500"
    assert dirs["video"] == tmp_path / "Runtime/video/control/step_000500"
    assert (dirs["metrics"] / "records.csv").is_file()


def test_parse_condition_variants_expands_runtime_control_paths():
    variants = parse_condition_variants("gt_7d_ldf,rootrefiner_7d_ldf,no_traj_ldf")

    assert [variant.name for variant in variants] == [
        "gt_7d_ldf",
        "rootrefiner_7d_ldf",
        "no_traj_ldf",
    ]
    assert variants[0].condition_path == "rootplan_7d"
    assert variants[0].use_root_refiner is False
    assert variants[0].force_no_traj is False
    assert variants[1].condition_path == "rootplan_7d"
    assert variants[1].use_root_refiner is True
    assert variants[1].force_no_traj is False
    assert variants[2].condition_path == "rootplan_7d"
    assert variants[2].use_root_refiner is False
    assert variants[2].force_no_traj is True


def test_aggregate_runtime_records_groups_condition_variants():
    summary = aggregate_runtime_records(
        [
            {
                "suite": "real",
                "mode": "real_route",
                "condition_variant": "gt_7d_ldf",
                "ADE": 1.0,
                "FDE": 2.0,
            },
            {
                "suite": "real",
                "mode": "real_route",
                "condition_variant": "rootrefiner_7d_ldf",
                "ADE": 3.0,
                "FDE": 4.0,
            },
            {
                "suite": "real",
                "mode": "real_route",
                "condition_variant": "no_traj_ldf",
                "ADE": 5.0,
                "FDE": 6.0,
            },
        ]
    )

    assert summary["by_condition_variant"]["gt_7d_ldf"]["ADE_mean"] == 1.0
    assert (
        summary["by_condition_variant"]["rootrefiner_7d_ldf"]["FDE_mean"] == 4.0
    )
    assert summary["by_suite_variant"]["real/no_traj_ldf"]["ADE_mean"] == 5.0


def test_visual_target_root_uses_plan_target_not_original_gt():
    original_gt = np.zeros((5, 3), dtype=np.float32)
    plan_times = np.asarray([0.0, 0.2], dtype=np.float32)
    plan_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )

    target = _visual_target_root_from_plan(
        original_gt_root=original_gt,
        plan_times=plan_times,
        plan_points_xyz=plan_points,
        target_frames=5,
        motion_fps=20.0,
    )

    assert target.shape == original_gt.shape
    assert np.allclose(target[0], plan_points[0])
    assert np.allclose(target[-1], plan_points[-1])
    assert not np.allclose(target, original_gt)


def test_resolve_traj_condition_source_makes_root_refiner_mode_explicit():
    assert resolve_traj_condition_source("rootplan_7d", None) == "route_7d"
    assert (
        resolve_traj_condition_source("rootplan_7d", object())
        == "root_refiner_7d"
    )
    assert resolve_traj_condition_source("legacy_xyz", None) == "legacy_xyz"
    assert (
        resolve_traj_condition_source("rootplan_7d", object(), no_traj=True)
        == "none"
    )


def test_csv_safe_record_serializes_nested_values_as_json():
    row = _csv_safe_record(
        {
            "suite": "turn",
            "rootplan_replan_commits": [0, 15],
            "rootplan_replan_sources": ["bench_old", "bench_new"],
            "bad": float("nan"),
        }
    )

    assert row["suite"] == "turn"
    assert json.loads(row["rootplan_replan_commits"]) == [0, 15]
    assert json.loads(row["rootplan_replan_sources"]) == ["bench_old", "bench_new"]
    assert row["bad"] is None


def test_aggregate_runtime_records_groups_checkpoint_selection_metrics():
    summary = aggregate_runtime_records(
        [
            {
                "suite": "humanml3d_control",
                "mode": "gt_traj",
                "ADE": 1.0,
                "FDE": 2.0,
                "path_arc": float("nan"),
            },
            {
                "suite": "humanml3d_control",
                "mode": "refiner_traj",
                "ADE": 3.0,
                "FDE": 4.0,
                "path_arc": 6.0,
            },
            {
                "suite": "route_edit",
                "mode": "replace_future",
                "ADE": 5.0,
                "FDE": 6.0,
                "path_arc": 8.0,
            },
        ]
    )

    assert summary["num_records"] == 3
    assert summary["ADE_mean"] == 3.0
    assert summary["ADE_count"] == 3
    assert summary["path_arc_mean"] == 7.0
    assert summary["path_arc_count"] == 2
    assert summary["by_suite"]["humanml3d_control"]["ADE_mean"] == 2.0
    assert summary["by_mode"]["gt_traj"]["FDE_mean"] == 2.0
    assert summary["by_suite_mode"]["route_edit/replace_future"]["ADE_mean"] == 5.0


def test_runtime_case_visuals_write_static_path_and_yaw_plots(tmp_path):
    frames = 8
    pred_root = np.zeros((frames, 3), dtype=np.float32)
    target_root = np.zeros((frames, 3), dtype=np.float32)
    pred_root[:, 2] = np.arange(frames, dtype=np.float32) * 0.12
    target_root[:, 2] = np.arange(frames, dtype=np.float32) * 0.10
    motion = np.zeros((frames, 263), dtype=np.float32)
    motion[:, 2] = pred_root[:, 2]

    _write_runtime_case_visuals(
        output_dir=tmp_path,
        case_name="case_a",
        pred_root=pred_root,
        target_root=target_root,
        motion_263=motion,
        split_tok=1,
    )

    assert (tmp_path / "case_a_plot_world_xz.png").is_file()
    assert (tmp_path / "case_a_plot_yaw.png").is_file()


class _FakeStreamModel:
    chunk_size = 1

    def init_generated(self, history_length, batch_size=1, num_denoise_steps=None):
        self.commit_index = 0
        self.generated = torch.zeros(1, 1, 1)
        self._traj_buf = TrajStreamBuffer(batch_size=1, buf_len=96)

    def stream_generate_step(self, step_input, first_chunk=True):
        self.commit_index += 1
        return {"generated": torch.zeros(1, 1, 263)}


class _FakeVae:
    def clear_cache(self):
        pass

    def stream_decode(self, generated, first_chunk=True):
        return torch.zeros(1, 1, 263)


class _CausalFakeVae:
    def clear_cache(self):
        pass

    def stream_decode(self, generated, first_chunk=True):
        num_frames = 1 if first_chunk else 4
        return torch.zeros(1, num_frames, 263)


class _RecordingRootRefinerRuntime:
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
                "start_commit_index": int(plan.start_commit_index),
                "history": history_motion_world_5d,
                "first_point": plan.points_xyz[0].copy(),
                "num_points": len(plan.points_xyz),
            }
        )
        points = torch.zeros(80, 3)
        points[:, 2] = torch.arange(80, dtype=torch.float32)
        return build_eval_root_plan_from_points(
            points,
            anchor_state=anchor_state,
            token_dt=token_dt,
            frames_per_token=4,
            source="fake_refiner",
        )


def test_babel_root_refiner_rebuilds_plan_when_text_segment_changes():
    feature = torch.zeros(9, 263)
    sample = {
        "token_length": 3,
        "feature_length": 9,
        "feature": feature,
        "traj": torch.zeros(9, 3),
        "text": ["walk", "run"],
        "token_text_end": [1, 3],
    }
    runtime = _RecordingRootRefinerRuntime()

    _run_babel(
        _FakeStreamModel(),
        _FakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=1,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="babel_timestamped",
        condition_path="rootplan_7d",
        root_refiner_runtime=runtime,
    )

    assert [(c["text"], c["start_commit_index"]) for c in runtime.calls] == [
        ("walk", 0),
        ("run", 1),
    ]


def test_step_root_refiner_sets_initial_root_plan():
    feature = torch.zeros(9, 263)
    sample = {
        "token_length": 3,
        "traj_length": 9,
        "feature": feature,
        "traj": torch.zeros(9, 3),
        "text": "walk forward",
        "token_mask": torch.ones(3),
        "traj_mask": torch.ones(9),
    }
    runtime = _RecordingRootRefinerRuntime()

    _run_step(
        _FakeStreamModel(),
        _FakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=1,
        mode="step_gtroot",
        condition_path="rootplan_7d",
        root_refiner_runtime=runtime,
    )

    assert [(c["text"], c["start_commit_index"]) for c in runtime.calls] == [
        ("walk forward", 0),
    ]


def test_babel_root_refiner_replan_receives_generated_root_history():
    feature = torch.zeros(9, 263)
    sample = {
        "token_length": 3,
        "feature_length": 9,
        "feature": feature,
        "traj": torch.zeros(9, 3),
        "text": ["walk", "run"],
        "token_text_end": [1, 3],
    }
    runtime = _RecordingRootRefinerRuntime()

    model = _FakeStreamModel()
    replan_events = []
    _run_babel(
        model,
        _CausalFakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=1,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="babel_timestamped",
        condition_path="rootplan_7d",
        root_refiner_runtime=runtime,
        replan_events=replan_events,
    )

    assert runtime.calls[0]["history"] is None
    assert runtime.calls[1]["history"] is not None
    assert runtime.calls[1]["history"].shape[-1] == 5
    assert len(replan_events) == 2
    assert replan_events[1]["text"] == "run"
    assert not hasattr(model, "_stream_eval_replan_events")


def test_real_gt_motion_7d_runner_records_root_plan_event():
    feature = torch.zeros(9, 263)
    sample = {
        "token_length": 3,
        "feature_length": 9,
        "feature": feature,
        "traj": torch.zeros(9, 3),
        "text": "walk forward",
    }
    root_plan_events = []
    replan_events = []

    _run_real(
        _FakeStreamModel(),
        _CausalFakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=2,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="real_gtroot",
        condition_path="rootplan_7d",
        gt_motion_7d=True,
        root_plan_events=root_plan_events,
        replan_events=replan_events,
    )

    assert root_plan_events
    assert root_plan_events[0]["source"] == "bench_real_gt_motion_7d"
    assert root_plan_events[0]["root_refiner"] is False
    assert replan_events[0]["source"] == "bench_real_gt_motion_7d"


def test_turn_rootplan_activates_composed_plan_for_delayed_replace():
    feature = torch.zeros(81, 263)
    sample = {
        "token_length": 20,
        "feature_length": 81,
        "feature": feature,
        "traj": torch.zeros(81, 3),
        "text": "walk forward",
    }
    model = _FakeStreamModel()
    replan_events = []

    _run_turn(
        model,
        _CausalFakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=2,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="turn_mid_update",
        angle=45.0,
        delay_tokens=0,
        blend_tokens=2,
        condition_path="rootplan_7d",
        root_refiner_runtime=None,
        replan_events=replan_events,
    )

    sources = [event["source"] for event in replan_events]
    assert sources == ["bench_old", "bench_composed"]
    assert "bench_new" not in sources
    assert not hasattr(model, "_stream_eval_replan_events")


def test_turn_rootplan_returns_composed_plan_as_visual_target():
    feature = torch.zeros(81, 263)
    traj = torch.zeros(81, 3)
    traj[:, 2] = torch.arange(81, dtype=torch.float32) * 0.05
    sample = {
        "token_length": 20,
        "feature_length": 81,
        "feature": feature,
        "traj": traj,
        "text": "walk forward",
    }
    root_plan_events = []

    _pm, _pr, _gr, target_t, target_pts, _ttfs = _run_turn(
        _FakeStreamModel(),
        _CausalFakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=2,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="turn_mid_update",
        angle=45.0,
        delay_tokens=0,
        blend_tokens=4,
        condition_path="rootplan_7d",
        root_refiner_runtime=None,
        root_plan_events=root_plan_events,
    )

    diagnostic_7d, _num_tokens = root_plan_events_to_diagnostic_arrays(root_plan_events)
    n = min(len(target_pts), len(diagnostic_7d))
    assert np.allclose(target_t[:n], np.arange(n, dtype=np.float32) / 20.0)
    np.testing.assert_allclose(target_pts[:n], diagnostic_7d[:n, :3], atol=1e-6)


def test_turn_force_no_traj_skips_rootplan_setup():
    feature = torch.zeros(81, 263)
    sample = {
        "token_length": 20,
        "feature_length": 81,
        "feature": feature,
        "traj": torch.zeros(81, 3),
        "text": "walk forward",
    }
    model = _FakeStreamModel()
    replan_events = []

    _run_turn(
        model,
        _CausalFakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=2,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="turn_mid_update",
        angle=45.0,
        delay_tokens=0,
        blend_tokens=2,
        condition_path="rootplan_7d",
        root_refiner_runtime=None,
        replan_events=replan_events,
        force_no_traj=True,
    )

    assert replan_events == []
    assert not model._traj_buf.has_active_plan()


def test_turn_root_refiner_does_not_rebuild_every_blend_token():
    feature = torch.zeros(81, 263)
    traj = torch.zeros(81, 3)
    traj[:, 2] = torch.arange(81, dtype=torch.float32) * 0.1
    sample = {
        "token_length": 20,
        "feature_length": 81,
        "feature": feature,
        "traj": traj,
        "text": "walk forward",
    }
    runtime = _RecordingRootRefinerRuntime()
    replan_events = []

    _run_turn(
        _FakeStreamModel(),
        _CausalFakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=2,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="turn_mid_update",
        angle=45.0,
        delay_tokens=0,
        blend_tokens=3,
        condition_path="rootplan_7d",
        root_refiner_runtime=runtime,
        replan_events=replan_events,
    )

    assert [(c["text"], c["start_commit_index"]) for c in runtime.calls] == [
        ("walk forward", 0),
        ("walk forward", 15),
    ]
    assert runtime.calls[1]["num_points"] < runtime.calls[0]["num_points"]
    assert [event["source"] for event in replan_events] == ["bench_old", "bench_composed"]


def test_turn_metric_target_follows_old_path_until_runtime_replacement():
    plan_t = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    old_pts = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    new_pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]],
        dtype=np.float32,
    )

    target_t, target_pts = build_turn_metric_target(
        old_times=plan_t,
        old_points_xyz=old_pts,
        new_times=plan_t,
        new_points_xyz=new_pts,
        target_frames=21,
        motion_fps=10.0,
        edit_commit=2,
        delay_tokens=3,
        blend_tokens=2,
        token_dt=0.1,
    )

    activation_frame = int(round((2 + 3 + 2) * 0.1 * 10.0))
    assert np.allclose(target_pts[activation_frame - 1], [0.0, 0.0, 0.6])
    assert target_pts[activation_frame, 0] > 0.0
    assert np.allclose(target_t, np.arange(21, dtype=np.float32) / 10.0)


def test_turn_metric_target_reanchors_new_route_to_effective_anchor():
    plan_t = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    old_pts = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    new_pts = np.array(
        [[10.0, 0.0, 0.0], [10.0, 0.0, 1.0], [10.0, 0.0, 2.0]],
        dtype=np.float32,
    )

    _, target_pts = build_turn_metric_target(
        old_times=plan_t,
        old_points_xyz=old_pts,
        new_times=plan_t,
        new_points_xyz=new_pts,
        target_frames=21,
        motion_fps=10.0,
        edit_commit=2,
        delay_tokens=3,
        blend_tokens=2,
        token_dt=0.1,
        new_anchor_xz=np.array([0.0, 0.5], dtype=np.float32),
    )

    activation_frame = int(round((2 + 3 + 2) * 0.1 * 10.0))
    assert np.allclose(target_pts[activation_frame - 1], [0.0, 0.0, 0.6])
    assert np.allclose(target_pts[activation_frame], [0.0, 0.0, 0.7])
