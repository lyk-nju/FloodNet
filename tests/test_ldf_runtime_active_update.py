import math

import torch

from eval.ldf.runtime_update.active_condition import (
    compose_active_window_segment,
    compose_active_window_world_condition,
)
from eval.ldf.runtime_update.diagnostics import (
    build_timeline_from_generated_traj7,
    validate_trajectory_diagnostics,
)
from eval.ldf.runtime_update.payload_builder import (
    build_active_window_root_plan,
    build_active_window_stream_payload,
    build_world_condition_stream_payload,
)
from eval.ldf.runtime_update.route_tracker import (
    RouteProgressTracker,
)
from utils.inference.timeline import RootFrameState, RootTimeline
from utils.local_frame import heading_dir_xz
from utils.motion_process import build_physical_7d_from_5d
from utils.token_frame import token_end_frame, token_range_to_frame_slice


def _traj7_from_xz_yaw(xz: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    y = torch.zeros(xz.shape[0], dtype=xz.dtype, device=xz.device)
    traj5 = torch.stack(
        [xz[:, 0], y, xz[:, 1], torch.cos(yaw), torch.sin(yaw)],
        dim=-1,
    )
    return build_physical_7d_from_5d(traj5)


def test_route_tracker_keeps_progress_monotonic_and_applies_lookahead():
    xz = torch.stack([torch.zeros(80), torch.linspace(0.0, 7.9, 80)], dim=-1)
    route = _traj7_from_xz_yaw(xz, torch.zeros(80))
    tracker = RouteProgressTracker(route, lookahead_m=0.35)

    first = tracker.project(
        current_xz=torch.tensor([0.0, 3.05]),
        current_yaw=torch.tensor(0.0),
    )
    second = tracker.project(
        current_xz=torch.tensor([0.0, 2.70]),
        current_yaw=torch.tensor(0.0),
    )

    assert second.route_index >= first.route_index
    assert second.future_index > second.route_index
    future_delta = xz[second.future_index] - torch.tensor([0.0, 2.70])
    assert torch.dot(future_delta, heading_dir_xz(torch.tensor(0.0))) > 0.25


def test_active_window_segment_starts_at_current_root_and_points_forward():
    route_xz = torch.stack(
        [
            0.4 * torch.sin(torch.linspace(0.0, 1.0, 120)),
            torch.linspace(0.0, 5.0, 120),
        ],
        dim=-1,
    )
    route_yaw = torch.atan2(
        torch.gradient(route_xz[:, 0])[0],
        torch.gradient(route_xz[:, 1])[0],
    )
    route = _traj7_from_xz_yaw(route_xz, route_yaw)
    generated = route.clone()
    generated[48:, 2] += 0.22
    current_frame = 55
    current_yaw = torch.atan2(generated[current_frame, 4], generated[current_frame, 3])

    result = compose_active_window_segment(
        route,
        generated,
        current_frame=current_frame,
        current_yaw=current_yaw,
        target_end_frame=95,
        lookahead_m=0.25,
        bridge_frames=10,
    )

    assert torch.allclose(
        result.segment_traj7[0, [0, 1, 2]],
        generated[current_frame, [0, 1, 2]],
        atol=1e-6,
    )
    xz = result.segment_traj7[:, [0, 2]]
    yaw = torch.atan2(result.segment_traj7[:, 4], result.segment_traj7[:, 3])
    delta = xz[1:] - xz[:-1]
    speed = torch.linalg.norm(delta, dim=-1)
    forward = (delta * heading_dir_xz(yaw[1:])).sum(-1)
    moving = speed > 1e-5
    assert bool((forward[moving] > -1e-4).all())
    assert float(torch.linalg.norm(xz[1] - xz[0]).item()) < 0.08


def test_active_window_segment_can_pin_route_progress_to_update_boundary():
    xz = torch.stack([torch.zeros(120), torch.linspace(0.0, 6.0, 120)], dim=-1)
    route = _traj7_from_xz_yaw(xz, torch.zeros(120))
    generated = route.clone()
    generated[70, 2] = route[30, 2]

    result = compose_active_window_segment(
        route,
        generated,
        current_frame=70,
        current_yaw=torch.tensor(0.0),
        target_end_frame=100,
        min_route_index=70,
        lookahead_m=0.20,
        bridge_frames=8,
    )

    assert result.route_index >= 70
    assert result.future_index > result.route_index


def test_active_window_segment_honors_requested_bridge_frames():
    xz = torch.stack([torch.zeros(160), torch.linspace(0.0, 8.0, 160)], dim=-1)
    route = _traj7_from_xz_yaw(xz, torch.zeros(160))
    generated = route.clone()
    current_frame = 72

    short = compose_active_window_segment(
        route,
        generated,
        current_frame=current_frame,
        current_yaw=torch.tensor(0.0),
        target_end_frame=130,
        min_route_index=current_frame,
        lookahead_m=0.20,
        bridge_frames=8,
    )
    long = compose_active_window_segment(
        route,
        generated,
        current_frame=current_frame,
        current_yaw=torch.tensor(0.0),
        target_end_frame=130,
        min_route_index=current_frame,
        lookahead_m=0.20,
        bridge_frames=24,
    )

    assert short.bridge_frames >= 8
    assert long.bridge_frames >= 24
    assert long.bridge_frames > short.bridge_frames



def test_active_window_segment_rebases_future_route_instead_of_snapping_back():
    route_xz = torch.stack([torch.zeros(180), torch.linspace(0.0, 9.0, 180)], dim=-1)
    route = _traj7_from_xz_yaw(route_xz, torch.zeros(180))
    generated = route.clone()
    generated[80:, 0] += 2.0
    generated[80:, 2] += 0.35

    result = compose_active_window_segment(
        route,
        generated,
        current_frame=90,
        current_yaw=torch.tensor(0.0),
        target_end_frame=150,
        min_route_index=90,
        lookahead_m=0.25,
        bridge_frames=12,
    )

    xz = result.segment_traj7[:, [0, 2]]
    speed = torch.linalg.norm(xz[1:] - xz[:-1], dim=-1)
    # The update should continue from the generated pose and preserve route-like
    # speed, not spend the bridge snapping laterally back to the offline route.
    assert torch.allclose(xz[0], generated[90, [0, 2]], atol=1e-6)
    assert int(result.segment_traj7.shape[0]) == 150 - 90 + 1
    assert float(torch.abs(xz[:20, 0] - generated[90, 0]).max().item()) < 0.15
    assert float(speed[:20].max().item()) < 0.12


def test_active_window_world_condition_keeps_render_and_feedback_route_continuous():
    route_xz = torch.stack([torch.zeros(320), torch.linspace(0.0, 16.0, 320)], dim=-1)
    route = _traj7_from_xz_yaw(route_xz, torch.zeros(320))
    generated = route.clone()
    current_frame = 236
    generated[: current_frame + 1, 0] += 1.2
    segment = compose_active_window_segment(
        route,
        generated[: current_frame + 1],
        current_frame=current_frame,
        current_yaw=torch.tensor(0.0),
        target_end_frame=319,
        min_route_index=current_frame,
        lookahead_m=0.25,
        bridge_frames=12,
    )

    legacy_patched = route.clone()
    legacy_patched[current_frame:] = segment.segment_traj7[: route.shape[0] - current_frame]
    legacy_speed = torch.linalg.norm(
        legacy_patched[1:, [0, 2]] - legacy_patched[:-1, [0, 2]], dim=-1
    )
    assert float(legacy_speed[current_frame - 1].item()) > 1.0

    world_condition = compose_active_window_world_condition(
        route,
        generated[: current_frame + 1],
        segment,
        current_frame=current_frame,
    )
    speed = torch.linalg.norm(
        world_condition[1:, [0, 2]] - world_condition[:-1, [0, 2]], dim=-1
    )

    assert torch.allclose(
        world_condition[: current_frame + 1, [0, 2]],
        generated[: current_frame + 1, [0, 2]],
        atol=1e-6,
    )
    assert torch.allclose(
        world_condition[current_frame, [0, 2]],
        world_condition[current_frame + 1, [0, 2]],
        atol=0.08,
    )
    assert float(speed.max().item()) < 0.12
    recomputed = build_physical_7d_from_5d(world_condition[:, :5])
    assert torch.allclose(world_condition[:, 5:7], recomputed[:, 5:7], atol=1e-6)


def test_active_window_world_condition_recomputes_delta_after_patch():
    route_xz = torch.stack([torch.zeros(80), torch.linspace(0.0, 4.0, 80)], dim=-1)
    route = _traj7_from_xz_yaw(route_xz, torch.zeros(80))
    generated = route.clone()
    current_frame = 20
    generated[: current_frame + 1, 0] += 0.4
    segment = compose_active_window_segment(
        route,
        generated[: current_frame + 1],
        current_frame=current_frame,
        current_yaw=torch.tensor(0.0),
        target_end_frame=60,
        min_route_index=current_frame,
        lookahead_m=0.20,
        bridge_frames=8,
    )
    corrupted = segment.segment_traj7.clone()
    corrupted[:, 5:7] = 123.0
    corrupted_segment = type(segment)(
        segment_traj7=corrupted,
        route_index=segment.route_index,
        future_index=segment.future_index,
        current_frame=segment.current_frame,
        bridge_frames=segment.bridge_frames,
    )

    world_condition = compose_active_window_world_condition(
        route,
        generated[: current_frame + 1],
        corrupted_segment,
        current_frame=current_frame,
    )

    recomputed = build_physical_7d_from_5d(world_condition[:, :5])
    assert torch.allclose(world_condition[:, 5:7], recomputed[:, 5:7], atol=1e-6)


def test_diagnostics_rejects_unreasonable_runtime_condition_speed():
    route_xz = torch.stack([torch.zeros(180), torch.linspace(0.0, 9.0, 180)], dim=-1)
    route = _traj7_from_xz_yaw(route_xz, torch.zeros(180))
    generated = route.clone()
    generated[80:, 0] += 2.0
    generated[80:, 2] += 0.35
    bad_segment = route[90:151].clone()
    bad_segment[0, [0, 2]] = generated[90, [0, 2]]

    ok, issues = validate_trajectory_diagnostics(
        bad_segment,
        reference_traj7=route[90:151],
        max_speed_scale=2.5,
        max_abs_speed=0.20,
    )

    assert not ok
    assert any("speed" in item for item in issues)


def test_world_condition_payload_uses_absolute_active_window_frames():
    route_xz = torch.stack([torch.zeros(160), torch.linspace(0.0, 8.0, 160)], dim=-1)
    world = _traj7_from_xz_yaw(route_xz, torch.zeros(160))
    timeline = RootTimeline(
        RootFrameState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )
    for commit in range(1, 40):
        frame = commit * 4
        timeline.append(
            RootFrameState(
                commit_idx=commit,
                world_xz=world[min(frame, world.shape[0] - 1), [0, 2]].clone(),
                world_yaw=torch.tensor(0.0),
                source="test",
            )
        )

    payload = build_world_condition_stream_payload(
        world,
        timeline,
        local_commit_index=31,
        absolute_commit_index=31,
        chunk_size=5,
        history_length=30,
        traj_horizon_tokens=20,
        frames_per_token=4,
    )

    assert payload is not None
    assert payload["traj_abs_start_token"] == 2
    assert payload["body_anchor_abs_token"] == 2
    assert payload["traj_substep_payloads"]
    # abs token 2 starts at frame 5, so local payload frame 112 is abs frame 117; body anchor state is commit 2, frame 8 in this test timeline.
    assert torch.allclose(
        payload["traj_cond_7d_frame"][0, 112, [0, 2]],
        world[117, [0, 2]] - world[8, [0, 2]],
        atol=1e-5,
    )


def test_world_condition_payload_uses_token_range_frame_slice_for_start0():
    route_xz = torch.stack([torch.zeros(120), torch.arange(120, dtype=torch.float32)], dim=-1)
    world = _traj7_from_xz_yaw(route_xz, torch.zeros(120))
    timeline = RootTimeline(
        RootFrameState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )
    for commit in range(1, 40):
        frame = token_end_frame(commit - 1, 4)
        timeline.append(
            RootFrameState(
                commit_idx=commit,
                world_xz=world[min(frame, world.shape[0] - 1), [0, 2]].clone(),
                world_yaw=torch.tensor(0.0),
                source="test",
            )
        )

    payload = build_world_condition_stream_payload(
        world,
        timeline,
        local_commit_index=0,
        absolute_commit_index=0,
        chunk_size=2,
        history_length=30,
        traj_horizon_tokens=3,
        frames_per_token=4,
    )

    assert payload is not None
    frame_slice = token_range_to_frame_slice(0, payload["traj_num_tokens"], 4)
    assert payload["traj_cond_7d_frame"].shape[1] == frame_slice.stop - frame_slice.start
    assert payload["traj_cond_7d_frame"].shape[1] == 17




def test_world_condition_payload_uses_generated_history_before_current_commit():
    route_xz = torch.stack([torch.zeros(200), torch.linspace(0.0, 10.0, 200)], dim=-1)
    future_world = _traj7_from_xz_yaw(route_xz, torch.zeros(200))
    generated_world = future_world.clone()
    generated_world[:130, 0] += 1.5
    timeline = RootTimeline(
        RootFrameState.initial(xz=(1.5, 0.0), yaw=0.0, dtype=torch.float32)
    )
    for commit in range(1, 45):
        frame = commit * 4
        timeline.append(
            RootFrameState(
                commit_idx=commit,
                world_xz=generated_world[min(frame, generated_world.shape[0] - 1), [0, 2]].clone(),
                world_yaw=torch.tensor(0.0),
                source="generated",
            )
        )

    payload = build_world_condition_stream_payload(
        future_world,
        timeline,
        local_commit_index=31,
        absolute_commit_index=31,
        chunk_size=5,
        history_length=30,
        traj_horizon_tokens=20,
        frames_per_token=4,
        generated_history_traj7=generated_world[:128],
    )

    assert payload is not None
    local = payload["traj_cond_7d_frame"][0]
    anchor = timeline.at_commit(payload["body_anchor_abs_token"])
    from utils.local_frame import uncanonicalize_7d

    world_payload = uncanonicalize_7d(
        local.unsqueeze(0),
        anchor.world_xz.unsqueeze(0),
        anchor.world_yaw.reshape(1),
    )[0]
    # abs token 2 starts at frame 5; local frame 112 corresponds to abs frame 117,
    # which is generated history at commit 31 and must not come from the future route.
    assert torch.allclose(world_payload[112, [0, 2]], generated_world[117, [0, 2]], atol=1e-5)
    assert not torch.allclose(world_payload[112, [0, 2]], future_world[117, [0, 2]], atol=1e-3)


def test_replay_timeline_maps_commit_to_last_committed_token_end_frame():
    xz = torch.stack([torch.zeros(140), torch.arange(140, dtype=torch.float32)], dim=-1)
    generated = _traj7_from_xz_yaw(xz, torch.zeros(140))

    timeline = build_timeline_from_generated_traj7(generated, frames_per_token=4)

    assert torch.allclose(timeline.at_commit(1).world_xz, generated[0, [0, 2]])
    assert torch.allclose(timeline.at_commit(31).world_xz, generated[120, [0, 2]])
    assert not torch.allclose(timeline.at_commit(31).world_xz, generated[123, [0, 2]])


def test_payload_builder_keeps_substep_payloads_with_history0_anchor():
    route_xz = torch.stack([torch.zeros(100), torch.linspace(0.0, 5.0, 100)], dim=-1)
    route = _traj7_from_xz_yaw(route_xz, torch.zeros(100))
    timeline = RootTimeline(
        RootFrameState.initial(
            xz=(0.0, 0.0),
            yaw=0.0,
            dtype=torch.float32,
        )
    )
    for commit in range(1, 40):
        frame = commit * 4
        timeline.append(
            RootFrameState(
                commit_idx=commit,
                world_xz=route[min(frame, route.shape[0] - 1), [0, 2]].clone(),
                world_yaw=torch.tensor(0.0),
                source="test",
            )
        )

    anchor = timeline.at_commit(20)
    plan = build_active_window_root_plan(
        route[80:],
        anchor_state=anchor,
        anchor_commit_idx=20,
        token_dt=0.20,
        frames_per_token=4,
    )
    payload = build_active_window_stream_payload(
        plan,
        timeline,
        local_commit_index=20,
        absolute_commit_index=20,
        chunk_size=4,
        history_length=30,
        traj_horizon_tokens=20,
    )

    assert payload is not None
    assert payload["body_anchor_abs_token"] == payload["traj_abs_start_token"]
    assert payload["traj_substep_payloads"]
    assert all(
        sub["body_anchor_abs_token"] == sub["traj_abs_start_token"]
        for sub in payload["traj_substep_payloads"]
    )
