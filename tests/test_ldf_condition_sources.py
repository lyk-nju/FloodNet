import math

import torch

from eval.ldf.experiments.condition_sources import (
    ConditionScenario,
    build_repeat_splice_condition,
    build_sample_condition,
    build_synthetic_condition,
    compose_constant_curvature_arc_traj7,
    compose_rotated_suffix_arc_updated_traj7,
    compose_rotated_suffix_updated_traj7,
)
from utils.motion_process import build_physical_7d_from_5d


def _traj7_from_xz_yaw(xz: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    y = torch.zeros(xz.shape[0], dtype=xz.dtype, device=xz.device)
    traj5 = torch.stack(
        [xz[:, 0], y, xz[:, 1], torch.cos(yaw), torch.sin(yaw)],
        dim=-1,
    )
    return build_physical_7d_from_5d(traj5)


def _xz_delta_yaw(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(delta[..., 0], delta[..., 1])


def _abs_angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs()


def test_sample_condition_returns_original_valid_prefix():
    xz = torch.tensor(
        [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [9.0, 9.0]],
        dtype=torch.float32,
    )
    yaw = torch.zeros(4, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    scenario = build_sample_condition(
        traj7,
        valid_frames=3,
        sample_name="000021",
        caption_index=1,
    )

    assert isinstance(scenario, ConditionScenario)
    assert scenario.name == "sample"
    assert scenario.base_sample_name == "000021"
    assert scenario.caption_index == 1
    assert scenario.update_frames == []
    assert torch.allclose(scenario.condition_traj7, traj7[:3], atol=1e-6)


def test_repeat_splice_center_symmetric_preserves_prefix_heading():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.4, 0.1],
            [0.6, 0.3],
            [0.7, 0.6],
        ],
        dtype=torch.float32,
    )
    prefix_yaw = torch.tensor(
        [0.0, math.pi / 4.0, math.pi / 2.0, -math.pi / 3.0, math.pi / 6.0],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, prefix_yaw)

    scenario = build_repeat_splice_condition(
        traj7,
        mode="center_symmetric",
        update_frame=3,
        source_start_frame=0,
        suffix_frames=4,
        derive_heading_from_path=True,
        sample_name="000021",
    )

    assert scenario.name == "repeat_splice:center_symmetric"
    assert scenario.update_frames == [3]
    assert torch.allclose(scenario.condition_traj7[:3, 3:5], traj7[:3, 3:5], atol=1e-6)
    assert int(scenario.condition_traj7.shape[0]) > int(traj7.shape[0])


def test_rotated_suffix_can_turn_straight_history_by_right_angle():
    xz = torch.stack(
        [torch.zeros(6), torch.arange(6, dtype=torch.float32)],
        dim=-1,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(6))

    updated = compose_rotated_suffix_updated_traj7(
        traj7,
        update_frame=3,
        source_start_frame=0,
        suffix_rotation_deg=90.0,
        turn_transition_frames=0,
    )

    out_xz = updated[:, [0, 2]]
    assert torch.allclose(out_xz[3], torch.tensor([0.0, 3.0]), atol=1e-6)
    assert torch.allclose(out_xz[4], torch.tensor([1.0, 3.0]), atol=1e-5)
    assert torch.allclose(out_xz[5], torch.tensor([2.0, 3.0]), atol=1e-5)
    suffix_yaw = torch.atan2(updated[3:6, 4], updated[3:6, 3])
    assert torch.allclose(
        suffix_yaw,
        torch.full_like(suffix_yaw, math.pi / 2.0),
        atol=1e-5,
    )


def test_rotated_suffix_turn_transition_progressively_bends_path():
    xz = torch.stack(
        [torch.zeros(10), torch.arange(10, dtype=torch.float32)],
        dim=-1,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(10))

    updated = compose_rotated_suffix_updated_traj7(
        traj7,
        update_frame=6,
        source_start_frame=0,
        suffix_rotation_deg=90.0,
        turn_transition_frames=3,
    )

    out_xz = updated[:, [0, 2]]
    first_suffix_delta = out_xz[7] - out_xz[6]
    middle_suffix_delta = out_xz[8] - out_xz[7]
    late_suffix_delta = out_xz[-1] - out_xz[-2]
    assert abs(float(first_suffix_delta[0])) < 1e-5
    assert float(first_suffix_delta[1]) > 0.9
    assert float(middle_suffix_delta[0]) > 0.0
    assert float(middle_suffix_delta[1]) > 0.0
    assert float(late_suffix_delta[0]) > 0.9
    assert abs(float(late_suffix_delta[1])) < 1e-4


def test_rotated_suffix_preserves_original_prefix_heading_before_update():
    xz = torch.stack(
        [torch.linspace(0.0, 0.9, 10), torch.arange(10, dtype=torch.float32)],
        dim=-1,
    )
    prefix_yaw = torch.tensor(
        [0.0, 0.3, -0.2, 0.7, -0.6, 0.1, 0.4, -0.1, 0.2, 0.0],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, prefix_yaw)

    updated = compose_rotated_suffix_updated_traj7(
        traj7,
        update_frame=6,
        source_start_frame=0,
        suffix_rotation_deg=45.0,
        turn_transition_frames=3,
    )

    assert torch.allclose(updated[:6, [0, 2]], traj7[:6, [0, 2]], atol=1e-6)
    assert torch.allclose(updated[:6, 3:5], traj7[:6, 3:5], atol=1e-6)


def test_repeat_splice_rotated_suffix_records_rotation_metadata():
    xz = torch.stack(
        [torch.zeros(8), torch.arange(8, dtype=torch.float32)],
        dim=-1,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(8))

    scenario = build_repeat_splice_condition(
        traj7,
        mode="rotated_suffix",
        update_frame=5,
        source_start_frame=1,
        suffix_rotation_deg=45.0,
        turn_transition_frames=4,
    )

    assert scenario.name == "repeat_splice:rotated_suffix"
    assert scenario.update_frames == [5]
    assert scenario.metadata["suffix_rotation_deg"] == 45.0
    assert scenario.metadata["turn_transition_frames"] == 4


def test_rotated_suffix_arc_preserves_prefix_and_turns_with_forward_arc():
    xz = torch.stack(
        [torch.zeros(10), torch.arange(10, dtype=torch.float32)],
        dim=-1,
    )
    prefix_yaw = torch.tensor(
        [0.0, 0.2, -0.1, 0.4, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, prefix_yaw)

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=6,
        source_start_frame=0,
        suffix_rotation_deg=90.0,
        turn_transition_frames=4,
        speed_window=4,
    )

    out_xz = updated[:, [0, 2]]
    assert torch.allclose(out_xz[:6], traj7[:6, [0, 2]], atol=1e-6)
    assert torch.allclose(updated[:6, 3:5], traj7[:6, 3:5], atol=1e-6)

    first_arc_delta = out_xz[7] - out_xz[6]
    middle_arc_delta = out_xz[9] - out_xz[8]
    post_arc_delta = out_xz[11] - out_xz[10]
    assert abs(float(first_arc_delta[0])) < 1e-5
    assert float(first_arc_delta[1]) > 0.9
    assert float(middle_arc_delta[0]) > 0.0
    assert float(middle_arc_delta[1]) > 0.0
    assert float(post_arc_delta[0]) > 0.9
    assert abs(float(post_arc_delta[1])) < 1e-4


def test_rotated_suffix_arc_blends_low_speed_source_boundary():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.001, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, 5.0],
            [0.0, 6.0],
            [0.0, 7.0],
            [0.0, 8.0],
            [0.0, 9.0],
        ],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(xz.shape[0], dtype=torch.float32))

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=8,
        source_start_frame=0,
        suffix_rotation_deg=90.0,
        turn_transition_frames=4,
        speed_window=4,
        suffix_blend_frames=4,
        suffix_min_speed_factor=0.25,
    )

    out_xz = updated[:, [0, 2]]
    delta = out_xz[1:] - out_xz[:-1]
    boundary = 8 + 4
    last_arc_delta = delta[boundary - 1]
    first_suffix_delta = delta[boundary]
    yaw_jump = _abs_angle_diff(
        _xz_delta_yaw(first_suffix_delta),
        _xz_delta_yaw(last_arc_delta),
    )
    speed_ratio = torch.linalg.norm(first_suffix_delta) / torch.linalg.norm(
        last_arc_delta
    )

    assert float(yaw_jump) < math.radians(5.0)
    assert float(speed_ratio) > 0.75


def test_rotated_suffix_arc_keeps_late_suffix_on_target_turn_direction():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, 5.0],
            [-0.5, 6.0],
            [-1.0, 7.0],
            [-1.5, 8.0],
            [-2.0, 9.0],
        ],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(xz.shape[0], dtype=torch.float32))
    update_frame = 8
    prefix_delta = xz[update_frame - 4 + 1:update_frame + 1] - xz[
        update_frame - 4:update_frame
    ]
    prefix_yaw = _xz_delta_yaw(prefix_delta.mean(dim=0))
    target_yaw = prefix_yaw + torch.tensor(math.radians(45.0))

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=0,
        suffix_rotation_deg=45.0,
        turn_transition_frames=4,
        speed_window=4,
        suffix_blend_frames=4,
        suffix_min_speed_factor=0.25,
    )

    out_xz = updated[:, [0, 2]]
    delta = out_xz[1:] - out_xz[:-1]
    late_delta = delta[-4:].mean(dim=0)
    late_yaw = _xz_delta_yaw(late_delta)

    assert float(_abs_angle_diff(late_yaw, target_yaw)) < math.radians(5.0)


def test_rotated_suffix_arc_preserves_repeat_shape_after_smooth_segment():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.1, 2.0],
            [0.4, 3.0],
            [0.7, 3.8],
            [1.1, 4.4],
            [1.4, 4.9],
            [1.6, 5.3],
            [1.7, 5.6],
        ],
        dtype=torch.float32,
    )
    source_yaw = torch.linspace(0.1, 0.8, xz.shape[0])
    traj7 = _traj7_from_xz_yaw(xz, source_yaw)
    update_frame = 6
    source_start = 2
    transition_frames = 3
    rotation_rad = torch.tensor(math.radians(45.0))

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=source_start,
        suffix_rotation_deg=45.0,
        turn_transition_frames=transition_frames,
        speed_window=3,
        suffix_blend_frames=4,
        suffix_min_speed_factor=0.0,
    )

    out_xz = updated[:, [0, 2]]
    repeat_start = update_frame + transition_frames
    repeat_delta = out_xz[repeat_start + 1:] - out_xz[repeat_start:-1]
    source_delta = xz[source_start + 1:update_frame + 1] - xz[
        source_start:update_frame
    ]
    expected_delta = torch.stack(
        [
            torch.cos(rotation_rad) * source_delta[:, 0]
            + torch.sin(rotation_rad) * source_delta[:, 1],
            -torch.sin(rotation_rad) * source_delta[:, 0]
            + torch.cos(rotation_rad) * source_delta[:, 1],
        ],
        dim=-1,
    )
    repeat_yaw = torch.atan2(
        updated[repeat_start + 1:repeat_start + (update_frame - source_start + 1), 4],
        updated[repeat_start + 1:repeat_start + (update_frame - source_start + 1), 3],
    )

    assert torch.allclose(repeat_delta, expected_delta, atol=1e-5)
    assert torch.allclose(
        repeat_yaw,
        source_yaw[source_start + 1:update_frame + 1] + rotation_rad,
        atol=1e-5,
    )


def test_rotated_suffix_arc_absorbs_low_speed_repeat_start_into_smooth():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.001, 0.0],
            [0.002, 0.0],
            [0.0, 1.0],
            [0.2, 2.0],
            [0.5, 3.0],
            [0.7, 4.0],
            [0.8, 5.0],
            [0.9, 6.0],
        ],
        dtype=torch.float32,
    )
    source_yaw = torch.zeros(xz.shape[0], dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, source_yaw)
    update_frame = 7
    transition_frames = 4
    rotation_rad = torch.tensor(math.radians(45.0))

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=0,
        suffix_rotation_deg=45.0,
        turn_transition_frames=transition_frames,
        speed_window=3,
        suffix_min_speed_factor=0.25,
    )

    repeat_start = update_frame + transition_frames
    repeat_delta = updated[repeat_start + 1:, [0, 2]] - updated[
        repeat_start:-1, [0, 2]
    ]
    source_delta = xz[1:update_frame + 1] - xz[:update_frame]
    expected_source_delta = source_delta[2:]
    expected_delta = torch.stack(
        [
            torch.cos(rotation_rad) * expected_source_delta[:, 0]
            + torch.sin(rotation_rad) * expected_source_delta[:, 1],
            -torch.sin(rotation_rad) * expected_source_delta[:, 0]
            + torch.cos(rotation_rad) * expected_source_delta[:, 1],
        ],
        dim=-1,
    )

    assert repeat_delta.shape == expected_delta.shape
    assert torch.allclose(repeat_delta, expected_delta, atol=1e-5)


def test_rotated_suffix_arc_smooth_matches_first_repeat_xz_tangent():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.7, 3.3],
            [0.7, 4.3],
            [0.7, 5.3],
            [0.7, 6.3],
        ],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(xz.shape[0], dtype=torch.float32))
    update_frame = 6
    transition_frames = 5

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=2,
        suffix_rotation_deg=45.0,
        turn_transition_frames=transition_frames,
        speed_window=4,
        suffix_min_speed_factor=0.0,
    )

    repeat_start = update_frame + transition_frames
    delta = updated[1:, [0, 2]] - updated[:-1, [0, 2]]
    last_smooth_yaw = _xz_delta_yaw(delta[repeat_start - 1])
    first_repeat_yaw = _xz_delta_yaw(delta[repeat_start])

    assert float(_abs_angle_diff(last_smooth_yaw, first_repeat_yaw)) < math.radians(1.0)


def test_rotated_suffix_arc_can_blend_repeat_heading_seam_without_moving_xz():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.7, 3.3],
            [0.7, 4.3],
            [0.7, 5.3],
            [0.7, 6.3],
        ],
        dtype=torch.float32,
    )
    source_yaw = torch.full((xz.shape[0],), math.radians(-70.0))
    traj7 = _traj7_from_xz_yaw(xz, source_yaw)
    update_frame = 6
    transition_frames = 5

    baseline = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=2,
        suffix_rotation_deg=45.0,
        turn_transition_frames=transition_frames,
        speed_window=4,
        suffix_min_speed_factor=0.0,
    )
    blended = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=2,
        suffix_rotation_deg=45.0,
        turn_transition_frames=transition_frames,
        speed_window=4,
        suffix_min_speed_factor=0.0,
        repeat_heading_blend_frames=4,
    )

    repeat_start = update_frame + transition_frames
    baseline_yaw = torch.atan2(baseline[:, 4], baseline[:, 3])
    blended_yaw = torch.atan2(blended[:, 4], blended[:, 3])
    baseline_jump = _abs_angle_diff(
        baseline_yaw[repeat_start + 1],
        baseline_yaw[repeat_start],
    )
    blended_jump = _abs_angle_diff(
        blended_yaw[repeat_start + 1],
        blended_yaw[repeat_start],
    )

    assert torch.allclose(baseline[:, [0, 1, 2]], blended[:, [0, 1, 2]], atol=1e-6)
    assert float(baseline_jump) > math.radians(20.0)
    assert float(blended_jump) < math.radians(1.0)


def test_rotated_suffix_arc_heading_residual_profile_uses_stable_heading_at_low_speed_update():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.3],
            [0.0, 0.6],
            [0.0, 0.9],
            [0.0, 1.2],
            [0.0, 1.5],
            [0.0, 1.8],
            [0.0, 2.1],
            [0.0, 2.4],
            [0.0, 2.7],
            [0.0, 3.0],
            [0.006, 3.0],  # low-speed sideways tail would corrupt path tangent
        ],
        dtype=torch.float32,
    )
    yaw = torch.zeros(xz.shape[0], dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)
    update_frame = 10
    transition_frames = 6

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=2,
        suffix_rotation_deg=45.0,
        turn_transition_frames=transition_frames,
        speed_window=4,
        suffix_min_speed_factor=0.0,
        arc_smooth_profile="heading_residual",
    )

    smooth_yaw = torch.atan2(updated[update_frame, 4], updated[update_frame, 3])
    assert float(_abs_angle_diff(smooth_yaw, torch.tensor(0.0))) < math.radians(1.0)


def test_rotated_suffix_arc_heading_residual_profile_preserves_reference_residual_in_smooth():
    frames = 30
    z = torch.arange(frames, dtype=torch.float32) * 0.2
    xz = torch.stack([torch.zeros(frames), z], dim=-1)
    residual = torch.deg2rad(
        torch.tensor(
            [
                0.0,
                8.0,
                0.0,
                -8.0,
                0.0,
                8.0,
                0.0,
                -8.0,
                0.0,
                8.0,
                0.0,
                -8.0,
                0.0,
                8.0,
                0.0,
                -8.0,
                0.0,
                8.0,
                0.0,
                -8.0,
                0.0,
                8.0,
                0.0,
                -8.0,
                0.0,
                8.0,
                0.0,
                -8.0,
                0.0,
                8.0,
            ],
            dtype=torch.float32,
        )
    )
    traj7 = _traj7_from_xz_yaw(xz, residual)
    update_frame = 22
    transition_frames = 12

    geometric = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=4,
        suffix_rotation_deg=30.0,
        turn_transition_frames=transition_frames,
        speed_window=8,
        suffix_min_speed_factor=0.0,
    )
    residual_profile = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=4,
        suffix_rotation_deg=30.0,
        turn_transition_frames=transition_frames,
        speed_window=8,
        suffix_min_speed_factor=0.0,
        arc_smooth_profile="heading_residual",
    )

    smooth_slice = slice(update_frame, update_frame + transition_frames)
    geometric_delta = geometric[1:, [0, 2]] - geometric[:-1, [0, 2]]
    residual_delta = residual_profile[1:, [0, 2]] - residual_profile[:-1, [0, 2]]
    geometric_tangent = _xz_delta_yaw(geometric_delta[smooth_slice])
    residual_tangent = _xz_delta_yaw(residual_delta[smooth_slice])
    geometric_yaw = torch.atan2(
        geometric[smooth_slice, 4],
        geometric[smooth_slice, 3],
    )
    residual_yaw = torch.atan2(
        residual_profile[smooth_slice, 4],
        residual_profile[smooth_slice, 3],
    )

    geometric_mismatch = _abs_angle_diff(geometric_yaw, geometric_tangent)
    residual_mismatch = _abs_angle_diff(residual_yaw, residual_tangent)
    assert float(geometric_mismatch.max()) < math.radians(0.1)
    assert float(residual_mismatch.max()) > math.radians(5.0)


def test_rotated_suffix_arc_smooth_uses_tail_average_and_first_repeat_xz_tangents():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [2.0, 5.0],
            [2.0, 6.0],
        ],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(xz.shape[0], dtype=torch.float32))
    update_frame = 7
    source_start = 2
    transition_frames = 5
    speed_window = 3
    rotation_rad = torch.tensor(math.radians(45.0))

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=source_start,
        suffix_rotation_deg=45.0,
        turn_transition_frames=transition_frames,
        speed_window=speed_window,
        suffix_min_speed_factor=0.0,
    )

    repeat_start = update_frame + transition_frames
    delta = updated[1:, [0, 2]] - updated[:-1, [0, 2]]
    source_delta = xz[source_start + 1:update_frame + 1] - xz[
        source_start:update_frame
    ]
    rotated_source_delta = torch.stack(
        [
            torch.cos(rotation_rad) * source_delta[:, 0]
            + torch.sin(rotation_rad) * source_delta[:, 1],
            -torch.sin(rotation_rad) * source_delta[:, 0]
            + torch.cos(rotation_rad) * source_delta[:, 1],
        ],
        dim=-1,
    )
    expected_start = (xz[update_frame - speed_window + 1:update_frame + 1]
                      - xz[update_frame - speed_window:update_frame]).mean(dim=0)
    expected_end = rotated_source_delta[0]

    start_delta = delta[update_frame]
    end_delta = delta[repeat_start - 1]

    assert float(
        _abs_angle_diff(_xz_delta_yaw(start_delta), _xz_delta_yaw(expected_start))
    ) < math.radians(0.1)
    assert float(
        _abs_angle_diff(_xz_delta_yaw(end_delta), _xz_delta_yaw(expected_end))
    ) < math.radians(0.1)
    assert float(torch.linalg.norm(start_delta) / torch.linalg.norm(expected_start)) > 0.99
    assert float(torch.linalg.norm(end_delta) / torch.linalg.norm(expected_end)) > 0.99


def test_rotated_suffix_arc_resamples_smooth_bridge_by_speed_ramp():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.30],
            [0.0, 0.60],
            [0.0, 0.90],
            [0.0, 1.20],
            [0.0, 1.50],
            [0.0, 1.54],
            [0.0, 1.58],
            [0.0, 1.62],
            [0.0, 1.66],
            [0.0, 1.70],
            [0.0, 1.74],
        ],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(xz.shape[0], dtype=torch.float32))
    update_frame = 10
    transition_frames = 10

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=0,
        suffix_rotation_deg=45.0,
        turn_transition_frames=transition_frames,
        speed_window=3,
        suffix_min_speed_factor=0.0,
    )

    smooth_xz = updated[
        update_frame:update_frame + transition_frames + 1,
        [0, 2],
    ]
    step_lengths = torch.linalg.norm(smooth_xz[1:] - smooth_xz[:-1], dim=-1)

    start_speed = float(step_lengths[:3].mean())
    end_speed = float(step_lengths[-3:].mean())
    assert end_speed > start_speed * 2.0


def test_rotated_suffix_arc_auto_transition_frames_use_path_length_and_side_speeds():
    frame_idx = torch.arange(90, dtype=torch.float32)
    xz = torch.stack([frame_idx * 0.003, frame_idx.square() * 0.0005], dim=-1)
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(xz.shape[0], dtype=torch.float32))
    update_frame = 70
    source_start = 20
    fixed_probe_frames = 48
    speed_window = 10

    probe = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=source_start,
        suffix_rotation_deg=45.0,
        turn_transition_frames=fixed_probe_frames,
        speed_window=speed_window,
        suffix_min_speed_factor=0.0,
    )
    probe_xz = probe[update_frame:update_frame + fixed_probe_frames + 1, [0, 2]]
    probe_length = torch.linalg.norm(probe_xz[1:] - probe_xz[:-1], dim=-1).sum()
    tail = xz[update_frame - speed_window:update_frame + 1]
    tail_speed = torch.linalg.norm(tail[1:] - tail[:-1], dim=-1).mean()
    source_delta = xz[source_start + 1:update_frame + 1] - xz[
        source_start:update_frame
    ]
    repeat_head_speed = torch.linalg.norm(
        source_delta[:speed_window],
        dim=-1,
    ).mean()
    target_speed = 0.5 * (tail_speed + repeat_head_speed)
    expected_frames = int(round(float(probe_length / target_speed)))
    expected_frames = max(12, min(expected_frames, 80))

    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=source_start,
        suffix_rotation_deg=45.0,
        turn_transition_frames=-1,
        speed_window=speed_window,
        suffix_min_speed_factor=0.0,
    )

    expected_total = update_frame + expected_frames + 1 + (update_frame - source_start)
    assert int(updated.shape[0]) == expected_total


def test_repeat_splice_rotated_suffix_arc_records_arc_metadata():
    xz = torch.stack(
        [torch.zeros(8), torch.arange(8, dtype=torch.float32)],
        dim=-1,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(8))

    scenario = build_repeat_splice_condition(
        traj7,
        mode="rotated_suffix_arc",
        update_frame=5,
        source_start_frame=1,
        suffix_rotation_deg=45.0,
        turn_transition_frames=4,
        arc_speed_window=3,
    )

    assert scenario.name == "repeat_splice:rotated_suffix_arc"
    assert scenario.update_frames == [5]
    assert scenario.metadata["suffix_rotation_deg"] == 45.0
    assert scenario.metadata["turn_transition_frames"] == 4
    assert scenario.metadata["arc_speed_window"] == 3


def test_rotated_suffix_arc_geometric_aligns_smooth_to_first_repeat_tangent():
    positions = []
    pos = torch.zeros(2, dtype=torch.float32)
    for frame in range(70):
        positions.append(pos.clone())
        if frame == 20:
            delta = torch.tensor([0.04, 0.0], dtype=torch.float32)
        else:
            delta = torch.tensor([0.0, 0.04], dtype=torch.float32)
        pos = pos + delta
    xz = torch.stack(positions)
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(70, dtype=torch.float32))

    update_frame = 60
    transition_frames = 12
    repeat_start = update_frame + transition_frames
    updated = compose_rotated_suffix_arc_updated_traj7(
        traj7,
        update_frame=update_frame,
        source_start_frame=20,
        suffix_rotation_deg=0.0,
        turn_transition_frames=transition_frames,
        speed_window=10,
        suffix_min_speed_factor=0.0,
    )

    smooth_last_delta = (
        updated[repeat_start, [0, 2]] - updated[repeat_start - 1, [0, 2]]
    )
    repeat_first_delta = (
        updated[repeat_start + 1, [0, 2]] - updated[repeat_start, [0, 2]]
    )
    seam_angle = _abs_angle_diff(
        _xz_delta_yaw(smooth_last_delta),
        _xz_delta_yaw(repeat_first_delta),
    )
    assert seam_angle <= math.radians(1.0)


def test_repeat_splice_rotated_suffix_arc_chain_records_each_update():
    xz = torch.stack(
        [torch.zeros(80), torch.arange(80, dtype=torch.float32) * 0.04],
        dim=-1,
    )
    traj7 = _traj7_from_xz_yaw(xz, torch.zeros(80, dtype=torch.float32))

    scenario = build_repeat_splice_condition(
        traj7,
        mode="rotated_suffix_arc_chain",
        update_frame=60,
        source_start_frame=30,
        turn_transition_frames=-1,
        arc_speed_window=10,
        suffix_min_speed_factor=0.0,
        repeat_count=3,
        repeat_angle_choices=(30.0, 60.0),
        repeat_seed=123,
    )

    assert scenario.name == "repeat_splice:rotated_suffix_arc_chain"
    assert len(scenario.update_frames) == 3
    assert scenario.update_frames == sorted(scenario.update_frames)
    assert len(scenario.metadata["chain_rotation_degrees"]) == 3
    assert set(scenario.metadata["chain_rotation_degrees"]).issubset({30.0, 60.0})
    assert len(scenario.metadata["chain_repeat_start_frames"]) == 3
    assert len(scenario.metadata["chain_transition_frames"]) == 3
    for update, repeat_start in zip(
        scenario.update_frames,
        scenario.metadata["chain_repeat_start_frames"],
    ):
        assert repeat_start > update
    assert int(scenario.condition_traj7.shape[0]) > int(traj7.shape[0])


def test_rotated_suffix_arc_chain_anchor_y_mode_prevents_repeated_y_drift():
    frame_idx = torch.arange(80, dtype=torch.float32)
    xz = torch.stack(
        [torch.zeros_like(frame_idx), frame_idx * 0.04],
        dim=-1,
    )
    y = 1.0 - frame_idx * 0.003
    yaw = torch.zeros_like(frame_idx)
    traj5 = torch.stack(
        [xz[:, 0], y, xz[:, 1], torch.cos(yaw), torch.sin(yaw)],
        dim=-1,
    )
    traj7 = build_physical_7d_from_5d(traj5)
    update_frame = 60

    scenario = build_repeat_splice_condition(
        traj7,
        mode="rotated_suffix_arc_chain",
        update_frame=update_frame,
        source_start_frame=30,
        suffix_rotation_deg=20.0,
        turn_transition_frames=12,
        arc_speed_window=10,
        suffix_min_speed_factor=0.0,
        repeat_count=3,
        repeat_angle_choices=(20.0,),
        arc_y_mode="anchor",
    )

    anchor_y = traj7[update_frame, 1]
    after_update_y = scenario.condition_traj7[update_frame:, 1]

    assert torch.allclose(after_update_y, anchor_y.expand_as(after_update_y), atol=1e-6)
    assert scenario.metadata["arc_y_mode"] == "anchor"


def test_synthetic_condition_can_mark_multiple_updates():
    scenario = build_synthetic_condition(
        preset="four_segment_curve",
        num_frames=120,
        update_frames=[30, 60, 90],
        sample_name="001168",
        caption_index=0,
    )

    assert scenario.name == "synthetic:four_segment_curve"
    assert scenario.base_sample_name == "001168"
    assert scenario.update_frames == [30, 60, 90]
    assert scenario.condition_traj7.shape == (120, 7)
    assert torch.all(scenario.condition_traj7[1:, 2] > scenario.condition_traj7[:-1, 2])


def test_constant_curvature_arc_has_uniform_turn_and_step_length():
    traj7 = compose_constant_curvature_arc_traj7(
        num_frames=101,
        arc_length=10.0,
        turn_degrees=30.0,
    )

    yaw = torch.atan2(traj7[:, 4], traj7[:, 3])
    yaw_step = yaw[1:] - yaw[:-1]
    xz_step = torch.linalg.norm(
        traj7[1:, [0, 2]] - traj7[:-1, [0, 2]],
        dim=-1,
    )

    assert traj7.shape == (101, 7)
    assert torch.allclose(yaw_step, yaw_step.mean().expand_as(yaw_step), atol=1e-6)
    assert torch.allclose(xz_step, xz_step.mean().expand_as(xz_step), atol=1e-5)
    assert float(traj7[-1, 0]) > 0.0
    assert float(traj7[-1, 2]) > 0.0


def test_synthetic_constant_arc_condition_metadata_records_turn():
    scenario = build_synthetic_condition(
        preset="constant_arc",
        num_frames=80,
        update_frames=[20, 40],
        total_forward=6.0,
        arc_turn_degrees=15.0,
        sample_name="001168",
    )

    assert scenario.name == "synthetic:constant_arc"
    assert scenario.condition_traj7.shape == (80, 7)
    assert scenario.update_frames == [20, 40]
    assert scenario.metadata["arc_turn_degrees"] == 15.0
