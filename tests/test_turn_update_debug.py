import math

import torch
import tools.run_stream_turn_update_debug as turn_debug

from tools.run_stream_turn_update_debug import (
    _build_second_source_from_anchor_boundary,
    _build_segment_source_from_runtime_anchor,
    _compose_rootrefiner_condition_traj7,
    _jsonify_debug_record,
    _mask_rootrefiner_condition_front,
    _rootrefiner_input_path_world_xz,
    _resolve_update_anchor_yaw_state,
    _path_yaw_from_xz,
    _override_heading_from_path_tangent,
    _compose_multi_rootrefiner_condition_traj7,
    apply_updated_traj_to_sample_batch,
    compose_anchor_local_updated_traj7,
    compose_center_symmetric_updated_traj7,
    compose_clean_s_curve_updated_traj7,
    compose_constant_arc_traj7,
    compose_four_segment_forward_curve_traj7,
    compose_forward_line_traj7,
    compose_legacy_s_turn_updated_traj7,
    compose_turn180_updated_traj7,
    condition_visual_mask,
)
from utils.conditions.root_refiner import RootRefinerPathCondition
from utils.inference.timeline import RootFrameState
from utils.local_frame import wrap_angle
from utils.motion_process import build_physical_7d_from_5d


def _line_traj7(num_frames: int) -> torch.Tensor:
    x = torch.arange(num_frames, dtype=torch.float32)
    y = torch.zeros(num_frames, dtype=torch.float32)
    z = torch.zeros(num_frames, dtype=torch.float32)
    yaw = torch.full((num_frames,), math.pi / 2.0, dtype=torch.float32)
    traj5 = torch.stack([x, y, z, torch.cos(yaw), torch.sin(yaw)], dim=-1)
    return build_physical_7d_from_5d(traj5)


def test_compose_constant_arc_uses_start_pose_and_uniform_turn():
    base = _line_traj7(10)
    out = compose_constant_arc_traj7(
        base,
        num_frames=101,
        arc_length=10.0,
        turn_degrees=20.0,
    )

    yaw = torch.atan2(out[:, 4], out[:, 3])
    yaw_step = torch.atan2(torch.sin(yaw[1:] - yaw[:-1]), torch.cos(yaw[1:] - yaw[:-1]))
    xz_step = torch.linalg.norm(out[1:, [0, 2]] - out[:-1, [0, 2]], dim=-1)

    assert out.shape == (101, 7)
    assert torch.allclose(out[0, :5], base[0, :5], atol=1e-6)
    assert torch.allclose(yaw_step, yaw_step.mean().expand_as(yaw_step), atol=1e-6)
    assert torch.allclose(xz_step, xz_step.mean().expand_as(xz_step), atol=1e-5)


def _traj7_from_xz_yaw(xz: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    y = torch.zeros(xz.shape[0], dtype=xz.dtype, device=xz.device)
    traj5 = torch.stack(
        [xz[:, 0], y, xz[:, 1], torch.cos(yaw), torch.sin(yaw)],
        dim=-1,
    )
    return build_physical_7d_from_5d(traj5)


class _DummyCondition:
    def __init__(self, path, path_valid_mask):
        self.path = path
        self.path_valid_mask = path_valid_mask


class _DummyPlan:
    pass


def test_rootrefiner_input_path_world_xz_uses_condition_path_and_anchor():
    condition = _DummyCondition(
        path=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=torch.float32,
        ),
        path_valid_mask=torch.tensor([1, 0, 1], dtype=torch.bool),
    )
    plan = _DummyPlan()
    plan.anchor_world_xz = torch.tensor([10.0, 20.0], dtype=torch.float32)
    plan.anchor_world_yaw = torch.tensor(math.pi / 2.0, dtype=torch.float32)

    world = _rootrefiner_input_path_world_xz(condition, plan)

    assert world.shape == (2, 2)
    assert torch.allclose(
        world,
        torch.tensor([[10.0, 20.0], [10.0, 18.0]], dtype=torch.float32),
        atol=1e-5,
    )


def test_compose_turn180_updated_traj_is_continuous_and_smoothly_turns_suffix():
    traj7 = _line_traj7(6)

    updated = compose_turn180_updated_traj7(
        traj7,
        update_frame=4,
        suffix_frames=8,
        turn_blend_frames=3,
    )

    assert updated.shape == (12, 7)
    assert torch.allclose(updated[4, :3], traj7[4, :3])
    # The source path moves +x; the updated route must not reverse instantly.
    assert updated[5, 0] > updated[4, 0]
    # After the smooth turn has completed, it should move in the opposite direction.
    assert updated[8, 0] < updated[7, 0]
    update_yaw = torch.atan2(updated[4, 4], updated[4, 3])
    old_update_yaw = torch.atan2(traj7[4, 4], traj7[4, 3])
    assert torch.allclose(wrap_angle(update_yaw - old_update_yaw), torch.tensor(0.0))
    final_yaw = torch.atan2(updated[-1, 4], updated[-1, 3])
    assert torch.allclose(
        torch.abs(wrap_angle(final_yaw - old_update_yaw)),
        torch.tensor(math.pi),
        atol=1e-5,
    )


def test_compose_forward_line_traj_moves_from_initial_pose_along_heading():
    xz = torch.tensor(
        [[2.0, 3.0], [10.0, 10.0], [20.0, 20.0]],
        dtype=torch.float32,
    )
    yaw = torch.zeros(3, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    out = compose_forward_line_traj7(
        traj7,
        num_frames=5,
        step_length=0.25,
    )

    assert out.shape == (5, 7)
    assert torch.allclose(out[:, 0], torch.full((5,), 2.0), atol=1e-6)
    assert torch.allclose(
        out[:, 2],
        torch.tensor([3.0, 3.25, 3.5, 3.75, 4.0]),
        atol=1e-6,
    )
    assert torch.allclose(out[:, 3], torch.ones(5), atol=1e-6)
    assert torch.allclose(out[:, 4], torch.zeros(5), atol=1e-6)


def test_condition_visual_mask_can_hide_prefix_before_update():
    mask = condition_visual_mask(
        num_frames=8,
        display_mode="future",
        update_frames=[3],
    )

    assert torch.equal(
        mask,
        torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.float32),
    )


def test_condition_visual_mask_full_keeps_all_condition_frames():
    mask = condition_visual_mask(
        num_frames=5,
        display_mode="full",
        update_frames=[3],
    )

    assert torch.equal(mask, torch.ones(5, dtype=torch.float32))


def test_compose_four_segment_forward_curve_is_continuous_and_mild():
    traj7 = compose_four_segment_forward_curve_traj7(
        num_frames=240,
        total_forward=4.2,
    )

    assert traj7.shape == (240, 7)
    assert torch.allclose(traj7[0, [0, 2]], torch.zeros(2), atol=1e-6)
    assert torch.all(traj7[1:, 2] > traj7[:-1, 2])
    assert float(traj7[:, 0].amax() - traj7[:, 0].amin()) < 0.6
    for frame in (60, 120, 180):
        assert torch.linalg.norm(traj7[frame, [0, 2]] - traj7[frame - 1, [0, 2]]) < 0.05


def test_override_heading_from_path_tangent_preserves_xyz_and_replaces_heading():
    xz = torch.tensor(
        [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]],
        dtype=torch.float32,
    )
    bad_yaw = torch.tensor([0.0, math.pi / 2.0, -math.pi, math.pi / 3.0])
    traj7 = _traj7_from_xz_yaw(xz, bad_yaw)

    out = _override_heading_from_path_tangent(traj7)

    assert torch.allclose(out[:, :3], traj7[:, :3], atol=1e-6)
    assert torch.allclose(out[:, 3], torch.ones(4), atol=1e-6)
    assert torch.allclose(out[:, 4], torch.zeros(4), atol=1e-6)


def test_override_heading_from_path_tangent_can_use_reference_path():
    noisy_xz = torch.tensor(
        [
            [0.00, 0.00],
            [0.08, 0.20],
            [-0.05, 0.40],
            [0.06, 0.60],
        ],
        dtype=torch.float32,
    )
    bad_yaw = torch.tensor([0.0, math.pi / 2.0, -math.pi, math.pi / 3.0])
    traj7 = _traj7_from_xz_yaw(noisy_xz, bad_yaw)
    reference_xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [0.0, 0.4],
            [0.0, 0.6],
        ],
        dtype=torch.float32,
    )
    reference = _traj7_from_xz_yaw(reference_xz, torch.zeros(4, dtype=torch.float32))

    out = _override_heading_from_path_tangent(traj7, path_xyz=reference[:, :3])

    assert torch.allclose(out[:, :3], traj7[:, :3], atol=1e-6)
    assert torch.allclose(out[:, 3], torch.ones(4), atol=1e-6)
    assert torch.allclose(out[:, 4], torch.zeros(4), atol=1e-6)


def test_compose_turn180_updated_traj_reanchors_new_route_in_update_local_frame():
    # The local new-route template starts by moving +Z with yaw=0. The original
    # frames after update deliberately move in another world direction so this
    # catches accidental use of ``traj7[update:]`` as the suffix source.
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [10.0, 10.0],
            [10.0, 9.0],
            [9.0, 9.0],
            [8.0, 9.0],
        ],
        dtype=torch.float32,
    )
    yaw = torch.tensor(
        [0.0, 0.0, 0.0, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    updated = compose_turn180_updated_traj7(
        traj7,
        update_frame=3,
        suffix_frames=5,
        turn_blend_frames=3,
    )

    assert torch.allclose(updated[3, [0, 2]], traj7[3, [0, 2]])
    assert torch.allclose(
        torch.atan2(updated[3, 4], updated[3, 3]),
        torch.tensor(math.pi / 2.0),
        atol=1e-6,
    )
    first_suffix_delta = updated[4, [0, 2]] - updated[3, [0, 2]]
    assert torch.allclose(first_suffix_delta, torch.tensor([1.0, 0.0]), atol=1e-5)


def test_compose_anchor_local_updated_traj_appends_source_route_in_update_frame():
    source_xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [10.0, 10.0],
            [10.0, 9.0],
        ],
        dtype=torch.float32,
    )
    yaw = torch.tensor(
        [0.0, 0.0, 0.0, math.pi / 2.0, math.pi / 2.0],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(source_xz, yaw)

    updated = compose_anchor_local_updated_traj7(
        traj7,
        update_frame=3,
        suffix_frames=4,
    )

    assert updated.shape == (7, 7)
    assert torch.allclose(updated[3, [0, 2]], traj7[3, [0, 2]])
    assert torch.allclose(
        torch.atan2(updated[3, 4], updated[3, 3]),
        torch.tensor(math.pi / 2.0),
        atol=1e-6,
    )
    first_suffix_delta = updated[4, [0, 2]] - updated[3, [0, 2]]
    assert torch.allclose(first_suffix_delta, torch.tensor([1.0, 0.0]), atol=1e-5)
    second_suffix_delta = updated[5, [0, 2]] - updated[4, [0, 2]]
    assert torch.allclose(second_suffix_delta, torch.tensor([1.0, 0.0]), atol=1e-5)


def test_compose_anchor_local_updated_traj_derives_suffix_heading_from_route_tangent():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [10.0, 10.0],
            [10.0, 9.0],
        ],
        dtype=torch.float32,
    )
    # Deliberately make source root heading disagree with the source path tangent.
    yaw = torch.full((5,), math.pi / 2.0, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    updated = compose_anchor_local_updated_traj7(
        traj7,
        update_frame=3,
        suffix_frames=4,
    )

    first_suffix_delta = updated[4, [0, 2]] - updated[3, [0, 2]]
    suffix_yaw = torch.atan2(updated[4, 4], updated[4, 3])
    assert torch.allclose(first_suffix_delta, torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(suffix_yaw, torch.tensor(math.pi / 2.0), atol=1e-5)


def test_compose_anchor_local_path_tangent_policy_keeps_boundary_tangent_continuous():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    yaw = torch.zeros(5, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    updated = compose_anchor_local_updated_traj7(
        traj7,
        update_frame=3,
        suffix_frames=4,
        anchor_yaw_policy="path_tangent",
        derive_heading_from_path=True,
    )

    pre_delta = updated[3, [0, 2]] - updated[2, [0, 2]]
    post_delta = updated[4, [0, 2]] - updated[3, [0, 2]]
    update_yaw = torch.atan2(updated[3, 4], updated[3, 3])
    assert torch.allclose(pre_delta, torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(post_delta, torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(update_yaw, torch.tensor(math.pi / 2.0), atol=1e-5)


def test_compose_anchor_local_path_heading_derivation_preserves_existing_prefix_positions():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 1.5],
            [2.0, 1.5],
            [3.0, 1.0],
            [4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    # Prefix root heading intentionally differs from path tangent. With path
    # heading derivation, positions stay fixed while heading is recomputed from
    # one route-wide tangent source.
    yaw = torch.tensor(
        [0.0, 0.2, 0.4, 0.7, 1.0, 1.2],
        dtype=torch.float32,
    )
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    updated = compose_anchor_local_updated_traj7(
        traj7,
        update_frame=4,
        suffix_frames=5,
        source_start_frame=1,
        anchor_yaw_policy="path_tangent",
        derive_heading_from_path=True,
        transition_frames=4,
    )

    assert torch.allclose(updated[:4, :3], traj7[:4, :3], atol=1e-6)


def test_compose_anchor_local_transition_smooths_post_update_heading_change():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ],
        dtype=torch.float32,
    )
    yaw = torch.zeros(6, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    sharp = compose_anchor_local_updated_traj7(
        traj7,
        update_frame=4,
        suffix_frames=5,
        source_start_frame=0,
        anchor_yaw_policy="path_tangent",
        derive_heading_from_path=True,
        transition_frames=0,
    )
    smooth = compose_anchor_local_updated_traj7(
        traj7,
        update_frame=4,
        suffix_frames=5,
        source_start_frame=0,
        anchor_yaw_policy="path_tangent",
        derive_heading_from_path=True,
        transition_frames=4,
    )

    sharp_yaw = torch.atan2(sharp[:, 4], sharp[:, 3])
    smooth_yaw = torch.atan2(smooth[:, 4], smooth[:, 3])
    sharp_step = torch.abs(wrap_angle(sharp_yaw[5] - sharp_yaw[4]))
    smooth_step = torch.abs(wrap_angle(smooth_yaw[5] - smooth_yaw[4]))
    assert smooth_step < sharp_step


def test_compose_anchor_local_can_resample_transition_duration():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [4.0, 1.0],
            [5.0, 1.0],
            [6.0, 1.0],
        ],
        dtype=torch.float32,
    )
    yaw = torch.zeros(8, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    base = compose_anchor_local_updated_traj7(
        traj7,
        update_frame=3,
        suffix_frames=6,
        source_start_frame=0,
        anchor_yaw_policy="path_tangent",
        derive_heading_from_path=True,
        transition_frames=2,
    )
    stretched = compose_anchor_local_updated_traj7(
        traj7,
        update_frame=3,
        suffix_frames=6,
        source_start_frame=0,
        anchor_yaw_policy="path_tangent",
        derive_heading_from_path=True,
        transition_frames=2,
        transition_output_frames=5,
    )

    assert stretched.shape[0] == base.shape[0] + 3
    assert torch.allclose(stretched[:3, :3], traj7[:3, :3], atol=1e-6)
    assert torch.allclose(stretched[3, :3], base[3, :3], atol=1e-6)
    assert torch.allclose(stretched[8, :3], base[5, :3], atol=1e-5)
    assert torch.allclose(stretched[9, :3], base[6, :3], atol=1e-5)


def test_compose_center_symmetric_reflects_history_around_update_anchor():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    yaw = torch.zeros(4, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    updated = compose_center_symmetric_updated_traj7(
        traj7,
        update_frame=3,
        suffix_frames=4,
        derive_heading_from_path=True,
    )

    expected_xz = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [4.0, 2.0],
        ],
        dtype=torch.float32,
    )
    assert updated.shape == (7, 7)
    assert torch.allclose(updated[:, [0, 2]], expected_xz, atol=1e-6)
    assert torch.allclose(updated[:3, :3], traj7[:3, :3], atol=1e-6)
    pre_delta = updated[3, [0, 2]] - updated[2, [0, 2]]
    post_delta = updated[4, [0, 2]] - updated[3, [0, 2]]
    assert torch.allclose(post_delta, pre_delta, atol=1e-6)


def test_compose_center_symmetric_can_resample_reflected_entry():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    yaw = torch.zeros(5, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    base = compose_center_symmetric_updated_traj7(
        traj7,
        update_frame=4,
        suffix_frames=5,
        transition_frames=2,
        derive_heading_from_path=True,
    )
    stretched = compose_center_symmetric_updated_traj7(
        traj7,
        update_frame=4,
        suffix_frames=5,
        transition_frames=2,
        transition_output_frames=4,
        derive_heading_from_path=True,
    )

    assert stretched.shape[0] == base.shape[0] + 2
    assert torch.allclose(stretched[:4, :3], traj7[:4, :3], atol=1e-6)
    assert torch.allclose(stretched[4, [0, 2]], base[4, [0, 2]], atol=1e-6)
    assert torch.allclose(stretched[8, [0, 2]], base[6, [0, 2]], atol=1e-6)
    assert torch.allclose(stretched[9, [0, 2]], base[7, [0, 2]], atol=1e-6)


def test_compose_center_symmetric_uses_extracted_history_without_padding_to_suffix():
    traj7 = _line_traj7(179)

    updated = compose_center_symmetric_updated_traj7(
        traj7,
        update_frame=169,
        suffix_frames=179,
        source_start_frame=20,
        transition_frames=48,
        transition_output_frames=80,
        derive_heading_from_path=True,
    )

    extracted_history = 169 - 20 + 1
    transition_extra = 80 - 48
    assert updated.shape == (169 + extracted_history + transition_extra, 7)


def test_compose_center_symmetric_preserves_prefix_heading_when_deriving_suffix():
    xz = torch.stack(
        [
            torch.arange(10, dtype=torch.float32),
            torch.zeros(10, dtype=torch.float32),
        ],
        dim=-1,
    )
    # Deliberately make authored heading differ from the xz tangent. The
    # update should keep existing prefix heading and only derive suffix heading.
    authored_yaw = torch.zeros(10, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, authored_yaw)

    updated = compose_center_symmetric_updated_traj7(
        traj7,
        update_frame=6,
        source_start_frame=2,
        suffix_frames=5,
        derive_heading_from_path=True,
    )

    assert torch.allclose(updated[:6, 3:5], traj7[:6, 3:5], atol=1e-6)


def test_compose_legacy_s_turn_preserves_resampled_s_turn_length_and_boundary():
    traj7 = _line_traj7(20)

    updated = compose_legacy_s_turn_updated_traj7(
        traj7,
        update_frame=10,
        suffix_frames=12,
        turn_blend_frames=4,
        transition_frames=3,
        transition_output_frames=6,
    )

    assert updated.shape == (25, 7)
    assert torch.allclose(updated[10, :3], traj7[10, :3], atol=1e-6)
    # The route starts by continuing the incoming path before the 180-degree
    # turn finishes, matching the old S-turn draft instead of reflecting history
    # into a separate "3"-shaped route.
    assert updated[11, 0] > updated[10, 0]
    assert updated[12, 0] > updated[11, 0]
    assert updated[-1, 0] < updated[-2, 0]


def test_compose_clean_s_curve_moves_forward_with_lateral_s_shape():
    traj7 = _line_traj7(20)

    updated = compose_clean_s_curve_updated_traj7(
        traj7,
        update_frame=10,
        suffix_frames=20,
        transition_frames=4,
        transition_output_frames=8,
    )

    assert updated.shape == (34, 7)
    assert torch.allclose(updated[10, :3], traj7[10, :3], atol=1e-6)
    anchor = updated[10, [0, 2]]
    rel = updated[10:, [0, 2]] - anchor
    # Original line heads +x, so x is forward and z is lateral.
    forward = rel[:, 0]
    lateral = rel[:, 1]
    assert torch.all(forward[1:] >= forward[:-1] - 1e-6)
    assert forward[-1] > forward[0] + 1.0
    assert lateral.amax() > 0.05
    assert lateral.amin() < -0.05
    assert torch.abs(lateral[-1]) < 1e-4


def test_route_update_dispatch_keeps_center_symmetric_on_sample_derived_path():
    traj7 = _line_traj7(24)

    expected = compose_center_symmetric_updated_traj7(
        traj7,
        update_frame=18,
        suffix_frames=20,
        source_start_frame=3,
        transition_frames=4,
        transition_output_frames=6,
    )
    actual = turn_debug.compose_updated_route_for_mode(
        traj7,
        composition_mode="center_symmetric",
        update_frame=18,
        suffix_frames=20,
        source_start_frame=3,
        transition_frames=4,
        transition_output_frames=6,
        anchor_yaw_policy="path_tangent",
        derive_heading_from_path=True,
        turn_blend_frames=24,
        update_lead_tokens=6,
        frames_per_token=4,
        first_straight_frames=40,
        arc_frames=48,
        arc_turn_degrees=35.0,
        step_lookback_frames=20,
        forward_frames=180,
        forward_step_length=0.015,
        four_segment_frames=240,
        four_segment_forward=4.2,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_compose_two_segment_arc_builds_straight_arc_straight_route():
    traj7 = _line_traj7(20)

    updated = turn_debug.compose_two_segment_arc_updated_traj7(
        traj7,
        update_frame=10,
        suffix_frames=80,
        first_straight_frames=12,
        arc_frames=24,
        turn_degrees=35.0,
    )

    assert updated.shape == (90, 7)
    assert torch.allclose(updated[10, :3], traj7[10, :3])
    early_delta = updated[15, [0, 2]] - updated[10, [0, 2]]
    assert early_delta[0] > 0.0
    assert torch.abs(early_delta[1]) < 1e-4

    yaw = torch.atan2(updated[:, 4], updated[:, 3])
    start_yaw = yaw[10]
    end_yaw = yaw[-1]
    assert torch.allclose(start_yaw, torch.tensor(math.pi / 2.0), atol=1e-4)
    assert torch.allclose(
        wrap_angle(end_yaw - start_yaw),
        torch.tensor(math.radians(35.0)),
        atol=0.08,
    )
    # Once the arc completes, the final straight segment should keep a stable
    # heading instead of continuing to curve.
    tail_yaw_delta = torch.abs(wrap_angle(yaw[-1] - yaw[-8]))
    assert tail_yaw_delta < 0.02


def test_compose_two_segment_arc_preserves_prefix_positions_and_uses_path_heading():
    xz = torch.stack(
        [torch.arange(12, dtype=torch.float32), torch.zeros(12)],
        dim=-1,
    )
    yaw = torch.zeros(12, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    updated = turn_debug.compose_two_segment_arc_updated_traj7(
        traj7,
        update_frame=8,
        suffix_frames=20,
        first_straight_frames=4,
        arc_frames=8,
        turn_degrees=-30.0,
    )

    assert torch.allclose(updated[:8, :3], traj7[:8, :3], atol=1e-6)
    update_yaw = torch.atan2(updated[8, 4], updated[8, 3])
    final_yaw = torch.atan2(updated[-1, 4], updated[-1, 3])
    assert torch.allclose(update_yaw, torch.tensor(math.pi / 2.0), atol=1e-4)
    assert wrap_angle(final_yaw - update_yaw) < 0.0


def test_compose_two_segment_arc_ignores_near_stationary_tail_for_step_length():
    xz = torch.zeros(16, 2, dtype=torch.float32)
    xz[:9, 0] = torch.arange(9, dtype=torch.float32)
    xz[9:, 0] = 8.0 + torch.arange(7, dtype=torch.float32) * 0.0002
    yaw = torch.full((16,), math.pi / 2.0, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    updated = turn_debug.compose_two_segment_arc_updated_traj7(
        traj7,
        update_frame=15,
        suffix_frames=20,
        first_straight_frames=10,
        arc_frames=6,
        turn_degrees=20.0,
        step_lookback_frames=6,
    )

    straight_distance = torch.linalg.norm(
        updated[25, [0, 2]] - updated[15, [0, 2]]
    )
    assert straight_distance > 5.0


def test_compose_two_segment_arc_ignores_near_stationary_tail_for_anchor_yaw():
    xz = torch.zeros(16, 2, dtype=torch.float32)
    xz[:9, 0] = torch.arange(9, dtype=torch.float32)
    xz[9:, 0] = 8.0
    xz[9:, 1] = -torch.arange(7, dtype=torch.float32) * 0.0002
    # Make the stored heading disagree with the earlier stable path direction;
    # the synthetic route should use robust path motion, not the noisy tail or
    # stored root heading.
    yaw = torch.zeros(16, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    updated = turn_debug.compose_two_segment_arc_updated_traj7(
        traj7,
        update_frame=15,
        suffix_frames=20,
        first_straight_frames=10,
        arc_frames=6,
        turn_degrees=20.0,
        step_lookback_frames=6,
    )

    first_delta = updated[16, [0, 2]] - updated[15, [0, 2]]
    first_yaw = torch.atan2(first_delta[0], first_delta[1])
    assert torch.allclose(first_yaw, torch.tensor(math.pi / 2.0), atol=1e-4)


def test_path_heading_derivation_ignores_near_zero_tail_tangent_noise():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0002, 0.0001],
            [2.0004, -0.0001],
        ],
        dtype=torch.float32,
    )

    yaw = _path_yaw_from_xz(xz)

    assert torch.allclose(yaw[2], yaw[1], atol=1e-6)
    assert torch.allclose(yaw[3], yaw[1], atol=1e-6)
    assert torch.allclose(yaw[4], yaw[1], atol=1e-6)


def test_center_symmetric_path_heading_only_derives_suffix_heading():
    xz = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    # Stored heading deliberately disagrees with the path tangent. For the
    # center-symmetric update, the authored prefix condition must remain intact;
    # only the new reflected suffix gets tangent-derived heading.
    yaw = torch.zeros(4, dtype=torch.float32)
    traj7 = _traj7_from_xz_yaw(xz, yaw)

    updated = compose_center_symmetric_updated_traj7(
        traj7,
        update_frame=3,
        suffix_frames=4,
        derive_heading_from_path=True,
    )

    boundary_prefix_yaw = torch.atan2(updated[2, 4], updated[2, 3])
    boundary_suffix_yaw = torch.atan2(updated[3, 4], updated[3, 3])
    assert torch.allclose(boundary_prefix_yaw, torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(boundary_suffix_yaw, torch.tensor(math.pi / 2.0), atol=1e-5)


def test_rootrefiner_dense_splice_can_use_decoded_anchor_frame_boundary():
    first = _line_traj7(150)
    second = _line_traj7(12).clone()
    second[:, 0] += 1000.0

    class Plan:
        pass

    first_plan = Plan()
    first_plan.valid_frames = int(first.shape[0])
    first_plan.waypoints_local_7d = first
    first_plan.anchor_world_xz = torch.zeros(2, dtype=torch.float32)
    first_plan.anchor_world_yaw = torch.tensor(0.0)
    first_plan.anchor_commit_idx = 0

    second_plan = Plan()
    second_plan.valid_frames = int(second.shape[0])
    second_plan.waypoints_local_7d = second
    second_plan.anchor_world_xz = torch.zeros(2, dtype=torch.float32)
    second_plan.anchor_world_yaw = torch.tensor(0.0)
    second_plan.anchor_commit_idx = 36
    second_plan.anchor_frame_idx = 140

    out = _compose_rootrefiner_condition_traj7(
        first_plan,
        second_plan,
        target_frames=150,
        frames_per_token=4,
    )

    assert torch.allclose(out[139, 0], first[139, 0])
    assert torch.allclose(out[140, 0], second[0, 0])
    assert torch.allclose(out[141, 0], second[1, 0])


def test_rootrefiner_dense_splice_can_use_generated_prefix_before_boundary():
    first = _line_traj7(150)
    second = _line_traj7(12).clone()
    second[:, 0] += 1000.0
    generated_prefix = _line_traj7(141).clone()
    generated_prefix[:, 0] += 900.0

    class Plan:
        pass

    first_plan = Plan()
    first_plan.valid_frames = int(first.shape[0])
    first_plan.waypoints_local_7d = first
    first_plan.anchor_world_xz = torch.zeros(2, dtype=torch.float32)
    first_plan.anchor_world_yaw = torch.tensor(0.0)
    first_plan.anchor_commit_idx = 0

    second_plan = Plan()
    second_plan.valid_frames = int(second.shape[0])
    second_plan.waypoints_local_7d = second
    second_plan.anchor_world_xz = torch.zeros(2, dtype=torch.float32)
    second_plan.anchor_world_yaw = torch.tensor(0.0)
    second_plan.anchor_commit_idx = 36
    second_plan.anchor_frame_idx = 140

    out = _compose_rootrefiner_condition_traj7(
        first_plan,
        second_plan,
        target_frames=150,
        frames_per_token=4,
        prefix_world_5d=generated_prefix[:, :5],
    )

    assert torch.allclose(out[139, 0], generated_prefix[139, 0])
    assert torch.allclose(out[140, 0], second[0, 0])
    assert torch.allclose(out[141, 0], second[1, 0])


def test_multi_rootrefiner_dense_splice_uses_each_anchor_frame_boundary():
    class Plan:
        pass

    plans = []
    for anchor, offset in ((0, 0.0), (10, 100.0), (20, 200.0), (30, 300.0)):
        plan = Plan()
        plan.valid_frames = 16
        local = _line_traj7(16).clone()
        local[:, 0] += offset
        plan.waypoints_local_7d = local
        plan.anchor_world_xz = torch.zeros(2, dtype=torch.float32)
        plan.anchor_world_yaw = torch.tensor(0.0)
        plan.anchor_commit_idx = 0
        plan.anchor_frame_idx = anchor
        plans.append(plan)

    out = _compose_multi_rootrefiner_condition_traj7(
        plans,
        target_frames=40,
        frames_per_token=4,
    )

    assert torch.allclose(out[9, 0], torch.tensor(9.0))
    assert torch.allclose(out[10, 0], torch.tensor(100.0))
    assert torch.allclose(out[19, 0], torch.tensor(109.0))
    assert torch.allclose(out[20, 0], torch.tensor(200.0))
    assert torch.allclose(out[29, 0], torch.tensor(209.0))
    assert torch.allclose(out[30, 0], torch.tensor(300.0))


def test_second_source_translation_keeps_suffix_continuous_from_anchor():
    route = _line_traj7(8)
    anchor_state = RootFrameState(
        commit_idx=3,
        world_xz=torch.tensor([10.0, 20.0], dtype=torch.float32),
        world_yaw=torch.tensor(math.pi / 2.0, dtype=torch.float32),
        source="test",
    )

    out = _build_second_source_from_anchor_boundary(
        route,
        boundary_frame=3,
        anchor_state=anchor_state,
        anchor_world_y=torch.tensor(1.5),
    )

    original_delta = route[4, [0, 2]] - route[3, [0, 2]]
    shifted_delta = out[1, [0, 2]] - out[0, [0, 2]]
    assert torch.allclose(out[0, [0, 2]], anchor_state.world_xz, atol=1e-6)
    assert torch.allclose(out[0, 1], torch.tensor(1.5), atol=1e-6)
    assert torch.allclose(shifted_delta, original_delta, atol=1e-6)


def test_runtime_segment_translate_starts_at_current_root_without_rotating_route():
    route = _line_traj7(8)
    anchor_state = RootFrameState(
        commit_idx=3,
        world_xz=torch.tensor([12.0, 20.0], dtype=torch.float32),
        world_yaw=torch.tensor(0.0, dtype=torch.float32),
        source="test",
    )

    out, debug = _build_segment_source_from_runtime_anchor(
        route,
        boundary_frame=3,
        end_frame=7,
        anchor_state=anchor_state,
        anchor_world_y=torch.tensor(1.5),
        reanchor_mode="current_root_translate",
    )

    assert torch.allclose(out[0, [0, 2]], anchor_state.world_xz, atol=1e-6)
    assert torch.allclose(out[1, [0, 2]] - out[0, [0, 2]], torch.tensor([1.0, 0.0]))
    assert torch.allclose(debug["route_boundary_xz"], torch.tensor([3.0, 0.0]))
    assert torch.allclose(debug["runtime_anchor_xz"], anchor_state.world_xz)


def test_runtime_segment_pose_starts_at_current_root_and_rotates_future_route():
    route = _line_traj7(8)
    anchor_state = RootFrameState(
        commit_idx=3,
        world_xz=torch.tensor([12.0, 20.0], dtype=torch.float32),
        world_yaw=torch.tensor(0.0, dtype=torch.float32),
        source="test",
    )

    out, debug = _build_segment_source_from_runtime_anchor(
        route,
        boundary_frame=3,
        end_frame=7,
        anchor_state=anchor_state,
        anchor_world_y=torch.tensor(1.5),
        reanchor_mode="current_root_pose",
    )

    assert torch.allclose(out[0, [0, 2]], anchor_state.world_xz, atol=1e-6)
    # The source route moves along +x/yaw=pi/2. Pose-relative update maps that
    # local forward motion onto the current actor yaw=0, i.e. +z.
    assert torch.allclose(
        out[1, [0, 2]] - out[0, [0, 2]],
        torch.tensor([0.0, 1.0]),
        atol=1e-6,
    )
    assert torch.allclose(torch.atan2(out[0, 4], out[0, 3]), torch.tensor(0.0))
    assert torch.allclose(debug["xz_offset"], torch.tensor([9.0, 20.0]))


def test_resolve_update_anchor_yaw_state_can_use_path_tangent():
    route = _line_traj7(8)
    anchor_state = RootFrameState(
        commit_idx=3,
        world_xz=torch.tensor([12.0, 20.0], dtype=torch.float32),
        world_yaw=torch.tensor(0.0, dtype=torch.float32),
        source="test",
    )
    segment, _ = _build_segment_source_from_runtime_anchor(
        route,
        boundary_frame=3,
        end_frame=7,
        anchor_state=anchor_state,
        anchor_world_y=torch.tensor(1.5),
        reanchor_mode="current_root_translate",
    )

    out, debug = _resolve_update_anchor_yaw_state(
        anchor_state,
        segment,
        mode="path_tangent",
        tangent_frames=4,
    )

    assert out.commit_idx == anchor_state.commit_idx
    assert torch.allclose(out.world_xz, anchor_state.world_xz)
    assert torch.allclose(out.world_yaw, torch.tensor(math.pi / 2.0), atol=1e-6)
    assert debug["mode"] == "path_tangent"
    assert abs(debug["yaw_delta"] - math.pi / 2.0) < 1e-6


def test_jsonify_debug_record_recurses_nested_tensors():
    out = _jsonify_debug_record(
        {
            "outer": {
                "scalar": torch.tensor(1.25),
                "vector": torch.tensor([1.0, 2.0]),
            },
            "items": [torch.tensor(3.0)],
        }
    )

    assert out == {"outer": {"scalar": 1.25, "vector": [1.0, 2.0]}, "items": [3.0]}


def test_mask_rootrefiner_condition_front_disables_front_control_only():
    condition = RootRefinerPathCondition(
        path=torch.arange(16, dtype=torch.float32).view(8, 2),
        path_valid_mask=torch.ones(8, dtype=torch.bool),
        path_control_mask=torch.ones(8, dtype=torch.bool),
        path_supervision_mask=torch.ones(12, dtype=torch.bool),
        path_features=torch.ones(5, dtype=torch.float32),
        path_features_raw=torch.ones(5, dtype=torch.float32),
        path_mode="dense_path",
        offset_start_frames=0,
    )

    masked, debug = _mask_rootrefiner_condition_front(
        condition,
        ratio=0.25,
        kind="control",
    )

    assert torch.equal(masked.path_valid_mask, condition.path_valid_mask)
    assert torch.equal(
        masked.path_control_mask,
        torch.tensor([0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.bool),
    )
    assert debug["masked_points"] == 2
    assert debug["valid_points"] == 8
    assert debug["kind"] == "control"


def test_mask_rootrefiner_condition_front_valid_keeps_tail_visible():
    condition = RootRefinerPathCondition(
        path=torch.arange(16, dtype=torch.float32).view(8, 2),
        path_valid_mask=torch.ones(8, dtype=torch.bool),
        path_control_mask=torch.ones(8, dtype=torch.bool),
        path_supervision_mask=torch.ones(12, dtype=torch.bool),
        path_features=torch.ones(5, dtype=torch.float32),
        path_features_raw=torch.ones(5, dtype=torch.float32),
        path_mode="dense_path",
        offset_start_frames=0,
    )

    masked, debug = _mask_rootrefiner_condition_front(
        condition,
        ratio=1.0,
        kind="valid",
    )

    assert torch.equal(
        masked.path_valid_mask,
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool),
    )
    assert torch.equal(masked.path_control_mask, masked.path_valid_mask)
    assert debug["masked_points"] == 7


def test_apply_updated_traj_to_sample_batch_updates_lengths_and_masks():
    updated = _line_traj7(9)
    sample_batch = {
        "traj_cond_7d": torch.zeros(1, 3, 7),
        "traj": torch.zeros(1, 3, 3),
        "traj_cond": torch.zeros(1, 3, 3),
        "traj_mask": torch.ones(1, 3),
        "traj_cond_mask": torch.ones(1, 3),
        "token_mask": torch.ones(1, 1),
    }

    out = apply_updated_traj_to_sample_batch(sample_batch, updated)

    assert out["traj_cond_7d"].shape == (1, 9, 7)
    assert out["traj"].shape == (1, 9, 3)
    assert out["traj_length"].item() == 9
    assert out["feature_length"].item() == 9
    assert out["token_length"].item() == 3
    assert out["traj_mask"].sum().item() == 9
