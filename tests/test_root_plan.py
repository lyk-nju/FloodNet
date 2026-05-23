"""Unit tests for utils/root_plan.py (T_A_03a).

Covers G14-G22 per docs/TODO.md §T_A_03 Unit tests:
    G14-G17: dual anchor regression on plan_local→body_window_local
    G18-G22: RootPlan slicing / overflow / hold-last / valid_frames consistency
"""

from __future__ import annotations

import math

import pytest
import torch

from utils.local_frame import canonicalize_7d, uncanonicalize_7d
from utils.root_plan import (
    RootPlan,
    plan_local_to_body_window_local,
    slice_plan_with_mask,
)
from utils.token_frame import num_frames_for_tokens

ATOL = 1e-5


def _dummy_plan(num_tokens_pred: int = 5,
                anchor_commit_idx: int = 20,
                anchor_world_xz=(0.0, 0.0),
                anchor_world_yaw: float = 0.0) -> RootPlan:
    """Construct a minimal RootPlan with deterministic waypoints."""
    n_valid = num_frames_for_tokens(num_tokens_pred)
    # waypoints_local_7d: x sweeps 0→n_valid m along +X, y=1, z=0, heading (1,0),
    # fwd_delta=1, yaw_delta=0. Trivial but useful for slicing tests.
    pts = torch.zeros(n_valid, 7, dtype=torch.float64)
    pts[:, 0] = torch.arange(n_valid, dtype=torch.float64)
    pts[:, 1] = 1.0
    pts[:, 3] = 1.0   # cos
    pts[:, 5] = 1.0   # fwd_delta
    return RootPlan(
        num_tokens_pred=num_tokens_pred,
        valid_frames=n_valid,
        waypoints_local_7d=pts,
        frame_dt=1 / 20.0,
        frames_per_token=4,
        anchor_commit_idx=anchor_commit_idx,
        anchor_world_xz=torch.tensor(anchor_world_xz, dtype=torch.float64),
        anchor_world_yaw=torch.tensor(anchor_world_yaw, dtype=torch.float64),
        source="debug",
    )


# ---------------------------------------------------------------------------
# G18-G22: RootPlan slicing
# ---------------------------------------------------------------------------


def test_G18_slice_in_valid_region_returns_real_waypoints_and_mask_all_true():
    plan = _dummy_plan(num_tokens_pred=5)   # valid_frames = 4*5 - 3 = 17
    traj, mask = slice_plan_with_mask(plan, current_plan_token=0, horizon_tokens=5)
    H = traj.shape[0]
    # token_range_to_frame_slice(0, 5) = slice(0, 4*5-3) = slice(0, 17), length=17
    assert H == 17
    assert mask.all()
    # First waypoint is plan.waypoints_local_7d[0]
    assert torch.allclose(traj[0], plan.waypoints_local_7d[0], atol=ATOL)
    # Last is plan.waypoints_local_7d[16]
    assert torch.allclose(traj[-1], plan.waypoints_local_7d[16], atol=ATOL)


def test_G19_overflow_partial_hold_last_with_zero_mask_in_overflow_region():
    plan = _dummy_plan(num_tokens_pred=5)   # valid_frames = 17
    # Request a slice that extends past valid_frames.
    traj, mask = slice_plan_with_mask(plan, frame_slice=slice(15, 25))
    H = traj.shape[0]
    assert H == 10
    # First 2 frames (15, 16) are valid; remaining 8 are overflow (hold-last).
    assert mask[:2].all()
    assert not mask[2:].any()
    # Valid region matches plan
    assert torch.allclose(traj[0], plan.waypoints_local_7d[15], atol=ATOL)
    assert torch.allclose(traj[1], plan.waypoints_local_7d[16], atol=ATOL)
    # Hold-last region all equals plan.waypoints_local_7d[16] (last valid)
    last = plan.waypoints_local_7d[16]
    for k in range(2, H):
        assert torch.allclose(traj[k], last, atol=ATOL)


def test_G20_fstart_beyond_valid_frames_full_hold_last_and_mask_all_false():
    plan = _dummy_plan(num_tokens_pred=5)
    # frame_slice fully past valid_frames=17
    traj, mask = slice_plan_with_mask(plan, frame_slice=slice(20, 30))
    H = traj.shape[0]
    assert H == 10
    assert not mask.any()
    last = plan.waypoints_local_7d[16]
    for k in range(H):
        assert torch.allclose(traj[k], last, atol=ATOL)


def test_G20b_fstart_beyond_with_hold_last_disabled_returns_all_zero():
    plan = _dummy_plan(num_tokens_pred=5)
    traj, mask = slice_plan_with_mask(
        plan, frame_slice=slice(20, 30), hold_last_on_overflow=False,
    )
    assert traj.shape == (10, 7)
    assert torch.equal(traj, torch.zeros_like(traj))
    assert not mask.any()


def test_G21_valid_frames_consistent_with_num_frames_for_tokens():
    for n in (1, 2, 3, 5, 20, 49):
        plan = _dummy_plan(num_tokens_pred=n)
        assert plan.valid_frames == num_frames_for_tokens(n)


def test_G22_current_plan_token_formula_via_anchor_difference():
    """Doc-test of G22: caller computes current_plan_token = head.commit_idx -
    plan.anchor_commit_idx. Here we verify slice_plan_with_mask honors that
    when caller passes (current_plan_token, horizon_tokens).
    """
    plan = _dummy_plan(num_tokens_pred=10, anchor_commit_idx=20)   # valid_frames = 37
    head_commit_idx = 35
    current_plan_token = head_commit_idx - plan.anchor_commit_idx   # = 15
    assert current_plan_token == 15

    traj, mask = slice_plan_with_mask(
        plan, current_plan_token=current_plan_token, horizon_tokens=5,
    )
    # token_range_to_frame_slice(15, 5) = slice(4*15-3, 4*(15+4)+1) = slice(57, 77), length=20
    assert traj.shape[0] == 20
    # First waypoint in this slice is plan.waypoints_local_7d[57], but
    # valid_frames=37 — so this slice is entirely past valid region.
    assert not mask.any()


def test_G18b_slice_via_explicit_frame_slice_matches_token_range():
    """frame_slice and (current_plan_token, horizon_tokens) call paths agree."""
    plan = _dummy_plan(num_tokens_pred=10)   # valid_frames=37

    traj_a, mask_a = slice_plan_with_mask(plan, current_plan_token=0, horizon_tokens=3)
    traj_b, mask_b = slice_plan_with_mask(plan, frame_slice=slice(0, 9))  # 4*3-3 = 9
    assert torch.equal(traj_a, traj_b)
    assert torch.equal(mask_a, mask_b)


def test_slice_preserves_device_dtype():
    """slice_plan_with_mask must use plan.waypoints_local_7d's device/dtype."""
    plan = _dummy_plan(num_tokens_pred=3)
    # Default constructor builds float64 on CPU
    traj, mask = slice_plan_with_mask(plan, current_plan_token=0, horizon_tokens=2)
    assert traj.dtype == plan.waypoints_local_7d.dtype == torch.float64
    assert traj.device == plan.waypoints_local_7d.device
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# G14-G17: dual anchor regression on plan_local_to_body_window_local
# ---------------------------------------------------------------------------


def test_G14_anchor_collision_traj_zero_means_body_anchor_equals_plan_target_at_slice_start():
    """G14: traj[0] ≈ (0, *, 0, 1, 0, *, *) iff body_anchor world pose ==
    plan target's world pose at plan_frame_slice.start.

    We construct: plan_anchor at world (1, 1) yaw 0; plan_frame_slice.start = 0
    so the target world pose at start IS the plan anchor itself.
    If body_anchor = plan_anchor (same world pose), then traj[0] should land
    on body-window-local origin.
    """
    plan_anchor_xz = torch.tensor([1.0, 1.0], dtype=torch.float64)
    plan_anchor_yaw = torch.tensor(0.0, dtype=torch.float64)
    # plan_local frame: first waypoint at plan_local origin (0,0) heading=(1,0)
    traj_plan_local = torch.zeros(3, 7, dtype=torch.float64)
    traj_plan_local[:, 3] = 1.0   # cos
    # Set y to a non-zero value to track preservation
    traj_plan_local[:, 1] = 0.7

    # body anchor coincides with plan anchor.
    body_anchor_xz = plan_anchor_xz.clone()
    body_anchor_yaw = plan_anchor_yaw.clone()

    traj_body_local = plan_local_to_body_window_local(
        traj_plan_local, plan_anchor_xz, plan_anchor_yaw,
        body_anchor_xz, body_anchor_yaw,
    )
    # traj[0] should be at body-window-local origin (xz=0, heading=(1,0)), y preserved
    assert torch.allclose(traj_body_local[0, 0], torch.tensor(0.0, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(traj_body_local[0, 1], torch.tensor(0.7, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(traj_body_local[0, 2], torch.tensor(0.0, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(traj_body_local[0, 3], torch.tensor(1.0, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(traj_body_local[0, 4], torch.tensor(0.0, dtype=torch.float64), atol=ATOL)


def test_G15_body_anchor_history0_distinct_from_head_traj0_is_nonzero_offset():
    """G15: body_anchor = history0 (in past), head/plan = current.
    traj[0] in body-window-local is the position of the current plan target
    expressed in history0-local — NOT origin, NOT closed-loop drift.
    """
    # history0 at world (0, 0) yaw 0
    body_anchor_xz = torch.tensor([0.0, 0.0], dtype=torch.float64)
    body_anchor_yaw = torch.tensor(0.0, dtype=torch.float64)
    # plan anchor at world (5, 0) yaw 0 — 5m ahead in world +X
    plan_anchor_xz = torch.tensor([5.0, 0.0], dtype=torch.float64)
    plan_anchor_yaw = torch.tensor(0.0, dtype=torch.float64)
    # plan-local: target at origin (means world = plan_anchor)
    traj_plan_local = torch.zeros(1, 7, dtype=torch.float64)
    traj_plan_local[0, 3] = 1.0   # cos

    traj_body_local = plan_local_to_body_window_local(
        traj_plan_local, plan_anchor_xz, plan_anchor_yaw,
        body_anchor_xz, body_anchor_yaw,
    )
    # In history0-local (= world here since body_anchor world pose is identity),
    # the plan target should appear at (5, 0).
    assert torch.allclose(traj_body_local[0, 0], torch.tensor(5.0, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(traj_body_local[0, 2], torch.tensor(0.0, dtype=torch.float64), atol=ATOL)
    # Heading unchanged because both anchors have yaw=0
    assert torch.allclose(traj_body_local[0, 3], torch.tensor(1.0, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(traj_body_local[0, 4], torch.tensor(0.0, dtype=torch.float64), atol=ATOL)
    # ⚠ The distance is the geometric distance from history0 → plan target in
    # world; it is NOT a closed-loop drift error.


def test_G16_target_current_error_oracle_case_is_zero():
    """G16: oracle case (closed-loop perfect) — head_state already at the world
    pose corresponding to plan target at current_plan_token. In body-window-local,
    plan target at current_plan_token equals head canonicalized to body anchor.

    Construct: body_anchor = world (0, 0) yaw 0; head_state = world (3, 0) yaw 0.
    Plan generated with anchor at head, so plan_anchor = head pose. Plan target
    at current_plan_token=0 is plan-local origin (head's world pose).
    Both should map to (3, 0) in body-window-local.
    """
    body_anchor_xz = torch.tensor([0.0, 0.0], dtype=torch.float64)
    body_anchor_yaw = torch.tensor(0.0, dtype=torch.float64)
    head_world_xz = torch.tensor([3.0, 0.0], dtype=torch.float64)
    head_world_yaw = torch.tensor(0.0, dtype=torch.float64)
    plan_anchor_xz = head_world_xz.clone()
    plan_anchor_yaw = head_world_yaw.clone()

    # Plan target at current_plan_token=0 is plan-local origin.
    traj_plan_local = torch.zeros(1, 7, dtype=torch.float64)
    traj_plan_local[0, 3] = 1.0   # cos

    traj_body_local = plan_local_to_body_window_local(
        traj_plan_local, plan_anchor_xz, plan_anchor_yaw,
        body_anchor_xz, body_anchor_yaw,
    )
    # head canonicalized to body-window-local
    head_5d = torch.zeros(1, 7, dtype=torch.float64)
    head_5d[0, 0] = head_world_xz[0]
    head_5d[0, 2] = head_world_xz[1]
    head_5d[0, 3] = torch.cos(head_world_yaw)
    head_5d[0, 4] = torch.sin(head_world_yaw)
    head_body_local = canonicalize_7d(head_5d, body_anchor_xz, body_anchor_yaw)

    # Oracle: plan target (at token 0) projected to body-local ≈ head projected to body-local.
    err_xz = (traj_body_local[0, [0, 2]] - head_body_local[0, [0, 2]]).norm()
    assert err_xz.item() < ATOL, f"oracle target_current_error_xz = {err_xz.item()}"


def test_G17_canonicalize_anchor_frame_origin_convention_doc():
    """G17 reminder: body_output_local[0] (= body window history0 frame) under
    canonicalize_7d must equal (0, y, 0, 1, 0, *, *). The convention is verified
    in test_local_frame.test_T08 / test_T11; we re-assert here to make the
    plan_local_to_body_window_local users aware of the invariant.
    """
    body_anchor_xz = torch.tensor([5.0, -1.0], dtype=torch.float64)
    body_anchor_yaw = torch.tensor(0.7, dtype=torch.float64)
    # The anchor frame in world: build a single-frame 7D at that world pose
    anchor_world = torch.zeros(1, 7, dtype=torch.float64)
    anchor_world[0, 0] = body_anchor_xz[0]
    anchor_world[0, 1] = 1.2   # arbitrary y
    anchor_world[0, 2] = body_anchor_xz[1]
    anchor_world[0, 3] = torch.cos(body_anchor_yaw)
    anchor_world[0, 4] = torch.sin(body_anchor_yaw)
    local = canonicalize_7d(anchor_world, body_anchor_xz, body_anchor_yaw)
    assert torch.allclose(local[0, 0], torch.tensor(0.0, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(local[0, 1], torch.tensor(1.2, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(local[0, 2], torch.tensor(0.0, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(local[0, 3], torch.tensor(1.0, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(local[0, 4], torch.tensor(0.0, dtype=torch.float64), atol=ATOL)


# ---------------------------------------------------------------------------
# RootPlan construction validation
# ---------------------------------------------------------------------------


def test_rootplan_rejects_bad_waypoint_shape():
    with pytest.raises(ValueError):
        RootPlan(
            num_tokens_pred=2,
            valid_frames=5,
            waypoints_local_7d=torch.zeros(5, 4),   # wrong last dim
            frame_dt=0.05,
            anchor_commit_idx=0,
            anchor_world_xz=torch.zeros(2),
            anchor_world_yaw=torch.tensor(0.0),
        )


def test_rootplan_rejects_too_few_waypoints_for_valid_frames():
    with pytest.raises(ValueError):
        RootPlan(
            num_tokens_pred=3,
            valid_frames=9,
            waypoints_local_7d=torch.zeros(5, 7),   # only 5 frames < 9
            frame_dt=0.05,
            anchor_commit_idx=0,
            anchor_world_xz=torch.zeros(2),
            anchor_world_yaw=torch.tensor(0.0),
        )


def test_plan_local_round_trip_with_rotated_anchors():
    """Sanity: canonicalize then uncanonicalize composition returns identity
    (test the dual-anchor helper composes the two T_A_01 ops as documented).
    """
    rng = torch.Generator().manual_seed(11)
    traj = torch.randn(4, 7, dtype=torch.float64, generator=rng)
    # Force cos/sin to be a valid unit-circle pair
    yaw = torch.randn(4, dtype=torch.float64, generator=rng)
    traj[:, 3] = torch.cos(yaw)
    traj[:, 4] = torch.sin(yaw)

    plan_anchor_xz = torch.tensor([0.0, 0.0], dtype=torch.float64)
    plan_anchor_yaw = torch.tensor(0.0, dtype=torch.float64)
    body_anchor_xz = torch.tensor([2.0, -1.5], dtype=torch.float64)
    body_anchor_yaw = torch.tensor(0.4, dtype=torch.float64)

    # plan_local → body_local
    body_local = plan_local_to_body_window_local(
        traj, plan_anchor_xz, plan_anchor_yaw,
        body_anchor_xz, body_anchor_yaw,
    )
    # body_local → plan_local should round-trip
    world_back = uncanonicalize_7d(body_local, body_anchor_xz, body_anchor_yaw)
    plan_local_back = canonicalize_7d(world_back, plan_anchor_xz, plan_anchor_yaw)
    assert torch.allclose(plan_local_back, traj, atol=1e-9)


# Used to keep `math` import meaningful for future tests; current ones use torch.
_ = math.pi
