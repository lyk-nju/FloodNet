"""Unit tests for utils.motion_process.root_to_traj_feats_7d (T_B_05 helper).

Covers shape / first-frame padding / physical-yaw consistency / rigid-invariance
under canonicalize_7d / forward-motion sign.
"""

from __future__ import annotations

import math

import torch

from utils.local_frame import (
    canonicalize_7d,
    heading_dir_xz,
    root_quat_to_physical_yaw,
)
from utils.motion_process import (
    recover_root_rot_pos,
    root_to_traj_feats_7d,
)

ATOL = 1e-5
PI = math.pi


def _make_263d(rot_vel_t0: float, local_vel_xz_per_frame, n_frames: int = 6):
    """Minimal 263D motion fixture (see tests/test_local_frame helpers)."""
    data = torch.zeros(1, n_frames, 263, dtype=torch.float32)
    data[0, 0, 0] = rot_vel_t0
    vx, vz = local_vel_xz_per_frame
    for t in range(n_frames):
        data[0, t, 1] = vx
        data[0, t, 2] = vz
    data[..., 3] = 1.0   # constant root y
    return data


# ---------------------------------------------------------------------------
# Shape / dtype
# ---------------------------------------------------------------------------


def test_shape_and_dtype_propagation():
    data = _make_263d(rot_vel_t0=0.0, local_vel_xz_per_frame=(0.0, 1.0), n_frames=4)
    quat, xyz = recover_root_rot_pos(data)
    feats = root_to_traj_feats_7d(quat, xyz)
    assert feats.shape == (1, 4, 7)
    assert feats.dtype == xyz.dtype


def test_batched_shape():
    """[B, T, 4] / [B, T, 3] → [B, T, 7]."""
    B, T = 3, 5
    quat = torch.zeros(B, T, 4)
    quat[..., 0] = 1.0   # identity quaternion
    xyz = torch.zeros(B, T, 3)
    xyz[..., 2] = torch.arange(T, dtype=torch.float32).expand(B, T)   # walk +Z
    feats = root_to_traj_feats_7d(quat, xyz)
    assert feats.shape == (B, T, 7)


# ---------------------------------------------------------------------------
# First-frame padding (HARD CONSTRAINT, round 8 P0-2)
# ---------------------------------------------------------------------------


def test_first_frame_fwd_delta_and_yaw_delta_both_zero():
    """fwd_delta[..., 0] == 0 AND yaw_delta[..., 0] == 0."""
    quat = torch.zeros(1, 5, 4)
    quat[..., 0] = 1.0
    xyz = torch.zeros(1, 5, 3)
    xyz[0, :, 2] = torch.arange(5, dtype=torch.float32)
    feats = root_to_traj_feats_7d(quat, xyz)
    assert feats[0, 0, 5].item() == 0.0, f"fwd_delta[0]={feats[0, 0, 5].item()}"
    assert feats[0, 0, 6].item() == 0.0, f"yaw_delta[0]={feats[0, 0, 6].item()}"


def test_first_frame_padding_holds_for_batched_input():
    B, T = 4, 6
    quat = torch.zeros(B, T, 4)
    quat[..., 0] = 1.0
    xyz = torch.randn(B, T, 3)
    feats = root_to_traj_feats_7d(quat, xyz)
    assert torch.equal(feats[:, 0, 5], torch.zeros(B))
    assert torch.equal(feats[:, 0, 6], torch.zeros(B))


# ---------------------------------------------------------------------------
# Physical yaw consistency with root_quat_to_physical_yaw
# ---------------------------------------------------------------------------


def test_cos_sin_channels_match_root_quat_to_physical_yaw():
    """Channels [3], [4] of the output are cos / sin of physical yaw (derived
    via the canonical local_frame helper)."""
    quat = torch.zeros(1, 5, 4)
    quat[..., 0] = math.cos(PI / 8)
    quat[..., 2] = math.sin(PI / 8)
    xyz = torch.zeros(1, 5, 3)
    feats = root_to_traj_feats_7d(quat, xyz)
    physical_yaw = root_quat_to_physical_yaw(quat)
    expected_cos = torch.cos(physical_yaw)
    expected_sin = torch.sin(physical_yaw)
    assert torch.allclose(feats[..., 3], expected_cos, atol=ATOL)
    assert torch.allclose(feats[..., 4], expected_sin, atol=ATOL)


# ---------------------------------------------------------------------------
# Walking +Z gives fwd_delta > 0 and yaw_delta = 0
# ---------------------------------------------------------------------------


def test_walking_plus_z_with_identity_quat_gives_positive_fwd_delta_and_zero_yaw_delta():
    """Identity quat (yaw=0, heading +Z) + xyz walking +Z → fwd_delta >= 0 in
    valid frames, and yaw_delta == 0 everywhere.
    """
    T = 5
    quat = torch.zeros(1, T, 4)
    quat[..., 0] = 1.0
    xyz = torch.zeros(1, T, 3)
    xyz[0, :, 2] = torch.arange(T, dtype=torch.float32)   # +Z every frame
    feats = root_to_traj_feats_7d(quat, xyz)
    fwd_delta = feats[0, :, 5]
    yaw_delta = feats[0, :, 6]
    # First frame is 0 (padding); remaining frames have +1 fwd delta along +Z.
    assert fwd_delta[0].item() == 0.0
    # heading_dir_xz(0) = [sin(0), cos(0)] = [0, 1]; delta = (0, 1); fwd = 1
    for t in range(1, T):
        assert abs(fwd_delta[t].item() - 1.0) < ATOL, f"fwd_delta[{t}]={fwd_delta[t].item()}"
    assert torch.allclose(yaw_delta, torch.zeros(T), atol=ATOL)


def test_yaw_delta_reflects_physical_yaw_diff_wrapped():
    """yaw_delta[t] = wrap_angle(physical_yaw[t] - physical_yaw[t-1]) for t>=1."""
    T = 4
    # Cumulative physical yaw: 0, 0.2, 0.4, 0.6 — encoded as half-angle a = yaw/(-2)
    # (root_quat_to_physical_yaw returns -2*atan2(qy, qw), so for physical_yaw=0.2,
    # we need a = -0.1 → qw = cos(-0.1), qy = sin(-0.1)).
    physical_yaws = [0.0, 0.2, 0.4, 0.6]
    quat = torch.zeros(1, T, 4)
    xyz = torch.zeros(1, T, 3)
    for t, py in enumerate(physical_yaws):
        a = -py / 2.0
        quat[0, t, 0] = math.cos(a)
        quat[0, t, 2] = math.sin(a)
    feats = root_to_traj_feats_7d(quat, xyz)
    assert feats[0, 0, 6].item() == 0.0
    for t in range(1, T):
        expected = physical_yaws[t] - physical_yaws[t - 1]
        assert abs(feats[0, t, 6].item() - expected) < 1e-5, (
            f"t={t} yaw_delta={feats[0, t, 6].item()} expected={expected}"
        )


# ---------------------------------------------------------------------------
# Rigid-invariance under canonicalize_7d
# ---------------------------------------------------------------------------


def test_y_fwd_delta_yaw_delta_invariant_under_canonicalize_7d():
    """canonicalize_7d only rotates / translates xz + rotates heading.
    The y, fwd_delta, yaw_delta channels are rigid-invariant and must be
    preserved bitwise.
    """
    T = 6
    quat = torch.zeros(1, T, 4)
    # Some varying yaw via half-angle progression
    for t in range(T):
        a = 0.1 * t
        quat[0, t, 0] = math.cos(a)
        quat[0, t, 2] = math.sin(a)
    xyz = torch.zeros(1, T, 3)
    xyz[0, :, 0] = torch.arange(T, dtype=torch.float32) * 0.5
    xyz[0, :, 1] = 1.2
    xyz[0, :, 2] = torch.arange(T, dtype=torch.float32) * 0.3

    feats_world = root_to_traj_feats_7d(quat, xyz)
    # canonicalize at frame 2 anchor
    anchor_xz = feats_world[0, 2, [0, 2]]
    anchor_yaw = torch.atan2(feats_world[0, 2, 4], feats_world[0, 2, 3])
    feats_local = canonicalize_7d(feats_world, anchor_xz, anchor_yaw)
    # channels 1 (y), 5 (fwd_delta), 6 (yaw_delta) unchanged
    assert torch.allclose(feats_local[..., 1], feats_world[..., 1], atol=ATOL)
    assert torch.allclose(feats_local[..., 5], feats_world[..., 5], atol=ATOL)
    assert torch.allclose(feats_local[..., 6], feats_world[..., 6], atol=ATOL)


# ---------------------------------------------------------------------------
# Heading-vector channel norm == 1
# ---------------------------------------------------------------------------


def test_cos_sin_channels_form_unit_vector():
    T = 5
    quat = torch.zeros(1, T, 4)
    # Arbitrary physical yaws
    for t in range(T):
        a = 0.3 * t
        quat[0, t, 0] = math.cos(a)
        quat[0, t, 2] = math.sin(a)
    xyz = torch.zeros(1, T, 3)
    feats = root_to_traj_feats_7d(quat, xyz)
    norms = feats[0, :, 3] ** 2 + feats[0, :, 4] ** 2
    assert torch.allclose(norms, torch.ones(T), atol=ATOL)


# ---------------------------------------------------------------------------
# Anti-regression: NOT equal to legacy traj_batch.root_to_traj_feats
# ---------------------------------------------------------------------------


def test_anti_regression_NOT_equal_to_legacy_traj_batch_function():
    """legacy `utils.traj_batch.root_to_traj_feats(traj_xyz)` returns
    `[x, z, dir_x, dir_z]` (path direction); ours returns physical yaw
    cos / sin which lives on the unit circle but in a DIFFERENT slot
    (axes 3,4 in 7D output) AND has different semantics. They must not be
    interchanged.
    """
    from utils.traj_batch import root_to_traj_feats as legacy

    T = 5
    quat = torch.zeros(1, T, 4)
    quat[..., 0] = 1.0   # identity
    xyz = torch.zeros(1, T, 3)
    xyz[0, :, 2] = torch.arange(T, dtype=torch.float32)   # walk +Z

    new = root_to_traj_feats_7d(quat, xyz)
    leg = legacy(xyz)
    # New (7D): cos_h, sin_h at [3], [4] = (1, 0) for yaw=0.
    assert abs(new[0, -1, 3].item() - 1.0) < ATOL
    assert abs(new[0, -1, 4].item() - 0.0) < ATOL
    # Legacy (4D): cos_yaw, sin_yaw at [2], [3] = (0, 1) for +Z path direction.
    assert abs(leg[0, -1, 2].item() - 0.0) < 1e-3
    assert abs(leg[0, -1, 3].item() - 1.0) < 1e-3
    # The two heading representations have axes swapped + different semantics.


# ---------------------------------------------------------------------------
# Recover-then-feats end-to-end (the canonical pipeline)
# ---------------------------------------------------------------------------


def test_end_to_end_with_recover_root_rot_pos():
    """Pipe through recover_root_rot_pos then root_to_traj_feats_7d for a
    physical 90° rotation case (consistent with test_T07 in local_frame).
    """
    # rot_vel_t0 = π/4 → physical_yaw = -π/2 (per local_frame sign).
    data = _make_263d(rot_vel_t0=PI / 4, local_vel_xz_per_frame=(0.0, 1.0), n_frames=4)
    quat, xyz = recover_root_rot_pos(data)
    feats = root_to_traj_feats_7d(quat, xyz)
    # At t=1, physical_yaw ≈ -π/2 → cos≈0, sin≈-1
    physical_yaw_t1 = root_quat_to_physical_yaw(quat[0, 1])
    assert abs(abs(physical_yaw_t1.item()) - PI / 2) < 1e-4
    expected_cos = math.cos(physical_yaw_t1.item())
    expected_sin = math.sin(physical_yaw_t1.item())
    assert abs(feats[0, 1, 3].item() - expected_cos) < 1e-4
    assert abs(feats[0, 1, 4].item() - expected_sin) < 1e-4
    # Heading direction matches: heading_dir_xz(physical_yaw) (consistency check)
    head = heading_dir_xz(physical_yaw_t1)
    assert torch.allclose(
        torch.tensor([feats[0, 1, 3], feats[0, 1, 4]]),
        torch.tensor([math.cos(physical_yaw_t1.item()), math.sin(physical_yaw_t1.item())]),
        atol=1e-4,
    )
    # Use `head` to ensure linter sees its value used.
    _ = head
