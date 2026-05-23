"""Unit tests for utils/local_frame.py (T_A_01).

Covers T01-T18 per docs/TODO.md §T_A_01 Unit tests.
"""

from __future__ import annotations

import math

import torch

from utils.local_frame import (
    canonicalize_5d,
    canonicalize_7d,
    heading_dir_xz,
    matrix_to_yaw,
    root_quat_to_physical_yaw,
    transform_xz_local_delta_to_world,
    transform_xz_local_to_world,
    transform_xz_world_to_local,
    uncanonicalize_7d,
    wrap_angle,
    yaw_to_matrix,
)

ATOL = 1e-5
PI = math.pi


def _t(x):
    return torch.as_tensor(x, dtype=torch.float64)


# ---------------------------------------------------------------------------
# T01-T05: basic yaw / matrix / heading_dir
# ---------------------------------------------------------------------------


def test_T01_yaw_to_matrix_zero_is_identity():
    R = yaw_to_matrix(_t(0.0))
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=ATOL)


def test_T02_yaw_to_matrix_self_inverse():
    yaw = _t(0.7)
    R = yaw_to_matrix(yaw) @ yaw_to_matrix(-yaw)
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=ATOL)


def test_T03_matrix_to_yaw_round_trip():
    yaws = _t([-PI + 0.1, -1.0, -0.3, 0.0, 0.3, 1.0, PI - 0.1])
    for y in yaws:
        recovered = matrix_to_yaw(yaw_to_matrix(y))
        assert torch.allclose(recovered, wrap_angle(y), atol=ATOL), (
            f"yaw={y.item():.4f} recovered={recovered.item():.4f}"
        )


def test_T04_matrix_to_yaw_nan_inf_fallback():
    R = torch.eye(3, dtype=torch.float64)
    R[0, 2] = float("nan")
    assert matrix_to_yaw(R).item() == 0.0
    R = torch.eye(3, dtype=torch.float64)
    R[2, 2] = float("inf")
    assert matrix_to_yaw(R).item() == 0.0


def test_T05_yaw_to_matrix_z_axis_projection_matches_heading_dir():
    for yaw_val in (0.0, 0.3, -0.7, PI / 4, -PI / 3):
        yaw = _t(yaw_val)
        z_axis = _t([0.0, 0.0, 1.0])
        rotated = yaw_to_matrix(yaw) @ z_axis
        xz_proj = torch.stack([rotated[0], rotated[2]], dim=-1)
        expected = heading_dir_xz(yaw)
        assert torch.allclose(xz_proj, expected, atol=ATOL), (
            f"yaw={yaw_val} xz_proj={xz_proj.tolist()} expected={expected.tolist()}"
        )


# ---------------------------------------------------------------------------
# T06/T07: physical yaw sign — constructed 263D motion via recover_root_rot_pos
# ---------------------------------------------------------------------------


def _make_263d_motion(rot_vel_t0: float, local_vel_xz_per_frame, n_frames: int = 4):
    """Build a minimal [1, T, 263] motion tensor.

    Layout (used by `utils.motion_process.recover_root_rot_pos`):
      data[..., 0] = root rotation velocity (per frame; cumsum after shift)
      data[..., 1:3] = root linear velocity xz (in local frame, shifted)
      data[..., 3] = root y
      ... (joint features ignored by recover_root_rot_pos)
    """
    data = torch.zeros(1, n_frames, 263, dtype=torch.float32)
    # set rot_vel at frame 0 so r_rot_ang[1:] = rot_vel_t0 (others = 0).
    data[0, 0, 0] = rot_vel_t0
    # local xz vel per frame
    for t in range(n_frames):
        vx, vz = local_vel_xz_per_frame
        data[0, t, 1] = vx
        data[0, t, 2] = vz
    data[..., 3] = 1.0  # constant root y, irrelevant to yaw
    return data


def test_T06_physical_yaw_case_A_zero_rotation():
    from utils.motion_process import recover_root_rot_pos

    data = _make_263d_motion(rot_vel_t0=0.0, local_vel_xz_per_frame=(0.0, 1.0))
    quat, pos = recover_root_rot_pos(data)
    yaw = root_quat_to_physical_yaw(quat)
    # All yaws should be ≈ 0 (no rotation).
    assert torch.allclose(yaw, torch.zeros_like(yaw), atol=1e-4), (
        f"expected ~0, got {yaw[0].tolist()}"
    )
    # heading_dir_xz(0) ≈ [0, 1] → +Z; displacement (frame 2 - frame 1) should be +Z.
    disp = pos[0, 2, [0, 2]] - pos[0, 1, [0, 2]]   # [x, z]
    disp_norm = disp / disp.norm().clamp(min=1e-8)
    head = heading_dir_xz(yaw[0, 1])
    assert torch.allclose(disp_norm.double(), head.double(), atol=1e-3), (
        f"disp_norm={disp_norm.tolist()} head={head.tolist()}"
    )


def test_T07_physical_yaw_case_B_quarter_turn_then_forward():
    """After a physical +π/2 rotation at frame 0, local +Z motion at subsequent
    frames should produce a displacement direction matching
    `heading_dir_xz(physical_yaw)`. This test LOCKS the sign convention in
    `root_quat_to_physical_yaw`.

    Note: HumanML3D stores `r_rot_ang` as a half-angle (the quaternion is built
    as `[cos(a), 0, sin(a), 0]` which encodes physical rotation `2a`). Hence
    `rot_vel_t0 = PI / 4` is what produces a physical 90° rotation.
    """
    from utils.motion_process import recover_root_rot_pos

    data = _make_263d_motion(rot_vel_t0=PI / 4, local_vel_xz_per_frame=(0.0, 1.0))
    quat, pos = recover_root_rot_pos(data)
    yaw = root_quat_to_physical_yaw(quat)
    yaw_t1 = yaw[0, 1].item()
    assert abs(abs(yaw_t1) - PI / 2) < 1e-4, f"expected |yaw|≈π/2, got {yaw_t1}"

    disp = pos[0, 2, [0, 2]] - pos[0, 1, [0, 2]]
    disp_norm = disp / disp.norm().clamp(min=1e-8)
    head = heading_dir_xz(yaw[0, 1])
    assert torch.allclose(disp_norm.double(), head.double(), atol=1e-3), (
        f"sign mismatch: disp_norm={disp_norm.tolist()} head={head.tolist()} "
        f"yaw={yaw_t1:.4f}. Flip the sign in root_quat_to_physical_yaw."
    )


# ---------------------------------------------------------------------------
# T08-T12: canonicalize / uncanonicalize
# ---------------------------------------------------------------------------


def _sample_7d(T: int = 5) -> torch.Tensor:
    """Random [T, 7] motion tensor in world frame."""
    g = torch.Generator().manual_seed(42)
    pos = torch.randn(T, 3, generator=g, dtype=torch.float64)
    yaw = torch.randn(T, generator=g, dtype=torch.float64)
    cos_h = torch.cos(yaw).unsqueeze(-1)
    sin_h = torch.sin(yaw).unsqueeze(-1)
    fwd = torch.randn(T, 1, generator=g, dtype=torch.float64)
    yaw_d = torch.randn(T, 1, generator=g, dtype=torch.float64)
    return torch.cat([pos, cos_h, sin_h, fwd, yaw_d], dim=-1)


def test_T08_canonicalize_7d_anchor_frame_origin():
    """traj_world's first frame should map to (0, y_anchor, 0, 1, 0, *, *)."""
    traj = _sample_7d(T=5)
    anchor_xz = traj[0, [0, 2]].clone()
    anchor_yaw = torch.atan2(traj[0, 4], traj[0, 3])  # physical yaw from cos/sin
    out = canonicalize_7d(traj, anchor_xz, anchor_yaw)
    assert torch.allclose(out[0, 0], _t(0.0), atol=ATOL)
    assert torch.allclose(out[0, 1], traj[0, 1], atol=ATOL)  # y preserved
    assert torch.allclose(out[0, 2], _t(0.0), atol=ATOL)
    assert torch.allclose(out[0, 3], _t(1.0), atol=ATOL)
    assert torch.allclose(out[0, 4], _t(0.0), atol=ATOL)


def test_T09_canonicalize_7d_round_trip():
    traj = _sample_7d(T=5)
    anchor_xz = _t([1.5, -0.7])
    anchor_yaw = _t(0.6)
    out = canonicalize_7d(traj, anchor_xz, anchor_yaw)
    recovered = uncanonicalize_7d(out, anchor_xz, anchor_yaw)
    assert torch.allclose(recovered, traj, atol=ATOL)


def test_T10_canonicalize_preserves_y_fwd_yawdelta():
    traj = _sample_7d(T=5)
    anchor_xz = _t([1.5, -0.7])
    anchor_yaw = _t(0.6)
    out = canonicalize_7d(traj, anchor_xz, anchor_yaw)
    # channels 1 (y), 5 (fwd), 6 (yaw_delta) untouched
    assert torch.allclose(out[..., 1], traj[..., 1], atol=ATOL)
    assert torch.allclose(out[..., 5], traj[..., 5], atol=ATOL)
    assert torch.allclose(out[..., 6], traj[..., 6], atol=ATOL)


def test_T11_canonicalize_5d_anchor_frame_origin():
    """5D = [x, y, z, cos_h, sin_h] anchor frame canonicalize."""
    g = torch.Generator().manual_seed(0)
    T = 4
    motion = torch.zeros(T, 5, dtype=torch.float64)
    pos = torch.randn(T, 3, generator=g, dtype=torch.float64)
    yaw = torch.randn(T, generator=g, dtype=torch.float64)
    motion[:, :3] = pos
    motion[:, 3] = torch.cos(yaw)
    motion[:, 4] = torch.sin(yaw)
    anchor_xz = motion[0, [0, 2]].clone()
    anchor_yaw = yaw[0].clone()
    out = canonicalize_5d(motion, anchor_xz, anchor_yaw)
    assert torch.allclose(out[0, 0], _t(0.0), atol=ATOL)
    assert torch.allclose(out[0, 1], motion[0, 1], atol=ATOL)  # y preserved
    assert torch.allclose(out[0, 2], _t(0.0), atol=ATOL)
    assert torch.allclose(out[0, 3], _t(1.0), atol=ATOL)
    assert torch.allclose(out[0, 4], _t(0.0), atol=ATOL)


def test_T12_batched_canonicalize_different_anchors():
    g = torch.Generator().manual_seed(7)
    B, T = 3, 4
    pos = torch.randn(B, T, 3, generator=g, dtype=torch.float64)
    yaw = torch.randn(B, T, generator=g, dtype=torch.float64)
    cos_h = torch.cos(yaw).unsqueeze(-1)
    sin_h = torch.sin(yaw).unsqueeze(-1)
    fwd = torch.randn(B, T, 1, generator=g, dtype=torch.float64)
    yaw_d = torch.randn(B, T, 1, generator=g, dtype=torch.float64)
    traj = torch.cat([pos, cos_h, sin_h, fwd, yaw_d], dim=-1)  # [B, T, 7]

    anchor_xz = traj[:, 0, [0, 2]].clone()                       # [B, 2]
    anchor_yaw = torch.atan2(traj[:, 0, 4], traj[:, 0, 3])       # [B]

    out = canonicalize_7d(traj, anchor_xz, anchor_yaw)
    # Each batch's first frame should be at origin with heading (1, 0).
    assert torch.allclose(out[:, 0, 0], torch.zeros(B, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(out[:, 0, 2], torch.zeros(B, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(out[:, 0, 3], torch.ones(B, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(out[:, 0, 4], torch.zeros(B, dtype=torch.float64), atol=ATOL)
    # Round-trip per batch.
    recovered = uncanonicalize_7d(out, anchor_xz, anchor_yaw)
    assert torch.allclose(recovered, traj, atol=ATOL)


# ---------------------------------------------------------------------------
# T13/T14: legacy regression — assert legacy != new physical heading
# ---------------------------------------------------------------------------


def test_T13_legacy_traj_batch_root_to_traj_feats_not_equal_to_physical_heading():
    """legacy `utils.traj_batch.root_to_traj_feats(traj_xyz)` outputs
    `[x, z, cos_yaw, sin_yaw]` where `(cos_yaw, sin_yaw)` is the path direction
    unit vector. Walking along +Z gives legacy (0, 1) at the moving frames.

    New physical heading for yaw=0 is `(cos(0), sin(0)) = (1, 0)`.
    These two must NOT be equal — that's the whole reason new 7D heading must
    be derived freshly from physical yaw, never reusing the legacy column.
    """
    from utils.traj_batch import root_to_traj_feats

    T = 5
    traj_xyz = torch.zeros(1, T, 3)
    traj_xyz[0, :, 2] = torch.arange(T, dtype=torch.float32)   # walk +Z
    feats = root_to_traj_feats(traj_xyz)   # [1, T, 4] = [x, z, cos_yaw, sin_yaw]
    legacy_cos = feats[0, -1, 2].item()    # last frame
    legacy_sin = feats[0, -1, 3].item()
    assert abs(legacy_cos - 0.0) < 1e-3 and abs(legacy_sin - 1.0) < 1e-3, (
        f"legacy (cos, sin) at last frame = ({legacy_cos}, {legacy_sin}); "
        f"expected (0, 1) for +Z path direction"
    )
    # Physical heading for yaw=0 (identity rotation) is (1, 0).
    physical_cos, physical_sin = 1.0, 0.0
    assert (legacy_cos, legacy_sin) != (physical_cos, physical_sin), (
        "legacy path-direction heading must NOT equal new physical-yaw heading; "
        "found equality — this means legacy heading column was reused as physical "
        "yaw cos/sin somewhere."
    )


def test_T14_legacy_extract_root_xz_phi_features_differs_from_physical_yaw():
    """legacy `extract_root_xz_phi_features_263` outputs cos_yaw=qw, sin_yaw=qy
    (half-angle of physical yaw). For physical_yaw=π/2, qw=cos(π/4)=√2/2,
    qy=sin(π/4)=√2/2, vs (cos(π/2), sin(π/2))=(0, 1). They must NOT match.
    """
    # Construct a 263D motion that has physical_yaw ≈ π/2 at frame ≥ 1.
    from utils.motion_process import extract_root_xz_phi_features_263, recover_root_rot_pos

    # rot_vel_t0 = π/4 → r_rot_ang = π/4 → physical yaw magnitude = π/2.
    # legacy 4D extractor reports cos_yaw = qw = cos(π/4) = √2/2, sin_yaw = qy = sin(π/4) = √2/2.
    data = _make_263d_motion(rot_vel_t0=PI / 4, local_vel_xz_per_frame=(0.0, 0.0))
    # call recover to get the quat (just to sanity check)
    quat, _ = recover_root_rot_pos(data)
    physical_yaw = root_quat_to_physical_yaw(quat)[0, 1]
    assert abs(abs(physical_yaw.item()) - PI / 2) < 1e-4

    # legacy 4D feature extraction
    feats_4d = extract_root_xz_phi_features_263(data[0].numpy())   # (T, 4) = [x, z, cos_yaw, sin_yaw]
    legacy_cos = feats_4d[1, 2]   # frame 1 cos_yaw (= qw = √2/2)
    legacy_sin = feats_4d[1, 3]   # frame 1 sin_yaw (= qy = √2/2)
    # Compare to physical heading at frame 1
    physical_cos = math.cos(physical_yaw.item())
    physical_sin = math.sin(physical_yaw.item())
    # legacy ≈ (√2/2, √2/2), physical ≈ (0, ±1): must differ.
    assert abs(legacy_cos - physical_cos) > 0.3 or abs(legacy_sin - physical_sin) > 0.3, (
        f"legacy=({legacy_cos:.4f}, {legacy_sin:.4f}) "
        f"physical=({physical_cos:.4f}, {physical_sin:.4f}) "
        f"should NOT be equivalent."
    )


# ---------------------------------------------------------------------------
# T15-T18: xz transform helpers
# ---------------------------------------------------------------------------


def test_T15_world_to_local_anchor_maps_to_origin():
    anchor_xz = _t([3.0, -2.0])
    anchor_yaw = _t(0.4)
    out = transform_xz_world_to_local(anchor_xz, anchor_xz, anchor_yaw)
    assert torch.allclose(out, torch.zeros(2, dtype=torch.float64), atol=ATOL)


def test_T16_world_to_local_round_trip():
    xz = _t([2.5, 1.3])
    anchor_xz = _t([1.0, -0.5])
    anchor_yaw = _t(-0.8)
    local = transform_xz_world_to_local(xz, anchor_xz, anchor_yaw)
    recovered = transform_xz_local_to_world(local, anchor_xz, anchor_yaw)
    assert torch.allclose(recovered, xz, atol=ATOL)


def test_T17_local_delta_to_world_zero_yaw_is_identity():
    delta = _t([0.0, 1.0])    # local +Z
    out = transform_xz_local_delta_to_world(delta, _t(0.0))
    assert torch.allclose(out, _t([0.0, 1.0]), atol=ATOL)


def test_T18_local_delta_to_world_quarter_yaw_to_world_x():
    """yaw=+π/2: local +Z delta → world +X delta (consistent with
    `yaw_to_matrix(π/2) @ [0,0,1] = [1, 0, 0]` and `heading_dir_xz(π/2) = [1, 0]`).
    """
    delta = _t([0.0, 1.0])    # local +Z
    out = transform_xz_local_delta_to_world(delta, _t(PI / 2))
    assert torch.allclose(out, _t([1.0, 0.0]), atol=ATOL)
