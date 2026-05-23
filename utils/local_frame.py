"""B-full local-frame geometry (sole world ↔ local convention module).

References:
- docs/design.md §0.3 (Anchor Convention v1).
- docs/TODO.md §T_A_01 (this module's spec) + §legacy heading 分析:
  legacy `cos_yaw/sin_yaw` in `utils/motion_process.py` (= raw quaternion `qw/qy`,
  half-angle) and `utils/traj_batch.py` (= path direction unit vector) are NOT
  `(cos(physical_yaw), sin(physical_yaw))`. New 7D heading must be derived via
  `root_quat_to_physical_yaw`, never reused from legacy fields.

Axis convention (Y-up, XZ ground plane, yaw around +Y, yaw=0 faces +Z):
    heading_dir_xz(yaw) = [sin(yaw), cos(yaw)]
    yaw_to_matrix(yaw) @ [0, 0, 1] xz-投影 = heading_dir_xz(yaw)

This module is **pure geometry**: it does not import motion_process / traj_batch
/ dataset code, and it does not own 263D recovery.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_PI = math.pi
_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Basic yaw / rotation matrix helpers
# ---------------------------------------------------------------------------


def wrap_angle(theta: Tensor) -> Tensor:
    """Wrap angle to [-pi, pi).

    Used by yaw-error metric, matrix_to_yaw, advance_head_from_body_window.
    """
    return (theta + _PI) % _TWO_PI - _PI


def yaw_to_matrix(yaw: Tensor) -> Tensor:
    """physical yaw → 3x3 rotation matrix (Y-up).

    Convention: yaw=0 faces +Z; yaw_to_matrix(yaw) @ [0, 0, 1] xz-投影 equals
    `heading_dir_xz(yaw)`. Output shape: yaw.shape + (3, 3).
    """
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    row0 = torch.stack([c, zero, s], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-s, zero, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_yaw(R: Tensor) -> Tensor:
    """3x3 rotation matrix → physical yaw, wrapped to [-pi, pi).

    Satisfies `matrix_to_yaw(yaw_to_matrix(yaw)) ≈ wrap_angle(yaw)` (T03).
    Indices follow our Y-up convention (NOT MuJoCo Z-up `atan2(R[1,0], R[0,0])`);
    they are locked by T05 (heading_dir_xz convention).

    NaN/Inf inputs return 0.0 (T04 fallback).
    """
    sin_yaw = R[..., 0, 2]
    cos_yaw = R[..., 2, 2]
    yaw = torch.atan2(sin_yaw, cos_yaw)
    finite = torch.isfinite(sin_yaw) & torch.isfinite(cos_yaw)
    yaw = torch.where(finite, yaw, torch.zeros_like(yaw))
    return wrap_angle(yaw)


def heading_dir_xz(yaw: Tensor) -> Tensor:
    """physical yaw → 2D unit forward direction on XZ plane.

    `heading_dir_xz(yaw) = [sin(yaw), cos(yaw)]` so that yaw=0 → (+Z, [0, 1]).
    Used by `fwd_delta = dot(delta_xz, heading_dir_xz(yaw))`.
    """
    return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


# ---------------------------------------------------------------------------
# physical yaw from recovered root quaternion
# ---------------------------------------------------------------------------


def root_quat_to_physical_yaw(root_quat: Tensor) -> Tensor:
    """Recovered root quaternion `[qw, qx, qy, qz]` → physical yaw (rad).

    `recover_root_rot_pos` (in `utils.motion_process`) returns
    `quat = [cos(a), 0, sin(a), 0]` (a = `r_rot_ang`, a half-angle), which
    encodes rotation `2a` around +Y under standard quaternion semantics.

    Sign: HumanML3D's `r_rot_ang` accumulates with a sign opposite to our
    `heading_dir_xz(yaw) = [sin(yaw), cos(yaw)]` convention (where +yaw=π/2
    means body faces +X). Empirically (T07): with `r_rot_ang` set to +π/4 and
    motion walking along local +Z, the recovered world displacement points
    along world -X — which matches `heading_dir_xz(-π/2) = [-1, 0]`, i.e.
    physical_yaw = -π/2 for r_rot_ang = +π/4.

    Therefore `physical_yaw = -2 * atan2(qy, qw)`. NaN/Inf → 0.0 fallback.
    """
    qw = root_quat[..., 0]
    qy = root_quat[..., 2]
    yaw = -2.0 * torch.atan2(qy, qw)
    finite = torch.isfinite(qw) & torch.isfinite(qy)
    yaw = torch.where(finite, yaw, torch.zeros_like(yaw))
    return wrap_angle(yaw)


# ---------------------------------------------------------------------------
# xz transform helpers (no y, no heading)
# ---------------------------------------------------------------------------


def _xz_rotate(x: Tensor, z: Tensor, c: Tensor, s: Tensor, *, inverse: bool):
    """Rotate (x, z) by anchor_yaw (forward) or -anchor_yaw (inverse).

    `c = cos(anchor_yaw)`, `s = sin(anchor_yaw)`. yaw_to_matrix(yaw) on xz:
        x' = c*x + s*z; z' = -s*x + c*z      (forward, rotate by +anchor_yaw)
        x' = c*x - s*z; z' =  s*x + c*z      (inverse, rotate by -anchor_yaw)
    """
    if inverse:
        return c * x - s * z, s * x + c * z
    return c * x + s * z, -s * x + c * z


def transform_xz_world_to_local(xz_world: Tensor,
                                anchor_xz: Tensor,
                                anchor_yaw: Tensor) -> Tensor:
    """world xz → local xz: `R_y(-anchor_yaw) @ (xz_world - anchor_xz)`.

    `xz_world: [..., 2]`. `anchor_xz: [..., 2]` and `anchor_yaw: [...]` broadcast
    against the leading dims of `xz_world` (without the trailing `2`).
    """
    diff_x = xz_world[..., 0] - anchor_xz[..., 0]
    diff_z = xz_world[..., 1] - anchor_xz[..., 1]
    c = torch.cos(anchor_yaw)
    s = torch.sin(anchor_yaw)
    x_local, z_local = _xz_rotate(diff_x, diff_z, c, s, inverse=True)
    return torch.stack([x_local, z_local], dim=-1)


def transform_xz_local_to_world(xz_local: Tensor,
                                anchor_xz: Tensor,
                                anchor_yaw: Tensor) -> Tensor:
    """Inverse of `transform_xz_world_to_local`: `R_y(anchor_yaw) @ xz + anchor`."""
    c = torch.cos(anchor_yaw)
    s = torch.sin(anchor_yaw)
    x_world_rel, z_world_rel = _xz_rotate(
        xz_local[..., 0], xz_local[..., 1], c, s, inverse=False,
    )
    x = x_world_rel + anchor_xz[..., 0]
    z = z_world_rel + anchor_xz[..., 1]
    return torch.stack([x, z], dim=-1)


def transform_xz_local_delta_to_world(delta_xz_local: Tensor,
                                       ref_world_yaw: Tensor) -> Tensor:
    """body-local xz delta → world xz delta (pure rotation, no translation).

    Used by `advance_head_from_body_window` (inference glue):
        `world_xz_new = world_xz_old + transform_xz_local_delta_to_world(...)`.
    `delta_xz_local: [..., 2]`, `ref_world_yaw: [...]`.
    """
    c = torch.cos(ref_world_yaw)
    s = torch.sin(ref_world_yaw)
    x_world, z_world = _xz_rotate(
        delta_xz_local[..., 0], delta_xz_local[..., 1], c, s, inverse=False,
    )
    return torch.stack([x_world, z_world], dim=-1)


# ---------------------------------------------------------------------------
# 5D / 7D canonicalize (the only world ↔ local API for motion / waypoint tensors)
# ---------------------------------------------------------------------------


def _broadcast_anchor(anchor: Tensor, traj: Tensor, *, last_dim: int) -> Tensor:
    """Insert a singleton T-axis into `anchor` so it broadcasts with
    `traj: [..., T, last_dim]`. `anchor: [..., last_dim]` or `[...]` for scalar.
    """
    if last_dim == 0:
        # anchor is a scalar (yaw): shape [..., ] → add a singleton on last to
        # broadcast against traj's T-axis.
        return anchor.unsqueeze(-1) if anchor.dim() >= 1 else anchor.reshape(1)
    return anchor.unsqueeze(-2)


def _apply_heading_rotation(traj: Tensor, anchor_yaw: Tensor, *, forward: bool):
    """Rotate (cos_h, sin_h) channels of `traj` by ±anchor_yaw, in-place safe.

    forward=True : world → local, theta_local = theta_world - anchor_yaw
    forward=False: local → world, theta_world = theta_local + anchor_yaw

    Mutates `traj[..., 3]` and `traj[..., 4]`.
    """
    cos_old = traj[..., 3].clone()
    sin_old = traj[..., 4].clone()
    cos_a = torch.cos(anchor_yaw)
    sin_a = torch.sin(anchor_yaw)
    if forward:
        traj[..., 3] = cos_old * cos_a + sin_old * sin_a
        traj[..., 4] = sin_old * cos_a - cos_old * sin_a
    else:
        traj[..., 3] = cos_old * cos_a - sin_old * sin_a
        traj[..., 4] = sin_old * cos_a + cos_old * sin_a


def _apply_xz_translate_rotate(traj: Tensor, anchor_xz: Tensor, anchor_yaw: Tensor,
                                *, forward: bool):
    """xz channels: world → local (subtract anchor + rotate -anchor_yaw) or
    local → world (rotate +anchor_yaw + add anchor). Mutates `traj[..., 0]` and
    `traj[..., 2]` (y at `traj[..., 1]` is left untouched).
    """
    c = torch.cos(anchor_yaw)
    s = torch.sin(anchor_yaw)
    if forward:
        diff_x = traj[..., 0] - anchor_xz[..., 0]
        diff_z = traj[..., 2] - anchor_xz[..., 1]
        x_new, z_new = _xz_rotate(diff_x, diff_z, c, s, inverse=True)
    else:
        x_rel, z_rel = _xz_rotate(traj[..., 0], traj[..., 2], c, s, inverse=False)
        x_new = x_rel + anchor_xz[..., 0]
        z_new = z_rel + anchor_xz[..., 1]
    traj[..., 0] = x_new
    traj[..., 2] = z_new


def canonicalize_5d(motion_5d_world: Tensor,
                    anchor_xz: Tensor,
                    anchor_yaw: Tensor) -> Tensor:
    """5D = [x, y, z, cos(physical_yaw), sin(physical_yaw)] world → local.

    `motion_5d_world: [..., T, 5]`; `anchor_xz: [..., 2]`; `anchor_yaw: [...]`.
    y (channel 1) is preserved (anchor convention §0.3: only xz / heading are
    canonicalized; physical y is kept).
    """
    out = motion_5d_world.clone()
    anchor_xz_b = anchor_xz.unsqueeze(-2)        # [..., 1, 2]
    anchor_yaw_b = anchor_yaw.unsqueeze(-1)      # [..., 1]
    _apply_xz_translate_rotate(out, anchor_xz_b, anchor_yaw_b, forward=True)
    _apply_heading_rotation(out, anchor_yaw_b, forward=True)
    return out


def canonicalize_7d(traj_7d_world: Tensor,
                    anchor_xz: Tensor,
                    anchor_yaw: Tensor) -> Tensor:
    """7D = [x, y, z, cos(physical_yaw), sin(physical_yaw), fwd, yaw_delta] world → local.

    `traj_7d_world: [..., T, 7]`; `anchor_xz: [..., 2]`; `anchor_yaw: [...]`.
    fwd / yaw_delta (channels 5, 6) are rigid-invariant and left untouched.
    """
    out = traj_7d_world.clone()
    anchor_xz_b = anchor_xz.unsqueeze(-2)
    anchor_yaw_b = anchor_yaw.unsqueeze(-1)
    _apply_xz_translate_rotate(out, anchor_xz_b, anchor_yaw_b, forward=True)
    _apply_heading_rotation(out, anchor_yaw_b, forward=True)
    return out


def uncanonicalize_7d(traj_7d_local: Tensor,
                      anchor_xz: Tensor,
                      anchor_yaw: Tensor) -> Tensor:
    """Inverse of `canonicalize_7d`: local → world. Round-trip identity T09.

    `traj_7d_local: [..., T, 7]`; `anchor_xz: [..., 2]`; `anchor_yaw: [...]`.
    """
    out = traj_7d_local.clone()
    anchor_xz_b = anchor_xz.unsqueeze(-2)
    anchor_yaw_b = anchor_yaw.unsqueeze(-1)
    _apply_heading_rotation(out, anchor_yaw_b, forward=False)
    _apply_xz_translate_rotate(out, anchor_xz_b, anchor_yaw_b, forward=False)
    return out


__all__ = [
    "wrap_angle",
    "yaw_to_matrix",
    "matrix_to_yaw",
    "heading_dir_xz",
    "root_quat_to_physical_yaw",
    "transform_xz_world_to_local",
    "transform_xz_local_to_world",
    "transform_xz_local_delta_to_world",
    "canonicalize_5d",
    "canonicalize_7d",
    "uncanonicalize_7d",
]
