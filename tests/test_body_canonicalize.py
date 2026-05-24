"""Unit tests for T_B_05 body-window canonicalize (docs/TODO.md §T_B_05, design §2.3)."""

from __future__ import annotations

import math

import torch

from utils.token_frame import token_body_window_left_frame, token_start_frame
from utils.training.body_canonicalize import apply_body_window_canonicalize


def _pose_feat(x, y, z, yaw, fwd=0.0, yawd=0.0):
    return torch.tensor([x, y, z, math.cos(yaw), math.sin(yaw), fwd, yawd])


# ---------------------------------------------------------------------------
# 2 (TODO): anchor frame → (0, y, 0, 1, 0, *, *); y / fwd / yaw_delta untouched
# ---------------------------------------------------------------------------


def test_anchor_frame_becomes_origin_identity_heading():
    B, T = 1, 10
    x0, y0, z0, yaw0 = 1.5, 0.9, -2.0, 0.7
    traj = torch.randn(B, T, 7)
    traj[0, 0] = _pose_feat(x0, y0, z0, yaw0, fwd=0.3, yawd=0.1)
    gt_xyz = torch.zeros(B, T, 3)
    gt_xyz[0, 0] = torch.tensor([x0, y0, z0])
    gt_yaw = torch.zeros(B, T)
    gt_yaw[0, 0] = yaw0
    # end=2, body_window_tokens=10 → bwl = token_start_frame(max(0,2-10)) = 0
    out, mask = apply_body_window_canonicalize(
        traj, torch.tensor([2]), gt_xyz, gt_yaw, torch.tensor([T]), body_window_tokens=10
    )
    assert mask[0] == 1.0
    a = out[0, 0]
    assert abs(float(a[0])) < 1e-5            # x → 0
    assert abs(float(a[1]) - y0) < 1e-5       # y unchanged
    assert abs(float(a[2])) < 1e-5            # z → 0
    assert abs(float(a[3]) - 1.0) < 1e-5      # cos_h → 1
    assert abs(float(a[4])) < 1e-5            # sin_h → 0
    assert abs(float(a[5]) - 0.3) < 1e-5      # fwd unchanged
    assert abs(float(a[6]) - 0.1) < 1e-5      # yaw_delta unchanged


# ---------------------------------------------------------------------------
# 5 (TODO): anchor uses token_body_window_left_frame, NOT active-window-left
# ---------------------------------------------------------------------------


def test_anchor_uses_body_window_left_frame():
    B, T = 1, 50
    end_token, bw = 30, 20
    bwl = token_body_window_left_frame(end_token, bw)   # token_start_frame(10) = 37
    assert bwl == token_start_frame(10) == 37
    x0, y0, z0, yaw0 = 4.0, 1.1, 3.0, -0.5
    traj = torch.randn(B, T, 7)
    traj[0, bwl] = _pose_feat(x0, y0, z0, yaw0)
    gt_xyz = torch.zeros(B, T, 3)
    gt_xyz[0, bwl] = torch.tensor([x0, y0, z0])
    gt_yaw = torch.zeros(B, T)
    gt_yaw[0, bwl] = yaw0
    out, mask = apply_body_window_canonicalize(
        traj, torch.tensor([end_token]), gt_xyz, gt_yaw, torch.tensor([T]),
        body_window_tokens=bw,
    )
    assert mask[0] == 1.0
    # the body-window-left frame (37) is the one mapped to the origin
    a = out[0, bwl]
    assert abs(float(a[0])) < 1e-5 and abs(float(a[2])) < 1e-5
    assert abs(float(a[3]) - 1.0) < 1e-5 and abs(float(a[4])) < 1e-5


# ---------------------------------------------------------------------------
# 3 (TODO): invalid sample → sample_loss_mask=0 but STILL canonicalized
# ---------------------------------------------------------------------------


def test_invalid_sample_masked_but_still_canonicalized():
    B, T = 1, 10
    valid_len = 3
    x0, y0, z0, yaw0 = 2.0, 0.8, -1.0, 0.4
    traj = torch.randn(B, T, 7)
    fallback_idx = valid_len - 1   # = 2
    traj[0, fallback_idx] = _pose_feat(x0, y0, z0, yaw0)
    gt_xyz = torch.zeros(B, T, 3)
    gt_xyz[0, fallback_idx] = torch.tensor([x0, y0, z0])
    gt_yaw = torch.zeros(B, T)
    gt_yaw[0, fallback_idx] = yaw0
    # end=50, bw=5 → bwl huge, clamp to T-1=9 >= valid_len 3 → invalid
    out, mask = apply_body_window_canonicalize(
        traj, torch.tensor([50]), gt_xyz, gt_yaw, torch.tensor([valid_len]),
        body_window_tokens=5,
    )
    assert mask[0] == 0.0                       # masked
    assert not torch.equal(out, traj)           # still canonicalized (not world)
    a = out[0, fallback_idx]                     # fallback anchor → origin
    assert abs(float(a[0])) < 1e-5 and abs(float(a[2])) < 1e-5


# ---------------------------------------------------------------------------
# 4 (TODO): rigid invariance — fwd/yaw_delta and y untouched everywhere
# ---------------------------------------------------------------------------


def test_rigid_invariant_channels_unchanged():
    B, T = 2, 12
    traj = torch.randn(B, T, 7)
    gt_xyz = torch.randn(B, T, 3)
    gt_yaw = torch.randn(B, T)
    out, _ = apply_body_window_canonicalize(
        traj, torch.tensor([8, 6]), gt_xyz, gt_yaw, torch.tensor([T, T]),
        body_window_tokens=10,
    )
    # y (1), fwd (5), yaw_delta (6) are rigid-invariant → identical
    assert torch.allclose(out[..., 1], traj[..., 1], atol=1e-6)
    assert torch.allclose(out[..., 5], traj[..., 5], atol=1e-6)
    assert torch.allclose(out[..., 6], traj[..., 6], atol=1e-6)


def test_all_valid_sample_loss_mask_one():
    B, T = 3, 20
    traj = torch.randn(B, T, 7)
    gt_xyz = torch.randn(B, T, 3)
    gt_yaw = torch.randn(B, T)
    _, mask = apply_body_window_canonicalize(
        traj, torch.tensor([5, 6, 7]), gt_xyz, gt_yaw, torch.tensor([T, T, T]),
        body_window_tokens=15,
    )
    assert torch.equal(mask, torch.ones(B))


def test_rejects_non_7d():
    import pytest
    with pytest.raises(ValueError):
        apply_body_window_canonicalize(
            torch.randn(1, 10, 4), torch.tensor([2]),
            torch.randn(1, 10, 3), torch.randn(1, 10), torch.tensor([10]),
            body_window_tokens=5,
        )
