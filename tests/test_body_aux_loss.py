"""Unit tests for T_B_06 body aux loss core (docs/TODO.md §T_B_06, design §2.4).

The 5 loss terms operate on already-recovered poses (no VAE), so they are fully
unit-testable. The VAE-decode wrapper (compute_body_aux_loss) needs a real VAE
and is verified at runtime, not here.
"""

from __future__ import annotations

import math

import torch

from utils.training.control_loss import (
    body_aux_loss_terms,
    derive_fwd_yaw_delta,
    masked_smooth_l1,
)

_W = {"root_xz": 1.0, "root_y": 0.3, "heading": 0.5, "fwd_delta": 0.1, "yaw_delta": 0.1}


def _poses(B=1, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    xyz = torch.randn(B, T, 3, generator=g, dtype=torch.float64)
    yaw = torch.randn(B, T, generator=g, dtype=torch.float64) * 0.5
    return xyz, yaw


def _full_mask(B, T):
    return torch.ones(B, T, dtype=torch.bool)


# 1. pred == gt → all 5 terms 0
def test_pred_equals_gt_zero_loss():
    xyz, yaw = _poses()
    total, terms = body_aux_loss_terms(xyz, yaw, xyz.clone(), yaw.clone(),
                                       _full_mask(*xyz.shape[:2]), _W)
    for k, v in terms.items():
        assert abs(float(v)) < 1e-9, f"{k}={float(v)}"
    assert abs(float(total)) < 1e-9


# 2. pred yaw off by π → L_heading ≈ 2 (cosine)
def test_heading_pi_offset_cosine():
    xyz, yaw = _poses()
    _, terms = body_aux_loss_terms(xyz, yaw + math.pi, xyz, yaw,
                                   _full_mask(*xyz.shape[:2]), _W, heading_form="cosine")
    assert abs(float(terms["heading"]) - 2.0) < 1e-6


# 3. pred xz off → L_root_xz matches smooth_l1
def test_root_xz_offset():
    xyz, yaw = _poses()
    pred = xyz.clone()
    pred[..., 0] += 2.0
    pred[..., 2] += 2.0   # both xz axes off by 2 → smooth_l1(2)=1.5 each, sum 3.0
    _, terms = body_aux_loss_terms(pred, yaw, xyz, yaw, _full_mask(*xyz.shape[:2]), _W)
    assert abs(float(terms["root_xz"]) - 3.0) < 1e-6


# 4. pred y off by 1 → L_root_y = smooth_l1(1)=0.5; weighted 0.3
def test_root_y_offset_and_weight():
    xyz, yaw = _poses()
    pred = xyz.clone()
    pred[..., 1] += 1.0
    total, terms = body_aux_loss_terms(pred, yaw, xyz, yaw, _full_mask(*xyz.shape[:2]),
                                       {**_W, "root_xz": 0, "heading": 0,
                                        "fwd_delta": 0, "yaw_delta": 0})
    assert abs(float(terms["root_y"]) - 0.5) < 1e-6      # smooth_l1(1, beta=1)
    assert abs(float(total) - 0.3 * 0.5) < 1e-6          # only root_y weighted


# 5. pred fwd_delta 0 vs GT nonzero → L_fwd_delta > 0
def test_fwd_delta_positive_when_pred_static():
    T = 8
    gt_xyz = torch.zeros(1, T, 3, dtype=torch.float64)
    # heading_dir_xz(0) = [sin0, cos0] = [0, 1] = +z, so move along +z (forward).
    gt_xyz[0, :, 2] = torch.arange(T, dtype=torch.float64)
    gt_yaw = torch.zeros(1, T, dtype=torch.float64)
    pred_xyz = torch.zeros(1, T, 3, dtype=torch.float64)     # static
    pred_yaw = torch.zeros(1, T, dtype=torch.float64)
    _, terms = body_aux_loss_terms(pred_xyz, pred_yaw, gt_xyz, gt_yaw,
                                   _full_mask(1, T), _W)
    assert float(terms["fwd_delta"]) > 0.0


# 6. changing pred in masked-out (active=0) region does not change loss.
# fwd/yaw_delta are cross-frame diffs, so the active window's leftmost delta uses
# the immediately-preceding (inactive) frame — keep that neighbor unchanged.
def test_inactive_region_does_not_affect_loss():
    xyz, yaw = _poses(T=12)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, 4:8] = True                         # active frames 4,5,6,7
    pred = xyz.clone()
    total1, _ = body_aux_loss_terms(pred, yaw, xyz, yaw, mask, _W)
    pred2 = pred.clone()
    pred2[0, :3] += 99.0       # frames 0,1,2 (frame 3 = active-left neighbor untouched)
    pred2[0, 8:] += 99.0       # frames 8.. (all inactive)
    total2, _ = body_aux_loss_terms(pred2, yaw, xyz, yaw, mask, _W)
    assert torch.allclose(total1, total2, atol=1e-9)


# 8. cosine vs smooth_l1 heading forms differ.
# For small angles |Δh|²=2(1-cos) makes smooth_l1 (quadratic regime) == cosine
# exactly; a LARGE offset pushes a heading component past beta=1 into smooth_l1's
# linear regime, where the two forms diverge.
def test_heading_form_cosine_vs_smooth_l1():
    B, T = 1, 8
    xyz = torch.zeros(B, T, 3, dtype=torch.float64)
    gt_yaw = torch.zeros(B, T, dtype=torch.float64)
    pred_yaw = torch.full((B, T), 2.0, dtype=torch.float64)   # ~115°, large
    _, t_cos = body_aux_loss_terms(xyz, pred_yaw, xyz, gt_yaw, _full_mask(B, T),
                                   _W, heading_form="cosine")
    _, t_sl1 = body_aux_loss_terms(xyz, pred_yaw, xyz, gt_yaw, _full_mask(B, T),
                                   _W, heading_form="smooth_l1")
    assert abs(float(t_cos["heading"]) - (1.0 - math.cos(2.0))) < 1e-6
    assert not math.isclose(float(t_cos["heading"]), float(t_sl1["heading"]), abs_tol=1e-3)


# sample_loss_mask zeroes an invalid sample's contribution (T_B_05 integration)
def test_sample_loss_mask_zeroes_invalid_sample():
    xyz, yaw = _poses(B=2, T=8)
    pred = xyz.clone()
    pred[1] += 5.0                       # sample 1 wildly off
    mask = _full_mask(2, 8)
    slm = torch.tensor([1.0, 0.0])       # sample 1 invalid
    _, terms_masked = body_aux_loss_terms(pred, yaw, xyz, yaw, mask, _W,
                                          sample_loss_mask=slm)
    # only sample 0 (pred==gt there) contributes → ~0
    assert abs(float(terms_masked["root_xz"])) < 1e-9


# derive_fwd_yaw_delta: first frame zero, matches manual
def test_derive_fwd_yaw_delta_first_frame_zero():
    xyz, yaw = _poses(T=6)
    fwd, yawd = derive_fwd_yaw_delta(xyz, yaw)
    assert abs(float(fwd[0, 0])) < 1e-9
    assert abs(float(yawd[0, 0])) < 1e-9


def test_masked_smooth_l1_basic():
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    gt = torch.zeros(1, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0]])   # ignore last
    # smooth_l1: 0.5, 1.5 (diff 1→0.5, diff2→1.5); masked-mean over 2 = 1.0
    out = masked_smooth_l1(pred.unsqueeze(-1), gt.unsqueeze(-1), mask)
    assert abs(float(out) - 1.0) < 1e-6
