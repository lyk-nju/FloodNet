"""RootRefiner loss helper tests."""

from __future__ import annotations

import torch

from utils.refiner.losses import (
    dense_path_control_loss,
    goal_point_control_loss,
    masked_mean,
    ordinal_duration_loss,
    second_order_diff_l2,
    soft_ordinal_targets,
    sparse_path_control_loss,
    smooth_l1_masked,
)


def test_smooth_l1_masked_ignores_masked_frames():
    pred = torch.zeros(2, 4, 3)
    gt = torch.ones(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, :2] = True

    loss = smooth_l1_masked(pred, gt, mask)

    assert abs(loss.item() - 0.5) < 1e-6


def test_smooth_l1_masked_zero_when_no_valid():
    pred = torch.randn(2, 4, 3)
    gt = torch.randn(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.bool)

    assert smooth_l1_masked(pred, gt, mask).item() == 0.0


def test_masked_mean_basic():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, True, False, False]])

    assert abs(masked_mean(values, mask).item() - 1.5) < 1e-6


def test_second_order_diff_l2_zero_on_linear_ramp():
    batch, frames, channels = 1, 6, 2
    ramp = (
        torch.arange(frames, dtype=torch.float32)
        .view(1, frames, 1)
        .expand(batch, frames, channels)
        .contiguous()
    )
    mask = torch.ones(batch, frames, dtype=torch.bool)

    assert second_order_diff_l2(ramp, mask).item() < 1e-9


def test_second_order_diff_l2_nonzero_on_curved():
    batch, frames = 1, 6
    curve = (torch.arange(frames, dtype=torch.float32) ** 2).view(1, frames, 1)
    mask = torch.ones(batch, frames, dtype=torch.bool)

    assert abs(second_order_diff_l2(curve, mask).item() - 4.0) < 1e-5


def test_soft_ordinal_targets_peak_at_target_and_decay_with_distance():
    target = torch.tensor([3])
    soft = soft_ordinal_targets(target, num_classes=7, sigma=1.0)

    assert soft.shape == (1, 7)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(1))
    assert soft[0].argmax().item() == 3
    assert soft[0, 2] > soft[0, 0]


def test_ordinal_duration_loss_zero_expected_when_distribution_matches_target():
    logits = torch.full((1, 5), -20.0)
    logits[0, 2] = 20.0
    target_num_tokens = torch.tensor([4])

    losses = ordinal_duration_loss(
        logits,
        target_num_tokens,
        min_tokens=2,
        sigma=1.0,
    )

    assert losses["expected"].item() < 1e-5
    assert torch.allclose(losses["expected_num_tokens"], torch.tensor([4.0]), atol=1e-4)


def test_dense_path_control_loss_ignores_offset_prefix():
    pred = torch.zeros(1, 5, 5)
    path = torch.zeros(1, 5, 2)
    path[:, :, 0] = torch.arange(5, dtype=torch.float32)
    supervision = torch.tensor([[False, False, True, True, True]])
    pred[0, 0, 0] = 100.0
    pred[0, 1, 0] = 100.0
    pred[0, 2:, 0] = torch.tensor([2.0, 3.0, 4.0])

    assert dense_path_control_loss(pred, path, supervision).item() < 1e-6


def test_sparse_path_control_loss_uses_control_and_supervision_intersection():
    pred = torch.zeros(1, 9, 5)
    path = torch.zeros(1, 5, 2)
    path[0, :, 0] = torch.arange(5, dtype=torch.float32)
    control = torch.tensor([[True, False, True, False, True]])
    supervision = torch.ones(1, 9, dtype=torch.bool)
    supervision[:, :2] = False
    offset = torch.tensor([2])
    pred[0, [2, 5, 8], 0] = torch.tensor([0.0, 2.0, 4.0])

    assert sparse_path_control_loss(pred, path, control, supervision, offset).item() < 1e-6


def test_goal_point_control_loss_uses_final_valid_frame_only():
    pred = torch.zeros(1, 5, 5)
    pred[0, 3, 0] = 4.0
    pred[0, 3, 2] = 2.0
    mask = torch.tensor([[True, True, True, True, False]])
    path = torch.zeros(1, 8, 2)
    path[0, -1] = torch.tensor([4.0, 2.0])
    control = torch.zeros(1, 8, dtype=torch.bool)
    control[0, -1] = True

    assert goal_point_control_loss(pred, mask, path, control).item() < 1e-6
