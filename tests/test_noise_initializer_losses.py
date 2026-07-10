from __future__ import annotations

import pytest
import torch

from utils.training.noise_initializer.losses import (
    anchored_root_xz_loss,
    delta_zT_l2_regularization,
    masked_mean,
)


def test_masked_mean_ignores_invalid_entries():
    values = torch.tensor([1.0, 5.0, 9.0])
    mask = torch.tensor([1.0, 0.0, 1.0])

    assert torch.equal(masked_mean(values, mask), torch.tensor(5.0))


def test_anchored_root_xz_loss_skips_history_and_matches_target_anchor():
    pred = torch.tensor(
        [
            [10.0, 0.0],
            [11.0, 0.0],
            [12.0, 0.0],
            [13.0, 0.0],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [100.0, 0.0],
            [101.0, 0.0],
            [103.0, 0.0],
            [106.0, 0.0],
        ]
    )
    mask = torch.ones(4)

    loss, parts = anchored_root_xz_loss(
        pred,
        target,
        mask,
        history_frames=2,
        lambda_vel=0.5,
        anchor_mode="target_anchor_abs",
    )

    assert torch.allclose(loss, torch.tensor(7.0))
    assert parts["traj_loss"] == pytest.approx(5.0)
    assert parts["vel_loss"] == pytest.approx(4.0)
    assert parts["history_frames"] == 2
    assert parts["optimized_frames"] == 2
    loss.backward()
    assert pred.grad is not None


def test_delta_zT_l2_regularization_penalizes_residual_size():
    delta = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    reg = delta_zT_l2_regularization(delta)

    assert torch.equal(reg, torch.tensor(7.5))


def test_anchored_root_xz_loss_absolute_preserves_predicted_world_position():
    pred = torch.tensor([[10.0, 0.0], [11.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    loss, parts = anchored_root_xz_loss(
        pred,
        target,
        torch.ones(2),
        history_frames=0,
        anchor_mode="absolute",
    )

    assert torch.equal(loss, torch.tensor(100.0))
    assert parts["anchor_mode"] == "absolute"


def test_anchored_root_xz_loss_rejects_unknown_anchor_mode():
    with pytest.raises(ValueError, match="anchor_mode"):
        anchored_root_xz_loss(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.ones(2),
            history_frames=0,
            anchor_mode="bad",
        )
