from __future__ import annotations

import pytest
import torch

from utils.training.noise_initializer.free_delta import optimize_free_delta


def test_optimize_free_delta_reduces_snapshot_loss():
    base_zT = torch.tensor([[[1.0, -1.0], [0.5, -0.5]]])
    target_zT = base_zT + 0.25

    result = optimize_free_delta(
        base_zT=base_zT,
        loss_fn=lambda frontier_zT: (frontier_zT - target_zT).pow(2).mean(),
        steps=80,
        lr=0.1,
        lambda_delta=0.0,
        max_delta_norm_ratio=None,
        log_every=20,
    )

    assert result.final_task_loss < result.initial_task_loss * 1e-3
    assert torch.allclose(result.frontier_zT, target_zT, atol=2e-3)
    assert result.loss_curve[0]["step"] == 0
    assert result.loss_curve[-1]["step"] == 80


def test_optimize_free_delta_emits_progress_rows():
    progress = []

    optimize_free_delta(
        base_zT=torch.ones(1, 1, 1),
        loss_fn=lambda frontier_zT: (frontier_zT - 1.1).pow(2).mean(),
        steps=5,
        lr=0.1,
        log_every=2,
        progress_fn=progress.append,
    )

    assert [row["step"] for row in progress] == [0, 2, 4, 5]


def test_optimize_free_delta_enforces_relative_norm_trust_region():
    base_zT = torch.ones(1, 2, 2)
    target_zT = base_zT + 10.0

    result = optimize_free_delta(
        base_zT=base_zT,
        loss_fn=lambda frontier_zT: (frontier_zT - target_zT).pow(2).mean(),
        steps=30,
        lr=0.2,
        lambda_delta=0.0,
        max_delta_norm_ratio=0.1,
        log_every=10,
    )

    base_norm = base_zT.norm().item()
    assert result.delta_zT.norm().item() <= 0.1 * base_norm + 1e-6
    assert result.clip_saturation_ratio > 0.0
    assert result.raw_delta_zT.norm() > result.delta_zT.norm()


def test_optimize_free_delta_rejects_non_positive_steps():
    with pytest.raises(ValueError, match="steps"):
        optimize_free_delta(
            base_zT=torch.zeros(1, 2, 2),
            loss_fn=lambda frontier_zT: frontier_zT.pow(2).mean(),
            steps=0,
            lr=0.1,
        )
