"""Unit tests for eval/root_refiner_benchmark.py (T_A_10).

Smoke / metric-correctness tests. Done-criteria accuracy thresholds
(num_token top-1 > 0.5, heading < 30°) require a trained ckpt (T_A_09) and are
NOT asserted here — random weights make those numbers meaningless. We only
verify the pipeline runs end-to-end with finite metrics + JSON/CSV output, and
that the metric math is correct on hand-built predictions.
"""

from __future__ import annotations

import json
import math

import torch

from datasets.refiner_dataset import RefinerDataset
from eval.root_refiner_benchmark import (
    compute_sample_metrics,
    run_benchmark,
    write_report,
)
from models.root_refiner import RootRefiner
from train_refiner import FrozenStubTextEncoder


# ---------------------------------------------------------------------------
# Per-sample metric correctness
# ---------------------------------------------------------------------------


def _unit_heading_wp(T: int, yaw_val: float = 0.0) -> torch.Tensor:
    wp = torch.zeros(T, 7)
    wp[:, 3] = math.cos(yaw_val)
    wp[:, 4] = math.sin(yaw_val)
    return wp


def test_perfect_prediction_yields_zero_errors():
    T = 10
    gt = _unit_heading_wp(T)
    gt[:, 0] = torch.arange(T, dtype=torch.float32)   # x ramp
    gt[:, 5] = 1.0   # fwd_delta
    pred = gt.clone()
    mask = torch.ones(T, dtype=torch.bool)
    m = compute_sample_metrics(pred, gt, mask)
    assert m["xyz_ADE"] < 1e-6
    assert m["xyz_FDE"] < 1e-6
    assert m["heading_error_deg"] < 1e-4
    assert m["fwd_speed_MAE"] < 1e-6
    assert m["yaw_rate_MAE"] < 1e-6


def test_xyz_ade_fde_known_offset():
    T = 5
    gt = _unit_heading_wp(T)
    pred = gt.clone()
    # Constant +1 offset in x on all frames → per-frame error = 1.
    pred[:, 0] += 1.0
    mask = torch.ones(T, dtype=torch.bool)
    m = compute_sample_metrics(pred, gt, mask)
    assert abs(m["xyz_ADE"] - 1.0) < 1e-5
    assert abs(m["xyz_FDE"] - 1.0) < 1e-5


def test_heading_error_deg_quarter_turn():
    T = 4
    gt = _unit_heading_wp(T, yaw_val=0.0)         # heading (1, 0)
    pred = _unit_heading_wp(T, yaw_val=math.pi / 2)  # heading (0, 1)
    mask = torch.ones(T, dtype=torch.bool)
    m = compute_sample_metrics(pred, gt, mask)
    assert abs(m["heading_error_deg"] - 90.0) < 1e-3


def test_masked_frames_excluded_from_metrics():
    T = 6
    gt = _unit_heading_wp(T)
    pred = gt.clone()
    pred[3:, 0] += 100.0   # large error, but those frames are masked out
    mask = torch.zeros(T, dtype=torch.bool)
    mask[:3] = True
    m = compute_sample_metrics(pred, gt, mask)
    assert m["xyz_ADE"] < 1e-6   # masked-out errors don't count


def test_empty_valid_returns_nan_metrics():
    T = 4
    gt = _unit_heading_wp(T)
    pred = gt.clone()
    mask = torch.zeros(T, dtype=torch.bool)
    m = compute_sample_metrics(pred, gt, mask)
    assert math.isnan(m["xyz_ADE"])
    assert math.isnan(m["heading_error_deg"])


# ---------------------------------------------------------------------------
# End-to-end smoke (random weights)
# ---------------------------------------------------------------------------


def _make_clip(T: int):
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 2] = 0.05   # +Z velocity
    motion[:, 3] = 1.0
    return {"motion_263": motion, "text": "walk forward"}


def test_run_benchmark_smoke_finite_metrics_and_report(tmp_path):
    clips = [_make_clip(50) for _ in range(6)]
    ds = RefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                         full_plan_ratio=1.0, seed=0)
    model = RootRefiner(d_model=32, n_layers=2, n_heads=4, ff_dim=64,
                         max_tokens=8, min_tokens=2, n_hist=8, n_path=16,
                         text_emb_dim=16, dropout=0.0)
    text_encoder = FrozenStubTextEncoder(emb_dim=16)

    result = run_benchmark(model, ds, text_encoder, device="cpu", max_samples=-1)
    summary = result["summary"]

    # All expected metric keys present.
    expected_keys = {
        "n_samples", "num_token_top1_accuracy", "num_token_top3_accuracy",
        "num_token_MAE", "xyz_ADE", "xyz_FDE", "heading_error_deg",
        "fwd_speed_MAE", "lateral_speed_MAE", "yaw_rate_MAE", "smoothness_acc_mean",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["n_samples"] == 6

    # Accuracy in [0, 1]; errors finite & non-negative (random weights, but
    # must not be NaN/Inf since inputs are well-formed).
    assert 0.0 <= summary["num_token_top1_accuracy"] <= 1.0
    assert 0.0 <= summary["num_token_top3_accuracy"] <= 1.0
    for k in ("xyz_ADE", "xyz_FDE", "heading_error_deg", "fwd_speed_MAE",
               "yaw_rate_MAE", "smoothness_acc_mean"):
        assert math.isfinite(summary[k]), f"{k} not finite: {summary[k]}"
        assert summary[k] >= 0.0

    # Report files written.
    write_report(result, tmp_path)
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "per_sample.csv").is_file()
    with (tmp_path / "summary.json").open() as f:
        loaded = json.load(f)
    assert loaded["n_samples"] == 6


def test_run_benchmark_max_samples_limit(tmp_path):
    clips = [_make_clip(50) for _ in range(10)]
    ds = RefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                         full_plan_ratio=1.0, seed=0)
    model = RootRefiner(d_model=32, n_layers=2, n_heads=4, ff_dim=64,
                         max_tokens=8, min_tokens=2, n_hist=8, n_path=16,
                         text_emb_dim=16, dropout=0.0)
    text_encoder = FrozenStubTextEncoder(emb_dim=16)
    result = run_benchmark(model, ds, text_encoder, device="cpu", max_samples=4)
    assert result["summary"]["n_samples"] == 4
    assert len(result["per_sample"]) == 4
