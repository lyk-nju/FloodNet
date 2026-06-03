from __future__ import annotations

import math

import numpy as np

from eval.stream_metrics import (
    build_stream_eval_summary,
    compute_root_jitter,
    compute_yaw_error,
)


def test_compute_yaw_error_wraps_angles():
    pred = np.array([math.pi - 0.1], dtype=np.float32)
    target = np.array([-math.pi + 0.1], dtype=np.float32)

    err = compute_yaw_error(pred, target)

    assert abs(err - 0.2) < 1e-5


def test_compute_root_jitter_is_zero_for_constant_velocity():
    root = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0],
         [4.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    assert compute_root_jitter(root) == 0.0


def test_build_stream_eval_summary_uses_stream_metric_keys():
    pred = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = pred.copy()
    target[-1, 0] = 4.0

    summary = build_stream_eval_summary(
        pred,
        target,
        pred_yaw=np.zeros(4, dtype=np.float32),
        target_yaw=np.zeros(4, dtype=np.float32),
    )

    assert summary["stream/root_ADE"] == 0.25
    assert summary["stream/root_FDE"] == 1.0
    assert summary["stream/yaw_error"] == 0.0
    assert "stream/jitter" in summary
    assert summary["stream/num_frames"] == 4
