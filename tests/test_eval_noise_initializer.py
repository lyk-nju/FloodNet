from __future__ import annotations

import torch

from tools.eval_noise_initializer import _metrics_for_token_window


def test_metrics_for_token_window_uses_causal_frame_slice():
    target = torch.zeros(17, 2)
    pred = target.clone()
    pred[5:9, 0] = 2.0
    mask = torch.ones(17)

    metrics = _metrics_for_token_window(
        pred,
        target,
        mask,
        start_token=2,
        num_tokens=1,
        frames_per_token=4,
    )

    assert metrics["ade"] == 2.0
    assert metrics["mse"] == 4.0
