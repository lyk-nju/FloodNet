from __future__ import annotations

import torch

from utils.token_frame import num_frames_for_tokens
from utils.training.ldf.validation_conditioning import build_windowed_metric_ground_truth


def test_windowed_metric_ground_truth_uses_model_batch_prefix():
    full_tokens = torch.arange(6 * 4, dtype=torch.float32).view(1, 6, 4)
    prefix_tokens = full_tokens[:, :3].clone()
    raw_frames = torch.arange(40 * 263, dtype=torch.float32).view(1, 40, 263)
    batch = {
        "token": full_tokens,
        "token_length": torch.tensor([6]),
        "feature": raw_frames,
        "feature_length": torch.tensor([40]),
    }
    model_batch = {
        "token": prefix_tokens,
        "token_length": torch.tensor([3]),
        "feature_length": torch.tensor([3]),
        "_window_global_start_token": torch.tensor([0]),
    }

    gt_token, gt_token_length, gt_feature, gt_feature_length = (
        build_windowed_metric_ground_truth(batch, model_batch)
    )

    assert torch.equal(gt_token, prefix_tokens)
    assert gt_token_length.tolist() == [3]
    assert len(gt_feature) == 1
    assert gt_feature_length == [num_frames_for_tokens(3)]
    assert torch.equal(gt_feature[0], raw_frames[0, : num_frames_for_tokens(3)])


def test_windowed_metric_ground_truth_clamps_prefix_to_valid_raw_frames():
    full_tokens = torch.zeros(1, 11, 4)
    raw_frames = torch.ones(1, 40, 263)
    batch = {
        "token": full_tokens,
        "token_length": torch.tensor([11]),
        "feature": raw_frames,
        "feature_length": torch.tensor([40]),
    }
    model_batch = {
        "token": full_tokens,
        "token_length": torch.tensor([11]),
        "feature_length": torch.tensor([11]),
        "_window_global_start_token": torch.tensor([0]),
    }

    _, _, gt_feature, gt_feature_length = build_windowed_metric_ground_truth(
        batch,
        model_batch,
    )

    assert num_frames_for_tokens(11) == 41
    assert gt_feature_length == [40]
    assert gt_feature[0].shape == (40, 263)
