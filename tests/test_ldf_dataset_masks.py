from __future__ import annotations

import numpy as np


def _feature(num_frames: int) -> np.ndarray:
    return np.zeros((num_frames, 263), dtype=np.float32)


def _token(num_tokens: int) -> np.ndarray:
    return np.zeros((num_tokens, 8), dtype=np.float32)


def test_humanml3d_dataset_emits_dense_traj_masks_without_sparse_config():
    from datasets.humanml3d import HumanML3DDataset

    ds = HumanML3DDataset.__new__(HumanML3DDataset)
    ds.split = "train"
    ds.window_length = 17
    ds.random_length = 0
    ds.smooth_traj_sigma = 0.0
    ds.traj_feat_dim = 4

    out = ds._process(
        {
            "dataset": "HumanML3D",
            "name": "sample",
            "feature": _feature(17),
            "token": _token(5),
        }
    )

    assert "token_mask" not in out
    assert np.all(out["traj_mask"] == 1.0)
    assert np.all(out["traj_cond_mask"] == 1.0)
    assert np.all(out["traj_loss_mask"] == 1.0)


def test_babel_dataset_emits_dense_traj_masks_without_sparse_config():
    from datasets.babel import BabelDataset

    ds = BabelDataset.__new__(BabelDataset)
    ds.split = "train"
    ds.window_length = 17
    ds.random_length = 0
    ds.feature_fps = 20.0
    ds.smooth_traj_sigma = 0.0
    ds.traj_feat_dim = 4

    out = ds._build_output(
        {
            "dataset": "BABEL",
            "name": "sample",
            "feature": _feature(17),
            "token": _token(5),
        },
        apply_crop=True,
    )

    assert "token_mask" not in out
    assert np.all(out["traj_mask"] == 1.0)
    assert np.all(out["traj_cond_mask"] == 1.0)
    assert np.all(out["traj_loss_mask"] == 1.0)


def test_generate_dataset_emits_dense_traj_masks_without_sparse_config():
    from datasets.generate import GenerateDataset

    ds = GenerateDataset.__new__(GenerateDataset)
    ds.dim = 263
    ds.token_dim = 8
    ds.feature_fps = 20.0
    ds.token_fps = 5.0
    ds.smooth_traj_sigma = 0.0

    out = ds._process(
        {
            "dataset": "generate",
            "name": "sample",
            "feature_length": 17,
            "token_length": 5,
            "text_data": [
                {
                    "caption": "walk forward",
                    "tokens": ["walk", "forward"],
                    "f_tag": 0.0,
                    "to_tag": 0.85,
                }
            ],
        }
    )

    assert "token_mask" not in out
    assert np.all(out["traj_mask"] == 1.0)
    assert np.all(out["traj_loss_mask"] == 1.0)
