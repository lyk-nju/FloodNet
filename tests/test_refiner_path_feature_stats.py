from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from utils.refiner.path_feature_stats import (
    PATH_FEATURE_NAMES,
    compute_sampling_config_hash,
    compute_stats_from_features,
    load_path_feature_stats,
    save_path_feature_stats,
    validate_path_feature_stats_meta,
)


def test_save_and_load_path_feature_stats_round_trip(tmp_path):
    mean = torch.arange(5, dtype=torch.float32)
    std = torch.arange(1, 6, dtype=torch.float32)
    meta = {
        "dataset": "humanml3d",
        "split": "train",
        "num_samples": 10,
        "sampling_config_hash": "abc",
        "n_path": 64,
        "goal_point_representation": "current_to_goal_line",
    }

    save_path_feature_stats(tmp_path, mean=mean, std=std, meta=meta)
    loaded = load_path_feature_stats(tmp_path)

    assert torch.equal(loaded.mean, mean)
    assert torch.equal(loaded.std, std)
    assert loaded.meta["feature_names"] == PATH_FEATURE_NAMES
    assert loaded.meta["sampling_config_hash"] == "abc"


def test_compute_stats_clamps_small_std():
    features = torch.ones(4, 5)
    mean, std = compute_stats_from_features(features)

    assert torch.equal(mean, torch.ones(5))
    assert torch.all(std >= 1e-6)


def test_sampling_config_hash_ignores_runtime_fields():
    cfg_a = {
        "model": {"n_path": 64, "min_tokens": 4, "max_tokens": 49, "frames_per_token": 4},
        "sampling": {"horizon_policy": "random"},
        "data": {"dataset": "humanml3d", "train_split_file": "train.txt"},
        "trainer": {"devices": [0]},
    }
    cfg_b = {
        **cfg_a,
        "trainer": {"devices": [0, 1, 2, 3]},
        "save_dir": "./elsewhere",
    }

    assert compute_sampling_config_hash(cfg_a) == compute_sampling_config_hash(cfg_b)


def test_validate_meta_rejects_hash_mismatch():
    meta = {
        "feature_names": PATH_FEATURE_NAMES,
        "sampling_config_hash": "old",
    }

    with pytest.raises(ValueError, match="path feature stats config hash mismatch"):
        validate_path_feature_stats_meta(meta, expected_hash="new")


def test_stats_files_are_numpy_and_json(tmp_path):
    save_path_feature_stats(
        tmp_path,
        mean=torch.zeros(5),
        std=torch.ones(5),
        meta={"sampling_config_hash": "abc"},
    )

    assert np.load(tmp_path / "path_features_mean.npy").shape == (5,)
    assert np.load(tmp_path / "path_features_std.npy").shape == (5,)
    meta = json.loads((tmp_path / "path_features_meta.json").read_text())
    assert meta["feature_names"] == PATH_FEATURE_NAMES


def test_compute_path_stats_cli_smoke(tmp_path):
    from scripts.compute_path_stats import main

    raw = tmp_path / "raw"
    ds = raw / "HumanML3D"
    (ds / "new_joint_vecs").mkdir(parents=True)
    (ds / "texts").mkdir(parents=True)
    (ds / "train.txt").write_text("000001\n")
    arr = np.zeros((60, 263), dtype=np.float32)
    arr[:, 2] = 0.1
    arr[:, 3] = 1.0
    np.save(ds / "new_joint_vecs" / "000001.npy", arr)
    (ds / "texts" / "000001.txt").write_text("walk forward#x#0#0\n")

    cfg = {
        "model": {
            "n_hist": 8,
            "n_path": 16,
            "min_tokens": 2,
            "max_tokens": 8,
            "frames_per_token": 4,
        },
        "training": {"sampling_mode_full_ratio": 1.0},
        "sampling": {
            "horizon_policy": "random",
            "path_condition": {
                "policy": "mixed",
                "ratios": {"dense_path": 1.0, "sparse_path": 0.0, "goal_point": 0.0},
            },
        },
        "data": {
            "raw_data_dir": str(raw),
            "dataset": "humanml3d",
            "train_split_file": "train.txt",
            "feature_path": "new_joint_vecs",
            "text_path": "texts",
            "normalize": False,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "stats"

    main([
        "--config",
        str(cfg_path),
        "--output-dir",
        str(out),
        "--num-samples",
        "3",
        "--seed",
        "7",
    ])

    assert (out / "path_features_mean.npy").is_file()
    assert (out / "path_features_std.npy").is_file()
    meta = json.loads((out / "path_features_meta.json").read_text())
    assert meta["num_samples"] == 3
    assert meta["feature_names"] == PATH_FEATURE_NAMES
