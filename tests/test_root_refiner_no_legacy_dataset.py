from __future__ import annotations

from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent


def test_root_refiner_code_has_no_legacy_dataset_imports():
    needles = (
        "datasets." "humanml3d" "_refiner",
        "HumanML3D" "RefinerDataset",
    )
    checked_roots = [
        _ROOT / "train_refiner.py",
        _ROOT / "eval" / "root_refiner" / "benchmark.py",
        _ROOT / "utils" / "training" / "root_refiner",
    ]
    offenders = []
    for path in checked_roots:
        files = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for file_path in files:
            text = file_path.read_text()
            for needle in needles:
                if needle in text:
                    offenders.append(f"{file_path.relative_to(_ROOT)}: {needle}")
    assert offenders == []


def test_legacy_refiner_dataset_file_removed():
    assert not (_ROOT / "datasets" / ("humanml3d" "_refiner.py")).exists()


def test_legacy_refiner_stats_tools_removed():
    assert not (_ROOT / "scripts" / "compute_5d_stats.py").exists()
    assert not (_ROOT / "scripts" / "compute_path_stats.py").exists()
    assert not (_ROOT / "utils" / "training" / "root_refiner" / "path_feature_stats.py").exists()


def test_train_refiner_rejects_legacy_dataset_target(tmp_path):
    from tests.helpers.humanml3d_fixture import write_humanml3d_fixture
    from train_refiner import build_datasets

    write_humanml3d_fixture(tmp_path, [{"name": "s1"}])
    cfg = {
        "model": {
            "target": "models.root_refiner.RootRefiner",
            "params": {
                "n_hist": 8,
                "n_path": 16,
                "min_frames": 5,
                "max_frames": 29,
            },
        },
        "optimizer": {
            "target": "AdamW",
            "params": {"lr": 1.0e-4, "weight_decay": 0.01},
        },
        "canonicalization": {
            "mode": "b_full",
            "anchor": "first_effective_frame",
            "full_plan_valid_history_frames": 1,
        },
        "loss_weights": {
            "pace": 1.0,
            "frame_pace": 0.1,
            "xyz": 5.0,
            "heading": 1.0,
            "fwd_delta": 0.5,
            "yaw_delta": 0.5,
            "smoothness": 0.0,
        },
        "sampling": {
            "full_plan_ratio": 1.0,
            "horizon_policy": "random",
            "path_condition": {
                "policy": "dense_path",
                "offset_start": {"enabled": False},
                "sparse_path": {"point_range": [3, 8]},
            },
        },
        "data": {
            "target": "datasets." "humanml3d" "_refiner.HumanML3D" "RefinerDataset",
            "collate_fn": "datasets." "humanml3d" "_refiner.collate_fn",
            "train_bs": 4,
            "val_bs": 4,
            "num_workers": 0,
            "raw_data_dir": str(tmp_path),
            "dataset": "humanml3d",
            "train_split_file": "train.txt",
            "feature_path": "new_joint_vecs",
            "text_path": "texts",
        },
    }

    with pytest.raises(ValueError, match="datasets.humanml3d.HumanML3DDataset"):
        build_datasets(cfg)
