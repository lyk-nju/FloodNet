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
        _ROOT / "scripts" / "compute_5d_stats.py",
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


def test_train_refiner_rejects_legacy_dataset_target(tmp_path):
    from tests.helpers.humanml3d_fixture import write_humanml3d_fixture
    from train_refiner import build_datasets
    from tests.test_train_refiner import _tiny_cfg

    write_humanml3d_fixture(tmp_path, [{"name": "s1"}])
    cfg = _tiny_cfg()
    cfg["data"].update(
        {
            "target": "datasets." "humanml3d" "_refiner.HumanML3D" "RefinerDataset",
            "collate_fn": "datasets." "humanml3d" "_refiner.collate_fn",
            "raw_data_dir": str(tmp_path),
            "dataset": "humanml3d",
            "train_split_file": "train.txt",
            "feature_path": "new_joint_vecs",
            "text_path": "texts",
            "normalize": False,
        }
    )

    with pytest.raises(ValueError, match="datasets.humanml3d.HumanML3DDataset"):
        build_datasets(cfg)
