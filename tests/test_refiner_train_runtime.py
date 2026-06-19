"""Run-control parity checks for train_refiner.py."""

from __future__ import annotations

import torch
import train_refiner as tr

from pathlib import Path
from omegaconf import OmegaConf
from tests.helpers.humanml3d_fixture import make_root_refiner_from_samples
from utils.initialize import load_config
from utils.training.root_refiner import FixedRefinerSampleDataset

_CFG_DIR = Path(__file__).resolve().parent.parent / "configs"
_ROOT = Path(__file__).resolve().parent.parent


def _clip(T=80):
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 2] = 0.1
    motion[:, 3] = 1.0
    return {"motion_263": motion, "text": "walk"}


def test_train_refiner_uses_project_config_loader_not_private_resolvers():
    source = (_ROOT / "train_refiner.py").read_text()

    assert "def _load_cfg" not in source
    assert "def resolve_cfg_interpolations" not in source
    assert "def _num_devices" not in source
    assert "def _safe_precision" not in source
    assert "load_config()" in source


def test_runtime_code_does_not_import_refiner_config_loader_from_train_entrypoint():
    checked = [
        _ROOT / "web_demo" / "runtime" / "model_loader.py",
        _ROOT / "eval" / "runtime" / "benchmark.py",
        _ROOT / "eval" / "root_refiner" / "benchmark.py",
    ]

    offenders = []
    for path in checked:
        source = path.read_text()
        if "from train_refiner import _load_cfg" in source:
            offenders.append(str(path.relative_to(_ROOT)))
        if "from train_refiner import resolve_cfg_interpolations" in source:
            offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == []


def test_root_refiner_train_config_resolves_through_project_loader():
    cfg = load_config(str(_CFG_DIR / "root_refiner_train.yaml"))
    resolved = OmegaConf.to_container(cfg.config, resolve=True)

    path = resolved["text_encoder"]["precomputed_text_emb_path"]
    assert "${" not in path
    assert path.startswith(resolved["data"]["raw_data_dir"])
    assert path.endswith("HumanML3D/t5_text_embeddings.pt")


def test_apply_fixed_overfit_replaces_train_and_val_with_cached_samples():
    source = make_root_refiner_from_samples([_clip(), _clip(T=90)], seed=0)
    cfg = {
        "fixed_overfit": {
            "enabled": True,
            "num_samples": 3,
            "mode_policy": "full",
            "force_no_path_aug": True,
            "val_on_train": True,
        }
    }

    train_ds, val_suites = tr.apply_fixed_overfit_datasets(source, [], cfg)

    assert isinstance(train_ds, FixedRefinerSampleDataset)
    assert len(val_suites) == 1
    assert val_suites[0]["name"] == "fixed_overfit"
    assert isinstance(val_suites[0]["dataset"], FixedRefinerSampleDataset)
    assert len(train_ds) == 3
    assert len(val_suites[0]["dataset"]) == 3
    assert torch.equal(train_ds[0]["path"], train_ds[0]["path"])


def test_apply_default_fixed_validation_replaces_only_val_with_all_samples():
    train_source = make_root_refiner_from_samples([_clip(), _clip(T=90)], seed=0)
    val_source = make_root_refiner_from_samples(
        [_clip(T=100), _clip(T=110), _clip(T=120)],
        seed=1,
    )

    train_ds, val_suites = tr.apply_default_fixed_validation_dataset(
        train_source,
        [{"name": "full_dense_max", "mode_policy": "full", "dataset": val_source}],
    )

    assert train_ds is train_source
    assert len(val_suites) == 1
    assert val_suites[0]["name"] == "full_dense_max"
    assert isinstance(val_suites[0]["dataset"], FixedRefinerSampleDataset)
    assert len(val_suites[0]["dataset"]) == len(val_source)
    assert torch.equal(
        val_suites[0]["dataset"][0]["path"],
        val_suites[0]["dataset"][0]["path"],
    )


def test_train_refiner_passes_trainer_precision_from_config():
    source = (_ROOT / "train_refiner.py").read_text()

    assert "trainer_kwargs.pop(\"precision\"" not in source
    assert "using 32-true" not in source
