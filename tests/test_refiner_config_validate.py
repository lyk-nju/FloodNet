"""RootRefiner config validation guards."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from utils.refiner.config_validate import validate_refiner_config


_CFG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load(name: str) -> dict:
    from train_refiner import _load_cfg

    return _load_cfg(str(_CFG_DIR / name))


def _minimal_cfg() -> dict:
    return {
        "model": {
            "target": "models.root_refiner.RootRefiner",
            "params": {
                "min_tokens": 4,
                "max_tokens": 49,
                "frames_per_token": 4,
            },
        },
        "optimizer": {
            "target": "AdamW",
            "params": {
                "lr": 1.0e-4,
                "weight_decay": 0.01,
            },
        },
        "data": {
            "target": "datasets.humanml3d_refiner.HumanML3DRefinerDataset",
            "collate_fn": "datasets.humanml3d_refiner.refiner_collate",
            "train_bs": 64,
            "val_bs": 64,
            "num_workers": 0,
        },
        "canonicalization": {
            "mode": "b_full",
            "anchor": "first_effective_frame",
            "full_plan_valid_history_frames": 1,
        },
        "sampling": {
            "horizon_policy": "random",
            "path_condition": {
                "policy": "mixed",
                "ratios": {
                    "dense_path": 0.5,
                    "sparse_path": 0.3,
                    "goal_point": 0.2,
                },
                "offset_start": {
                    "enabled": True,
                    "prob": 0.3,
                    "max_frames": 40,
                    "apply_to": ["dense_path", "sparse_path"],
                },
                "sparse_path": {
                    "point_range": [3, 8],
                },
            },
        },
        "loss_weights": {
            "num_token": 1.0,
            "num_token_soft": 0.1,
            "xyz": 5.0,
            "heading": 1.0,
            "fwd_delta": 0.5,
            "yaw_delta": 0.5,
            "smoothness": 0.0,
        },
    }


def test_shipped_refiner_configs_are_valid():
    validate_refiner_config(_load("root_refiner.yaml"))
    validate_refiner_config(_load("root_refiner_train.yaml"))
    validate_refiner_config(_load("root_refiner_train_fixed_val_no_random_token.yaml"))


def test_rejects_invalid_token_range():
    cfg = _minimal_cfg()
    cfg["model"]["params"]["min_tokens"] = 50

    with pytest.raises(ValueError, match="min_tokens"):
        validate_refiner_config(cfg)


def test_rejects_invalid_frames_per_token():
    cfg = _minimal_cfg()
    cfg["model"]["params"]["frames_per_token"] = 0

    with pytest.raises(ValueError, match="frames_per_token"):
        validate_refiner_config(cfg)


def test_rejects_unknown_horizon_policy():
    cfg = _minimal_cfg()
    cfg["sampling"]["horizon_policy"] = "median"

    with pytest.raises(ValueError, match="horizon_policy"):
        validate_refiner_config(cfg)


def test_rejects_unimplemented_bucketed_horizon_policy():
    cfg = _minimal_cfg()
    cfg["sampling"]["horizon_policy"] = "bucketed"

    with pytest.raises(ValueError, match="bucketed"):
        validate_refiner_config(cfg)


def test_rejects_legacy_num_token_policy_key():
    cfg = _minimal_cfg()
    cfg.setdefault("data", {})["num_token_policy"] = "random"

    with pytest.raises(ValueError, match="data.num_token_policy"):
        validate_refiner_config(cfg)


def test_rejects_legacy_training_block():
    cfg = _minimal_cfg()
    cfg["training"] = {"batch_size": 64, "lr": 1.0e-4, "total_steps": 1000}

    with pytest.raises(ValueError, match="training is legacy"):
        validate_refiner_config(cfg)


def test_rejects_model_without_target_params():
    cfg = _minimal_cfg()
    cfg["model"] = {"min_tokens": 4, "max_tokens": 49}

    with pytest.raises(ValueError, match="model.target"):
        validate_refiner_config(cfg)


def test_rejects_legacy_path_aug_key():
    cfg = _minimal_cfg()
    cfg["path_aug"] = {"trim_prob": 0.3}

    with pytest.raises(ValueError, match="path_aug"):
        validate_refiner_config(cfg)


def test_rejects_legacy_loss_weight_names():
    cfg = _minimal_cfg()
    cfg["loss_weights"]["speed"] = 1.0

    with pytest.raises(ValueError, match="legacy"):
        validate_refiner_config(cfg)


def test_rejects_unsupported_canonicalization_contract():
    cfg = _minimal_cfg()
    cfg["canonicalization"]["mode"] = "partial"

    with pytest.raises(ValueError, match="canonicalization.mode"):
        validate_refiner_config(cfg)
