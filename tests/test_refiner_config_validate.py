"""RootRefiner config validation guards."""

from __future__ import annotations

import pytest

from pathlib import Path
from omegaconf import OmegaConf
from utils.initialize import load_config
from utils.training.root_refiner.config_validate import validate_refiner_config
from utils.training.root_refiner.sampling_schedule import apply_training_schedule_to_cfg


_CFG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load(name: str) -> dict:
    cfg = load_config(str(_CFG_DIR / name)).config
    apply_training_schedule_to_cfg(cfg)
    return OmegaConf.to_container(
        cfg,
        resolve=True,
    )


def _minimal_cfg() -> dict:
    return {
        "model": {
            "target": "models.root_refiner.RootRefiner",
            "params": {
                "min_frames": 13,
                "max_frames": 193,
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
            "target": "datasets.humanml3d.HumanML3DDataset",
            "collate_fn": "utils.training.root_refiner.collate_fn",
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
            "pace": 1.0,
            "frame_pace": 0.1,
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
    validate_refiner_config(_load("root_refiner_train_root_branch_finetune.yaml"))
    validate_refiner_config(
        _load("root_refiner_train_scheduled_200k_root_branch_50k.yaml")
    )


def test_shipped_refiner_configs_validate_before_omegaconf_resolution():
    for name in (
        "root_refiner.yaml",
        "root_refiner_train.yaml",
        "root_refiner_train_root_branch_finetune.yaml",
        "root_refiner_train_scheduled_200k_root_branch_50k.yaml",
    ):
        validate_refiner_config(load_config(str(_CFG_DIR / name)).config)


def test_scheduled_refiner_config_derives_trainer_max_steps_before_resolution():
    cfg = load_config(
        str(_CFG_DIR / "root_refiner_train_scheduled_200k_root_branch_50k.yaml")
    ).config

    schedule = apply_training_schedule_to_cfg(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert schedule.total_steps == 450000
    assert resolved["trainer"]["max_steps"] == 450000
    assert resolved["lr_scheduler"]["params"]["num_training_steps"] == 450000


def test_rejects_history_condition_config():
    cfg = _minimal_cfg()
    cfg["sampling"]["history_condition"] = {
        "policy": "mixed",
        "ratios": {
            "only_current": 0.2,
            "short_history": 0.4,
            "full_history": 0.4,
        },
    }

    with pytest.raises(ValueError, match="history_condition"):
        validate_refiner_config(cfg)


def test_rejects_invalid_frame_range():
    cfg = _minimal_cfg()
    cfg["model"]["params"]["min_frames"] = 194

    with pytest.raises(ValueError, match="frame range"):
        validate_refiner_config(cfg)


def test_rejects_legacy_token_model_keys():
    cfg = _minimal_cfg()
    cfg["model"]["params"]["frames_per_token"] = 4

    with pytest.raises(ValueError, match="legacy token"):
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


@pytest.mark.parametrize("key", ["normalize", "stats_dir", "path_feature_stats_dir"])
def test_rejects_legacy_normalize_data_keys(key):
    cfg = _minimal_cfg()
    cfg["data"][key] = True if key == "normalize" else "deps/refiner_stats"

    with pytest.raises(ValueError, match=key):
        validate_refiner_config(cfg)


def test_rejects_legacy_training_block():
    cfg = _minimal_cfg()
    cfg["training"] = {"batch_size": 64, "lr": 1.0e-4, "total_steps": 1000}

    with pytest.raises(ValueError, match="training is legacy"):
        validate_refiner_config(cfg)


def test_rejects_model_without_target_params():
    cfg = _minimal_cfg()
    cfg["model"] = {"min_frames": 13, "max_frames": 193}

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


def test_rejects_unknown_freeze_refiner_module():
    cfg = _minimal_cfg()
    cfg["freeze"] = {"refiner_modules": ["duration_head", "not_a_module"]}

    with pytest.raises(ValueError, match="freeze.refiner_modules"):
        validate_refiner_config(cfg)


def test_rejects_unknown_training_schedule_freeze_refiner_module():
    cfg = _minimal_cfg()
    cfg["training_schedule"] = {
        "enabled": True,
        "phases": [
            {
                "name": "bad_freeze",
                "steps": 100,
                "freeze": {"refiner_modules": ["not_a_module"]},
            },
        ],
    }

    with pytest.raises(ValueError, match="training_schedule.phases"):
        validate_refiner_config(cfg)


def test_rejects_training_schedule_without_positive_steps():
    cfg = _minimal_cfg()
    cfg["training_schedule"] = {
        "enabled": True,
        "phases": [
            {
                "name": "empty",
                "steps": 0,
            },
        ],
    }

    with pytest.raises(ValueError, match="training_schedule.phases"):
        validate_refiner_config(cfg)


def test_rejects_unsupported_canonicalization_contract():
    cfg = _minimal_cfg()
    cfg["canonicalization"]["mode"] = "partial"

    with pytest.raises(ValueError, match="canonicalization.mode"):
        validate_refiner_config(cfg)
