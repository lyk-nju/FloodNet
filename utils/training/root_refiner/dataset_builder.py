"""RootRefiner dataset construction from training config."""

from __future__ import annotations

import copy
import logging

from pathlib import Path
from typing import Mapping

from omegaconf import OmegaConf

from datasets.humanml3d import HumanML3DDataset
from utils.training.root_refiner.batch_builder import (
    FixedRefinerSampleDataset,
    RootRefinerDataset,
    build_fixed_samples,
)
from utils.training.root_refiner.config_validate import validate_refiner_config

log = logging.getLogger(__name__)


DATASET_DEFAULTS: dict[str, dict[str, str]] = {
    "humanml3d": {
        "subdir": "HumanML3D",
        "feature_path": "new_joint_vecs",
        "text_path": "texts",
        "split_file": "train.txt",
    },
    "babel": {
        "subdir": "BABEL_streamed",
        "feature_path": "motions",
        "text_path": "texts",
        "split_file": "train_processed.txt",
    },
}

DEFAULT_VALIDATION_SUITES = (
    {
        "name": "full_dense_max",
        "mode_policy": "full",
        "full_plan_ratio": 1.0,
        "horizon_policy": "max",
        "path_condition_policy": "dense_path",
        "offset_start_enabled": False,
    },
    {
        "name": "sliding_dense_random",
        "mode_policy": "sliding",
        "full_plan_ratio": 0.0,
        "horizon_policy": "random",
        "path_condition_policy": "dense_path",
        "offset_start_enabled": False,
    },
    {
        "name": "sliding_dense_max",
        "mode_policy": "sliding",
        "full_plan_ratio": 0.0,
        "horizon_policy": "max",
        "path_condition_policy": "dense_path",
        "offset_start_enabled": False,
    },
)


def normalize_validation_suites_in_cfg(cfg: dict) -> list[dict]:
    """Ensure cfg.validation.suites is explicit and return copied configs."""
    validation = cfg.setdefault("validation", {})
    suites = validation.get("suites") or list(DEFAULT_VALIDATION_SUITES)
    normalized = []
    for suite in suites:
        if not isinstance(suite, dict):
            raise ValueError(f"validation suite must be a mapping, got {suite!r}")
        item = copy.deepcopy(suite)
        if not item.get("name"):
            raise ValueError(f"validation suite missing name: {suite!r}")
        normalized.append(item)
    validation["suites"] = normalized
    return normalized


def resolve_dataset_dir(raw_data_dir: str | Path, dataset: str = "humanml3d") -> Path:
    defaults = _dataset_defaults(dataset)
    root = Path(raw_data_dir)
    candidate = root / defaults["subdir"]
    if candidate.is_dir():
        return candidate
    return root


def build_humanml3d_dataset_cfg(
    *,
    raw_data_dir: str | Path,
    dataset: str = "humanml3d",
    split_file: str | None = None,
    feature_path: str | None = None,
    token_path: str | None = None,
    text_path: str | None = None,
    min_length: int = 1,
    max_length: int = 10**9,
    window_length: int | None = None,
    random_length: int = 0,
    traj_feat_dim: int = 4,
    smooth_traj_sigma: float = 0.0,
    return_text_all: bool = False,
    debug: bool = False,
):
    defaults = _dataset_defaults(dataset)
    dataset_dir = resolve_dataset_dir(raw_data_dir, dataset)
    split_path = dataset_dir / (split_file or defaults["split_file"])
    return OmegaConf.create(
        {
            "debug": bool(debug),
            "data": {
                "train_meta_paths": [str(split_path)],
                "val_meta_paths": [str(split_path)],
                "test_meta_paths": [str(split_path)],
                "feature_path": feature_path or defaults["feature_path"],
                "token_path": token_path,
                "text_path": text_path or defaults["text_path"],
                "min_length": int(min_length),
                "max_length": int(max_length),
                "window_length": int(window_length or max_length),
                "random_length": int(random_length),
                "traj_feat_dim": int(traj_feat_dim),
                "smooth_traj_sigma": float(smooth_traj_sigma),
                "return_text_all": bool(return_text_all),
            },
        }
    )


def build_root_refiner_dataset(
    cfg: Mapping,
    split_file: str | None = None,
    *,
    split: str = "train",
    seed: int | None = None,
    randomize_caption: bool = True,
    validation_suite: Mapping | None = None,
) -> RootRefinerDataset:
    validate_refiner_config(cfg)
    data_cfg = _section(cfg, "data")
    data_target = data_cfg.get("target", "datasets.humanml3d.HumanML3DDataset")
    if data_target != "datasets.humanml3d.HumanML3DDataset":
        raise ValueError(
            "RootRefiner data.target must be "
            "'datasets.humanml3d.HumanML3DDataset'; "
            f"got {data_target!r}."
        )

    model_cfg = _section(_section(cfg, "model"), "params")
    sampling_cfg = _section(cfg, "sampling")
    path_condition_cfg = _section(sampling_cfg, "path_condition")
    offset_cfg = _section(path_condition_cfg, "offset_start")
    sparse_cfg = _section(path_condition_cfg, "sparse_path")

    full_plan_ratio = sampling_cfg.get("full_plan_ratio", 0.5)
    horizon_policy = sampling_cfg.get("horizon_policy", "random")
    path_condition_policy = path_condition_cfg.get("policy", "dense_path")
    path_condition_ratios = path_condition_cfg.get("ratios")
    offset_start_enabled = bool(offset_cfg.get("enabled", False))
    offset_start_prob = float(offset_cfg.get("prob", 0.0))
    if validation_suite is not None:
        full_plan_ratio = validation_suite.get("full_plan_ratio", full_plan_ratio)
        horizon_policy = validation_suite.get("horizon_policy", horizon_policy)
        path_condition_policy = validation_suite.get(
            "path_condition_policy", path_condition_policy
        )
        path_condition_ratios = validation_suite.get(
            "path_condition_ratios", path_condition_ratios
        )
        offset_start_enabled = bool(
            validation_suite.get("offset_start_enabled", offset_start_enabled)
        )
        default_offset_prob = offset_start_prob if offset_start_enabled else 0.0
        offset_start_prob = float(
            validation_suite.get("offset_start_prob", default_offset_prob)
        )

    dataset_name = str(data_cfg.get("dataset", "humanml3d"))
    max_length = int(data_cfg.get("max_length", 10**9))
    raw_dataset_cfg = build_humanml3d_dataset_cfg(
        raw_data_dir=data_cfg["raw_data_dir"],
        dataset=dataset_name,
        split_file=split_file or _default_split_file(data_cfg, split, dataset_name),
        feature_path=data_cfg.get("feature_path"),
        token_path=data_cfg.get("token_path"),
        text_path=data_cfg.get("text_path"),
        min_length=int(data_cfg.get("min_length", 1)),
        max_length=max_length,
        window_length=int(data_cfg.get("window_length", max_length)),
        random_length=int(data_cfg.get("random_length", 0)),
        traj_feat_dim=int(data_cfg.get("traj_feat_dim", 4)),
        smooth_traj_sigma=float(data_cfg.get("smooth_traj_sigma", 0.0)),
        return_text_all=True,
        debug=bool(cfg.get("debug", False)),
    )
    raw_dataset = HumanML3DDataset(raw_dataset_cfg, split=split)
    dataset = RootRefinerDataset(
        raw_dataset,
        n_hist=model_cfg["n_hist"],
        n_path=model_cfg["n_path"],
        max_frames=int(model_cfg["max_frames"]),
        min_frames=int(model_cfg["min_frames"]),
        full_plan_ratio=full_plan_ratio,
        horizon_policy=horizon_policy,
        path_condition_policy=path_condition_policy,
        path_condition_ratios=path_condition_ratios,
        offset_start_enabled=offset_start_enabled,
        offset_start_prob=offset_start_prob,
        offset_start_max_frames=int(offset_cfg.get("max_frames", 40)),
        offset_start_apply_to=tuple(
            offset_cfg.get("apply_to", ("dense_path", "sparse_path"))
        ),
        sparse_path_point_range=tuple(sparse_cfg.get("point_range", (3, 8))),
        seed=seed,
        randomize_caption=randomize_caption,
    )
    if (
        validation_suite is not None
        and validation_suite.get("mode_policy") == "sliding"
    ):
        dataset.valid_indices = [
            idx
            for idx in dataset.valid_indices
            if idx in dataset.sliding_eligible_indices
        ]
    return dataset


def build_datasets(cfg: dict, seed: int | None = None):
    """Build train and optional validation datasets from cfg.data."""
    validate_refiner_config(cfg)
    data_cfg = cfg.get("data", {})
    train_split = data_cfg.get("train_split_file", "train.txt")
    val_split = data_cfg.get("val_split_file")
    train_ds = _build_one_dataset(cfg, train_split, seed=seed)
    if not val_split:
        return train_ds, []
    suites = []
    for suite_cfg in normalize_validation_suites_in_cfg(cfg):
        dataset = _build_one_dataset(
            cfg,
            val_split,
            seed=0,
            randomize_caption=False,
            validation_suite=suite_cfg,
        )
        suites.append(
            {
                "name": suite_cfg["name"],
                "mode_policy": suite_cfg.get("mode_policy", "random"),
                "dataset": dataset,
            }
        )
    return train_ds, suites


def apply_fixed_overfit_datasets(train_ds, val_suites, cfg: dict):
    """Replace train/val datasets with pre-sampled deterministic diagnostics."""
    fixed_cfg = cfg.get("fixed_overfit") or {}
    if not fixed_cfg.get("enabled", False):
        return train_ds, val_suites

    kwargs = {
        "num_samples": int(fixed_cfg.get("num_samples", 64)),
        "mode_policy": fixed_cfg.get("mode_policy", "mixed"),
        "force_no_path_aug": bool(fixed_cfg.get("force_no_path_aug", True)),
        "force_text_idx": fixed_cfg.get("force_text_idx", 0),
    }
    train_samples = build_fixed_samples(train_ds, **kwargs)
    fixed_train = FixedRefinerSampleDataset(train_samples)

    if bool(fixed_cfg.get("val_on_train", True)) or not val_suites:
        fixed_val = FixedRefinerSampleDataset(train_samples)
    else:
        val_samples = build_fixed_samples(val_suites[0]["dataset"], **kwargs)
        fixed_val = FixedRefinerSampleDataset(val_samples)

    log.info(
        "fixed_overfit enabled: samples=%d mode_policy=%s force_no_path_aug=%s "
        "val_on_train=%s",
        len(fixed_train),
        kwargs["mode_policy"],
        kwargs["force_no_path_aug"],
        bool(fixed_cfg.get("val_on_train", True)),
    )
    return fixed_train, [
        {
            "name": "fixed_overfit",
            "mode_policy": kwargs["mode_policy"],
            "dataset": fixed_val,
        }
    ]


def apply_default_fixed_validation_dataset(train_ds, val_suites):
    """Replace validation suites with deterministic fixed samples."""
    if not val_suites:
        return train_ds, []
    fixed_suites = []
    for suite in val_suites:
        name = suite["name"]
        dataset = suite["dataset"]
        if len(dataset) == 0:
            log.warning("validation suite %s has no samples; skipping.", name)
            continue
        if isinstance(dataset, FixedRefinerSampleDataset):
            fixed_dataset = dataset
        else:
            force_anchor_frame = None
            if (
                suite.get("mode_policy") == "sliding"
                and getattr(dataset, "horizon_policy", None) == "max"
                and hasattr(dataset, "n_hist")
            ):
                force_anchor_frame = int(dataset.n_hist) - 1
            val_samples = build_fixed_samples(
                dataset,
                num_samples=len(dataset),
                mode_policy=suite.get("mode_policy", "random"),
                force_no_path_aug=True,
                force_text_idx=0,
                force_anchor_frame=force_anchor_frame,
            )
            fixed_dataset = FixedRefinerSampleDataset(val_samples)
        fixed_suites.append({**suite, "dataset": fixed_dataset})
        log.info(
            "fixed validation suite=%s samples=%d mode_policy=%s force_no_path_aug=True",
            name,
            len(fixed_dataset),
            suite.get("mode_policy", "random"),
        )
    return train_ds, fixed_suites


def _build_one_dataset(
    cfg: dict,
    split_file: str,
    *,
    seed: int | None = None,
    randomize_caption: bool = True,
    validation_suite: dict | None = None,
):
    split_name = "val" if validation_suite is not None else "train"
    return build_root_refiner_dataset(
        cfg,
        split_file,
        split=split_name,
        seed=seed,
        randomize_caption=randomize_caption,
        validation_suite=validation_suite,
    )


def _dataset_defaults(dataset: str) -> dict[str, str]:
    key = str(dataset).lower()
    if key not in DATASET_DEFAULTS:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {list(DATASET_DEFAULTS)}"
        )
    return DATASET_DEFAULTS[key]


def _default_split_file(data_cfg: Mapping, split: str, dataset: str) -> str:
    if split == "train":
        return data_cfg.get("train_split_file") or _dataset_defaults(dataset)[
            "split_file"
        ]
    if split == "val":
        return data_cfg.get("val_split_file") or _dataset_defaults(dataset)[
            "split_file"
        ]
    if split == "test":
        return data_cfg.get("test_split_file") or _dataset_defaults(dataset)[
            "split_file"
        ]
    return _dataset_defaults(dataset)["split_file"]


def _section(cfg: Mapping, key: str) -> Mapping:
    value = cfg.get(key, {}) if isinstance(cfg, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "DATASET_DEFAULTS",
    "DEFAULT_VALIDATION_SUITES",
    "apply_default_fixed_validation_dataset",
    "apply_fixed_overfit_datasets",
    "build_datasets",
    "build_humanml3d_dataset_cfg",
    "build_root_refiner_dataset",
    "normalize_validation_suites_in_cfg",
    "resolve_dataset_dir",
]
