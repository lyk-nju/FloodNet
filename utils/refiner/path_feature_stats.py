"""Path feature normalization statistics for RootRefiner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


PATH_FEATURE_NAMES = [
    "path_length",
    "start_dx",
    "start_dz",
    "start_distance",
    "chord_length",
]


@dataclass(frozen=True)
class PathFeatureStats:
    mean: Tensor
    std: Tensor
    meta: dict[str, Any]


def _sampling_relevant_config(cfg: dict[str, Any]) -> dict[str, Any]:
    model = cfg.get("model", {}) or {}
    data = cfg.get("data", {}) or {}
    return {
        "model": {
            "n_path": model.get("n_path"),
            "min_tokens": model.get("min_tokens"),
            "max_tokens": model.get("max_tokens"),
            "frames_per_token": model.get("frames_per_token"),
        },
        "sampling": cfg.get("sampling", {}) or {},
        "data": {
            "dataset": data.get("dataset"),
            "train_split_file": data.get("train_split_file"),
        },
    }


def compute_sampling_config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(
        _sampling_relevant_config(cfg),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_stats_from_features(features: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    features = torch.as_tensor(features, dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != len(PATH_FEATURE_NAMES):
        raise ValueError(
            f"path feature tensor must be [N, {len(PATH_FEATURE_NAMES)}], "
            f"got {tuple(features.shape)}"
        )
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp(min=float(eps))
    return mean, std


def save_path_feature_stats(
    output_dir: str | Path,
    *,
    mean: Tensor,
    std: Tensor,
    meta: dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mean = torch.as_tensor(mean, dtype=torch.float32)
    std = torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-6)
    if mean.shape != (len(PATH_FEATURE_NAMES),) or std.shape != (len(PATH_FEATURE_NAMES),):
        raise ValueError("path feature mean/std must be shape (5,)")
    full_meta = dict(meta)
    full_meta["feature_names"] = PATH_FEATURE_NAMES
    np.save(output / "path_features_mean.npy", mean.cpu().numpy())
    np.save(output / "path_features_std.npy", std.cpu().numpy())
    (output / "path_features_meta.json").write_text(
        json.dumps(full_meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_path_feature_stats(stats_dir: str | Path) -> PathFeatureStats:
    stats = Path(stats_dir)
    mean = torch.as_tensor(np.load(stats / "path_features_mean.npy"), dtype=torch.float32)
    std = torch.as_tensor(np.load(stats / "path_features_std.npy"), dtype=torch.float32).clamp(min=1e-6)
    meta = json.loads((stats / "path_features_meta.json").read_text(encoding="utf-8"))
    if mean.shape != (len(PATH_FEATURE_NAMES),) or std.shape != (len(PATH_FEATURE_NAMES),):
        raise ValueError("path feature mean/std must be shape (5,)")
    return PathFeatureStats(mean=mean, std=std, meta=meta)


def validate_path_feature_stats_meta(
    meta: dict[str, Any],
    *,
    expected_hash: str,
    allow_mismatch: bool = False,
) -> None:
    feature_names = meta.get("feature_names")
    if feature_names != PATH_FEATURE_NAMES:
        raise ValueError(
            f"path feature stats feature_names mismatch: {feature_names!r}"
        )
    actual_hash = meta.get("sampling_config_hash")
    if actual_hash != expected_hash and not allow_mismatch:
        raise ValueError(
            "path feature stats config hash mismatch: "
            f"expected {expected_hash}, got {actual_hash}"
        )


__all__ = [
    "PATH_FEATURE_NAMES",
    "PathFeatureStats",
    "compute_sampling_config_hash",
    "compute_stats_from_features",
    "save_path_feature_stats",
    "load_path_feature_stats",
    "validate_path_feature_stats_meta",
]
