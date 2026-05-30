#!/usr/bin/env python
"""Compute RootRefiner path feature normalization stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_refiner import _load_cfg, build_datasets, resolve_cfg_interpolations  # noqa: E402
from utils.refiner.path_feature_stats import (  # noqa: E402
    PATH_FEATURE_NAMES,
    compute_sampling_config_hash,
    compute_stats_from_features,
    save_path_feature_stats,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    cfg = resolve_cfg_interpolations(_load_cfg(args.config))
    cfg.setdefault("data", {})
    cfg["data"]["normalize"] = False
    train_ds, _ = build_datasets(cfg, seed=args.seed)
    if len(train_ds) == 0:
        raise ValueError("cannot compute path feature stats from an empty train split")

    features = []
    for i in range(int(args.num_samples)):
        sample = train_ds[i % len(train_ds)]
        features.append(sample["path_features"])
    feature_tensor = torch.stack(features)
    mean, std = compute_stats_from_features(feature_tensor)

    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    meta = {
        "dataset": data_cfg.get("dataset", "humanml3d"),
        "split": data_cfg.get("train_split_file", "train.txt"),
        "num_samples": int(args.num_samples),
        "sampling_config_hash": compute_sampling_config_hash(cfg),
        "n_path": int(model_cfg.get("n_path", 0)),
        "goal_point_representation": "current_to_goal_line",
        "feature_names": PATH_FEATURE_NAMES,
    }
    save_path_feature_stats(args.output_dir, mean=mean, std=std, meta=meta)


if __name__ == "__main__":
    main()
