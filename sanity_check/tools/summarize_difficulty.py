#!/usr/bin/env python3
"""Summarize HumanML3D sample difficulty and split into bins.

This script scans a HumanML3D metadata list (e.g. train.txt / test_min.txt),
loads per-sample motion features and trajectory features, computes a few simple
trajectory difficulty metrics, and writes a ranked table plus optional bin lists.

Expected directory layout (same as FloodNet datasets):
  <dataset_root>/new_joint_vecs/<name>.npy
  <dataset_root>/TOKENS_.../<name>.npy
  <dataset_root>/texts/<name>.txt

Usage example:
  python tools/summarize_humanml3d_difficulty.py \
    --meta /path/to/HumanML3D/train.txt \
    --out /path/to/outputs/difficulty.json \
    --bins easy,medium,hard
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils.motion_process import extract_root_trajectory_263, extract_root_trajectory_length


def load_feature(path: Path) -> np.ndarray:
    feat = np.load(path)
    if np.isnan(feat).any():
        raise ValueError(f"NaN values found in {path}")
    return feat


def compute_metrics(
    feature: np.ndarray,
    min_path_length: float,
    min_speed_std: float,
) -> Dict[str, float]:
    traj = extract_root_trajectory_263(feature)
    xz = traj[:, [0, 2]]
    traj_length = extract_root_trajectory_length(feature, xy_only=True)
    if len(xz) < 2:
        return {
            "traj_length": float(len(xz)),
            "path_length": float(traj_length),
            "displacement": 0.0,
            "turn_count": 0.0,
            "turn_angle_sum": 0.0,
            "speed_std": 0.0,
            "mean_speed": 0.0,
            "xz_area": 0.0,
            "curvature_proxy": 0.0,
            "stationary_ratio": 1.0,
            "path_to_disp": 0.0,
        }

    diffs = np.diff(xz, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(traj_length)
    displacement = float(np.linalg.norm(xz[-1] - xz[0]))
    speed_std = float(step_lengths.std() if len(step_lengths) > 1 else 0.0)
    mean_speed = float(step_lengths.mean() if len(step_lengths) > 0 else 0.0)
    stationary_ratio = float(np.mean(step_lengths < 1e-4)) if len(step_lengths) > 0 else 1.0
    if len(xz) >= 3:
        x_min, z_min = np.min(xz[:, 0]), np.min(xz[:, 1])
        x_max, z_max = np.max(xz[:, 0]), np.max(xz[:, 1])
        xz_area = float(max(x_max - x_min, 0.0) * max(z_max - z_min, 0.0))
    else:
        xz_area = 0.0

    # If the motion is very short or the velocity variation is too small,
    # heading-based metrics become unreliable and should be removed.
    suppress_heading_metrics = (
        path_length < min_path_length
        or speed_std < min_speed_std
        or displacement < 1e-3
        or stationary_ratio > 0.85
    )

    if suppress_heading_metrics:
        turn_count = 0.0
        turn_angle_sum = 0.0
        curvature_proxy = 0.0
    else:
        turn_count = 0.0
        turn_angle_sum = 0.0
        curvature_proxy = 0.0
        if len(diffs) >= 2:
            v1 = diffs[:-1]
            v2 = diffs[1:]
            n1 = np.linalg.norm(v1, axis=1, keepdims=True)
            n2 = np.linalg.norm(v2, axis=1, keepdims=True)
            valid = (n1.squeeze(-1) > 1e-8) & (n2.squeeze(-1) > 1e-8)
            cos = np.zeros(len(v1), dtype=np.float32)
            denom = (n1.squeeze(-1) * n2.squeeze(-1))
            cos[valid] = np.sum(v1[valid] * v2[valid], axis=1) / np.clip(denom[valid], 1e-8, None)
            cos = np.clip(cos, -1.0, 1.0)
            turn_angles = np.arccos(cos)
            turn_count = float(np.sum(turn_angles > math.radians(15.0)))
            turn_angle_sum = float(turn_angles.sum())
            curvature_proxy = float(turn_angle_sum / max(path_length, 1e-6))

    # Hard cap heading-based metrics for readability and to avoid stationary-motion noise.
    turn_count = float(min(turn_count, 3.0))
    curvature_proxy = float(min(curvature_proxy, 3.0))

    path_to_disp = float(path_length / max(displacement, 1e-6))
    return {
        "traj_length": float(len(xz)),
        "path_length": path_length,
        "displacement": displacement,
        "turn_count": turn_count,
        "turn_angle_sum": turn_angle_sum,
        "speed_std": speed_std,
        "mean_speed": mean_speed,
        "xz_area": xz_area,
        "curvature_proxy": curvature_proxy,
        "stationary_ratio": stationary_ratio,
        "path_to_disp": path_to_disp,
    }


def normalize(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    mn = float(arr.min())
    mx = float(arr.max())
    if abs(mx - mn) < 1e-8:
        return [0.0] * len(values)
    return ((arr - mn) / (mx - mn)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to HumanML3D meta text file")
    parser.add_argument("--out", required=True, help="Output json path")
    parser.add_argument("--feature-path", default="new_joint_vecs", help="Relative feature dir under dataset root")
    parser.add_argument("--bins", default="easy,medium,hard", help="Comma-separated difficulty bins")
    parser.add_argument("--hard-top-ratio", type=float, default=0.3, help="Top ratio to mark as hard")
    parser.add_argument("--easy-bottom-ratio", type=float, default=0.3, help="Bottom ratio to mark as easy")
    parser.add_argument("--max-frames", type=int, default=200, help="Drop samples longer than this many frames")
    parser.add_argument("--min-path-length", type=float, default=1.0, help="Suppress heading metrics when path length is below this threshold")
    parser.add_argument("--min-speed-std", type=float, default=1e-3, help="Suppress heading metrics when speed std is below this threshold")
    args = parser.parse_args()

    meta = Path(args.meta).resolve()
    dataset_root = meta.parent
    feature_dir = dataset_root / args.feature_path
    names = [line.strip() for line in meta.read_text().splitlines() if line.strip()]

    records: List[Dict[str, float]] = []
    filtered_long = 0
    filtered_missing = 0
    filtered_error = 0
    for name in names:
        feat_path = feature_dir / f"{name}.npy"
        if not feat_path.exists():
            filtered_missing += 1
            continue
        try:
            feat = load_feature(feat_path)
            if feat.shape[0] > args.max_frames:
                filtered_long += 1
                continue
            metrics = compute_metrics(
                feat,
                min_path_length=args.min_path_length,
                min_speed_std=args.min_speed_std,
            )
            metrics["name"] = name
            records.append(metrics)
        except Exception as e:
            filtered_error += 1
            print(f"[skip] {name}: {e}")

    if not records:
        raise RuntimeError("No valid samples found.")

    path_lengths = normalize([r["path_length"] for r in records])
    turn_counts = normalize([r["turn_count"] for r in records])
    path_ratios = normalize([r["path_to_disp"] for r in records])
    curvature_proxies = normalize([r["curvature_proxy"] for r in records])
    speed_stds = normalize([r["speed_std"] for r in records])
    mean_speeds = normalize([r["mean_speed"] for r in records])
    xz_areas = normalize([r["xz_area"] for r in records])
    stationary_ratios = normalize([r["stationary_ratio"] for r in records])

    for i, r in enumerate(records):
        r["difficulty"] = (
            0.23 * path_lengths[i]
            + 0.18 * turn_counts[i]
            + 0.12 * curvature_proxies[i]
            + 0.10 * speed_stds[i]
            + 0.10 * mean_speeds[i]
            + 0.28 * xz_areas[i]
            + 0.05 * (1.0 - stationary_ratios[i])
        )

    records.sort(key=lambda x: x["difficulty"], reverse=True)

    n = len(records)
    hard_n = max(1, int(round(n * args.hard_top_ratio)))
    easy_n = max(1, int(round(n * args.easy_bottom_ratio)))
    hard = records[:hard_n]
    easy = records[-easy_n:]
    medium = records[hard_n:-easy_n] if hard_n + easy_n < n else []

    bins = [b.strip() for b in args.bins.split(",") if b.strip()]
    out = {
        "meta": str(meta),
        "dataset_root": str(dataset_root),
        "max_frames": args.max_frames,
        "min_path_length": args.min_path_length,
        "min_speed_std": args.min_speed_std,
        "filtered": {
            "long": filtered_long,
            "missing": filtered_missing,
            "error": filtered_error,
        },
        "counts": {
            "total": n,
            "easy": len(easy),
            "medium": len(medium),
            "hard": len(hard),
        },
        "records": records,
        "bins": {},
    }
    bin_names = {}
    if len(bins) >= 3:
        bin_names[bins[0]] = [r["name"] for r in easy]
        bin_names[bins[1]] = [r["name"] for r in medium]
        bin_names[bins[2]] = [r["name"] for r in hard]
        out["bins"] = bin_names

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Also write plain-text bin files next to the JSON for easy dataset splits.
    for bin_name, names_in_bin in bin_names.items():
        txt_path = out_path.parent / f"{out_path.stem}_{bin_name}.txt"
        txt_path.write_text("\n".join(names_in_bin) + ("\n" if names_in_bin else ""))

    print(f"Saved {len(records)} records to {out_path}")
    print(
        f"Filtered out {filtered_long} long / {filtered_missing} missing / {filtered_error} error samples"
    )
    for bin_name, names_in_bin in bin_names.items():
        txt_path = out_path.parent / f"{out_path.stem}_{bin_name}.txt"
        print(f"Saved {len(names_in_bin)} names to {txt_path}")

    print("Top 10 hard samples:")
    for r in hard[:10]:
        print(
            f"  {r['name']}: diff={r['difficulty']:.4f}, path={r['path_length']:.3f}, "
            f"turns={r['turn_count']:.0f}, ratio={r['path_to_disp']:.3f}, "
            f"curv={r['curvature_proxy']:.4f}, speed_std={r['speed_std']:.4f}, "
            f"mean_speed={r['mean_speed']:.4f}, xz_area={r['xz_area']:.4f}"
        )


if __name__ == "__main__":
    main()
