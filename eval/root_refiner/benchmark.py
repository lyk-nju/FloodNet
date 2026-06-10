"""Refiner standalone benchmark (T_A_10, Benchmark A).

Evaluates RootRefiner predictions against RefinerDataset targets WITHOUT the
body model. Computes the metric suite from docs/TODO.md §T_A_10:

    num_token_top1_accuracy / num_token_top3_accuracy / num_token_MAE
    xyz_ADE / xyz_FDE
    heading_error_deg (median)
    fwd_speed_MAE      (per-frame fwd_delta channel; "speed" is the metric label)
    lateral_speed_MAE  (metric label; per-frame lateral displacement MAE,
                        not meters/second)
    yaw_rate_MAE       (per-frame yaw_delta channel)
    smoothness_acc_mean

Outputs a JSON summary + per-sample CSV.

⚠ Done-criteria thresholds (num_token top-1 > 0.5, heading_error_deg < 30°
median) require a TRAINED checkpoint (T_A_09). With random weights the pipeline
runs end-to-end but the numbers are meaningless — the smoke test only checks
that metrics are finite and the report is written.

References:
- docs/TODO.md §T_A_10 lines 1441-1478.
- docs/design.md §10.3 (Benchmark A metric definitions).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common.artifacts import (  # noqa: E402
    ensure_dir,
    standard_eval_artifact_dirs,
    write_eval_json,
)
from eval.common.visualization import (  # noqa: E402
    plot_xz_trajectories,
    plot_yaw_series,
    yaw_from_7d,
)
from eval.common.json import json_sanitize, write_json_strict  # noqa: E402
from eval.root_refiner.adapters import (  # noqa: E402
    DURATION_GROUNDTRUTH,
    DURATION_PRED,
    ROOT_REFINER_ARTIFACT_NAMES,
    build_root_refiner_sample_metadata,
    normalize_duration_mode,
)
from eval.root_refiner.metrics import (  # noqa: E402
    _heading_to_yaw,
    _lateral_component,
    compute_sample_metrics,
)
from utils.refiner.path_feature_stats import compute_sampling_config_hash  # noqa: E402


@dataclass(frozen=True)
class RootRefinerEvalSuite:
    name: str
    path_modes: tuple[str | None, ...]
    default_max_samples: int | None = None
    force_no_path_aug: bool = True
    duration_mode: str = DURATION_PRED

    @property
    def oracle_duration(self) -> bool:
        """Legacy compatibility alias for groundtruth-duration eval."""
        return self.duration_mode == DURATION_GROUNDTRUTH

    def to_json_dict(self) -> dict:
        return {
            "name": self.name,
            "path_modes": [
                mode if mode is not None else "mixed" for mode in self.path_modes
            ],
            "default_max_samples": self.default_max_samples,
            "force_no_path_aug": self.force_no_path_aug,
            "duration_mode": self.duration_mode,
            "oracle_duration": self.oracle_duration,
        }


_ROOT_REFINER_SUITES = {
    "smoke": RootRefinerEvalSuite(
        name="smoke",
        path_modes=(None,),
        default_max_samples=50,
        force_no_path_aug=True,
    ),
    "standard": RootRefinerEvalSuite(
        name="standard",
        path_modes=("dense_path", "sparse_path", "goal_point"),
        default_max_samples=None,
        force_no_path_aug=True,
    ),
    "standard_oracle": RootRefinerEvalSuite(
        name="standard_oracle",
        path_modes=("dense_path", "sparse_path", "goal_point"),
        default_max_samples=None,
        force_no_path_aug=True,
        duration_mode=DURATION_GROUNDTRUTH,
    ),
    "standard_groundtruth": RootRefinerEvalSuite(
        name="standard_groundtruth",
        path_modes=("dense_path", "sparse_path", "goal_point"),
        default_max_samples=None,
        force_no_path_aug=True,
        duration_mode=DURATION_GROUNDTRUTH,
    ),
    "stress": RootRefinerEvalSuite(
        name="stress",
        path_modes=("sparse_path", "goal_point"),
        default_max_samples=None,
        force_no_path_aug=False,
    ),
    "full_route": RootRefinerEvalSuite(
        name="full_route",
        path_modes=("dense_path",),
        default_max_samples=None,
        force_no_path_aug=True,
        duration_mode=DURATION_GROUNDTRUTH,
    ),
}


def resolve_suite_config(suite: str) -> RootRefinerEvalSuite:
    key = str(suite)
    if key not in _ROOT_REFINER_SUITES:
        valid = ", ".join(sorted(_ROOT_REFINER_SUITES))
        raise ValueError(f"unknown RootRefiner eval suite {key!r}; expected one of: {valid}")
    return _ROOT_REFINER_SUITES[key]


# ---------------------------------------------------------------------------
# Aggregate benchmark
# ---------------------------------------------------------------------------


def _get_eval_sample(
    dataset,
    idx: int,
    *,
    force_path_mode: str | None,
    force_no_path_aug: bool,
    force_mode: str | None = None,
    force_num_tokens: int | None = None,
    force_anchor_frame: int | None = None,
) -> dict:
    if not hasattr(dataset, "get_sample"):
        if force_path_mode is not None:
            raise ValueError("dataset does not support force_path_mode")
        return dataset[idx]

    kwargs = {
        "force_no_path_aug": bool(force_no_path_aug),
        "force_text_idx": 0,
    }
    if force_path_mode is not None:
        kwargs["force_path_mode"] = force_path_mode
    if force_mode is not None:
        kwargs["force_mode"] = force_mode
    if force_num_tokens is not None:
        kwargs["force_num_tokens"] = int(force_num_tokens)
    if force_anchor_frame is not None:
        kwargs["force_anchor_frame"] = int(force_anchor_frame)
    try:
        return dataset.get_sample(idx, **kwargs)
    except TypeError as exc:
        if force_path_mode is None:
            raise
        raise ValueError(
            "dataset.get_sample must support force_path_mode for RootRefiner "
            f"suite path-mode evaluation; got {force_path_mode!r}"
        ) from exc


def _to_int(value, default: int = 0) -> int:
    if value is None:
        return int(default)
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def _dataset_clip_for_eval_index(dataset, idx: int):
    if not hasattr(dataset, "_clips"):
        return None
    clip_idx = (
        int(dataset.valid_indices[idx]) if hasattr(dataset, "valid_indices") else int(idx)
    )
    return dataset._clips[clip_idx]


def _clip_metadata_for_eval_index(dataset, idx: int) -> dict:
    clip = _dataset_clip_for_eval_index(dataset, idx)
    if not isinstance(clip, Mapping):
        return {}
    meta = {}
    raw_id = clip.get("raw_id", clip.get("name"))
    if raw_id is not None:
        meta["raw_id"] = raw_id
    for key in ("split_index", "split_file", "dataset"):
        value = clip.get(key)
        if value is not None:
            meta[key] = value
    return meta


def build_eval_task_specs(dataset, max_samples: int = -1) -> list[dict]:
    """Freeze underlying RootRefiner tasks before path-mode bucketing.

    Each spec captures the dataset index, full/sliding mode, target duration,
    and anchor frame. Path-mode suites can then rebuild dense/sparse/goal
    conditions over identical tasks instead of comparing different random
    anchors or horizons.
    """
    if hasattr(dataset, "reset_rng"):
        dataset.reset_rng()
    n = len(dataset) if max_samples < 0 else min(int(max_samples), len(dataset))
    specs = []
    for idx in range(n):
        sample = _get_eval_sample(
            dataset,
            idx,
            force_path_mode=None,
            force_no_path_aug=True,
        )
        mode = str(sample.get("mode", "sample"))
        num_tokens = _to_int(sample.get("num_tokens"), default=0)
        anchor_frame = _to_int(sample.get("anchor_frame"), default=0)
        specs.append(
            {
                "idx": idx,
                **_clip_metadata_for_eval_index(dataset, idx),
                "mode": mode,
                "num_tokens": num_tokens,
                "anchor_frame": anchor_frame,
                "task_key": f"{idx}:{mode}:{num_tokens}:{anchor_frame}",
            }
        )
    return specs


def _max_tokens_for_full_route(dataset, idx: int, anchor_frame: int = 0) -> int:
    clip = _dataset_clip_for_eval_index(dataset, idx)
    if clip is None:
        sample = _get_eval_sample(
            dataset,
            idx,
            force_path_mode="dense_path",
            force_no_path_aug=True,
            force_mode="full",
            force_anchor_frame=anchor_frame,
        )
        return _to_int(sample.get("num_tokens"), default=0)
    motion = clip["motion_263"]
    T = int(motion.shape[0])
    remaining = max(0, T - int(anchor_frame))
    frames_per_token = int(getattr(dataset, "frames_per_token", 4))
    max_tokens = int(getattr(dataset, "max_tokens", 49))
    return min(max_tokens, (remaining + frames_per_token - 1) // frames_per_token)


def build_full_route_task_specs(
    dataset,
    *,
    max_samples: int = -1,
    raw_ids: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """Build deterministic full-clip RootRefiner tasks.

    These specs answer the full-route diagnostic question: given a complete GT
    route from clip start, can RootRefiner reproduce the whole future root plan?
    `sample_XXXXXX` artifact names remain eval-order IDs; raw dataset identity
    is carried by `raw_id` / `split_index` in the spec and metadata.
    """
    wanted = None if raw_ids is None else {str(raw_id) for raw_id in raw_ids}
    specs = []
    max_n = None if int(max_samples) < 0 else int(max_samples)
    for idx in range(len(dataset)):
        meta = _clip_metadata_for_eval_index(dataset, idx)
        raw_id = str(meta.get("raw_id", meta.get("name", idx)))
        if wanted is not None and raw_id not in wanted:
            continue
        num_tokens = _max_tokens_for_full_route(dataset, idx, anchor_frame=0)
        spec = {
            "idx": idx,
            **meta,
            "mode": "full",
            "num_tokens": num_tokens,
            "anchor_frame": 0,
            "task_key": f"{raw_id}:full:{num_tokens}:0",
        }
        specs.append(spec)
        if wanted is None and max_n is not None and len(specs) >= max_n:
            break
    if wanted is not None:
        found = {str(spec.get("raw_id", spec.get("name", spec["idx"]))) for spec in specs}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"raw_id(s) not found in eval split: {missing}")
        if max_n is not None:
            specs = specs[:max_n]
    return specs


def _resolve_cli_force_path_mode(
    *,
    task_specs: list[dict] | None,
    suite: str | None,
) -> str | None:
    """Return CLI-only forced path mode for direct full-route runs.

    `--suite full_route` already applies the suite's dense_path mode through
    `run_suite_benchmark()`. Direct `--full_route` / `--raw_id` invocations
    bypass suite path modes, so they must force dense_path here to preserve the
    full-route diagnostic contract.
    """
    if task_specs is not None and suite is None:
        return "dense_path"
    return None


def _nanmean_from_samples(per_sample: list[dict], key: str) -> float:
    vals = [s[key] for s in per_sample if key in s and not math.isnan(s[key])]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _nanmedian_from_samples(per_sample: list[dict], key: str) -> float:
    vals = sorted(s[key] for s in per_sample if key in s and not math.isnan(s[key]))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2)


def summarize_per_sample(
    per_sample: list[dict],
    *,
    duration_mode: str | None = None,
    oracle_duration: bool = False,
) -> dict:
    resolved_duration_mode = resolve_duration_mode(
        duration_mode=duration_mode,
        oracle_duration=oracle_duration,
    )
    use_groundtruth_duration = resolved_duration_mode == DURATION_GROUNDTRUTH
    n = len(per_sample)
    unique_task_keys = {
        str(s.get("task_key", s.get("idx", i)))
        for i, s in enumerate(per_sample)
    }
    n_unique = len(unique_task_keys)
    return {
        "n_samples": n_unique,
        "n_records": n,
        "n_unique_tasks": n_unique,
        "duration_mode": resolved_duration_mode,
        "oracle_duration": use_groundtruth_duration,
        "num_token_top1_accuracy": (
            sum(int(s.get("num_token_top1_hit", 0)) for s in per_sample) / n
            if n else float("nan")
        ),
        "num_token_top3_accuracy": (
            sum(int(s.get("num_token_top3_hit", 0)) for s in per_sample) / n
            if n else float("nan")
        ),
        "num_token_MAE": (
            sum(
                abs(int(s["pred_num_tokens"]) - int(s["gt_num_tokens"]))
                for s in per_sample
            ) / n
            if n else float("nan")
        ),
        "xyz_ADE": _nanmean_from_samples(per_sample, "xyz_ADE"),
        "xyz_FDE": _nanmean_from_samples(per_sample, "xyz_FDE"),
        "heading_error_deg": _nanmedian_from_samples(per_sample, "heading_error_deg"),
        "fwd_speed_MAE": _nanmean_from_samples(per_sample, "fwd_speed_MAE"),
        "lateral_speed_MAE": _nanmean_from_samples(per_sample, "lateral_speed_MAE"),
        "yaw_rate_MAE": _nanmean_from_samples(per_sample, "yaw_rate_MAE"),
        "smoothness_acc_mean": _nanmean_from_samples(per_sample, "smoothness"),
    }


def resolve_duration_mode(
    *,
    duration_mode: str | None = None,
    oracle_duration: bool = False,
) -> str:
    if duration_mode is None:
        return DURATION_GROUNDTRUTH if oracle_duration else DURATION_PRED
    resolved = normalize_duration_mode(duration_mode)
    if oracle_duration and resolved != DURATION_GROUNDTRUTH:
        raise ValueError(
            "conflicting duration options: --oracle_duration is compatible only "
            f"with duration_mode={DURATION_GROUNDTRUTH!r}"
        )
    return resolved


def resolve_suite_duration_mode(
    suite_cfg: RootRefinerEvalSuite,
    *,
    duration_mode: str | None = None,
    oracle_duration: bool = False,
) -> str:
    if duration_mode is not None or oracle_duration:
        return resolve_duration_mode(
            duration_mode=duration_mode,
            oracle_duration=oracle_duration,
        )
    return normalize_duration_mode(suite_cfg.duration_mode)


_CKPT_CONFIG_CONTRACT_KEYS = (
    ("model", "params", "n_hist"),
    ("model", "params", "n_path"),
    ("model", "params", "max_tokens"),
    ("model", "params", "min_tokens"),
    ("model", "params", "frames_per_token"),
)


def _nested_get(mapping, path: tuple[str, ...]):
    cur = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def validate_ckpt_eval_config_compatible(ckpt_cfg: dict, eval_cfg: dict) -> None:
    """Fail fast when benchmark --config disagrees with checkpoint hparams."""
    mismatches = []
    for path in _CKPT_CONFIG_CONTRACT_KEYS:
        ckpt_value = _nested_get(ckpt_cfg, path)
        eval_value = _nested_get(eval_cfg, path)
        if ckpt_value is None or eval_value is None:
            continue
        if ckpt_value != eval_value:
            dotted = ".".join(path)
            mismatches.append(f"{dotted}: ckpt={ckpt_value!r} eval={eval_value!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            "RootRefiner benchmark config mismatch between checkpoint hparams "
            f"and --config: {joined}. Use the training config for this checkpoint."
        )


def _as_cpu_tensor(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _valid_rows(value, mask=None) -> np.ndarray:
    arr = _as_cpu_tensor(value).numpy()
    if mask is None:
        return arr
    m = _as_cpu_tensor(mask).bool().numpy().reshape(-1)
    if arr.shape[0] != m.shape[0]:
        return arr
    return arr[m]


def _path_mode_to_artifact_route_mode(path_mode: str | None) -> str:
    mapping = {
        "dense_path": "dense_gt",
        "sparse_path": "sparse_gt",
        "goal_point": "user_polyline",
        None: "dense_gt",
        "mixed": "dense_gt",
    }
    return mapping.get(path_mode, "user_polyline")


def _denormalize_path_xz(path_xz, wp_mean, wp_std, wp_norm_idx) -> torch.Tensor:
    out = _as_cpu_tensor(path_xz).float().clone()
    if wp_mean is None or wp_std is None or wp_norm_idx is None:
        return out
    mean = _as_cpu_tensor(wp_mean).float()
    std = _as_cpu_tensor(wp_std).float().clamp(min=1e-6)
    idx_set = set(_as_cpu_tensor(wp_norm_idx).long().tolist())
    if 0 in idx_set and out.shape[-1] >= 1:
        out[..., 0] = out[..., 0] * std[0] + mean[0]
    if 2 in idx_set and out.shape[-1] >= 2:
        out[..., 1] = out[..., 1] * std[2] + mean[2]
    return out


def _sample_anchor_world(sample: dict) -> tuple[list[float], float]:
    xz = sample.get("anchor_xz_world", None)
    yaw = sample.get("anchor_yaw_world", None)
    if xz is None:
        anchor_xz = [0.0, 0.0]
    else:
        anchor_arr = _as_cpu_tensor(xz).float().reshape(-1)
        anchor_xz = [float(anchor_arr[0].item()), float(anchor_arr[1].item())]
    anchor_yaw = 0.0 if yaw is None else float(_as_cpu_tensor(yaw).float().item())
    return anchor_xz, anchor_yaw


def _pred_duration_mask(model, used_tokens: int, mask_len: int) -> torch.Tensor:
    valid_eff = int(model.frames_per_token) * int(used_tokens) - (
        int(model.frames_per_token) - 1
    )
    return torch.arange(mask_len, dtype=torch.long) < max(0, int(valid_eff))


def _rootplan_artifact_payload(
    root_7d,
    *,
    duration_mode: str,
    valid_mask,
    pred_num_tokens: int,
    frames_per_token: int,
    source: str,
    anchor_world_xz: list[float],
    anchor_world_yaw: float,
) -> dict:
    root = _valid_rows(root_7d, valid_mask)
    return {
        "schema_version": "root_refiner_rootplan_artifact.v1",
        "duration_mode": normalize_duration_mode(duration_mode),
        "source": str(source),
        "coordinate_frame": "anchor_local",
        "frames_per_token": int(frames_per_token),
        "pred_num_tokens": int(pred_num_tokens),
        "valid_frames": int(root.shape[0]),
        "anchor_commit_idx": 0,
        "anchor_world_xz": [float(anchor_world_xz[0]), float(anchor_world_xz[1])],
        "anchor_world_yaw": float(anchor_world_yaw),
        "waypoints_local_7d": root.astype(np.float32).tolist(),
    }


def _write_root_refiner_sample_artifacts(
    sample_dir: str | Path,
    *,
    sample: dict,
    sample_id: str,
    metrics: dict,
    pred_by_duration: dict[str, dict],
    gt_root_7d: torch.Tensor,
    gt_mask: torch.Tensor,
    frames_per_token: int,
    wp_mean=None,
    wp_std=None,
    wp_norm_idx=None,
) -> None:
    out = ensure_dir(sample_dir)
    shared_names = ROOT_REFINER_ARTIFACT_NAMES["shared"]

    path_physical = _denormalize_path_xz(
        sample.get("path", torch.zeros(0, 2)),
        wp_mean,
        wp_std,
        wp_norm_idx,
    )
    path_valid_mask = _as_cpu_tensor(
        sample.get(
            "path_valid_mask",
            torch.ones(path_physical.shape[0], dtype=torch.bool),
        )
    ).bool()
    path_control_mask = _as_cpu_tensor(
        sample.get(
            "path_control_mask",
            torch.zeros(path_physical.shape[0], dtype=torch.bool),
        )
    ).bool()
    route_input = _valid_rows(path_physical, path_valid_mask)
    route_valid_mask = _valid_rows(path_valid_mask, path_valid_mask).astype(bool)
    route_control_mask = _valid_rows(path_control_mask, path_valid_mask).astype(bool)
    np.save(out / "route_input.npy", route_input.astype(np.float32))
    np.save(out / "route_valid_mask.npy", route_valid_mask)
    np.save(out / "route_control_mask.npy", route_control_mask)
    np.save(out / "gt_root_7d.npy", _valid_rows(gt_root_7d, gt_mask).astype(np.float32))

    path_mode = str(sample.get("path_mode", "dense_path"))
    offset_frame = _to_int(sample.get("offset_start_frames"), default=0)
    frames_per_token_i = int(frames_per_token)
    offset_token = offset_frame // max(1, frames_per_token_i)
    anchor_frame = _to_int(sample.get("anchor_frame"), default=0)
    gt_slice_start = anchor_frame
    gt_slice_end = gt_slice_start + int(_as_cpu_tensor(gt_mask).bool().sum().item())
    anchor_world_xz, anchor_world_yaw = _sample_anchor_world(sample)
    metadata = build_root_refiner_sample_metadata(
        sample_id=sample_id,
        route_mode=_path_mode_to_artifact_route_mode(path_mode),
        duration_mode=str(metrics.get("duration_mode", DURATION_PRED)),
        offset_frame=offset_frame,
        offset_token=offset_token,
        anchor_world_xz=anchor_world_xz,
        anchor_world_yaw=anchor_world_yaw,
        gt_slice_start=gt_slice_start,
        gt_slice_end=gt_slice_end,
        extra={
            "raw_id": sample.get("raw_id", metrics.get("raw_id")),
            "split_index": sample.get("split_index", metrics.get("split_index")),
            "split_file": sample.get("split_file", metrics.get("split_file")),
            "dataset": sample.get("dataset", metrics.get("dataset")),
            "task_key": metrics.get("task_key"),
            "path_mode": path_mode,
            "mode": sample.get("mode"),
            "text": sample.get("text"),
            "gt_num_tokens": int(metrics.get("gt_num_tokens", 0)),
        },
    )
    write_eval_json(out / shared_names["metadata"], metadata)
    write_eval_json(out / shared_names["metrics"], metrics)

    xz_series: dict[str, object] = {
        "route_input": route_input,
        "gt_root": _valid_rows(gt_root_7d, gt_mask),
    }
    yaw_series: dict[str, object] = {"gt_yaw": yaw_from_7d(_valid_rows(gt_root_7d, gt_mask))}

    for duration, pred_payload in pred_by_duration.items():
        names = ROOT_REFINER_ARTIFACT_NAMES[duration]
        root = _as_cpu_tensor(pred_payload["root_7d"]).float()
        mask = _as_cpu_tensor(pred_payload["mask"]).bool()
        pred_num_tokens = int(pred_payload["pred_num_tokens"])
        valid_root = _valid_rows(root, mask)
        np.save(out / names["pred_root_7d"], valid_root.astype(np.float32))
        write_eval_json(
            out / names["rootplan"],
            _rootplan_artifact_payload(
                root,
                duration_mode=duration,
                valid_mask=mask,
                pred_num_tokens=pred_num_tokens,
                frames_per_token=frames_per_token_i,
                source="root_refiner_benchmark",
                anchor_world_xz=anchor_world_xz,
                anchor_world_yaw=anchor_world_yaw,
            ),
        )
        xz_series[duration] = valid_root
        yaw_series[f"{duration}_yaw"] = yaw_from_7d(valid_root)

    plot_xz_trajectories(
        out / shared_names["plot_xz"],
        xz_series,
        point_series={
            "route_control": route_input[route_control_mask],
        },
        title=f"RootRefiner {sample_id}",
    )
    plot_yaw_series(
        out / shared_names["plot_yaw"],
        yaw_series,
        title=f"RootRefiner {sample_id}",
    )


@torch.no_grad()
def run_benchmark(
    model,
    dataset,
    text_encoder,
    device="cpu",
    max_samples: int = -1,
    oracle_duration: bool = False,
    duration_mode: str | None = None,
    force_path_mode: str | None = None,
    force_no_path_aug: bool = True,
    task_specs: list[dict] | None = None,
    artifact_dir: str | Path | None = None,
    artifact_max_samples: int = 0,
) -> dict:
    """Run inference over `dataset` and aggregate the metric suite.

    `model` is a RootRefiner (eval mode). `text_encoder` must have
    `.encode(list[str], device=...) -> [B, text_emb_dim]`.

    `duration_mode`: `pred_duration` uses model-predicted duration;
    `groundtruth_duration` feeds GT num_tokens so trajectory metrics
    (xyz_ADE/FDE, heading, ...) measure the WAYPOINT DECODER ALONE under the
    correct horizon, isolating it from num_token-head prediction error.
    `oracle_duration` is a deprecated alias for `groundtruth_duration`.
    Default = real inference (model picks its expected-round duration, so
    trajectory metrics use the common GT/predicted prefix).
    top-k duration metrics still use classification logits; num_token_MAE uses
    the actual predicted duration that drives inference.

    Returns dict with `summary` (aggregate metrics) and `per_sample` (list).
    """
    duration_mode_resolved = resolve_duration_mode(
        duration_mode=duration_mode,
        oracle_duration=oracle_duration,
    )
    use_groundtruth_duration = duration_mode_resolved == DURATION_GROUNDTRUTH
    model = model.to(device).eval()
    min_tokens = model.min_tokens

    # Reproducibility: when no frozen task list is provided, get_sample advances
    # the dataset RNG every call, so reset it to the base seed for comparable
    # single-pass/oracle runs.
    if task_specs is None and hasattr(dataset, "reset_rng"):
        dataset.reset_rng()

    if task_specs is None:
        n = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))
        task_specs = [{"idx": idx} for idx in range(n)]
    else:
        n = len(task_specs)

    per_sample = []
    artifact_count = 0

    for spec in task_specs:
        idx = int(spec["idx"])
        # force_no_path_aug controls whether the benchmark measures clean path
        # conditions or intentionally stresses offset/sparse path augmentation.
        # force_text_idx=0 pins the canonical first caption (training randomizes
        # over all captions for augmentation; eval must stay comparable).
        sample = _get_eval_sample(
            dataset,
            idx,
            force_path_mode=force_path_mode,
            force_no_path_aug=force_no_path_aug,
            force_mode=spec.get("mode"),
            force_num_tokens=spec.get("num_tokens"),
            force_anchor_frame=spec.get("anchor_frame"),
        )
        text_emb = text_encoder.encode([sample["text"]], device=device)
        from utils.motion_process import build_physical_7d_from_normalized_5d
        wp_mean = getattr(dataset, "_wp_mean", None)
        wp_std = getattr(dataset, "_wp_std", None)
        wp_norm_idx = getattr(dataset, "_wp_norm_idx", None)

        def _forward_duration(mode_name: str) -> tuple[dict, torch.Tensor, torch.Tensor]:
            teacher_num_tokens = (
                sample["num_tokens"].reshape(1).to(device)
                if mode_name == DURATION_GROUNDTRUTH
                else None
            )
            output = model(
                text_emb=text_emb,
                path=sample["path"].unsqueeze(0).to(device),
                path_valid_mask=sample["path_valid_mask"].unsqueeze(0).to(device),
                path_control_mask=sample["path_control_mask"].unsqueeze(0).to(device),
                path_mode=[sample.get("path_mode", "dense_path")],
                path_features=sample["path_features"].unsqueeze(0).to(device),
                history_motion=sample["history_motion"].unsqueeze(0).to(device),
                history_mask=sample["history_mask"].unsqueeze(0).to(device),
                offset_start_frames=sample.get(
                    "offset_start_frames",
                    torch.tensor(0, dtype=torch.long),
                ).reshape(1).to(device),
                num_tokens=teacher_num_tokens,
            )
            # Model emits NORMALIZED 5D. Assemble physical 7D at the boundary
            # (unnormalize xyz → unit heading → append fwd_delta / yaw_delta) so
            # metrics are computed in physical space.
            pred_root = build_physical_7d_from_normalized_5d(
                output["waypoints"][0].cpu(), wp_mean, wp_std, wp_norm_idx,
            )
            used_tokens = int(output["used_num_tokens"][0].detach().cpu().item())
            valid_mask = (
                _as_cpu_tensor(sample.get("target_mask", sample.get("waypoints_mask"))).bool()
                if mode_name == DURATION_GROUNDTRUTH
                else _pred_duration_mask(model, used_tokens, pred_root.shape[0])
            )
            return output, pred_root, valid_mask

        out, pred_wp, duration_mask = _forward_duration(duration_mode_resolved)
        logits = out["num_token_logits"][0]                  # [K]
        # GT `target_waypoints` is PHYSICAL-then-z-scored, so unnormalize its xyz
        # the same way and re-derive its deltas.
        gt_source = sample.get("target_waypoints", sample.get("waypoints"))
        gt5_norm = gt_source[..., :5]
        gt_wp = build_physical_7d_from_normalized_5d(
            gt5_norm, wp_mean, wp_std, wp_norm_idx,
        )                                                    # [max_frames, 7] physical
        gt_mask = _as_cpu_tensor(sample.get("target_mask", sample.get("waypoints_mask"))).bool()
        mask = duration_mask
        if not use_groundtruth_duration:
            mask = gt_mask & duration_mask.bool()

        # num_token metrics.
        gt_class = int(sample["num_tokens"].item()) - min_tokens
        gt_class = max(0, min(gt_class, logits.shape[-1] - 1))
        pred_class = int(logits.argmax().item())
        pred_num_tokens = int(out["pred_num_tokens"][0].detach().cpu().item())
        top3 = torch.topk(logits, k=min(3, logits.shape[-1])).indices.tolist()
        top1_hit = int(pred_class == gt_class)
        top3_hit = int(gt_class in top3)

        m = compute_sample_metrics(pred_wp, gt_wp, mask)
        m["idx"] = idx
        m["task_key"] = spec.get("task_key", f"{idx}")
        m["mode"] = sample.get("mode", spec.get("mode"))
        m["anchor_frame"] = _to_int(sample.get("anchor_frame"), default=0)
        m["path_mode"] = sample.get("path_mode", force_path_mode or "mixed")
        for meta_key in ("raw_id", "split_index", "split_file", "dataset"):
            value = sample.get(meta_key, spec.get(meta_key))
            if value is not None:
                m[meta_key] = value
        m["num_token_top1_hit"] = top1_hit
        m["num_token_top3_hit"] = top3_hit
        m["pred_num_tokens"] = pred_num_tokens
        m["argmax_num_tokens"] = pred_class + min_tokens
        m["gt_num_tokens"] = gt_class + min_tokens
        m["duration_mode"] = duration_mode_resolved
        per_sample.append(m)

        if artifact_dir is not None and (
            int(artifact_max_samples) <= 0 or artifact_count < int(artifact_max_samples)
        ):
            used_num_tokens = int(out["used_num_tokens"][0].detach().cpu().item())
            pred_by_duration = {
                duration_mode_resolved: {
                    "root_7d": pred_wp,
                    "mask": (
                        duration_mask
                        if duration_mode_resolved == DURATION_PRED
                        else gt_mask
                    ),
                    "pred_num_tokens": used_num_tokens,
                }
            }
            for extra_duration in (DURATION_PRED, DURATION_GROUNDTRUTH):
                if extra_duration in pred_by_duration:
                    continue
                extra_out, extra_wp, extra_mask = _forward_duration(extra_duration)
                extra_pred_num_tokens = int(
                    extra_out["used_num_tokens"][0].detach().cpu().item()
                )
                pred_by_duration[extra_duration] = {
                    "root_7d": extra_wp,
                    "mask": extra_mask,
                    "pred_num_tokens": extra_pred_num_tokens,
                }
            sample_id = f"sample_{artifact_count:06d}"
            _write_root_refiner_sample_artifacts(
                Path(artifact_dir) / sample_id,
                sample=sample,
                sample_id=sample_id,
                metrics=m,
                pred_by_duration=pred_by_duration,
                gt_root_7d=gt_wp,
                gt_mask=gt_mask,
                frames_per_token=int(model.frames_per_token),
                wp_mean=wp_mean,
                wp_std=wp_std,
                wp_norm_idx=wp_norm_idx,
            )
            artifact_count += 1

    summary = summarize_per_sample(
        per_sample,
        duration_mode=duration_mode_resolved,
    )
    return {"summary": summary, "per_sample": per_sample}


def _suite_sample_limit(
    suite_cfg: RootRefinerEvalSuite,
    max_samples: int,
) -> int:
    if max_samples is not None and int(max_samples) > 0:
        return int(max_samples)
    if suite_cfg.default_max_samples is not None:
        return int(suite_cfg.default_max_samples)
    return -1


def build_refiner_dataset_from_clips(
    cfg: dict,
    clips,
    *,
    dataset_cls,
    seed: int = 0,
):
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg["model"]["params"]
    sampling_cfg = cfg.get("sampling", {}) or {}
    path_condition_cfg = sampling_cfg.get("path_condition", {}) or {}
    offset_cfg = path_condition_cfg.get("offset_start", {}) or {}
    sparse_cfg = path_condition_cfg.get("sparse_path", {}) or {}
    normalize = bool(data_cfg.get("normalize", False))
    path_feature_stats_dir = data_cfg.get("path_feature_stats_dir") if normalize else None
    return dataset_cls(
        clips,
        n_hist=model_cfg["n_hist"],
        n_path=model_cfg["n_path"],
        max_tokens=model_cfg["max_tokens"],
        min_tokens=model_cfg["min_tokens"],
        frames_per_token=model_cfg["frames_per_token"],
        full_plan_ratio=sampling_cfg.get("full_plan_ratio", 0.5),
        horizon_policy=sampling_cfg.get("horizon_policy", "random"),
        path_condition_policy=path_condition_cfg.get("policy", "dense_path"),
        path_condition_ratios=path_condition_cfg.get("ratios"),
        offset_start_enabled=bool(offset_cfg.get("enabled", False)),
        offset_start_prob=float(offset_cfg.get("prob", 0.0)),
        offset_start_max_frames=int(offset_cfg.get("max_frames", 40)),
        offset_start_apply_to=tuple(
            offset_cfg.get("apply_to", ("dense_path", "sparse_path"))
        ),
        sparse_path_point_range=tuple(sparse_cfg.get("point_range", (3, 8))),
        normalize=normalize,
        stats_dir=data_cfg.get("stats_dir") if normalize else None,
        path_feature_stats_dir=path_feature_stats_dir,
        sampling_config_hash=(
            compute_sampling_config_hash(cfg)
            if path_feature_stats_dir is not None
            else None
        ),
        seed=seed,
    )


def _path_mode_label(path_mode: str | None) -> str:
    return path_mode if path_mode is not None else "mixed"


def _add_prefixed_summary(
    target: dict,
    *,
    prefix: str,
    summary: dict,
) -> None:
    for key, value in summary.items():
        target[f"{prefix}/{key}"] = value


def build_eval_payload(
    result: dict,
    *,
    suite: str = "single",
    suite_config: RootRefinerEvalSuite | None = None,
    runs: list[dict] | None = None,
) -> dict:
    if result.get("schema_version") == "root_refiner_eval.v1":
        return result
    return {
        "schema_version": "root_refiner_eval.v1",
        "evaluator": "root_refiner",
        "suite": str(suite),
        "suite_config": (
            suite_config.to_json_dict() if suite_config is not None else None
        ),
        "summary": result["summary"],
        "runs": runs or [],
        "per_sample": result["per_sample"],
    }


def _json_sanitize(value):
    return json_sanitize(value)


def run_suite_benchmark(
    model,
    dataset,
    text_encoder,
    *,
    suite: str,
    device="cpu",
    max_samples: int = -1,
    duration_mode: str | None = None,
    oracle_duration: bool = False,
    artifact_dir: str | Path | None = None,
    artifact_max_samples: int = 0,
    task_specs: list[dict] | None = None,
) -> dict:
    suite_cfg = resolve_suite_config(suite)
    duration_mode_resolved = resolve_suite_duration_mode(
        suite_cfg,
        duration_mode=duration_mode,
        oracle_duration=oracle_duration,
    )
    sample_limit = _suite_sample_limit(suite_cfg, max_samples)
    if task_specs is None:
        task_specs = build_eval_task_specs(dataset, max_samples=sample_limit)
    runs: list[dict] = []
    per_sample: list[dict] = []

    for path_mode in suite_cfg.path_modes:
        label = _path_mode_label(path_mode)
        run = run_benchmark(
            model,
            dataset,
            text_encoder,
            device=device,
            max_samples=sample_limit,
            duration_mode=duration_mode_resolved,
            force_path_mode=path_mode,
            force_no_path_aug=suite_cfg.force_no_path_aug,
            task_specs=task_specs,
            artifact_dir=(
                Path(artifact_dir) / label if artifact_dir is not None else None
            ),
            artifact_max_samples=artifact_max_samples,
        )
        run_samples = []
        for sample in run["per_sample"]:
            sample = dict(sample)
            sample["suite"] = suite_cfg.name
            sample["run_name"] = label
            sample["path_mode"] = label
            run_samples.append(sample)
        runs.append(
            {
                "name": label,
                "path_mode": label,
                "summary": run["summary"],
            }
        )
        per_sample.extend(run_samples)

    summary = summarize_per_sample(
        per_sample,
        duration_mode=duration_mode_resolved,
    )
    summary["n_samples"] = len(task_specs)
    summary["n_records"] = len(per_sample)
    summary["n_unique_tasks"] = len(task_specs)
    summary["num_runs"] = len(runs)
    for run in runs:
        _add_prefixed_summary(
            summary,
            prefix=f"path_mode/{run['path_mode']}",
            summary=run["summary"],
        )

    return build_eval_payload(
        {"summary": summary, "per_sample": per_sample},
        suite=suite_cfg.name,
        suite_config=RootRefinerEvalSuite(
            name=suite_cfg.name,
            path_modes=suite_cfg.path_modes,
            default_max_samples=suite_cfg.default_max_samples,
            force_no_path_aug=suite_cfg.force_no_path_aug,
            duration_mode=duration_mode_resolved,
        ),
        runs=runs,
    )


def _write_per_sample_csv(path: str | Path, per_sample: list[dict]) -> None:
    if not per_sample:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for sample in per_sample for key in sample.keys()})
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(per_sample)


def write_report(
    result: dict,
    output_dir: str | Path,
    *,
    suite: str | None = None,
    run_id: str | None = None,
    write_standard_layout: bool = True,
) -> None:
    """Write legacy files plus run_eval-style RootRefiner artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite_tag = str(suite or result.get("suite", "single"))
    payload = _json_sanitize(build_eval_payload(result, suite=suite_tag))
    if suite is not None:
        payload["suite"] = str(suite)
    if run_id is not None:
        payload["run_id"] = str(run_id)
    write_json_strict(out / "metrics.json", payload)
    write_json_strict(out / "summary.json", payload["summary"])
    per_sample = payload["per_sample"]
    _write_per_sample_csv(out / "per_sample.csv", per_sample)

    if write_standard_layout:
        run_tag = str(run_id or payload.get("run_id") or "latest")
        dirs = standard_eval_artifact_dirs(
            out,
            evaluator="RootRefiner",
            probe_tag=str(payload.get("suite", suite_tag)),
            run_id=run_tag,
            artifact_kinds=("metrics", "per_sample"),
        )
        write_json_strict(dirs["metrics"] / "metrics.json", payload)
        write_json_strict(dirs["metrics"] / "summary.json", payload["summary"])
        _write_per_sample_csv(dirs["per_sample"] / "per_sample.csv", per_sample)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_model_from_ckpt(ckpt_path: str, device: str):
    """Load RootRefiner weights from a Lightning checkpoint or raw state_dict."""
    from train_refiner import RefinerLightningModule

    ckpt = torch.load(ckpt_path, map_location=device)
    if "hyper_parameters" in ckpt and "cfg" in ckpt["hyper_parameters"]:
        cfg = ckpt["hyper_parameters"]["cfg"]
        module = RefinerLightningModule(cfg)
        module.load_state_dict(ckpt["state_dict"])
        return module.refiner, module.text_encoder, cfg
    raise ValueError(
        f"checkpoint {ckpt_path} missing hyper_parameters.cfg; "
        "pass a Lightning checkpoint saved by train_refiner.py"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/root_refiner.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/refiner_bench")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--split_file", type=str, default=None,
                         help="Eval split; defaults to data.val_split_file or the dataset default.")
    parser.add_argument(
        "--oracle_duration",
        action="store_true",
        default=False,
        help=(
            "Deprecated alias for --duration_mode groundtruth_duration: "
            "teacher-force GT num_tokens for trajectory metrics."
        ),
    )
    parser.add_argument(
        "--duration_mode",
        type=str,
        choices=(DURATION_PRED, DURATION_GROUNDTRUTH),
        default=None,
        help=(
            "Duration mode for RootRefiner trajectory evaluation. "
            "Defaults to the selected suite's mode, or pred_duration in "
            "legacy single-pass mode."
        ),
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=sorted(_ROOT_REFINER_SUITES),
        default=None,
        help="Optional layered eval suite. Omit to preserve legacy single-pass behavior.",
    )
    parser.add_argument(
        "--save_artifacts",
        action="store_true",
        default=False,
        help="Save per-sample RootRefiner route/root/plot artifacts under output_dir/samples.",
    )
    parser.add_argument(
        "--artifact_max_samples",
        type=int,
        default=20,
        help="Maximum per-run artifact samples when --save_artifacts is set; <=0 means all.",
    )
    parser.add_argument(
        "--full_route",
        action="store_true",
        default=False,
        help=(
            "Evaluate full-clip routes: force mode=full, anchor_frame=0, "
            "and num_tokens=max valid tokens for each selected clip."
        ),
    )
    parser.add_argument(
        "--raw_id",
        action="append",
        default=None,
        help=(
            "Restrict full-route eval to a raw dataset id, e.g. --raw_id 000021. "
            "May be repeated. Implies --full_route."
        ),
    )
    args = parser.parse_args(argv)

    from datasets.humanml3d_refiner import HumanML3DRefinerDataset as RefinerDataset
    from scripts.compute_5d_stats import load_clips_from_dir

    from train_refiner import resolve_cfg_interpolations

    from train_refiner import _load_cfg
    cfg = _load_cfg(args.config)
    # A-P0-1: resolve ${data.raw_data_dir} etc. (the model cfg comes from the
    # ckpt's saved hparams, which train_refiner already resolved).
    cfg = resolve_cfg_interpolations(cfg)

    model, text_encoder, ckpt_cfg = _load_model_from_ckpt(args.ckpt, args.device)
    validate_ckpt_eval_config_compatible(ckpt_cfg, cfg)

    data_cfg = cfg.get("data", {})
    split_file = args.split_file or data_cfg.get("val_split_file")
    clips = load_clips_from_dir(
        data_cfg["raw_data_dir"],
        dataset=data_cfg.get("dataset", "humanml3d"),
        split_file=split_file,
        feature_path=data_cfg.get("feature_path"),
        text_path=data_cfg.get("text_path"),
    )
    dataset = build_refiner_dataset_from_clips(
        cfg,
        clips,
        dataset_cls=RefinerDataset,
        seed=0,
    )
    task_specs = None
    if args.full_route or args.raw_id or args.suite == "full_route":
        task_specs = build_full_route_task_specs(
            dataset,
            max_samples=args.max_samples,
            raw_ids=args.raw_id,
        )
    run_duration_mode = args.duration_mode
    if (
        task_specs is not None
        and args.suite is None
        and run_duration_mode is None
        and not args.oracle_duration
    ):
        run_duration_mode = DURATION_GROUNDTRUTH
    cli_force_path_mode = _resolve_cli_force_path_mode(
        task_specs=task_specs,
        suite=args.suite,
    )
    run_id = Path(args.ckpt).stem.replace("=", "_")
    suite_tag = args.suite or (
        "full_route" if args.full_route or args.raw_id or task_specs is not None else "single"
    )
    legacy_sample_dir = Path(args.output_dir) / "samples"
    standard_sample_dir = None
    if args.save_artifacts:
        standard_sample_dir = standard_eval_artifact_dirs(
            args.output_dir,
            evaluator="RootRefiner",
            probe_tag=suite_tag,
            run_id=run_id,
            artifact_kinds=("samples",),
        )["samples"]

    if args.suite:
        result = run_suite_benchmark(
            model,
            dataset,
            text_encoder,
            suite=args.suite,
            device=args.device,
            max_samples=args.max_samples,
            duration_mode=args.duration_mode,
            oracle_duration=args.oracle_duration,
            artifact_dir=standard_sample_dir,
            artifact_max_samples=args.artifact_max_samples,
            task_specs=task_specs,
        )
    else:
        result = run_benchmark(model, dataset, text_encoder, device=args.device,
                                max_samples=args.max_samples,
                                oracle_duration=args.oracle_duration,
                                duration_mode=run_duration_mode,
                                force_path_mode=cli_force_path_mode,
                                task_specs=task_specs,
                                artifact_dir=(
                                    standard_sample_dir
                                    if args.save_artifacts else None
                                ),
                                artifact_max_samples=args.artifact_max_samples)
    if (
        args.save_artifacts
        and standard_sample_dir is not None
        and standard_sample_dir.exists()
        and standard_sample_dir != legacy_sample_dir
    ):
        shutil.copytree(standard_sample_dir, legacy_sample_dir, dirs_exist_ok=True)
    write_report(result, args.output_dir, suite=suite_tag, run_id=run_id)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
