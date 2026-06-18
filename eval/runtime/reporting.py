"""Reporting helpers for runtime stream benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from eval.common.artifacts import ensure_dir, standard_eval_artifact_dirs
from eval.common.json import json_sanitize, write_json_strict


def _json_sanitize(value):
    from eval.common.json import json_sanitize

    return json_sanitize(value)


def write_stream_summary(path, summary: dict) -> None:
    write_json_strict(path, summary)


def _csv_safe_record(record: dict) -> dict:
    """Return a CSV row with nested values encoded as strict JSON strings."""
    row = {}
    for key, value in record.items():
        clean = json_sanitize(value)
        if isinstance(clean, (dict, list)):
            row[key] = json.dumps(clean, separators=(",", ":"), allow_nan=False)
        else:
            row[key] = clean
    return row


_AGGREGATE_METRIC_KEYS = (
    "ADE",
    "FDE",
    "ADE_vs_metric_target",
    "FDE_vs_metric_target",
    "turn_post_switch_ADE",
    "turn_post_switch_FDE",
    "ADE_vs_original_gt",
    "FDE_vs_original_gt",
    "ADE_vs_root_condition",
    "FDE_vs_root_condition",
    "refiner_ADE",
    "refiner_FDE",
    "path_arc",
    "path_chamfer",
    "heading_path_error_deg",
    "lateral_velocity_ratio",
)

_RUNTIME_RECORD_FIELDS = (
    "suite",
    "mode",
    "sample_id",
    "base_case_name",
    "case_name",
    "condition_variant",
    "ADE",
    "FDE",
    "ADE_vs_metric_target",
    "FDE_vs_metric_target",
    "turn_post_switch_ADE",
    "turn_post_switch_FDE",
    "path_arc",
    "path_chamfer",
    "chamfer_type",
    "lateral_velocity_ratio",
    "heading_path_error_deg",
    "target_source",
    "ADE_vs_original_gt",
    "FDE_vs_original_gt",
    "ADE_vs_root_condition",
    "FDE_vs_root_condition",
    "refiner_ADE",
    "refiner_FDE",
    "root_condition_num_tokens",
    "root_condition_num_frames",
    "traj_condition_path",
    "condition_source",
    "root_refiner_enabled",
    "rootplan_replan_count",
    "rootplan_replan_commits",
    "rootplan_replan_sources",
    "turn_edit_commit",
    "turn_delay_tokens",
    "turn_blend_tokens",
    "turn_requested_effective_commit",
    "turn_effective_commit",
    "turn_activation_commit",
    "turn_post_switch_start_frame",
    "turn_post_switch_num_frames",
    "turn_target_source",
)



def _is_finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _aggregate_record_group(records: list[dict]) -> dict:
    out = {"num_records": int(len(records))}
    for key in _AGGREGATE_METRIC_KEYS:
        vals = [float(rec[key]) for rec in records if _is_finite_number(rec.get(key))]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std())
        out[f"{key}_count"] = int(arr.size)
    return out


def aggregate_runtime_records(records: list[dict]) -> dict:
    """Aggregate runtime benchmark records for checkpoint selection."""
    summary = _aggregate_record_group(records)
    by_suite: dict[str, list[dict]] = {}
    by_mode: dict[str, list[dict]] = {}
    by_suite_mode: dict[str, list[dict]] = {}
    by_condition_variant: dict[str, list[dict]] = {}
    by_suite_variant: dict[str, list[dict]] = {}
    for rec in records:
        suite = str(rec.get("suite", "unknown"))
        mode = str(rec.get("mode", "unknown"))
        variant = str(rec.get("condition_variant", "default"))
        by_suite.setdefault(suite, []).append(rec)
        by_mode.setdefault(mode, []).append(rec)
        by_suite_mode.setdefault(f"{suite}/{mode}", []).append(rec)
        by_condition_variant.setdefault(variant, []).append(rec)
        by_suite_variant.setdefault(f"{suite}/{variant}", []).append(rec)
    summary["by_suite"] = {
        key: _aggregate_record_group(vals) for key, vals in sorted(by_suite.items())
    }
    summary["by_mode"] = {
        key: _aggregate_record_group(vals) for key, vals in sorted(by_mode.items())
    }
    summary["by_suite_mode"] = {
        key: _aggregate_record_group(vals)
        for key, vals in sorted(by_suite_mode.items())
    }
    summary["by_condition_variant"] = {
        key: _aggregate_record_group(vals)
        for key, vals in sorted(by_condition_variant.items())
    }
    summary["by_suite_variant"] = {
        key: _aggregate_record_group(vals)
        for key, vals in sorted(by_suite_variant.items())
    }
    return summary


def _write_runtime_records_csv(path, records: list[dict]) -> None:
    if not records:
        return
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", newline="") as fc:
        writer = csv.DictWriter(
            fc,
            fieldnames=list(_RUNTIME_RECORD_FIELDS),
            extrasaction="ignore",
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(_csv_safe_record(rec))


def write_runtime_report(
    *,
    output_dir,
    run_id: str,
    suite_tag: str,
    payload: dict,
    records: list[dict],
    artifact_kinds=("metrics",),
) -> dict:
    """Write legacy runtime summary plus run_eval-style metric artifacts."""
    legacy_root = ensure_dir(Path(output_dir) / str(run_id))
    write_stream_summary(legacy_root / "summary.json", payload)
    _write_runtime_records_csv(legacy_root / "summary.csv", records)

    dirs = standard_eval_artifact_dirs(
        output_dir,
        evaluator="Runtime",
        probe_tag=str(suite_tag),
        run_id=str(run_id),
        artifact_kinds=artifact_kinds,
    )
    write_stream_summary(dirs["metrics"] / "summary.json", payload)
    _write_runtime_records_csv(dirs["metrics"] / "records.csv", records)
    return {"legacy_root": legacy_root, **dirs}


def prepare_runtime_media_dirs(
    *,
    output_dir: str | Path,
    run_id: str,
    suite_tag: str,
    render_video: bool,
    save_plots: bool,
    enabled: bool = True,
) -> dict[str, str | None]:
    """Prepare legacy runtime media dirs for non-debug runtime runs.

    The runtime debug matrix writes into ``runtime/<ckpt>/<run_id>`` via
    ``RuntimeArtifactLayout``. Creating the older ``<output>/<run_id>`` folder
    for debug runs leaves an empty timestamp directory, so callers can disable
    this helper there.
    """

    if not enabled:
        return {
            "out_root": None,
            "video_dir": None,
            "plot_dir": None,
            "standard_video_dir": None,
            "standard_plot_dir": None,
        }

    out_root = ensure_dir(Path(output_dir) / str(run_id))
    standard_media_dirs = standard_eval_artifact_dirs(
        output_dir,
        evaluator="Runtime",
        probe_tag=suite_tag,
        run_id=run_id,
        artifact_kinds=("plot", "video"),
        create=False,
    )
    video_dir = ensure_dir(out_root / "videos") if render_video else None
    plot_dir = ensure_dir(out_root / "plots") if save_plots else None
    standard_video_dir = (
        ensure_dir(standard_media_dirs["video"]) if render_video else None
    )
    standard_plot_dir = (
        ensure_dir(standard_media_dirs["plot"]) if save_plots else None
    )
    return {
        "out_root": str(out_root),
        "video_dir": str(video_dir) if video_dir is not None else None,
        "plot_dir": str(plot_dir) if plot_dir is not None else None,
        "standard_video_dir": (
            str(standard_video_dir) if standard_video_dir is not None else None
        ),
        "standard_plot_dir": (
            str(standard_plot_dir) if standard_plot_dir is not None else None
        ),
    }



__all__ = [
    "_csv_safe_record",
    "aggregate_runtime_records",
    "prepare_runtime_media_dirs",
    "write_runtime_report",
    "write_stream_summary",
]
