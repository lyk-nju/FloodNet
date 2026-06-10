"""Report helpers for LDF stream-eval checkpoint comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from FloodNet.eval.common.json import write_json_strict
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from eval.common.json import write_json_strict


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "stream_gt/root_ADE": ("stream_gt/root_ADE", "traj/ADE_mean"),
    "stream_gt/root_FDE": ("stream_gt/root_FDE", "traj/FDE_mean"),
    "stream_gt/path_arc_ADE": ("stream_gt/path_arc_ADE", "path/arc_ADE_mean"),
    "stream_gt/jitter": ("stream_gt/jitter", "traj/jitter_mean"),
    "stream_gt/yaw_error": (
        "stream_gt/yaw_error",
        "stream_gt/yaw_error_mean",
    ),
    "stream_gt/control_L2": (
        "stream_gt/control_L2",
        "control/Control_L2_dist_mean",
    ),
    "stream_gt/foot_skating": (
        "stream_gt/foot_skating",
        "control/Skating_Ratio_mean",
    ),
    "stream_gt/chunk_boundary_root_jump": (
        "stream_gt/chunk_boundary_root_jump",
        "stream_boundary/root_jump_mean",
    ),
    "stream_vs_offline/root_ADE": (
        "stream_vs_offline/root_ADE",
        "stream_vs_offline/root_ade_mean",
    ),
    "stream_vs_offline/feature_L2": (
        "stream_vs_offline/feature_L2",
        "stream_vs_offline/feature_l2_mean",
    ),
    "stream_no_traj/root_ADE": (
        "stream_no_traj/root_ADE",
        "stream_no_traj/root_ADE_mean",
    ),
    "stream_no_traj/root_FDE": (
        "stream_no_traj/root_FDE",
        "stream_no_traj/root_FDE_mean",
    ),
    "control_gain/root_ADE_delta": (
        "control_gain/root_ADE_delta",
        "control_gain/root_ADE_delta_mean",
        "traj/ctrl_delta_ADE_mean",
    ),
    "control_gain/root_FDE_delta": (
        "control_gain/root_FDE_delta",
        "control_gain/root_FDE_delta_mean",
    ),
}


def _load_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"summary payload must be a dict: {path}")
    return payload


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("summary payload field 'summary' must be a dict")
    return summary


def _metric_value(summary: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        if key not in summary:
            continue
        value = summary[key]
        if value is None:
            return None
        try:
            return round(float(value), 10)
        except (TypeError, ValueError):
            return None
    return None


def extract_stream_selection_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    summary = _summary(payload)
    return {
        metric: _metric_value(summary, aliases)
        for metric, aliases in METRIC_ALIASES.items()
    }


def _tag(payload: dict[str, Any], path: str | Path) -> str:
    return str(payload.get("probe_tag") or Path(path).stem)


def _side(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "tag": _tag(payload, path),
        "ckpt": payload.get("ckpt"),
        "num_samples": payload.get("num_samples"),
        "num_runs": payload.get("num_runs"),
        "metrics": extract_stream_selection_metrics(payload),
    }


def _delta(candidate_value: float | None, baseline_value: float | None) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    return round(float(candidate_value) - float(baseline_value), 10)


def _thresholds(
    *,
    max_root_ade_regression: float | None = None,
    max_root_fde_regression: float | None = None,
    max_path_arc_regression: float | None = None,
    max_yaw_error_regression: float | None = None,
    max_jitter_regression: float | None = None,
    max_foot_skating_regression: float | None = None,
    max_root_jump_regression: float | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if max_root_ade_regression is not None:
        out["stream_gt/root_ADE"] = float(max_root_ade_regression)
    if max_root_fde_regression is not None:
        out["stream_gt/root_FDE"] = float(max_root_fde_regression)
    if max_path_arc_regression is not None:
        out["stream_gt/path_arc_ADE"] = float(max_path_arc_regression)
    if max_yaw_error_regression is not None:
        out["stream_gt/yaw_error"] = float(max_yaw_error_regression)
    if max_jitter_regression is not None:
        out["stream_gt/jitter"] = float(max_jitter_regression)
    if max_foot_skating_regression is not None:
        out["stream_gt/foot_skating"] = float(max_foot_skating_regression)
    if max_root_jump_regression is not None:
        out["stream_gt/chunk_boundary_root_jump"] = float(max_root_jump_regression)
    return out


def build_checkpoint_comparison(
    baseline_summary: str | Path,
    candidate_summary: str | Path,
    *,
    max_root_ade_regression: float | None = None,
    max_root_fde_regression: float | None = None,
    max_path_arc_regression: float | None = None,
    max_yaw_error_regression: float | None = None,
    max_jitter_regression: float | None = None,
    max_foot_skating_regression: float | None = None,
    max_root_jump_regression: float | None = None,
) -> dict[str, Any]:
    baseline_payload = _load_payload(baseline_summary)
    candidate_payload = _load_payload(candidate_summary)
    baseline = _side(baseline_summary, baseline_payload)
    candidate = _side(candidate_summary, candidate_payload)

    deltas = {
        metric: _delta(candidate["metrics"].get(metric), baseline["metrics"].get(metric))
        for metric in METRIC_ALIASES
    }
    thresholds = _thresholds(
        max_root_ade_regression=max_root_ade_regression,
        max_root_fde_regression=max_root_fde_regression,
        max_path_arc_regression=max_path_arc_regression,
        max_yaw_error_regression=max_yaw_error_regression,
        max_jitter_regression=max_jitter_regression,
        max_foot_skating_regression=max_foot_skating_regression,
        max_root_jump_regression=max_root_jump_regression,
    )
    failures = []
    for metric, max_allowed in thresholds.items():
        delta = deltas.get(metric)
        if delta is None:
            failures.append(
                {
                    "metric": metric,
                    "delta": None,
                    "max_allowed_regression": max_allowed,
                }
            )
        elif delta > max_allowed:
            failures.append(
                {
                    "metric": metric,
                    "delta": delta,
                    "max_allowed_regression": max_allowed,
                }
            )

    return {
        "kind": "ldf_stream_checkpoint_comparison",
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
        "thresholds": thresholds,
        "decision": {
            "passed": not failures,
            "failures": failures,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two LDF stream-eval summary.json files."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-root-ade-regression", type=float, default=None)
    parser.add_argument("--max-root-fde-regression", type=float, default=None)
    parser.add_argument("--max-path-arc-regression", type=float, default=None)
    parser.add_argument("--max-yaw-error-regression", type=float, default=None)
    parser.add_argument("--max-jitter-regression", type=float, default=None)
    parser.add_argument("--max-foot-skating-regression", type=float, default=None)
    parser.add_argument("--max-root-jump-regression", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_checkpoint_comparison(
        args.baseline,
        args.candidate,
        max_root_ade_regression=args.max_root_ade_regression,
        max_root_fde_regression=args.max_root_fde_regression,
        max_path_arc_regression=args.max_path_arc_regression,
        max_yaw_error_regression=args.max_yaw_error_regression,
        max_jitter_regression=args.max_jitter_regression,
        max_foot_skating_regression=args.max_foot_skating_regression,
        max_root_jump_regression=args.max_root_jump_regression,
    )
    write_json_strict(args.out, report)
    return 0 if report["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
