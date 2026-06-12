"""Unified stream benchmark runner (Task 002).

Usage::

    python eval/stream_benchmark.py \\
        --config configs/stream.yaml \\
        --ckpt outputs/step_460000.ckpt \\
        --vae_ckpt outputs/vae_1d_z4_step=300000.ckpt \\
        --raw_data_dir /path/to/raw_data \\
        --preset smoke \\
        --render_video
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

DEFAULT_RUNTIME_OUTPUT_DIR = os.path.join(_project_root, "eval", "output_eval")

import numpy as np
import torch
import random
try:
    from lightning import seed_everything
except ImportError:
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
from torch_ema import ExponentialMovingAverage
from omegaconf import OmegaConf

from eval.common.json import json_sanitize, write_json_strict
from eval.common.artifacts import ensure_dir, standard_eval_artifact_dirs
from eval.common.visualization import (
    plot_xz_trajectories,
    plot_yaw_series,
    yaw_from_root_path,
)
from utils.initialize import check_state_dict, instantiate, load_config
from utils.motion_process import (
    convert_motion_to_joints,
    extract_root_trajectory_263,
)
from utils.token_frame import (
    token_start_frame,
)
from utils.render_skeleton import get_humanml3d_chains, render_simple_skeleton_video
from eval.runtime.cases import get_cases
from eval.runtime.artifacts import (
    RuntimeArtifactLayout,
    infer_ckpt_tag,
    read_experiment_root_plan,
    write_experiment_metrics,
    write_experiment_root_plan,
    write_runtime_debug_report,
    write_root_diagnostic_artifacts,
    write_root_diagnostics_summary,
)
from eval.runtime.experiments import (
    DEFAULT_ROOT_SOURCES,
    build_default_runtime_experiments,
    filter_runtime_experiments,
    parse_csv_ints,
    parse_csv_strings,
    runtime_debug_root_refiner_history_policy,
    runtime_debug_mode_for_source,
    runtime_debug_turn_plan_policy,
    summarize_numeric_records,
)
from eval.runtime.metrics import (
    build_plan_metrics,
    compute_ade,
    compute_fde,
    compute_plan_targets,
    compute_root_condition_diagnostics,
    estimate_body_yaw,
)
from eval.runtime.root_sources import (
    normalize_root_source,
    root_source_metadata,
    runtime_debug_condition_source,
)
from eval.runtime.runners import (
    build_turn_metric_target as _build_turn_metric_target,
    root_plan_events_to_diagnostic_arrays,
    run_babel_case as _run_babel,
    run_real_case as _run_real,
    run_step_case as _run_step,
    run_turn_case as _run_turn,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── 7D RootPlan streaming helpers ──────────────────────────────────────

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
    "turn_post_switch_ADE",
    "turn_post_switch_FDE",
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


@dataclass(frozen=True)
class ConditionVariant:
    name: str
    condition_path: str
    use_root_refiner: bool = False
    force_no_traj: bool = False


_CONDITION_VARIANT_ALIASES = {
    "gt_7d": "gt_7d_ldf",
    "route_7d": "gt_7d_ldf",
    "gt_7d_ldf": "gt_7d_ldf",
    "rootrefiner_7d": "rootrefiner_7d_ldf",
    "root_refiner_7d": "rootrefiner_7d_ldf",
    "rootrefiner_7d_ldf": "rootrefiner_7d_ldf",
    "root_refiner_7d_ldf": "rootrefiner_7d_ldf",
    "no_traj": "no_traj_ldf",
    "no_traj_ldf": "no_traj_ldf",
    "legacy_xyz": "legacy_xyz_ldf",
    "legacy_xyz_ldf": "legacy_xyz_ldf",
}


def _condition_variant_from_name(name: str) -> ConditionVariant:
    key = _CONDITION_VARIANT_ALIASES.get(str(name).strip())
    if key is None:
        valid = ", ".join(sorted(set(_CONDITION_VARIANT_ALIASES)))
        raise ValueError(f"unknown condition variant {name!r}; expected one of: {valid}")
    if key == "gt_7d_ldf":
        return ConditionVariant(name=key, condition_path="rootplan_7d")
    if key == "rootrefiner_7d_ldf":
        return ConditionVariant(
            name=key,
            condition_path="rootplan_7d",
            use_root_refiner=True,
        )
    if key == "no_traj_ldf":
        return ConditionVariant(
            name=key,
            condition_path="rootplan_7d",
            force_no_traj=True,
        )
    if key == "legacy_xyz_ldf":
        return ConditionVariant(name=key, condition_path="legacy_xyz")
    raise AssertionError(key)


def parse_condition_variants(
    spec: str | None,
    *,
    include_root_refiner: bool = True,
) -> list[ConditionVariant]:
    """Parse runtime benchmark condition variants."""
    raw = "auto" if spec is None else str(spec).strip()
    if raw in {"", "auto"}:
        names = ["gt_7d_ldf", "no_traj_ldf"]
        if include_root_refiner:
            names.insert(1, "rootrefiner_7d_ldf")
    else:
        names = [item.strip() for item in raw.split(",") if item.strip()]
    variants = [_condition_variant_from_name(name) for name in names]
    seen = set()
    out = []
    for variant in variants:
        if variant.name in seen:
            continue
        seen.add(variant.name)
        out.append(variant)
    return out


def _variant_case_name(case_name: str, variant: ConditionVariant) -> str:
    return f"{case_name}__{variant.name}"


def _is_legacy_no_traj_case(case) -> bool:
    return str(getattr(case, "mode", "")).endswith("_no_traj")


def _visual_target_root_from_plan(
    *,
    original_gt_root: np.ndarray,
    plan_times: np.ndarray | None,
    plan_points_xyz: np.ndarray | None,
    target_frames: int,
    motion_fps: float,
) -> np.ndarray:
    """Return the same time-sampled target trajectory used by plan metrics."""
    original = np.asarray(original_gt_root, dtype=np.float32)
    target_frames = int(target_frames)
    if target_frames <= 0:
        return original[:0].copy()
    if plan_times is None or plan_points_xyz is None:
        return original[:target_frames].copy()
    target_time, _target_arc = compute_plan_targets(
        np.asarray(plan_times, dtype=np.float32),
        np.asarray(plan_points_xyz, dtype=np.float32),
        target_frames,
        float(motion_fps),
    )
    return target_time.astype(np.float32, copy=False)


def _turn_activation_commit_from_events(
    replan_events: list[dict] | None,
    *,
    fallback_commit: int,
) -> int:
    """Return the commit where the composed turn RootPlan actually became active."""
    fallback = int(fallback_commit)
    for event in reversed(replan_events or []):
        if str(event.get("source", "")) != "bench_composed":
            continue
        try:
            return int(event.get("commit", fallback))
        except (TypeError, ValueError):
            return fallback
    return fallback


def _add_turn_post_switch_metrics(
    rec: dict,
    *,
    pred_root_xyz: np.ndarray,
    plan_times: np.ndarray,
    plan_points_xyz: np.ndarray,
    target_frames: int,
    motion_fps: float,
    activation_commit: int,
) -> None:
    """Add ADE/FDE measured only after the turn condition actually switches."""
    target_time, _target_arc = compute_plan_targets(
        np.asarray(plan_times, dtype=np.float32),
        np.asarray(plan_points_xyz, dtype=np.float32),
        int(target_frames),
        float(motion_fps),
    )
    pred = np.asarray(pred_root_xyz, dtype=np.float32)
    n = min(len(pred), len(target_time))
    start_frame = int(token_start_frame(int(activation_commit)))
    start_frame = max(0, start_frame)
    num_frames = max(0, n - start_frame)
    rec["turn_post_switch_start_frame"] = int(start_frame)
    rec["turn_post_switch_num_frames"] = int(num_frames)
    if num_frames <= 0:
        rec["turn_post_switch_ADE"] = float("nan")
        rec["turn_post_switch_FDE"] = float("nan")
        return
    rec["turn_post_switch_ADE"] = compute_ade(
        pred[start_frame:n],
        target_time[start_frame:n],
    )
    rec["turn_post_switch_FDE"] = compute_fde(
        pred[start_frame:n],
        target_time[start_frame:n],
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


def _parse_runtime_debug_devices(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _split_runtime_debug_indices(
    num_specs: int,
    devices: tuple[str, ...],
) -> list[list[int]]:
    if int(num_specs) <= 0 or not devices:
        return []
    chunks = [[] for _ in devices]
    for idx in range(int(num_specs)):
        chunks[idx % len(devices)].append(idx)
    return [chunk for chunk in chunks if chunk]


def _strip_cli_option(argv: list[str], option: str) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    prefix = f"{option}="
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            skip_next = True
            continue
        if arg.startswith(prefix):
            continue
        stripped.append(arg)
    return stripped


def _build_runtime_debug_specs_from_args(args) -> list:
    root_sources = tuple(
        normalize_root_source(source)
        for source in parse_csv_strings(
            args.runtime_debug_root_sources,
            default=DEFAULT_ROOT_SOURCES,
        )
    )
    families = parse_csv_strings(
        args.runtime_debug_families,
        default=("web_stream", "rotation", "turn"),
    )
    specs = build_default_runtime_experiments(
        sample_id=str(args.runtime_debug_sample_id),
        root_sources=root_sources,
        rotation_degrees=parse_csv_ints(
            args.runtime_debug_rotation_degrees,
            default=(10, 20, 30, 40, 50, 60, 70, 80, 90),
        ),
        turn_delay_tokens=parse_csv_ints(
            args.runtime_debug_turn_delay_tokens,
            default=(5, 10, 20),
        ),
        turn_blend_tokens=parse_csv_ints(
            args.runtime_debug_turn_blend_tokens,
            default=(2, 4, 8),
        ),
    )
    return filter_runtime_experiments(specs, families=families)


def _select_runtime_debug_specs(specs: list, raw_indices: str | None) -> list:
    if not raw_indices:
        return list(specs)
    selected = []
    for idx in parse_csv_ints(raw_indices, default=tuple(range(len(specs)))):
        if int(idx) < 0 or int(idx) >= len(specs):
            raise ValueError(f"runtime debug spec index out of range: {idx}")
        selected.append(specs[int(idx)])
    return selected


def _collect_runtime_debug_outputs(
    layout: RuntimeArtifactLayout,
    specs: list,
) -> tuple[list[dict], dict[tuple[str, tuple[str, ...]], dict[str, tuple[np.ndarray, int]]]]:
    records: list[dict] = []
    root_plan_cache: dict[tuple[str, tuple[str, ...]], dict[str, tuple[np.ndarray, int]]] = {}
    for spec in specs:
        root_source = normalize_root_source(spec.root_source)
        metrics_path = (
            layout.experiment_dir(root_source, spec.family, *spec.parts) / "metrics.json"
        )
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing runtime debug metrics: {metrics_path}")
        with metrics_path.open() as f:
            records.append(json.load(f))
        if root_source in {"gtroot", "rootrefiner"}:
            loaded = read_experiment_root_plan(
                layout,
                root_source=root_source,
                family=spec.family,
                parts=spec.parts,
            )
            if loaded is not None:
                root_plan_cache.setdefault((spec.family, spec.parts), {})[
                    root_source
                ] = loaded
    return records, root_plan_cache


def _write_runtime_debug_final_report(
    *,
    debug_layout: RuntimeArtifactLayout,
    args,
    run_id: str,
    specs: list,
    all_recs: list[dict],
    root_plan_cache: dict[tuple[str, tuple[str, ...]], dict[str, tuple[np.ndarray, int]]],
) -> tuple[dict[str, Path], dict[str, Path]]:
    root_diag_records = []
    for (family, parts), by_source in sorted(root_plan_cache.items()):
        if "gtroot" not in by_source or "rootrefiner" not in by_source:
            continue
        gt_root_7d, gt_num_tokens = by_source["gtroot"]
        pred_root_7d, pred_num_tokens = by_source["rootrefiner"]
        diag = compute_root_condition_diagnostics(
            gt_root_7d,
            pred_root_7d,
            gt_num_tokens=gt_num_tokens,
            pred_num_tokens=pred_num_tokens,
        )
        diag.update(
            {
                "family": family,
                "parts": list(parts),
                "experiment": "/".join(parts) if parts else family,
            }
        )
        root_diag_records.append(diag)
        write_root_diagnostic_artifacts(
            debug_layout,
            family=family,
            parts=parts,
            metrics=diag,
            gt_root_7d=gt_root_7d,
            pred_root_7d=pred_root_7d,
            gt_num_tokens=gt_num_tokens,
            pred_num_tokens=pred_num_tokens,
        )

    aggregate = aggregate_runtime_records(all_recs)
    root_diag_summary = summarize_numeric_records(root_diag_records)
    root_diag_paths = write_root_diagnostics_summary(
        debug_layout,
        summary=root_diag_summary,
        records=root_diag_records,
    )
    devices = _parse_runtime_debug_devices(getattr(args, "runtime_debug_devices", ""))
    summary = {
        "run_id": run_id,
        "config": args.config,
        "ckpt": args.ckpt,
        "vae_ckpt": args.vae_ckpt,
        "root_refiner_config": args.root_refiner_config,
        "root_refiner_ckpt": args.root_refiner_ckpt,
        "runtime_debug_matrix": True,
        "runtime_debug_sample_id": args.runtime_debug_sample_id,
        "runtime_debug_devices": list(devices),
        "aggregate": aggregate,
        "root_diagnostics": root_diag_summary,
        "summary": aggregate,
        "records": all_recs,
    }
    debug_report_paths = write_runtime_debug_report(
        debug_layout,
        manifest={
            "run_id": run_id,
            "config": args.config,
            "ckpt": args.ckpt,
            "vae_ckpt": args.vae_ckpt,
            "root_refiner_config": args.root_refiner_config,
            "root_refiner_ckpt": args.root_refiner_ckpt,
            "runtime_debug_matrix": True,
            "runtime_debug_sample_id": args.runtime_debug_sample_id,
            "runtime_debug_devices": list(devices),
            "root_sources": sorted({spec.root_source for spec in specs}),
            "num_experiments": len(specs),
        },
        summary=summary,
        records=all_recs,
        source_payloads={
            root_source: root_source_metadata(root_source)
            for root_source in sorted({spec.root_source for spec in specs})
        },
    )
    return debug_report_paths, root_diag_paths


def _run_runtime_debug_distributed(
    *,
    args,
    run_id: str,
    debug_layout: RuntimeArtifactLayout,
    specs: list,
) -> None:
    devices = _parse_runtime_debug_devices(args.runtime_debug_devices)
    chunks = _split_runtime_debug_indices(len(specs), devices)
    if not chunks:
        raise ValueError("--runtime_debug_devices was provided but no work was assigned")

    base_argv = list(sys.argv[1:])
    for opt in (
        "--runtime_debug_devices",
        "--runtime_debug_run_id",
        "--runtime_debug_spec_indices",
        "--runtime_debug_worker_id",
    ):
        base_argv = _strip_cli_option(base_argv, opt)

    workers = []
    for worker_id, indices in enumerate(chunks):
        device = devices[worker_id]
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_argv,
            "--runtime_debug_run_id",
            str(run_id),
            "--runtime_debug_spec_indices",
            ",".join(str(idx) for idx in indices),
            "--runtime_debug_worker_id",
            str(worker_id),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        print(
            f"[runtime-debug] worker {worker_id} gpu={device} "
            f"specs={','.join(str(idx) for idx in indices)}",
            flush=True,
        )
        workers.append((worker_id, subprocess.Popen(cmd, env=env)))

    failed = []
    for worker_id, proc in workers:
        ret = proc.wait()
        if ret != 0:
            failed.append((worker_id, ret))
    if failed:
        raise SystemExit(f"runtime debug worker failure(s): {failed}")

    all_recs, root_plan_cache = _collect_runtime_debug_outputs(debug_layout, specs)
    debug_report_paths, root_diag_paths = _write_runtime_debug_final_report(
        debug_layout=debug_layout,
        args=args,
        run_id=run_id,
        specs=specs,
        all_recs=all_recs,
        root_plan_cache=root_plan_cache,
    )
    print(f"\nRuntime debug: {debug_report_paths['summary']}")
    print(f"Runtime debug CSV: {debug_report_paths['records']}")
    print(f"Root diagnostics: {root_diag_paths['summary']}")


def resolve_traj_condition_source(
    condition_path: str,
    root_refiner_runtime=None,
    *,
    no_traj: bool = False,
    gt_motion_7d: bool = False,
) -> str:
    if no_traj:
        return "none"
    if condition_path == "rootplan_7d":
        if gt_motion_7d:
            return "gt_motion_7d"
        return "rootrefiner_7d" if root_refiner_runtime is not None else "route_derived_7d"
    return str(condition_path)


def _reset_eval_runtime_trace(model) -> None:
    """Backward-compatible cleanup for older callers that stored trace on model."""
    if hasattr(model, "_stream_eval_replan_events"):
        delattr(model, "_stream_eval_replan_events")


# ── model loading ──────────────────────────────────────────────────────

def _load_vae(cfg, device):
    vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False,
                      **cfg.test_vae.params)
    ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in ckpt:
        vae.load_state_dict(ckpt["state_dict"], strict=True)
        ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(vae.parameters())
    else:
        vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _load_model(cfg, ckpt_path, device):
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False,
                        **cfg.model.params)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(ckpt["state_dict"].keys())
    cn_missing = not any(k.startswith("controlnet.") for k in ckpt_keys)
    strict = not cn_missing
    result = model.load_state_dict(ckpt["state_dict"], strict=strict)
    if not strict and result.missing_keys:
        if any("controlnet." in k for k in result.missing_keys):
            model.controlnet.init_from_backbone(model.model)
    if "ema_state" in ckpt:
        n_shadow = len(ckpt["ema_state"]["shadow_params"])
        ema_params = [p for p in model.parameters() if p.requires_grad]
        if len(ema_params) != n_shadow:
            ema_params = list(model.parameters())
        ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(ema_params)
    model.to(device).eval()
    return model


# ── sample loading ─────────────────────────────────────────────────────

def _load_humanml3d_sample(raw_data_dir, sample_id):
    data_dir = os.path.join(raw_data_dir, "HumanML3D")
    feat = np.load(os.path.join(data_dir, "new_joint_vecs", f"{sample_id}.npy")).astype(np.float32)
    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            text_data.append({
                "caption": parts[0],
                "tokens": parts[1].split(" ") if len(parts) > 1 else [],
                "f_tag": float(parts[2]) if len(parts) > 2 else 0.0,
                "to_tag": float(parts[3]) if len(parts) > 3 else 0.0,
            })
    traj_xyz = extract_root_trajectory_263(feat)
    token = np.load(os.path.join(
        data_dir, "TOKENS_20251030_085836_vae_wan_z4", f"{sample_id}.npy")).astype(np.float32)
    return {
        "name": sample_id, "dataset": "HumanML3D",
        "feature": torch.from_numpy(feat).float(), "feature_length": len(feat),
        "token": torch.from_numpy(token).float(), "token_length": len(token),
        "text": text_data[0]["caption"],
        "traj": torch.from_numpy(traj_xyz).float(), "traj_length": len(traj_xyz),
        "token_mask": torch.ones(len(token), dtype=torch.float32),
        "traj_mask": torch.ones(len(traj_xyz), dtype=torch.float32),
    }


def _load_babel_sample(raw_data_dir, sample_id):
    data_dir = os.path.join(raw_data_dir, "BABEL_streamed")
    feat = np.load(os.path.join(data_dir, "motions", f"{sample_id}.npy")).astype(np.float32)
    token = np.load(os.path.join(data_dir, "TOKENS_20251030_085836_vae_wan_z4",
                                 f"{sample_id}.npy")).astype(np.float32)
    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            ft = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
            tt = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0.0
            text_data.append({"caption": parts[0].strip(),
                              "f_tag": 0.0 if np.isnan(ft) else ft,
                              "to_tag": 0.0 if np.isnan(tt) else tt})
    return {"feature": feat, "token": token, "text_data": text_data, "name": sample_id}


def _merge_babel(raw_data_dir, sample_ids):
    parts = [_load_babel_sample(raw_data_dir, sid) for sid in sample_ids]
    feat = np.concatenate([p["feature"] for p in parts], axis=0)
    token = np.concatenate([p["token"] for p in parts], axis=0)
    tf, tt = len(feat), len(token)
    text_data, feat_ofs = [], 0
    feat_fps = 20.0
    for p in parts:
        for td in p["text_data"]:
            ft, ttag = td["f_tag"], td["to_tag"]
            if ft == 0.0 and ttag == 0.0:
                af, at = feat_ofs / feat_fps, (feat_ofs + len(p["feature"])) / feat_fps
            else:
                af, at = feat_ofs / feat_fps + ft, feat_ofs / feat_fps + ttag
            text_data.append({"caption": td["caption"], "f_tag": af, "to_tag": at})
        feat_ofs += len(p["feature"])
    texts, fte, cursor = [], [], 0
    for td in text_data:
        a_start = max(0, int(td["f_tag"] * feat_fps + 0.5))
        a_end = int(td["to_tag"] * feat_fps + 0.5) if td["to_tag"] > 0 else tf
        if a_end <= a_start:
            continue
        if a_start > cursor:
            texts.append(""); fte.append(min(a_start, tf)); cursor = a_start
        texts.append(td["caption"]); fte.append(min(a_end, tf)); cursor = a_end
    if cursor < tf:
        texts.append(""); fte.append(tf)
    if not texts:
        texts = [td["caption"] or "" for td in text_data] or [""]; fte = [tf]
    token_te = [max(0, min(tt, (ef - 1 + 3) // 4 + 1)) for ef in fte]
    traj = extract_root_trajectory_263(feat)
    return {
        "name": sample_ids[0].rsplit("_", 1)[0], "dataset": "BABEL_streamed",
        "feature": torch.from_numpy(feat).float(), "feature_length": tf,
        "token": torch.from_numpy(token).float(), "token_length": tt,
        "text": texts, "traj": torch.from_numpy(traj).float(), "traj_length": len(traj),
        "token_text_end": token_te, "feature_text_end": fte,
        "token_mask": torch.ones(tt, dtype=torch.float32),
        "traj_mask": torch.ones(len(traj), dtype=torch.float32),
    }


# ── case runners ───────────────────────────────────────────────────────

def _motion_video_overlay_kwargs(target_root):
    """Build trajectory overlay kwargs for the skeleton action video."""
    render_setting = {
        "cond_traj_show_full": True,
        "traj_mask_point_radius": 3,
        "cond_traj_point_radius": 4,
    }
    kwargs = {"render_setting": render_setting}
    if target_root is None:
        return kwargs
    target_root = np.asarray(target_root, dtype=np.float32)
    if target_root.ndim != 2 or target_root.shape[0] == 0 or target_root.shape[1] < 3:
        return kwargs
    kwargs["traj_xz"] = target_root[:, [0, 2]]
    kwargs["traj_mask"] = np.ones((target_root.shape[0],), dtype=np.float32)
    return kwargs


def _transform_joints_to_runtime_world(
    joint_positions,
    *,
    pred_root,
    pred_yaw_offset: float = 0.0,
) -> np.ndarray:
    """Put rendered joints in the same world frame as runtime metrics."""

    joints = np.asarray(joint_positions, dtype=np.float32).copy()
    pred = np.asarray(pred_root, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"joint_positions must be [T,J,3], got {joints.shape}")
    if pred.ndim != 2 or pred.shape[-1] < 3:
        raise ValueError(f"pred_root must be [T,>=3], got {pred.shape}")
    n = min(len(joints), len(pred))
    joints = joints[:n]
    pred = pred[:n]
    yaw = float(pred_yaw_offset)
    if abs(yaw) > 1e-8:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        x = joints[..., 0].copy()
        z = joints[..., 2].copy()
        joints[..., 0] = c * x + s * z
        joints[..., 2] = -s * x + c * z
    root_xz = joints[:, 0, [0, 2]]
    offset_xz = pred[:, [0, 2]] - root_xz
    offset_y = pred[:, 1] - joints[:, 0, 1]
    joints[:, :, 0] += offset_xz[:, 0, None]
    joints[:, :, 2] += offset_xz[:, 1, None]
    joints[:, :, 1] += offset_y[:, None]
    return joints.astype(np.float32)


def _render_runtime_case_video(
    *,
    motion_263,
    pred_root,
    target_root,
    save_path,
    pred_yaw_offset: float = 0.0,
):
    overlay = _motion_video_overlay_kwargs(target_root)
    render_setting = overlay.get("render_setting", {})
    joints = convert_motion_to_joints(np.asarray(motion_263), dim=263)
    joints_world = _transform_joints_to_runtime_world(
        joints,
        pred_root=pred_root,
        pred_yaw_offset=pred_yaw_offset,
    )
    traj_xz = overlay.get("traj_xz")
    if traj_xz is not None:
        traj_xz = np.asarray(traj_xz, dtype=np.float32)[: len(joints_world)]
    traj_mask = overlay.get("traj_mask")
    if traj_mask is not None:
        traj_mask = np.asarray(traj_mask, dtype=np.float32)[: len(joints_world)]
    render_simple_skeleton_video(
        data=joints_world,
        chains=get_humanml3d_chains(),
        out_path=str(save_path),
        fps=render_setting.get("fps", 20),
        traj_mask=traj_mask,
        traj_mask_point_radius=int(render_setting.get("traj_mask_point_radius", 3)),
        traj_xz=traj_xz,
        cond_traj_mask=traj_mask,
        cond_traj_point_radius=int(render_setting.get("cond_traj_point_radius", 4)),
        cond_traj_show_full=bool(render_setting.get("cond_traj_show_full", True)),
    )


def _render_traj_video(pred_root, target_root, out_path, title, *, split_tok=None):
    """Render animated XZ trajectory comparison video."""
    _n = min(len(pred_root), len(target_root))
    if _n <= 1:
        return
    import matplotlib
    matplotlib.use("Agg")
    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    _f2, _a2 = plt.subplots(figsize=(7, 7))
    _all_x = [target_root[:_n, 0], pred_root[:_n, 0]]
    _all_z = [target_root[:_n, 2], pred_root[:_n, 2]]
    _xl = (min(a.min() for a in _all_x) - 0.5, max(a.max() for a in _all_x) + 0.5)
    _zl = (min(a.min() for a in _all_z) - 0.5, max(a.max() for a in _all_z) + 0.5)
    _wr = FFMpegWriter(fps=20)
    _sf = max(1, _n // 150)
    with _wr.saving(_f2, out_path, dpi=100):
        for _f in range(1, _n + 1, _sf):
            _a2.clear()
            _a2.plot(target_root[:min(_f, _n), 0], target_root[:min(_f, _n), 2],
                     "g-", lw=1.5, alpha=0.7, label="target")
            _a2.plot(pred_root[:min(_f, _n), 0], pred_root[:min(_f, _n), 2],
                     "r-", lw=1.5, alpha=0.7, label="pred")
            _a2.plot(target_root[0, 0], target_root[0, 2], "go", ms=6)
            if split_tok is not None:
                _sb = max(1, 1 + 4 * (split_tok - 1))
                if 0 < _sb < _n:
                    _a2.axvline(x=target_root[min(_sb, _n - 1), 0],
                                color="gray", ls="--", alpha=0.5, label="split")
            _a2.set_xlim(_xl); _a2.set_ylim(_zl)
            _a2.set_aspect("equal")
            _a2.legend(loc="upper right")
            _a2.set_title(f"{title}  f{min(_f,_n)}/{_n}")
            _wr.grab_frame()
    plt.close(_f2)


def _write_runtime_case_visuals(
    output_dir,
    *,
    case_name: str,
    pred_root,
    target_root,
    motion_263=None,
    split_tok: int | None = None,
    pred_yaw_offset: float = 0.0,
) -> None:
    out = ensure_dir(output_dir)
    split_frame = None if split_tok is None else int(token_start_frame(int(split_tok)))
    boundary_frames = [] if split_frame is None else [split_frame]
    plot_xz_trajectories(
        out / f"{case_name}_plot_world_xz.png",
        {
            "target": target_root,
            "pred": pred_root,
        },
        title=str(case_name),
        boundary_frames=boundary_frames,
    )
    if motion_263 is not None:
        try:
            pred_yaw = estimate_body_yaw(np.asarray(motion_263)) + float(pred_yaw_offset)
        except Exception:
            pred_yaw = yaw_from_root_path(pred_root)
    else:
        pred_yaw = yaw_from_root_path(pred_root)
    plot_yaw_series(
        out / f"{case_name}_plot_yaw.png",
        {
            "target_yaw": yaw_from_root_path(target_root),
            "pred_yaw": pred_yaw,
        },
        title=str(case_name),
        boundary_frames=boundary_frames,
    )


# ── main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Unified stream benchmark runner")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--raw_data_dir", required=True)
    p.add_argument("--output_dir", default=DEFAULT_RUNTIME_OUTPUT_DIR)
    p.add_argument("--preset", default="smoke")
    p.add_argument("--suites", default=None)
    p.add_argument("--render_video", action="store_true", default=False)
    p.add_argument("--no_save_plots", action="store_true", default=False)
    p.add_argument("--history_length", type=int, default=30)
    p.add_argument("--traj_horizon_tokens", type=int, default=20)
    p.add_argument("--num_denoise_steps", type=int, default=10)
    p.add_argument("--waypoint_dt", type=float, default=0.05)
    p.add_argument("--token_dt", type=float, default=0.20)
    p.add_argument("--motion_fps", type=float, default=20.0)
    p.add_argument(
        "--traj_condition_path",
        choices=("rootplan_7d", "legacy_xyz"),
        default="rootplan_7d",
        help="Trajectory conditioning path for stream_generate_step.",
    )
    p.add_argument("--root_refiner_config", default=None)
    p.add_argument("--root_refiner_ckpt", default=None)
    p.add_argument("--root_refiner_path_mode", default="dense_path")
    p.add_argument("--root_refiner_non_strict", action="store_true", default=False)
    p.add_argument(
        "--condition_variants",
        default="auto",
        help=(
            "Comma-separated runtime condition variants: gt_7d_ldf, "
            "rootrefiner_7d_ldf, no_traj_ldf, legacy_xyz_ldf, or auto. "
            "auto runs gt_7d_ldf/no_traj_ldf and includes rootrefiner_7d_ldf "
            "when --root_refiner_config/--root_refiner_ckpt are provided."
        ),
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--precomputed_text_emb_path", default=None)
    p.add_argument(
        "--runtime_debug_matrix",
        action="store_true",
        default=False,
        help=(
            "Run the new runtime debug matrix. Defaults to gtroot/rootrefiner; "
            "pass --runtime_debug_root_sources gtroot,rootrefiner,notraj to include no-traj."
        ),
    )
    p.add_argument(
        "--runtime_debug_sample_id",
        default="001168",
        help="HumanML3D sample id for --runtime_debug_matrix.",
    )
    p.add_argument(
        "--runtime_debug_root_sources",
        default=",".join(DEFAULT_ROOT_SOURCES),
        help="Comma-separated root sources for --runtime_debug_matrix.",
    )
    p.add_argument(
        "--runtime_debug_devices",
        default="",
        help=(
            "Comma-separated GPU ids for multi-process runtime debug matrix. "
            "Empty keeps single-process execution."
        ),
    )
    p.add_argument("--runtime_debug_run_id", default=None, help=argparse.SUPPRESS)
    p.add_argument("--runtime_debug_spec_indices", default=None, help=argparse.SUPPRESS)
    p.add_argument("--runtime_debug_worker_id", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--runtime_debug_families",
        default="web_stream,rotation,turn",
        help="Comma-separated families: web_stream, rotation, turn.",
    )
    p.add_argument(
        "--runtime_debug_rotation_degrees",
        default="10,20,30,40,50,60,70,80,90",
        help="Comma-separated rotation degrees for runtime debug rotation family.",
    )
    p.add_argument(
        "--runtime_debug_turn_delay_tokens",
        default="5,10,20",
        help="Comma-separated delay-token values for runtime debug turn family.",
    )
    p.add_argument(
        "--runtime_debug_turn_blend_tokens",
        default="2,4,8",
        help="Comma-separated blend-token values for runtime debug turn family.",
    )
    args = p.parse_args()

    if args.runtime_debug_matrix and not (args.root_refiner_config and args.root_refiner_ckpt):
        p.error(
            "--runtime_debug_matrix requires --root_refiner_config and "
            "--root_refiner_ckpt so gtroot/rootrefiner diagnostics can be compared"
        )

    run_id = args.runtime_debug_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_layout = RuntimeArtifactLayout(
        output_root=args.output_dir,
        ckpt_tag=infer_ckpt_tag(args.ckpt),
        run_id=run_id,
    )
    runtime_debug_specs = (
        _build_runtime_debug_specs_from_args(args) if args.runtime_debug_matrix else []
    )
    if (
        args.runtime_debug_matrix
        and _parse_runtime_debug_devices(args.runtime_debug_devices)
        and not args.runtime_debug_spec_indices
    ):
        _run_runtime_debug_distributed(
            args=args,
            run_id=run_id,
            debug_layout=debug_layout,
            specs=runtime_debug_specs,
        )
        return

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(config_path=args.config)
    OmegaConf.update(cfg.config, "test_vae_ckpt", args.vae_ckpt)
    if args.precomputed_text_emb_path:
        OmegaConf.update(cfg.config, "model.params.use_precomputed_text_emb", True)
        OmegaConf.update(cfg.config, "model.params.precomputed_text_emb_path",
                         args.precomputed_text_emb_path)

    print(f"Loading VAE ...")
    vae = _load_vae(cfg, dev)
    print(f"Loading model ...")
    model = _load_model(cfg, args.ckpt, dev)
    root_refiner_runtime = None
    if args.root_refiner_config or args.root_refiner_ckpt:
        if not (args.root_refiner_config and args.root_refiner_ckpt):
            p.error("--root_refiner_config and --root_refiner_ckpt must be provided together")
        from utils.refiner.runtime import RootRefinerRuntime

        print("Loading RootRefiner runtime ...")
        root_refiner_runtime = RootRefinerRuntime.from_config(
            config_path=args.root_refiner_config,
            ckpt_path=args.root_refiner_ckpt,
            device=dev,
            strict=not args.root_refiner_non_strict,
            path_mode=args.root_refiner_path_mode,
        )

    condition_variants = parse_condition_variants(
        args.condition_variants,
        include_root_refiner=root_refiner_runtime is not None,
    )
    if any(variant.use_root_refiner for variant in condition_variants) and root_refiner_runtime is None:
        p.error(
            "condition variant rootrefiner_7d_ldf requires "
            "--root_refiner_config and --root_refiner_ckpt"
        )

    suites_list = [s.strip() for s in args.suites.split(",")] if args.suites else None
    suite_tag = "_".join(suites_list) if suites_list else args.preset
    media_dirs = prepare_runtime_media_dirs(
        output_dir=args.output_dir,
        run_id=run_id,
        suite_tag=suite_tag,
        render_video=bool(args.render_video),
        save_plots=not bool(args.no_save_plots),
        enabled=not bool(args.runtime_debug_matrix),
    )
    vdir = media_dirs["video_dir"]
    pdir = media_dirs["plot_dir"]
    standard_vdir = media_dirs["standard_video_dir"]
    standard_pdir = media_dirs["standard_plot_dir"]

    if args.runtime_debug_matrix:
        all_specs = runtime_debug_specs
        specs = _select_runtime_debug_specs(
            all_specs,
            args.runtime_debug_spec_indices,
        )
        print(
            f"{len(specs)} runtime debug experiment(s)  "
            f"sample={args.runtime_debug_sample_id}"
        )
        sample_cache = {}
        all_recs = []
        root_plan_cache: dict[tuple[str, tuple[str, ...]], dict[str, tuple[np.ndarray, int]]] = {}

        for spec in specs:
            root_source = normalize_root_source(spec.root_source)
            if spec.sample_id not in sample_cache:
                sample_cache[spec.sample_id] = _load_humanml3d_sample(
                    args.raw_data_dir,
                    spec.sample_id,
                )
            sample = sample_cache[spec.sample_id]
            print(
                f"\n--- {root_source}/{spec.family}/"
                f"{'/'.join(spec.parts) if spec.parts else spec.name} ---"
            )
            seed_everything(args.seed)
            variant_root_refiner = (
                root_refiner_runtime if root_source == "rootrefiner" else None
            )
            replan_events: list[dict] = []
            root_plan_events: list[dict] = []
            kw = dict(
                hl=args.history_length,
                nds=args.num_denoise_steps,
                hz=args.traj_horizon_tokens,
                tdt=args.token_dt,
                wpdt=args.waypoint_dt,
                fps=args.motion_fps,
                condition_path="rootplan_7d",
                root_refiner_runtime=variant_root_refiner,
                replan_events=replan_events,
                root_plan_events=root_plan_events,
            )
            motion_yaw_offset = 0.0
            split_tok = None

            if spec.family in {"web_stream", "rotation"}:
                mode, gt_motion_7d, force_no_traj = runtime_debug_mode_for_source(root_source)
                pm, pr, gr, pt, pp, motion_yaw_offset = _run_real(
                    model,
                    vae,
                    sample,
                    dev,
                    mode=mode,
                    rotate_plan_deg=float(spec.params.get("rotate_plan_deg", 0.0)),
                    gt_motion_7d=gt_motion_7d,
                    force_no_traj=force_no_traj,
                    **kw,
                )
                target_frames = sample["feature_length"]
                target_source = "runtime_debug_gtroot_target" if gt_motion_7d else "runtime_debug_route_target"
            elif spec.family == "turn":
                angle = float(spec.params.get("update_angle", 30.0))
                delay_tokens = int(spec.params.get("mid_update_delay_tokens", 20))
                blend_tokens = int(spec.params.get("mid_update_blend_tokens", 4))
                pm, pr, gr, pt, pp, target_frames = _run_turn(
                    model,
                    vae,
                    sample,
                    dev,
                    mode=f"turn_{spec.name}",
                    angle=angle,
                    delay_tokens=delay_tokens,
                    blend_tokens=blend_tokens,
                    force_no_traj=False,
                    **kw,
                )
                split_tok = 15
                target_source = "runtime_debug_turn_target"
            else:
                raise ValueError(f"unknown runtime debug family {spec.family!r}")

            rec = build_plan_metrics(
                pr,
                original_gt_root=gr,
                plan_times=pt,
                plan_points_xyz=pp,
                target_frames=target_frames,
                motion_fps=args.motion_fps,
                motion_263=pm,
                motion_yaw_offset=motion_yaw_offset,
                target_source=target_source,
            )
            rec.update(
                {
                    "suite": "runtime_debug",
                    "mode": spec.family,
                    "sample_id": spec.sample_id,
                    "base_case_name": spec.name,
                    "case_name": f"{root_source}/{spec.family}/"
                    f"{'/'.join(spec.parts) if spec.parts else spec.name}",
                    "condition_variant": root_source,
                    "root_source": root_source,
                    "experiment_family": spec.family,
                    "experiment_name": spec.name,
                    "experiment_parts": list(spec.parts),
                    "traj_condition_path": "rootplan_7d",
                    "root_refiner_enabled": variant_root_refiner is not None,
                    "condition_source": runtime_debug_condition_source(
                        root_source,
                        family=spec.family,
                    ),
                    "timeline_semantics": "token_start_frame",
                    "turn_plan_policy": runtime_debug_turn_plan_policy(
                        spec.family,
                        root_source=root_source,
                    ),
                    "root_refiner_history_policy": runtime_debug_root_refiner_history_policy(
                        root_source,
                        family=spec.family,
                    ),
                    "rootplan_replan_count": len(replan_events),
                    "rootplan_replan_commits": [
                        int(event.get("commit", 0)) for event in replan_events
                    ],
                    "rootplan_replan_sources": [
                        str(event.get("source", "")) for event in replan_events
                    ],
                }
            )
            if spec.family == "turn":
                turn_edit_commit = 15
                turn_delay_tokens = int(spec.params.get("mid_update_delay_tokens", 20))
                turn_blend_tokens = int(spec.params.get("mid_update_blend_tokens", 4))
                turn_effective_commit = turn_edit_commit + turn_delay_tokens
                turn_activation_commit = _turn_activation_commit_from_events(
                    replan_events,
                    fallback_commit=turn_effective_commit,
                )
                _add_turn_post_switch_metrics(
                    rec,
                    pred_root_xyz=pr,
                    plan_times=pt,
                    plan_points_xyz=pp,
                    target_frames=target_frames,
                    motion_fps=args.motion_fps,
                    activation_commit=turn_activation_commit,
                )
                rec.update(
                    {
                        "turn_edit_commit": turn_edit_commit,
                        "turn_delay_tokens": turn_delay_tokens,
                        "turn_blend_tokens": turn_blend_tokens,
                        "turn_requested_effective_commit": turn_effective_commit,
                        "turn_effective_commit": turn_effective_commit,
                        "turn_activation_commit": turn_activation_commit,
                        "turn_target_source": "runtime_debug_turn_target",
                    }
                )
            all_recs.append(rec)
            write_experiment_metrics(
                debug_layout,
                root_source=root_source,
                family=spec.family,
                parts=spec.parts,
                metrics=rec,
            )
            print(f"  ADE={rec.get('ADE', float('nan')):.4f}  FDE={rec.get('FDE', float('nan')):.4f}")

            visual_target_root = _visual_target_root_from_plan(
                original_gt_root=gr,
                plan_times=pt,
                plan_points_xyz=pp,
                target_frames=len(pr),
                motion_fps=args.motion_fps,
            )
            leaf_dir = debug_layout.experiment_dir(root_source, spec.family, *spec.parts)
            if not args.no_save_plots and pr is not None and visual_target_root is not None:
                _write_runtime_case_visuals(
                    leaf_dir / "plots",
                    case_name=spec.name,
                    pred_root=pr,
                    target_root=visual_target_root,
                    motion_263=pm,
                    split_tok=split_tok,
                    pred_yaw_offset=motion_yaw_offset,
                )
            if args.render_video and pm is not None and pm.size > 0:
                video_dir = ensure_dir(leaf_dir / "video")
                mp4 = video_dir / f"{spec.name}.mp4"
                _render_runtime_case_video(
                    motion_263=pm,
                    pred_root=pr,
                    target_root=visual_target_root,
                    save_path=mp4,
                    pred_yaw_offset=motion_yaw_offset,
                )
                print(f"    video: {mp4}")

            root_7d_world, num_tokens = root_plan_events_to_diagnostic_arrays(
                root_plan_events
            )
            if root_source in {"gtroot", "rootrefiner"} and len(root_7d_world) > 0:
                write_experiment_root_plan(
                    debug_layout,
                    root_source=root_source,
                    family=spec.family,
                    parts=spec.parts,
                    root_7d_world=root_7d_world,
                    num_tokens=num_tokens,
                )
                root_plan_cache.setdefault((spec.family, spec.parts), {})[root_source] = (
                    root_7d_world,
                    num_tokens,
                )

        if args.runtime_debug_worker_id is not None:
            print(
                f"[runtime-debug] worker {args.runtime_debug_worker_id} "
                f"completed {len(all_recs)} experiment(s)"
            )
            return

        root_diag_records = []
        for (family, parts), by_source in sorted(root_plan_cache.items()):
            if "gtroot" not in by_source or "rootrefiner" not in by_source:
                continue
            gt_root_7d, gt_num_tokens = by_source["gtroot"]
            pred_root_7d, pred_num_tokens = by_source["rootrefiner"]
            diag = compute_root_condition_diagnostics(
                gt_root_7d,
                pred_root_7d,
                gt_num_tokens=gt_num_tokens,
                pred_num_tokens=pred_num_tokens,
            )
            diag.update(
                {
                    "family": family,
                    "parts": list(parts),
                    "experiment": "/".join(parts) if parts else family,
                }
            )
            root_diag_records.append(diag)
            write_root_diagnostic_artifacts(
                debug_layout,
                family=family,
                parts=parts,
                metrics=diag,
                gt_root_7d=gt_root_7d,
                pred_root_7d=pred_root_7d,
                gt_num_tokens=gt_num_tokens,
                pred_num_tokens=pred_num_tokens,
            )

        aggregate = aggregate_runtime_records(all_recs)
        root_diag_summary = summarize_numeric_records(root_diag_records)
        write_root_diagnostics_summary(
            debug_layout,
            summary=root_diag_summary,
            records=root_diag_records,
        )
        summary = {
            "run_id": run_id,
            "config": args.config,
            "ckpt": args.ckpt,
            "vae_ckpt": args.vae_ckpt,
            "root_refiner_config": args.root_refiner_config,
            "root_refiner_ckpt": args.root_refiner_ckpt,
            "runtime_debug_matrix": True,
            "runtime_debug_sample_id": args.runtime_debug_sample_id,
            "aggregate": aggregate,
            "root_diagnostics": root_diag_summary,
            "summary": aggregate,
            "records": all_recs,
        }
        debug_report_paths = write_runtime_debug_report(
            debug_layout,
            manifest={
                "run_id": run_id,
                "config": args.config,
                "ckpt": args.ckpt,
                "vae_ckpt": args.vae_ckpt,
                "root_refiner_config": args.root_refiner_config,
                "root_refiner_ckpt": args.root_refiner_ckpt,
                "runtime_debug_matrix": True,
                "runtime_debug_sample_id": args.runtime_debug_sample_id,
                "root_sources": sorted({spec.root_source for spec in specs}),
                "num_experiments": len(specs),
            },
            summary=summary,
            records=all_recs,
            source_payloads={
                root_source: root_source_metadata(root_source)
                for root_source in sorted({spec.root_source for spec in specs})
            },
        )
        print(f"\nRuntime debug: {debug_report_paths['summary']}")
        print(f"Runtime debug CSV: {debug_report_paths['records']}")
        print(f"Root diagnostics: {debug_layout.run_dir / 'root_diagnostics' / 'summary.json'}")
        return

    cases = get_cases(suites=suites_list, preset=args.preset)
    if any(variant.force_no_traj for variant in condition_variants):
        cases = [case for case in cases if not _is_legacy_no_traj_case(case)]
    print(
        f"{len(cases)} case(s)  preset={args.preset}  suites={suites_list} "
        f"condition_variants={[variant.name for variant in condition_variants]}"
    )

    all_recs = []
    _pm = _pr = _gr = _split = None
    for case in cases:
        if case.dataset == "babel" and case.sample_ids:
            sample = _merge_babel(args.raw_data_dir, case.sample_ids)
        elif case.dataset == "babel":
            sample = _load_babel_sample(args.raw_data_dir, case.sample_id)
        else:
            sample = _load_humanml3d_sample(args.raw_data_dir, case.sample_id)
        gr_base = extract_root_trajectory_263(sample["feature"].numpy())

        for variant in condition_variants:
            display_case_name = _variant_case_name(case.name, variant)
            print(
                f"\n--- {display_case_name} "
                f"({case.suite}/{case.mode}/{variant.name}) ---"
            )
            seed_everything(args.seed)

            variant_root_refiner = (
                root_refiner_runtime if variant.use_root_refiner else None
            )
            replan_events: list[dict] = []
            kw = dict(hl=args.history_length, nds=args.num_denoise_steps,
                      hz=args.traj_horizon_tokens, tdt=args.token_dt,
                      wpdt=args.waypoint_dt, fps=args.motion_fps,
                      condition_path=variant.condition_path,
                      root_refiner_runtime=variant_root_refiner,
                      replan_events=replan_events,
                      force_no_traj=variant.force_no_traj)
            visual_target_root = None
            motion_yaw_offset = 0.0

            if case.suite == "step":
                pm, pr, gr = _run_step(model, vae, sample, dev, mode=case.mode, **kw)
                step_plan_times = np.arange(len(gr), dtype=np.float32) / 20.0
                rec = build_plan_metrics(
                    pr, original_gt_root=gr,
                    plan_times=step_plan_times,
                    plan_points_xyz=gr,
                    target_frames=sample["feature_length"], motion_fps=args.motion_fps,
                    motion_263=pm, target_source="original_gt_root",
                )
                visual_target_root = _visual_target_root_from_plan(
                    original_gt_root=gr,
                    plan_times=step_plan_times,
                    plan_points_xyz=gr,
                    target_frames=len(pr),
                    motion_fps=args.motion_fps,
                )
            elif case.suite == "real":
                _rot = case.mode_kwargs.get("rotate_plan_deg", 0.0)
                _gt_motion_7d = bool(case.mode_kwargs.get("gt_motion_7d", False))
                pm, pr, gr, pt, pp, motion_yaw_offset = _run_real(
                    model, vae, sample, dev, mode=case.mode,
                    rotate_plan_deg=float(_rot),
                    gt_motion_7d=_gt_motion_7d,
                    **kw)
                rec = build_plan_metrics(
                    pr, original_gt_root=gr, plan_times=pt, plan_points_xyz=pp,
                    target_frames=sample["feature_length"], motion_fps=args.motion_fps,
                    motion_263=pm,
                    motion_yaw_offset=motion_yaw_offset,
                )
                visual_target_root = _visual_target_root_from_plan(
                    original_gt_root=gr,
                    plan_times=pt,
                    plan_points_xyz=pp,
                    target_frames=len(pr),
                    motion_fps=args.motion_fps,
                )
            elif case.suite == "turn":
                ang = case.mode_kwargs.get("update_angle", 30.0)
                dt_val = case.mode_kwargs.get("mid_update_delay_tokens", 20)
                db_val = case.mode_kwargs.get("mid_update_blend_tokens", 4)
                if isinstance(dt_val, str):
                    dt_val = int(dt_val.split(",")[0])
                turn_delay_tokens = int(dt_val)
                turn_blend_tokens = int(db_val)
                turn_edit_commit = 15
                turn_effective_commit = turn_edit_commit + turn_delay_tokens
                pm, pr, gr, pt, pp, ttfs = _run_turn(
                    model, vae, sample, dev, mode=case.mode,
                    angle=ang, delay_tokens=int(dt_val),
                    blend_tokens=int(db_val), **kw)
                turn_activation_commit = _turn_activation_commit_from_events(
                    replan_events,
                    fallback_commit=turn_effective_commit,
                )
                rec = build_plan_metrics(
                    pr, original_gt_root=gr, plan_times=pt, plan_points_xyz=pp,
                    target_frames=ttfs, motion_fps=args.motion_fps,
                    motion_263=pm, target_source="scheduled_turn_target",
                )
                _add_turn_post_switch_metrics(
                    rec,
                    pred_root_xyz=pr,
                    plan_times=pt,
                    plan_points_xyz=pp,
                    target_frames=ttfs,
                    motion_fps=args.motion_fps,
                    activation_commit=turn_activation_commit,
                )
                visual_target_root = _visual_target_root_from_plan(
                    original_gt_root=gr,
                    plan_times=pt,
                    plan_points_xyz=pp,
                    target_frames=len(pr),
                    motion_fps=args.motion_fps,
                )
                rec["turn_edit_commit"] = turn_edit_commit
                rec["turn_delay_tokens"] = turn_delay_tokens
                rec["turn_blend_tokens"] = turn_blend_tokens
                rec["turn_requested_effective_commit"] = turn_effective_commit
                rec["turn_effective_commit"] = turn_effective_commit
                rec["turn_activation_commit"] = turn_activation_commit
                rec["turn_target_source"] = "scheduled_turn_target"
            elif case.suite == "babel":
                pm, pr, gr, pt, pp = _run_babel(
                    model, vae, sample, dev, mode=case.mode, **kw)
                rec = build_plan_metrics(
                    pr, original_gt_root=gr, plan_times=pt, plan_points_xyz=pp,
                    target_frames=sample["feature_length"], motion_fps=args.motion_fps,
                    motion_263=pm,
                )
                visual_target_root = _visual_target_root_from_plan(
                    original_gt_root=gr,
                    plan_times=pt,
                    plan_points_xyz=pp,
                    target_frames=len(pr),
                    motion_fps=args.motion_fps,
                )
            else:
                print(f"  SKIP: unknown suite {case.suite}")
                continue

            rec["suite"] = case.suite; rec["mode"] = case.mode
            rec["sample_id"] = case.sample_id
            rec["base_case_name"] = case.name
            rec["case_name"] = display_case_name
            rec["condition_variant"] = variant.name
            rec["traj_condition_path"] = variant.condition_path
            rec["root_refiner_enabled"] = variant_root_refiner is not None
            rec["condition_source"] = resolve_traj_condition_source(
                variant.condition_path,
                variant_root_refiner,
                no_traj=variant.force_no_traj or str(case.mode).endswith("_no_traj"),
                gt_motion_7d=bool(case.mode_kwargs.get("gt_motion_7d", False)),
            )
            rec["rootplan_replan_count"] = len(replan_events)
            rec["rootplan_replan_commits"] = [
                int(event.get("commit", 0)) for event in replan_events
            ]
            rec["rootplan_replan_sources"] = [
                str(event.get("source", "")) for event in replan_events
            ]
            all_recs.append(rec)
            print(f"  ADE={rec.get('ADE', float('nan')):.4f}  FDE={rec.get('FDE', float('nan')):.4f}")

            _pm, _pr, _gr = pm, pr, visual_target_root
            _split = 15 if case.suite == "turn" else None

            if pdir and _pr is not None and _gr is not None:
                _write_runtime_case_visuals(
                    pdir,
                    case_name=display_case_name,
                    pred_root=_pr,
                    target_root=_gr,
                    motion_263=_pm,
                    split_tok=_split,
                    pred_yaw_offset=motion_yaw_offset,
                )
                if standard_pdir:
                    for name in (
                        f"{display_case_name}_plot_world_xz.png",
                        f"{display_case_name}_plot_yaw.png",
                    ):
                        src = os.path.join(pdir, name)
                        if os.path.exists(src):
                            shutil.copy2(src, os.path.join(standard_pdir, name))

            if args.render_video and _pm is not None and _pm.size > 0:
                mp4 = os.path.join(vdir, f"{display_case_name}.mp4")
                _render_runtime_case_video(
                    motion_263=_pm,
                    pred_root=_pr,
                    target_root=_gr,
                    save_path=mp4,
                    pred_yaw_offset=motion_yaw_offset,
                )
                print(f"    video: {mp4}")
                if standard_vdir:
                    shutil.copy2(mp4, os.path.join(standard_vdir, f"{display_case_name}.mp4"))
                if _pr is not None and _gr is not None:
                    traj_mp4 = os.path.join(vdir, f"{display_case_name}_traj.mp4")
                    _render_traj_video(
                        _pr, _gr, traj_mp4, display_case_name, split_tok=_split)
                    if standard_vdir:
                        shutil.copy2(
                            traj_mp4,
                            os.path.join(standard_vdir, f"{display_case_name}_traj.mp4"),
                        )

    aggregate = aggregate_runtime_records(all_recs)
    summary = {"run_id": run_id, "config": args.config, "ckpt": args.ckpt,
               "vae_ckpt": args.vae_ckpt, "waypoint_dt": args.waypoint_dt,
               "traj_horizon_tokens": args.traj_horizon_tokens,
               "history_length": args.history_length,
               "traj_condition_path": args.traj_condition_path,
               "root_refiner_config": args.root_refiner_config,
               "root_refiner_ckpt": args.root_refiner_ckpt,
               "root_refiner_path_mode": args.root_refiner_path_mode,
               "root_refiner_available": root_refiner_runtime is not None,
               "condition_variants": [variant.name for variant in condition_variants],
               "condition_sources": sorted(
                   {str(rec.get("condition_source", "")) for rec in all_recs}
               ),
               "aggregate": aggregate,
               # Backward-compatible alias for scripts that consumed the nested
               # field added during the eval package refactor.
               "summary": aggregate,
               "records": all_recs}
    report_dirs = write_runtime_report(
        output_dir=args.output_dir,
        run_id=run_id,
        suite_tag=suite_tag,
        payload=summary,
        records=all_recs,
        artifact_kinds=("metrics", "plot", "video"),
    )
    debug_source_payloads = {}
    for variant in condition_variants:
        root_source = normalize_root_source(variant.name)
        debug_source_payloads[root_source] = root_source_metadata(root_source)
    debug_report_paths = write_runtime_debug_report(
        debug_layout,
        manifest={
            "run_id": run_id,
            "config": args.config,
            "ckpt": args.ckpt,
            "vae_ckpt": args.vae_ckpt,
            "root_refiner_config": args.root_refiner_config,
            "root_refiner_ckpt": args.root_refiner_ckpt,
            "preset": args.preset,
            "suites": suites_list,
            "condition_variants": [variant.name for variant in condition_variants],
        },
        summary=summary,
        records=all_recs,
        source_payloads=debug_source_payloads,
    )
    print(f"\nSummary: {report_dirs['legacy_root'] / 'summary.json'}")
    if all_recs:
        print(f"CSV: {report_dirs['legacy_root'] / 'summary.csv'}")
        print(f"Runtime metrics: {report_dirs['metrics'] / 'summary.json'}")
        print(f"Runtime debug: {debug_report_paths['summary']}")


if __name__ == "__main__":
    main()
