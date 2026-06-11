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
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
from utils.inference_glue import InferenceGlueState, InferenceGlueTimeline
from utils.local_frame import canonicalize_7d
from utils.motion_process import (
    StreamJointRecovery263,
    append_traj_deltas_5d_to_7d,
    extract_root_trajectory_263,
)
from utils.root_plan import RootPlan
from utils.runtime_rootplan import build_rootplan_stream_payload_from_buffer
from utils.stream_rollout import (
    StreamTextSegment, StreamTextRolloutController,
    build_stream_step_model_input,
    build_stream_suffix_conditioning,
)
from utils.stream_traj import (
    StreamTrajectoryPlan,
    assign_uniform_timestamps,
    blend_future_trajs,
    reanchor_stream_plan_to_xz,
    resample_polyline_by_arclength,
    sample_plan_future,
    sample_plan_by_time,
    sample_timestamped_trajectory,
    smoothstep01,
)
from utils.token_frame import (
    num_tokens_for_frame_len,
    token_start_frame,
)
from utils.visualize import render_single_video
from eval.runtime.cases import get_cases
from eval.runtime.metrics import (
    build_plan_metrics,
    compute_plan_targets,
    estimate_body_yaw,
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
    "turn_effective_commit",
    "turn_activation_commit",
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


def resolve_traj_condition_source(
    condition_path: str,
    root_refiner_runtime=None,
    *,
    no_traj: bool = False,
) -> str:
    if no_traj:
        return "none"
    if condition_path == "rootplan_7d":
        return "root_refiner_7d" if root_refiner_runtime is not None else "route_7d"
    return str(condition_path)


def _infer_physical_yaw_from_points(points_xyz: torch.Tensor) -> torch.Tensor:
    yaw_values = []
    last_yaw = points_xyz.new_tensor(0.0)
    n = int(points_xyz.shape[0])
    for i in range(n):
        if i < n - 1:
            delta = points_xyz[i + 1, [0, 2]] - points_xyz[i, [0, 2]]
        elif i > 0:
            delta = points_xyz[i, [0, 2]] - points_xyz[i - 1, [0, 2]]
        else:
            delta = points_xyz.new_zeros(2)
        if torch.linalg.norm(delta) > 1e-6:
            last_yaw = torch.atan2(delta[0], delta[1])
        yaw_values.append(last_yaw)
    return torch.stack(yaw_values) if yaw_values else points_xyz.new_zeros(0)


def build_eval_root_plan_from_points(
    points_xyz,
    *,
    anchor_state: InferenceGlueState,
    token_dt: float,
    frames_per_token: int = 4,
    source: str = "eval_route",
) -> RootPlan:
    """Convert a world-space route sampled at frame cadence into a 7D RootPlan."""
    points = torch.as_tensor(
        points_xyz,
        device=anchor_state.world_xz.device,
        dtype=torch.float32,
    )
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points_xyz must be [T,3], got {tuple(points.shape)}")
    yaw = _infer_physical_yaw_from_points(points)
    traj_5d_world = torch.cat(
        [points, torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)],
        dim=-1,
    )
    traj_7d_world = append_traj_deltas_5d_to_7d(
        traj_5d_world,
        physical_yaw=yaw,
    )
    anchor_xz = anchor_state.world_xz.to(device=points.device, dtype=torch.float32)
    anchor_yaw = anchor_state.world_yaw.to(device=points.device, dtype=torch.float32)
    traj_7d_local = canonicalize_7d(traj_7d_world, anchor_xz, anchor_yaw)
    valid_frames = int(traj_7d_local.shape[0])
    return RootPlan(
        num_tokens_pred=num_tokens_for_frame_len(valid_frames, frames_per_token),
        valid_frames=valid_frames,
        waypoints_local_7d=traj_7d_local,
        frame_dt=float(token_dt) / float(frames_per_token),
        frames_per_token=int(frames_per_token),
        anchor_commit_idx=int(anchor_state.commit_idx),
        anchor_world_xz=anchor_xz,
        anchor_world_yaw=anchor_yaw,
        source=str(source),
    )


def build_rootplan_stream_step_payload(
    model,
    timeline: InferenceGlueTimeline,
    *,
    history_length: int,
    traj_horizon_tokens: int,
    absolute_commit_index: int | None = None,
) -> dict | None:
    """Build the direct 7D payload consumed by stream_generate_step."""
    traj_buf = getattr(model, "_traj_buf", None)
    chunk_size = int(getattr(model, "chunk_size", 1))
    local_commit = int(getattr(model, "commit_index", 0))
    absolute_commit = (
        local_commit if absolute_commit_index is None else int(absolute_commit_index)
    )
    return build_rootplan_stream_payload_from_buffer(
        traj_buf,
        timeline,
        local_commit_index=local_commit,
        absolute_commit_index=absolute_commit,
        chunk_size=chunk_size,
        history_length=history_length,
        traj_horizon_tokens=traj_horizon_tokens,
    )


def _new_eval_timeline() -> InferenceGlueTimeline:
    return InferenceGlueTimeline(
        InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )


def _append_eval_timeline_state(
    timeline: InferenceGlueTimeline,
    *,
    commit_idx: int,
    recovery: StreamJointRecovery263,
) -> None:
    commit_idx = int(commit_idx)
    if commit_idx <= timeline.head.commit_idx:
        return
    root = np.asarray(recovery.r_pos_accum, dtype=np.float32)
    timeline.append(
        InferenceGlueState(
            commit_idx=commit_idx,
            world_xz=torch.tensor(root[[0, 2]], dtype=torch.float32),
            world_yaw=torch.tensor(-2.0 * float(recovery.r_rot_ang_accum), dtype=torch.float32),
            source="stream_eval",
        )
    )


def _reset_eval_runtime_trace(model) -> None:
    """Backward-compatible cleanup for older callers that stored trace on model."""
    if hasattr(model, "_stream_eval_replan_events"):
        delattr(model, "_stream_eval_replan_events")


def _append_eval_root_history(root_5d_history: list, frame_idx: int, recovery) -> int:
    root = np.asarray(recovery.r_pos_accum, dtype=np.float32).copy()
    yaw = -2.0 * float(recovery.r_rot_ang_accum)
    root5d = np.asarray(
        [root[0], root[1], root[2], np.cos(yaw), np.sin(yaw)],
        dtype=np.float32,
    )
    root_5d_history.append((int(frame_idx), root5d))
    return int(frame_idx) + 1


def _get_eval_root_refiner_history_5d(root_5d_history, anchor_commit: int):
    if not root_5d_history:
        return None
    from utils.token_frame import token_start_frame

    anchor_frame = token_start_frame(max(0, int(anchor_commit)))
    frames = [
        np.asarray(root5d, dtype=np.float32)
        for frame_idx, root5d in root_5d_history
        if int(frame_idx) <= anchor_frame
    ]
    if not frames:
        return None
    return np.stack(frames, axis=0).astype(np.float32)


def _clear_model_traj_state(model) -> None:
    traj_buf = getattr(model, "_traj_buf", None)
    if traj_buf is None:
        return
    if hasattr(traj_buf, "reset"):
        traj_buf.reset()
    if hasattr(traj_buf, "clear"):
        traj_buf.clear()


def _set_eval_root_plan(
    model,
    timeline: InferenceGlueTimeline,
    stream_plan: StreamTrajectoryPlan,
    *,
    text: str,
    token_dt: float,
    root_refiner_runtime=None,
    root_5d_history=None,
    replan_events=None,
    frames_per_token: int = 4,
) -> bool:
    if not timeline.has_exact_state(int(stream_plan.start_commit_index)):
        return False
    anchor_state = timeline.at_commit(int(stream_plan.start_commit_index))
    root_plan_input = reanchor_stream_plan_to_xz(
        stream_plan,
        anchor_state.world_xz.detach().cpu().numpy(),
    )
    if root_refiner_runtime is not None:
        root_plan = root_refiner_runtime.build_root_plan(
            text=text,
            plan=root_plan_input,
            anchor_state=anchor_state,
            token_dt=token_dt,
            history_motion_world_5d=_get_eval_root_refiner_history_5d(
                root_5d_history,
                int(stream_plan.start_commit_index),
            ),
        )
    else:
        root_plan = build_eval_root_plan_from_points(
            root_plan_input.points_xyz,
            anchor_state=anchor_state,
            token_dt=token_dt,
            frames_per_token=frames_per_token,
            source=root_plan_input.source or "eval_route",
        )
    model._traj_buf.set_root_plan(root_plan)
    if isinstance(replan_events, list):
        replan_events.append(
            {
                "commit": int(stream_plan.start_commit_index),
                "text": str(text),
                "source": str(stream_plan.source),
                "root_refiner": root_refiner_runtime is not None,
            }
        )
    return True


def _slice_stream_plan_from_commit(
    plan_times: np.ndarray,
    plan_points_xyz: np.ndarray,
    *,
    start_commit_index: int,
    token_dt: float,
    waypoint_dt: float,
    version: int,
    source: str,
) -> StreamTrajectoryPlan:
    """Build a future-only stream plan anchored at an absolute commit index."""
    plan_times = np.asarray(plan_times, dtype=np.float32)
    plan_points_xyz = np.asarray(plan_points_xyz, dtype=np.float32)
    start_commit_index = int(start_commit_index)
    elapsed = max(0.0, float(start_commit_index) * float(token_dt))
    if plan_times.size == 0:
        times = np.asarray([0.0, float(waypoint_dt)], dtype=np.float32)
        points = np.zeros((2, 3), dtype=np.float32)
    else:
        end_time = max(elapsed, float(plan_times[-1]))
        npt = max(2, int(round((end_time - elapsed) / max(float(waypoint_dt), 1e-6))) + 1)
        query_abs = elapsed + np.arange(npt, dtype=np.float32) * np.float32(waypoint_dt)
        points = sample_plan_by_time(plan_times, plan_points_xyz, query_abs)
        times = query_abs - np.float32(elapsed)
    return StreamTrajectoryPlan(
        times=times.astype(np.float32),
        points_xyz=points.astype(np.float32),
        start_commit_index=start_commit_index,
        version=int(version),
        source=str(source),
    )


def _build_turn_metric_target(
    *,
    old_times: np.ndarray,
    old_points_xyz: np.ndarray,
    new_times: np.ndarray,
    new_points_xyz: np.ndarray,
    target_frames: int,
    motion_fps: float,
    edit_commit: int,
    delay_tokens: int,
    blend_tokens: int,
    token_dt: float,
    new_anchor_xz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the time target that matches web delayed-replace RootPlan input.

    The replacement plan is authored in absolute route coordinates but runtime
    reanchors plan-local t=0 to the effective update anchor before converting it
    to a RootPlan. Metrics must apply the same XZ translation, otherwise turn
    ADE/FDE measures a jump that the model never receives as input.
    """
    target_t = np.arange(int(target_frames), dtype=np.float32) / np.float32(motion_fps)
    effective_t = (float(edit_commit) + float(delay_tokens)) * float(token_dt)
    activation_t = (
        float(edit_commit) + float(delay_tokens) + float(blend_tokens)
    ) * float(token_dt)
    old_target = sample_plan_by_time(old_times, old_points_xyz, target_t)
    new_target = sample_plan_by_time(new_times, new_points_xyz, target_t)
    if new_anchor_xz is None:
        anchor_point = sample_plan_by_time(
            old_times,
            old_points_xyz,
            np.asarray([effective_t], dtype=np.float32),
        )[0]
        anchor_xz = anchor_point[[0, 2]]
    else:
        anchor_xz = np.asarray(new_anchor_xz, dtype=np.float32).reshape(2)
    new_zero = sample_plan_by_time(
        new_times,
        new_points_xyz,
        np.asarray([effective_t], dtype=np.float32),
    )[0]
    offset = anchor_xz - new_zero[[0, 2]]
    new_target = new_target.copy()
    new_target[:, [0, 2]] += offset[None, :]
    use_new = target_t >= np.float32(activation_t)
    target = old_target.copy()
    target[use_new] = new_target[use_new]
    return target_t.astype(np.float32), target.astype(np.float32)


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

def _run_step(model, vae, sample, device, *, hl, nds, mode, **_kw):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    hz = int(_kw.get("hz", 20))
    tdt = float(_kw.get("tdt", 0.20))
    fps = float(_kw.get("fps", 20.0))
    condition_path = str(_kw.get("condition_path", "rootplan_7d"))
    root_refiner_runtime = _kw.get("root_refiner_runtime")
    replan_events = _kw.get("replan_events")
    force_no_traj = bool(_kw.get("force_no_traj", False))
    text = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
    timeline = _new_eval_timeline()
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    _clear_model_traj_state(model)
    root_5d_history, frame_idx = [], 0
    _bs = {  # batch-style wrapper for build_stream_suffix_conditioning
        "traj": sample["traj"].unsqueeze(0),
        "token_length": torch.tensor([tl]),
        "traj_length": torch.tensor([sample["traj_length"]]),
        "token_mask": sample["token_mask"].unsqueeze(0),
        "traj_mask": sample["traj_mask"].unsqueeze(0),
    }
    if condition_path == "rootplan_7d" and mode != "step_no_traj" and not force_no_traj:
        plan = StreamTrajectoryPlan(
            times=np.arange(len(sample["traj"]), dtype=np.float32) / fps,
            points_xyz=sample["traj"].numpy().astype(np.float32),
            start_commit_index=0,
            version=0,
            source="bench_step",
        )
        _set_eval_root_plan(
            model,
            timeline,
            plan,
            text=text,
            token_dt=tdt,
            root_refiner_runtime=root_refiner_runtime,
            root_5d_history=root_5d_history,
            replan_events=replan_events,
        )
    sr = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, _roots = [], []
    fc = True
    for ci in range(tl):
        if mode == "step_no_traj" or force_no_traj:
            ti = None
        elif condition_path == "rootplan_7d":
            ti = build_rootplan_stream_step_payload(
                model,
                timeline,
                history_length=hl,
                traj_horizon_tokens=hz,
                absolute_commit_index=ci,
            )
        elif condition_path == "legacy_xyz" and mode == "step_predroot":
            ti = build_stream_suffix_conditioning(_bs, ci, prefer_xyz=True)
            if ti is not None and len(_roots) > 0:
                pr_cur = _roots[-1]
                gt_ci = sample["traj"].numpy()[min(ci * 4, len(sample["traj"]) - 1)].astype(np.float32)
                offset = pr_cur.astype(np.float32) - gt_ci
                t = ti["traj"]
                if torch.is_tensor(t):
                    ti["traj"] = t + torch.from_numpy(offset).float().to(t)
        elif condition_path == "legacy_xyz":
            ti = build_stream_suffix_conditioning(_bs, ci, prefer_xyz=True)
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        sp = build_stream_step_model_input(
            text,
            traj_input=ti)
        out = model.stream_generate_step(sp, first_chunk=fc)
        dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                 first_chunk=fc)[0].float().cpu().numpy())
        fc = False
        for frm in dec:
            sr.process_frame(frm)
            _roots.append(sr.r_pos_accum.copy())
            frame_idx = _append_eval_root_history(root_5d_history, frame_idx, sr)
        _append_eval_timeline_state(
            timeline,
            commit_idx=ci + 1,
            recovery=sr,
        )
        decs.append(dec)
    vae.clear_cache()
    pm = np.concatenate(decs, axis=0)[:tfs] if decs else np.zeros((0, 263))
    pr = np.array(_roots, dtype=np.float32)[:tfs] if _roots else np.zeros((0, 3))
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    return pm, pr, gr


def _run_real(model, vae, sample, device, *, hl, nds, hz, tdt, wpdt, fps, mode,
              rotate_plan_deg=0.0, condition_path="rootplan_7d",
              root_refiner_runtime=None, replan_events=None, force_no_traj=False):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gr_arr = sample["traj"].numpy()
    text = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
    timeline = _new_eval_timeline()
    dur = (sample["feature_length"] - 1) / fps
    npt = max(2, int(round(dur / wpdt)) + 1)
    plan_pts = resample_polyline_by_arclength(gr_arr, npt)
    plan_t = assign_uniform_timestamps(npt, wpdt)
    if rotate_plan_deg:
        plan_pts = _rotate_xz(plan_pts, plan_pts[0], float(rotate_plan_deg))
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    _clear_model_traj_state(model)
    root_5d_history, frame_idx = [], 0
    if condition_path == "rootplan_7d" and mode != "real_no_traj" and not force_no_traj:
        _set_eval_root_plan(
            model,
            timeline,
            StreamTrajectoryPlan(
                times=plan_t,
                points_xyz=plan_pts,
                start_commit_index=0,
                version=0,
                source="bench_real",
            ),
            text=text,
            token_dt=tdt,
            root_refiner_runtime=root_refiner_runtime,
            root_5d_history=root_5d_history,
            replan_events=replan_events,
        )
    sr = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, _roots, fc = [], [], True
    for ci in range(tl):
        if mode == "real_no_traj" or force_no_traj:
            ti = None
        elif condition_path == "rootplan_7d":
            ti = build_rootplan_stream_step_payload(
                model,
                timeline,
                history_length=hl,
                traj_horizon_tokens=hz,
                absolute_commit_index=ci,
            )
        elif condition_path == "legacy_xyz":
            cr = np.zeros(3, dtype=np.float32)
            if mode == "real_gtroot":
                cr = gr_arr[min(ci * 4, len(gr_arr) - 1)].astype(np.float32)
            else:
                cr[[0, 2]] = sr.r_pos_accum[[0, 2]].astype(np.float32)
            ft = sample_plan_future(StreamTrajectoryPlan(times=plan_t, points_xyz=plan_pts, start_commit_index=0, version=0, source="bench"), current_commit=ci, current_root_xyz=cr, horizon_tokens=hz, token_dt=tdt, reanchor_to_current_root=True)
            ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0),
                  "token_mask": torch.ones(1, hz)}
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        sp = build_stream_step_model_input(
            text,
            traj_input=ti)
        out = model.stream_generate_step(sp, first_chunk=fc)
        dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                 first_chunk=fc)[0].float().cpu().numpy())
        fc = False
        for frm in dec:
            sr.process_frame(frm)
            _roots.append(sr.r_pos_accum.copy())
            frame_idx = _append_eval_root_history(root_5d_history, frame_idx, sr)
        _append_eval_timeline_state(
            timeline,
            commit_idx=ci + 1,
            recovery=sr,
        )
        decs.append(dec)
    vae.clear_cache()
    pm = np.concatenate(decs, axis=0)[:tfs] if decs else np.zeros((0, 263))
    pr = np.array(_roots, dtype=np.float32)[:tfs] if _roots else np.zeros((0, 3))
    return pm, pr, gr, plan_t, plan_pts


def _rotate_xz(points, anchor, deg):
    pts = np.asarray(points, dtype=np.float32).copy()
    anc = np.asarray(anchor, dtype=np.float32).reshape(3)
    c, s = float(np.cos(np.deg2rad(deg))), float(np.sin(np.deg2rad(deg)))
    rel = pts[:, [0, 2]] - anc[[0, 2]][None, :]
    pts[:, 0], pts[:, 2] = anc[0] + c * rel[:, 0] - s * rel[:, 1], anc[2] + s * rel[:, 0] + c * rel[:, 1]
    return pts


def _run_turn(model, vae, sample, device, *, hl, nds, hz, tdt, wpdt, fps, mode, angle,
              delay_tokens=20, blend_tokens=4, condition_path="rootplan_7d",
              root_refiner_runtime=None, replan_events=None, force_no_traj=False):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gr_arr = sample["traj"].numpy()
    text = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
    timeline = _new_eval_timeline()
    dur = (sample["feature_length"] - 1) / fps
    npt = max(2, int(round(dur / wpdt)) + 1)
    plan_pts = resample_polyline_by_arclength(gr_arr, npt)
    plan_t = assign_uniform_timestamps(npt, wpdt)
    split_tok, sf = 15, max(1, 1 + 4 * 14)
    rot_pts = np.concatenate(
        [plan_pts[:sf], _rotate_xz(plan_pts[sf:], plan_pts[sf - 1], angle)], axis=0)
    rot_t = np.arange(len(rot_pts), dtype=np.float32) * wpdt
    ed = int(delay_tokens) if isinstance(delay_tokens, (int, float)) else 20
    eb = int(blend_tokens) if isinstance(blend_tokens, (int, float)) else 4
    # Extend past the blend zone for post-turn observation.
    extra = max(0, split_tok + ed + eb - tl)
    total_tl = tl + extra + 8
    total_tfs = 1 + 4 * (total_tl - 1) if total_tl > 1 else 1
    # Extend plans to cover the full query horizon past the blend zone.
    _needed_wp = max(len(plan_pts), int((total_tl + hz) * tdt / wpdt) + 2)
    for _p_arr, _p_name in [(plan_pts, "plan"), (rot_pts, "rot")]:
        _n = len(_p_arr)
        if _needed_wp > _n:
            _start_wp = max(0, _n - 5)
            _vel = _p_arr[-1] - _p_arr[_start_wp]
            _denom = max(1, _n - 1 - _start_wp)  # actual intervals spanned
            _step = _vel / float(_denom)
            _n_extra = _needed_wp - _n
            _p_new = _p_arr[-1][None, :] + np.arange(1, _n_extra + 1, dtype=np.float32)[:, None] * _step[None, :]
            if _p_name == "plan":
                plan_pts = np.concatenate([plan_pts, _p_new.astype(np.float32)], axis=0)
            else:
                rot_pts = np.concatenate([rot_pts, _p_new.astype(np.float32)], axis=0)
    plan_t = assign_uniform_timestamps(len(plan_pts), wpdt)
    rot_t = np.arange(len(rot_pts), dtype=np.float32) * wpdt
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    _clear_model_traj_state(model)
    root_5d_history, frame_idx = [], 0
    # Build old/new plans with web-demo effective-commit semantics.
    edit_commit = split_tok
    effective_commit = edit_commit + ed
    old_plan = StreamTrajectoryPlan(
        times=plan_t, points_xyz=plan_pts,
        start_commit_index=0, version=0, source="bench_old",
    )
    new_plan = _slice_stream_plan_from_commit(
        rot_t,
        rot_pts,
        start_commit_index=effective_commit,
        token_dt=tdt,
        waypoint_dt=wpdt,
        version=1,
        source="bench_new",
    )
    if condition_path == "rootplan_7d" and not force_no_traj:
        _set_eval_root_plan(
            model,
            timeline,
            old_plan,
            text=text,
            token_dt=tdt,
            root_refiner_runtime=root_refiner_runtime,
            root_5d_history=root_5d_history,
            replan_events=replan_events,
        )
    new_root_plan_active = False
    sr = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, _roots, fc = [], [], True
    for ci in range(total_tl):
        offset = ci - edit_commit
        if force_no_traj:
            ti = None
        elif condition_path == "rootplan_7d":
            if (
                not new_root_plan_active
                and offset >= ed + eb
                and timeline.has_exact_state(effective_commit)
            ):
                new_root_plan_active = _set_eval_root_plan(
                    model,
                    timeline,
                    new_plan,
                    text=text,
                    token_dt=tdt,
                    root_refiner_runtime=root_refiner_runtime,
                    root_5d_history=root_5d_history,
                    replan_events=replan_events,
                )
            ti = build_rootplan_stream_step_payload(
                model,
                timeline,
                history_length=hl,
                traj_horizon_tokens=hz,
                absolute_commit_index=ci,
            )
        elif condition_path == "legacy_xyz":
            cr = np.zeros(3, dtype=np.float32)
            cr[[0, 2]] = sr.r_pos_accum[[0, 2]].astype(np.float32)
            _reanchor = dict(current_commit=ci, current_root_xyz=cr,
                             horizon_tokens=hz, token_dt=tdt, reanchor_to_current_root=True)
            old_ft = sample_plan_future(old_plan, **_reanchor)
            if offset < ed:
                ft = old_ft
            elif offset < ed + eb and eb > 0:
                new_ft = sample_plan_future(new_plan, **_reanchor)
                w = smoothstep01(float(offset - ed) / eb)
                ft = blend_future_trajs(old_ft, new_ft, w)
            else:
                ft = sample_plan_future(new_plan, **_reanchor)
            ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0), "token_mask": torch.ones(1, hz)}
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        sp = build_stream_step_model_input(
            text,
            traj_input=ti)
        out = model.stream_generate_step(sp, first_chunk=fc)
        dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                 first_chunk=fc)[0].float().cpu().numpy())
        fc = False
        for frm in dec:
            sr.process_frame(frm)
            _roots.append(sr.r_pos_accum.copy())
            frame_idx = _append_eval_root_history(root_5d_history, frame_idx, sr)
        _append_eval_timeline_state(
            timeline,
            commit_idx=ci + 1,
            recovery=sr,
        )
        decs.append(dec)
    vae.clear_cache()
    pm = np.concatenate(decs, axis=0)[:total_tfs] if decs else np.zeros((0, 263))
    pr = np.array(_roots, dtype=np.float32)[:total_tfs] if _roots else np.zeros((0, 3))
    effective_frame = token_start_frame(effective_commit)
    if len(pr) > 0:
        anchor_frame = int(np.clip(effective_frame, 0, len(pr) - 1))
        new_anchor_xz = pr[anchor_frame, [0, 2]]
    else:
        new_anchor_xz = None
    target_t, target_pts = _build_turn_metric_target(
        old_times=plan_t,
        old_points_xyz=plan_pts,
        new_times=rot_t,
        new_points_xyz=rot_pts,
        target_frames=total_tfs,
        motion_fps=fps,
        edit_commit=edit_commit,
        delay_tokens=ed,
        blend_tokens=eb,
        token_dt=tdt,
        new_anchor_xz=new_anchor_xz,
    )
    return pm, pr, gr, target_t, target_pts, total_tfs


def _run_babel(model, vae, sample, device, *, hl, nds, hz, tdt, wpdt, fps, mode,
               condition_path="rootplan_7d", root_refiner_runtime=None,
               replan_events=None, force_no_traj=False):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gr_arr = sample["traj"].numpy()
    timeline = _new_eval_timeline()
    dur = (sample["feature_length"] - 1) / fps
    npt = max(2, int(round(dur / wpdt)) + 1)
    plan_pts = resample_polyline_by_arclength(gr_arr, npt)
    plan_t = assign_uniform_timestamps(npt, wpdt)
    segs = [StreamTextSegment(text=t, token_end=te)
            for t, te in zip(sample["text"], sample["token_text_end"])]
    tc = StreamTextRolloutController(segs)
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    _clear_model_traj_state(model)
    root_5d_history, frame_idx = [], 0
    active_root_refiner_text = None
    if condition_path == "rootplan_7d" and mode != "babel_no_traj" and not force_no_traj:
        initial_text = sample["text"][0] if sample["text"] else ""
        if _set_eval_root_plan(
            model,
            timeline,
            _slice_stream_plan_from_commit(
                plan_t,
                plan_pts,
                start_commit_index=0,
                token_dt=tdt,
                waypoint_dt=wpdt,
                version=0,
                source="bench_babel",
            ),
            text=initial_text,
            token_dt=tdt,
            root_refiner_runtime=root_refiner_runtime,
            root_5d_history=root_5d_history,
            replan_events=replan_events,
        ):
            active_root_refiner_text = initial_text
    sr = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, _roots, fc = [], [], True
    for ci in range(tl):
        txt = tc.get_text_for_commit_index(ci)
        if mode == "babel_no_traj" or force_no_traj:
            ti = None
        elif condition_path == "rootplan_7d":
            if (
                root_refiner_runtime is not None
                and txt != active_root_refiner_text
                and timeline.has_exact_state(ci)
            ):
                refreshed = _set_eval_root_plan(
                    model,
                    timeline,
                    _slice_stream_plan_from_commit(
                        plan_t,
                        plan_pts,
                        start_commit_index=ci,
                        token_dt=tdt,
                        waypoint_dt=wpdt,
                        version=ci,
                        source="bench_babel_text",
                    ),
                    text=txt,
                    token_dt=tdt,
                    root_refiner_runtime=root_refiner_runtime,
                    root_5d_history=root_5d_history,
                    replan_events=replan_events,
                )
                if refreshed:
                    active_root_refiner_text = txt
            ti = build_rootplan_stream_step_payload(
                model,
                timeline,
                history_length=hl,
                traj_horizon_tokens=hz,
                absolute_commit_index=ci,
            )
        elif condition_path == "legacy_xyz" and mode == "babel_timestamped":
            cr = np.zeros(3, dtype=np.float32)
            cr[[0, 2]] = sr.r_pos_accum[[0, 2]].astype(np.float32)
            qt = (float(ci) * tdt + np.arange(hz, dtype=np.float32) * tdt)
            gt_t = np.arange(len(gr_arr), dtype=np.float32) / 20.0
            ft = sample_timestamped_trajectory(gt_t, gr_arr, qt)
            anc = sample_timestamped_trajectory(gt_t, gr_arr, np.asarray([qt[0]], dtype=np.float32))[0]
            ft = cr + (ft - anc.astype(np.float32))
            ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0), "token_mask": torch.ones(1, hz)}
        elif condition_path == "legacy_xyz":
            cr = np.zeros(3, dtype=np.float32)
            cr[[0, 2]] = sr.r_pos_accum[[0, 2]].astype(np.float32)
            ft = sample_plan_future(StreamTrajectoryPlan(times=plan_t, points_xyz=plan_pts, start_commit_index=0, version=0, source="bench"), current_commit=ci, current_root_xyz=cr, horizon_tokens=hz, token_dt=tdt, reanchor_to_current_root=True)
            ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0), "token_mask": torch.ones(1, hz)}
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        sp = build_stream_step_model_input(txt, traj_input=ti)
        out = model.stream_generate_step(sp, first_chunk=fc)
        dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                 first_chunk=fc)[0].float().cpu().numpy())
        fc = False
        for frm in dec:
            sr.process_frame(frm)
            _roots.append(sr.r_pos_accum.copy())
            frame_idx = _append_eval_root_history(root_5d_history, frame_idx, sr)
        _append_eval_timeline_state(
            timeline,
            commit_idx=ci + 1,
            recovery=sr,
        )
        decs.append(dec)
    vae.clear_cache()
    pm = np.concatenate(decs, axis=0)[:tfs] if decs else np.zeros((0, 263))
    pr = np.array(_roots, dtype=np.float32)[:tfs] if _roots else np.zeros((0, 3))
    return pm, pr, gr, plan_t, plan_pts


def _render_traj_video(pred_root, target_root, out_path, title, *, split_tok=None):
    """Render animated XZ trajectory comparison video."""
    _n = min(len(pred_root), len(target_root))
    if _n <= 1:
        return
    import matplotlib
    matplotlib.use("Agg")
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
            pred_yaw = estimate_body_yaw(np.asarray(motion_263))
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
    p.add_argument("--output_dir", default="outputs/stream_benchmark")
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
    args = p.parse_args()

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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, run_id)
    suites_list = [s.strip() for s in args.suites.split(",")] if args.suites else None
    suite_tag = "_".join(suites_list) if suites_list else args.preset
    standard_media_dirs = standard_eval_artifact_dirs(
        args.output_dir,
        evaluator="Runtime",
        probe_tag=suite_tag,
        run_id=run_id,
        artifact_kinds=("plot", "video"),
        create=False,
    )
    vdir = os.path.join(out_root, "videos") if args.render_video else None
    pdir = os.path.join(out_root, "plots") if not args.no_save_plots else None
    standard_vdir = (
        str(ensure_dir(standard_media_dirs["video"])) if args.render_video else None
    )
    standard_pdir = (
        str(ensure_dir(standard_media_dirs["plot"]))
        if not args.no_save_plots
        else None
    )
    os.makedirs(out_root, exist_ok=True)
    if vdir: os.makedirs(vdir, exist_ok=True)
    if pdir: os.makedirs(pdir, exist_ok=True)

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
                pm, pr, gr, pt, pp = _run_real(
                    model, vae, sample, dev, mode=case.mode,
                    rotate_plan_deg=float(_rot), **kw)
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
                turn_activation_commit = turn_effective_commit + turn_blend_tokens
                pm, pr, gr, pt, pp, ttfs = _run_turn(
                    model, vae, sample, dev, mode=case.mode,
                    angle=ang, delay_tokens=int(dt_val),
                    blend_tokens=int(db_val), **kw)
                rec = build_plan_metrics(
                    pr, original_gt_root=gr, plan_times=pt, plan_points_xyz=pp,
                    target_frames=ttfs, motion_fps=args.motion_fps,
                    motion_263=pm, target_source="scheduled_turn_target",
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
                render_single_video(motion=_pm, save_path=mp4, dim=263, render_setting={})
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
    print(f"\nSummary: {report_dirs['legacy_root'] / 'summary.json'}")
    if all_recs:
        print(f"CSV: {report_dirs['legacy_root'] / 'summary.csv'}")
        print(f"Runtime metrics: {report_dirs['metrics'] / 'summary.json'}")


if __name__ == "__main__":
    main()
