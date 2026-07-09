#!/usr/bin/env python3
"""Replay active-window runtime update payload construction from debug NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.ldf.runtime_update.diagnostics import (
    build_timeline_from_generated_traj7,
    payload_local_to_world,
    save_replay_outputs,
    trajectory_diagnostics,
    validate_trajectory_diagnostics,
    yaw_from_7d,
)
from utils.inference.runtime_update.active_condition import (
    compose_active_window_segment,
    compose_active_window_world_condition,
)
from utils.inference.runtime_update.payload_builder import (
    build_world_condition_stream_payload,
)
from utils.inference.runtime_update.route_tracker import RouteProgressTracker
from utils.token_frame import token_end_frame, token_range_to_frame_slice


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--frames_per_token", type=int, default=4)
    parser.add_argument("--token_dt", type=float, default=0.20)
    parser.add_argument("--history_length", type=int, default=30)
    parser.add_argument("--horizon_tokens", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument("--lookahead_m", type=float, default=0.25)
    parser.add_argument("--bridge_frames", type=int, default=12)
    parser.add_argument("--max_abs_condition_speed", type=float, default=0.20)
    parser.add_argument("--max_speed_scale", type=float, default=3.0)
    parser.add_argument("--min_heading_tangent_dot", type=float, default=-0.05)
    parser.add_argument(
        "--no_fail_on_validation",
        action="store_true",
        help="Save diagnostics but return success even when active-window validation fails.",
    )
    return parser.parse_args()


def _first_int(npz, key: str, default: int = -1) -> int:
    if key not in npz:
        return int(default)
    value = np.asarray(npz[key]).reshape(-1)
    if value.size == 0:
        return int(default)
    return int(value[0])


def _extract_old_payload(npz, payload_idx: int) -> dict | None:
    name = f"payload_{payload_idx:02d}"
    if f"{name}_traj_cond_7d_frame" not in npz:
        return None
    return {
        "traj_abs_start_token": _first_int(npz, f"{name}_traj_abs_start_token"),
        "body_anchor_abs_token": _first_int(npz, f"{name}_body_anchor_abs_token"),
        "traj_num_tokens": _first_int(npz, f"{name}_traj_num_tokens"),
        "traj_cond_7d_frame": np.asarray(npz[f"{name}_traj_cond_7d_frame"]),
        "traj_cond_frame_mask": np.asarray(npz[f"{name}_traj_cond_frame_mask"]),
    }


def _tensor_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().float().numpy().astype(np.float32)


def main() -> int:
    args = _parse_args()
    npz_path = Path(args.input_npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    route = torch.from_numpy(np.asarray(data["offline_route_traj7"])).float()
    generated = torch.from_numpy(np.asarray(data["generated_traj7"])).float()
    update_commits = [int(v) for v in np.asarray(data["update_commits"]).reshape(-1)]
    switch_frames = [int(v) for v in np.asarray(data["switch_frames"]).reshape(-1)]
    update_frames = [int(v) for v in np.asarray(data["update_frames"]).reshape(-1)]
    payload_commits = [int(v) for v in np.asarray(data["payload_commits"]).reshape(-1)]

    timeline = build_timeline_from_generated_traj7(
        generated,
        frames_per_token=int(args.frames_per_token),
    )
    tracker = RouteProgressTracker(route, lookahead_m=float(args.lookahead_m))
    active_plans = []
    segments = []
    segment_records = []
    target_frames = int(route.shape[0])
    yaw = yaw_from_7d(generated)
    composed = route.clone()
    validation_issues: list[str] = []

    for update_idx, update_commit in enumerate(update_commits):
        switch_frame = (
            switch_frames[update_idx]
            if update_idx < len(switch_frames)
            else min(update_commit * int(args.frames_per_token), target_frames - 1)
        )
        segment_end = (
            update_frames[update_idx + 1]
            if update_idx + 1 < len(update_frames)
            else target_frames - 1
        )
        anchor_frame = max(0, min(int(switch_frame), int(generated.shape[0]) - 1))
        anchor_state = timeline.at_commit(int(update_commit))
        anchor_state = type(anchor_state)(
            commit_idx=int(update_commit),
            world_xz=generated[anchor_frame, [0, 2]].clone(),
            world_yaw=yaw[anchor_frame].clone(),
            source="active_window_replay_anchor",
        )
        segment = compose_active_window_segment(
            route,
            generated,
            current_frame=anchor_frame,
            current_yaw=anchor_state.world_yaw,
            target_end_frame=int(target_frames - 1),
            lookahead_m=float(args.lookahead_m),
            bridge_frames=int(args.bridge_frames),
            min_route_index=int(switch_frame),
            tracker=tracker,
        )
        segments.append(segment)
        composed = compose_active_window_world_condition(
            composed,
            generated[: anchor_frame + 1],
            segment,
            current_frame=anchor_frame,
        )
        active_plans.append((int(update_commit), composed.clone()))
        reference = route[
            int(segment.route_index) : min(
                int(route.shape[0]),
                int(segment.route_index) + int(segment.segment_traj7.shape[0]),
            )
        ]
        validation_ok, segment_issues = validate_trajectory_diagnostics(
            segment.segment_traj7,
            reference_traj7=reference if int(reference.shape[0]) >= 2 else None,
            max_speed_scale=float(args.max_speed_scale),
            max_abs_speed=float(args.max_abs_condition_speed),
            min_heading_tangent_dot=float(args.min_heading_tangent_dot),
        )
        for issue in segment_issues:
            validation_issues.append(f"update {update_idx}: {issue}")
        segment_records.append(
            {
                "update_idx": int(update_idx),
                "update_commit": int(update_commit),
                "switch_frame": int(switch_frame),
                "anchor_frame": int(anchor_frame),
                "segment_end": int(segment_end),
                "route_index": int(segment.route_index),
                "future_index": int(segment.future_index),
                "bridge_frames": int(segment.bridge_frames),
                "diagnostics": trajectory_diagnostics(segment.segment_traj7),
                "validation_ok": bool(validation_ok),
                "validation_issues": list(segment_issues),
            }
        )

    payload_tracker = RouteProgressTracker(route, lookahead_m=float(args.lookahead_m))
    new_payloads: list[dict | None] = []
    old_world_payloads = []
    new_world_payloads = []
    payload_debug_records: list[dict] = []
    for payload_idx, commit in enumerate(payload_commits):
        active = None
        for update_commit, plan in active_plans:
            if int(commit) >= int(update_commit):
                active = plan
            else:
                break
        if active is None:
            new_payloads.append(None)
            payload_debug_records.append({"payload_idx": int(payload_idx), "has_payload": False})
            continue
        generated_prefix_frames = min(
            int(generated.shape[0]),
            token_end_frame(int(commit) - 1, int(args.frames_per_token)) + 1,
        )
        payload_generated_history = generated[:generated_prefix_frames]
        current_payload_frame = max(0, int(payload_generated_history.shape[0]) - 1)
        payload_segment = compose_active_window_segment(
            route,
            payload_generated_history,
            current_frame=current_payload_frame,
            current_yaw=yaw_from_7d(payload_generated_history[current_payload_frame : current_payload_frame + 1])[0],
            target_end_frame=int(target_frames - 1),
            lookahead_m=float(args.lookahead_m),
            bridge_frames=int(args.bridge_frames),
            min_route_index=current_payload_frame,
            tracker=payload_tracker,
        )
        condition_for_payload = route.clone()
        payload_patch_end = min(
            int(condition_for_payload.shape[0]),
            current_payload_frame + int(payload_segment.segment_traj7.shape[0]),
        )
        if payload_patch_end > current_payload_frame:
            condition_for_payload[current_payload_frame:payload_patch_end] = payload_segment.segment_traj7[
                : payload_patch_end - current_payload_frame
            ]
        payload = build_world_condition_stream_payload(
            condition_for_payload,
            timeline,
            local_commit_index=int(commit),
            absolute_commit_index=int(commit),
            chunk_size=int(args.chunk_size),
            history_length=int(args.history_length),
            traj_horizon_tokens=int(args.horizon_tokens),
            frames_per_token=int(args.frames_per_token),
            generated_history_traj7=payload_generated_history,
        )
        new_payloads.append(payload)
        old_payload = _extract_old_payload(data, payload_idx)
        old_world = payload_local_to_world(old_payload, timeline) if old_payload else None
        new_world = payload_local_to_world(payload, timeline) if payload else None
        if old_world is not None:
            old_world_payloads.append(old_world)
        payload_validation_ok = None
        payload_validation_issues: list[str] = []
        payload_diag = None
        future_payload_diag = None
        if new_world is not None:
            new_world_payloads.append(new_world)
            payload_diag = trajectory_diagnostics(new_world)
            payload_abs_start = int(payload.get("traj_abs_start_token", 0)) if payload else 0
            payload_num_tokens = int(payload.get("traj_num_tokens", 0)) if payload else 0
            payload_frame_slice = token_range_to_frame_slice(
                payload_abs_start,
                payload_num_tokens,
                int(args.frames_per_token),
            )
            payload_frame_start = int(payload_frame_slice.start)
            payload_current_frame = token_end_frame(
                int(commit) - 1,
                int(args.frames_per_token),
            )
            future_local_start = max(0, min(int(new_world.shape[0]) - 1, payload_current_frame - payload_frame_start))
            future_world = new_world[future_local_start:]
            future_payload_diag = trajectory_diagnostics(future_world)
            payload_validation_ok, payload_validation_issues = validate_trajectory_diagnostics(
                future_world,
                max_abs_speed=float(args.max_abs_condition_speed),
                min_heading_tangent_dot=float(args.min_heading_tangent_dot),
            )
            for issue in payload_validation_issues:
                validation_issues.append(f"payload {payload_idx}: {issue}")
        payload_debug_records.append(
            {
                "payload_idx": int(payload_idx),
                "commit": int(commit),
                "has_payload": payload is not None,
                "world_diagnostics": payload_diag,
                "future_world_diagnostics": future_payload_diag,
                "validation_ok": payload_validation_ok,
                "validation_issues": list(payload_validation_issues),
            }
        )

    save_payload = {
        "offline_route_traj7": _tensor_np(route),
        "generated_traj7": _tensor_np(generated),
        "composed_world_condition_traj7": _tensor_np(composed),
        "update_commits": np.asarray(update_commits, dtype=np.int64),
        "switch_frames": np.asarray(switch_frames, dtype=np.int64),
        "payload_commits": np.asarray(payload_commits, dtype=np.int64),
    }
    for idx, segment in enumerate(segments):
        save_payload[f"active_segment_{idx:02d}_traj7"] = _tensor_np(segment.segment_traj7)
    payload_records = []
    for idx, payload in enumerate(new_payloads):
        debug_record = payload_debug_records[idx] if idx < len(payload_debug_records) else {"payload_idx": int(idx)}
        if payload is None:
            payload_records.append(dict(debug_record, has_payload=False))
            continue
        name = f"active_payload_{idx:02d}"
        save_payload[f"{name}_traj_cond_7d_frame"] = _tensor_np(payload["traj_cond_7d_frame"])
        save_payload[f"{name}_traj_cond_frame_mask"] = (
            payload["traj_cond_frame_mask"].detach().cpu().float().numpy().astype(np.float32)
        )
        save_payload[f"{name}_traj_abs_start_token"] = np.asarray(
            [int(payload.get("traj_abs_start_token", -1))],
            dtype=np.int64,
        )
        save_payload[f"{name}_body_anchor_abs_token"] = np.asarray(
            [int(payload.get("body_anchor_abs_token", -1))],
            dtype=np.int64,
        )
        payload_records.append(
            {
                "payload_idx": int(idx),
                "commit": int(payload_commits[idx]),
                "has_payload": True,
                "traj_abs_start_token": int(payload.get("traj_abs_start_token", -1)),
                "body_anchor_abs_token": int(payload.get("body_anchor_abs_token", -1)),
                "traj_num_tokens": int(payload.get("traj_num_tokens", -1)),
                "num_subpayloads": int(len(payload.get("traj_substep_payloads", []))),
                "world_diagnostics": debug_record.get("world_diagnostics"),
                "future_world_diagnostics": debug_record.get("future_world_diagnostics"),
                "validation_ok": debug_record.get("validation_ok"),
                "validation_issues": debug_record.get("validation_issues", []),
            }
        )

    summary = {
        "input_npz": str(npz_path),
        "frames_per_token": int(args.frames_per_token),
        "history_length": int(args.history_length),
        "horizon_tokens": int(args.horizon_tokens),
        "chunk_size": int(args.chunk_size),
        "lookahead_m": float(args.lookahead_m),
        "bridge_frames": int(args.bridge_frames),
        "segments": segment_records,
        "payloads": payload_records,
        "composed_diagnostics": trajectory_diagnostics(composed),
        "validation_ok": not validation_issues,
        "validation_issues": validation_issues,
        "validation_thresholds": {
            "max_abs_condition_speed": float(args.max_abs_condition_speed),
            "max_speed_scale": float(args.max_speed_scale),
            "min_heading_tangent_dot": float(args.min_heading_tangent_dot),
        },
    }
    plot_series = {
        "offline_route": route,
        "generated": generated,
        "active_composed": composed,
    }
    if old_world_payloads:
        plot_series["old_payload0_world"] = old_world_payloads[0]
    if new_world_payloads:
        plot_series["active_payload0_world"] = new_world_payloads[0]
    outputs = save_replay_outputs(
        args.out_dir,
        npz_payload=save_payload,
        summary=summary,
        plot_series=plot_series,
        update_frames=switch_frames,
    )
    summary["outputs"] = outputs
    Path(outputs["summary"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_issues and not bool(args.no_fail_on_validation):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
