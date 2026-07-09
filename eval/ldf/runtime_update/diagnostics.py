"""Diagnostics for active-window runtime update replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from eval.ldf.experiments.artifacts import plot_7d_xz_heading
from utils.inference.timeline import RootFrameState, RootTimeline
from utils.local_frame import heading_dir_xz, uncanonicalize_7d


def yaw_from_7d(traj7: torch.Tensor) -> torch.Tensor:
    return torch.atan2(traj7[:, 4], traj7[:, 3])


def build_timeline_from_generated_traj7(
    generated_traj7: torch.Tensor,
    *,
    frames_per_token: int = 4,
) -> RootTimeline:
    generated = generated_traj7.detach().cpu().float()
    if generated.dim() != 2 or generated.shape[-1] < 5:
        raise ValueError(
            f"generated_traj7 must be [T,>=5], got {tuple(generated.shape)}"
        )
    yaw = yaw_from_7d(generated)
    timeline = RootTimeline(
        RootFrameState(
            commit_idx=0,
            world_xz=generated[0, [0, 2]].clone(),
            world_yaw=yaw[0].clone(),
            source="replay_generated",
        )
    )
    max_commit = max(1, int(np.ceil(float(generated.shape[0]) / float(frames_per_token))))
    for commit in range(1, max_commit + 1):
        frame = min(commit * int(frames_per_token) - 1, int(generated.shape[0]) - 1)
        timeline.append(
            RootFrameState(
                commit_idx=int(commit),
                world_xz=generated[frame, [0, 2]].clone(),
                world_yaw=yaw[frame].clone(),
                source="replay_generated",
            )
        )
    return timeline


def payload_local_to_world(payload: Mapping[str, object], timeline) -> torch.Tensor | None:
    if not payload or "traj_cond_7d_frame" not in payload:
        return None
    local = payload["traj_cond_7d_frame"]
    if isinstance(local, np.ndarray):
        local_t = torch.from_numpy(local).float()
    elif torch.is_tensor(local):
        local_t = local.detach().cpu().float()
    else:
        return None
    if local_t.dim() == 3:
        local_t = local_t[0]
    anchor_token = int(payload.get("body_anchor_abs_token", payload.get("traj_abs_start_token", 0)))
    anchor = timeline.at_commit(anchor_token)
    return uncanonicalize_7d(
        local_t.unsqueeze(0),
        anchor.world_xz.detach().cpu().float().unsqueeze(0),
        anchor.world_yaw.detach().cpu().float().reshape(1),
    )[0]


def trajectory_diagnostics(traj7: torch.Tensor) -> dict[str, float]:
    traj = traj7.detach().cpu().float()
    if traj.shape[0] < 2:
        return {
            "num_frames": float(traj.shape[0]),
            "min_speed": 0.0,
            "max_speed": 0.0,
            "min_heading_tangent_dot": 0.0,
        }
    xz = traj[:, [0, 2]]
    yaw = yaw_from_7d(traj)
    delta = xz[1:] - xz[:-1]
    speed = torch.linalg.norm(delta, dim=-1)
    heading = heading_dir_xz(yaw[1:])
    dot = (delta * heading).sum(-1) / speed.clamp_min(1e-8)
    moving = speed > 1e-5
    dot_moving = dot[moving] if bool(moving.any()) else dot
    return {
        "num_frames": float(traj.shape[0]),
        "min_speed": float(speed.min().item()),
        "max_speed": float(speed.max().item()),
        "mean_speed": float(speed.mean().item()),
        "min_heading_tangent_dot": float(dot_moving.min().item()),
        "mean_heading_tangent_dot": float(dot_moving.mean().item()),
    }




def validate_trajectory_diagnostics(
    traj7: torch.Tensor,
    *,
    reference_traj7: torch.Tensor | None = None,
    max_speed_scale: float = 3.0,
    max_abs_speed: float | None = None,
    min_heading_tangent_dot: float = -1e-4,
) -> tuple[bool, list[str]]:
    """Return whether a runtime condition segment is physically plausible.

    This is intentionally conservative and used as an offline gate before LDF
    generation. It catches the failure mode where a short bridge snaps from the
    generated pose back to a distant global route, producing impossible root
    speeds or heading/tangent reversal.
    """
    traj = traj7.detach().cpu().float()
    issues: list[str] = []
    if traj.dim() != 2 or traj.shape[-1] < 5:
        return False, [f"traj7 must be [T,>=5], got {tuple(traj.shape)}"]
    if int(traj.shape[0]) < 2:
        return True, issues

    xz = traj[:, [0, 2]]
    yaw = yaw_from_7d(traj)
    delta = xz[1:] - xz[:-1]
    speed = torch.linalg.norm(delta, dim=-1)
    moving = speed > 1e-5
    if bool(moving.any()):
        heading = heading_dir_xz(yaw[1:])
        dot = (delta * heading).sum(-1) / speed.clamp_min(1e-8)
        min_dot = float(dot[moving].min().item())
        if min_dot < float(min_heading_tangent_dot):
            issues.append(
                f"heading_tangent_dot below threshold: {min_dot:.6f} < {float(min_heading_tangent_dot):.6f}"
            )

    max_speed = float(speed.max().item())
    thresholds: list[float] = []
    if max_abs_speed is not None:
        thresholds.append(float(max_abs_speed))
    if reference_traj7 is not None and int(reference_traj7.shape[0]) >= 2:
        ref = reference_traj7.detach().cpu().float()
        ref_speed = torch.linalg.norm(ref[1:, [0, 2]] - ref[:-1, [0, 2]], dim=-1)
        ref_moving = ref_speed[ref_speed > 1e-5]
        if bool(ref_moving.numel() > 0):
            thresholds.append(float(ref_moving.mean().item()) * float(max_speed_scale))
    if thresholds:
        threshold = max(thresholds)
        if max_speed > threshold:
            issues.append(f"speed exceeds threshold: {max_speed:.6f} > {threshold:.6f}")

    return not issues, issues


def save_replay_outputs(
    out_dir: str | Path,
    *,
    npz_payload: dict[str, np.ndarray],
    summary: dict,
    plot_series: Mapping[str, torch.Tensor | np.ndarray],
    update_frames: list[int],
) -> dict[str, str]:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    npz_path = path / "active_window_replay.npz"
    np.savez(npz_path, **npz_payload)
    summary_path = path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_path = plot_7d_xz_heading(
        path / "composed_world_condition_xz_heading.png",
        plot_series,
        update_frames=update_frames,
        title="active-window replay: route / generated / composed",
    )
    return {
        "npz": str(npz_path),
        "summary": str(summary_path),
        "plot": str(plot_path),
    }


__all__ = [
    "build_timeline_from_generated_traj7",
    "payload_local_to_world",
    "save_replay_outputs",
    "trajectory_diagnostics",
    "validate_trajectory_diagnostics",
    "yaw_from_7d",
]
