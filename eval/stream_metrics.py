"""Unified stream benchmark metrics (Task 002).

Implements: ADE, FDE, path_arc, symmetric path_chamfer,
lateral_velocity_ratio, heading_path_error_deg, plus plan-target helpers.
"""

from __future__ import annotations

import numpy as np


# ── base trajectory metrics ────────────────────────────────────────────


def compute_ade(pred_root_xyz: np.ndarray, target_root_xyz: np.ndarray) -> float:
    """Mean XZ-plane Euclidean distance over the shorter sequence."""
    n = min(len(pred_root_xyz), len(target_root_xyz))
    if n == 0:
        return float("nan")
    p = pred_root_xyz[:n]
    t = target_root_xyz[:n]
    pxz = p[:, [0, 2]] if p.shape[1] >= 3 else p
    txz = t[:, [0, 2]] if t.shape[1] >= 3 else t
    return float(np.mean(np.linalg.norm(pxz - txz, axis=1)))


def compute_fde(pred_root_xyz: np.ndarray, target_root_xyz: np.ndarray) -> float:
    """XZ Euclidean distance at the last frame of the shorter sequence."""
    n = min(len(pred_root_xyz), len(target_root_xyz))
    if n == 0:
        return float("nan")
    p = pred_root_xyz[n - 1]
    t = target_root_xyz[n - 1]
    pxz = p[[0, 2]] if p.shape[0] >= 3 else p
    txz = t[[0, 2]] if t.shape[0] >= 3 else t
    return float(np.linalg.norm(pxz - txz))


# ── path shape metrics ─────────────────────────────────────────────────


def _resample_by_arc(path: np.ndarray, num_samples: int) -> np.ndarray:
    xz = path[:, [0, 2]] if path.shape[1] >= 3 else path
    segs = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    cum = np.concatenate([np.zeros(1), np.cumsum(segs)])
    total = cum[-1]
    if total < 1e-8 or num_samples <= 1:
        return np.tile(xz[:1], (num_samples, 1))
    q = np.linspace(0.0, total, num_samples)
    return np.column_stack([np.interp(q, cum, xz[:, d]) for d in range(xz.shape[1])])


def compute_path_arc(
    pred_root_xyz: np.ndarray, target_arc_xyz: np.ndarray,
    num_samples: int = 100,
) -> float:
    """Arc-length reparameterized ADE."""
    n = min(len(pred_root_xyz), len(target_arc_xyz))
    if n < 2:
        return compute_ade(pred_root_xyz, target_arc_xyz)
    rp = _resample_by_arc(pred_root_xyz[:n], num_samples)
    rt = _resample_by_arc(target_arc_xyz[:n], num_samples)
    return float(np.mean(np.linalg.norm(rp - rt, axis=1)))


def compute_path_chamfer(
    pred_root_xyz: np.ndarray, target_arc_xyz: np.ndarray,
    *,
    chamfer_type: str = "symmetric",
) -> float:
    """Path Chamfer distance (default symmetric)."""
    from scipy.spatial import cKDTree
    n = min(len(pred_root_xyz), len(target_arc_xyz))
    if n < 2:
        return compute_ade(pred_root_xyz, target_arc_xyz)
    p = pred_root_xyz[:n][:, [0, 2]] if pred_root_xyz.shape[1] >= 3 else pred_root_xyz[:n]
    t = target_arc_xyz[:n][:, [0, 2]] if target_arc_xyz.shape[1] >= 3 else target_arc_xyz[:n]
    tree_t = cKDTree(t)
    tree_p = cKDTree(p)
    d_pt, _ = tree_t.query(p)
    if chamfer_type == "symmetric":
        d_tp, _ = tree_p.query(t)
        return float(0.5 * (np.mean(d_pt) + np.mean(d_tp)))
    return float(np.mean(d_pt))


# ── motion quality metrics ─────────────────────────────────────────────


def estimate_body_yaw(motion_263: np.ndarray) -> np.ndarray:
    """Approximate body yaw from root rotation velocity in 263D format."""
    rot_vel_y = motion_263[:, 0]
    dt = 1.0 / 20.0
    return np.cumsum(rot_vel_y * dt)


def compute_lateral_velocity_ratio(motion_263: np.ndarray) -> float:
    """Mean |lateral| / mean speed in body-local frame."""
    if len(motion_263) < 3:
        return float("nan")
    from utils.motion_process import extract_root_trajectory_263_torch
    import torch as _torch
    root = extract_root_trajectory_263_torch(
        _torch.from_numpy(motion_263).float().unsqueeze(0)
    )[0].cpu().numpy()
    vel_xz = np.diff(root[:, [0, 2]], axis=0)
    yaw = estimate_body_yaw(motion_263)
    yaw_mid = 0.5 * (yaw[:-1] + yaw[1:])
    cos_y, sin_y = np.cos(yaw_mid), np.sin(yaw_mid)
    lateral = -sin_y * vel_xz[:, 0] + cos_y * vel_xz[:, 1]
    speed = np.linalg.norm(vel_xz, axis=1)
    valid = speed > 0.02
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(lateral[valid])) / (np.mean(speed[valid]) + 1e-8))


def compute_heading_path_error_deg(
    motion_263: np.ndarray, target_root_xyz: np.ndarray,
) -> float:
    """Mean |body-yaw - path-direction| in degrees."""
    n = min(len(motion_263), len(target_root_xyz))
    if n < 3:
        return float("nan")
    yaw = estimate_body_yaw(motion_263[:n])
    tvel = np.diff(target_root_xyz[:n, [0, 2]], axis=0)
    path_yaw = np.arctan2(tvel[:, 1], tvel[:, 0])
    yaw_mid = 0.5 * (yaw[:-1] + yaw[1:])
    speeds = np.linalg.norm(tvel, axis=1)
    valid = speeds > 0.02
    if valid.sum() == 0:
        return float("nan")
    err = np.abs(np.arctan2(
        np.sin(yaw_mid[valid] - path_yaw[valid]),
        np.cos(yaw_mid[valid] - path_yaw[valid]),
    ))
    return float(np.mean(err) * 180.0 / np.pi)


# ── plan-target construction ───────────────────────────────────────────


def compute_plan_targets(
    plan_times: np.ndarray,
    plan_points_xyz: np.ndarray,
    target_frames: int,
    motion_fps: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build time-aligned and arc-aligned plan targets.

    Returns ``(target_time, target_arc)``, each shape ``(target_frames, 3)``.
    """
    from utils.stream_traj import sample_plan_by_time
    frame_times = np.arange(target_frames, dtype=np.float32) / motion_fps
    target_time = sample_plan_by_time(
        np.asarray(plan_times, dtype=np.float32),
        np.asarray(plan_points_xyz, dtype=np.float32),
        frame_times,
    )
    target_arc = _resample_by_arc(
        np.asarray(plan_points_xyz, dtype=np.float32), target_frames,
    )
    if target_arc.shape[1] == 2:
        target_arc = np.c_[
            target_arc[:, 0],
            np.zeros(len(target_arc), dtype=np.float32),
            target_arc[:, 1],
        ]
    return target_time.astype(np.float32), target_arc.astype(np.float32)


def build_plan_metrics(
    pred_root_xyz: np.ndarray,
    *,
    original_gt_root: np.ndarray | None,
    plan_times: np.ndarray,
    plan_points_xyz: np.ndarray,
    target_frames: int,
    motion_fps: float = 20.0,
    motion_263: np.ndarray | None = None,
    chamfer_type: str = "symmetric",
    target_source: str = "sampled_plan_target",
) -> dict:
    """Compute a unified metrics dict for a single benchmark case."""
    target_time, target_arc = compute_plan_targets(
        plan_times, plan_points_xyz, target_frames, motion_fps,
    )
    n = min(len(pred_root_xyz), len(target_time))
    pred = pred_root_xyz[:n]
    tt = target_time[:n]
    ta = target_arc[:n]

    metrics = {
        "ADE": compute_ade(pred, tt),
        "FDE": compute_fde(pred, tt),
        "path_arc": compute_path_arc(pred, ta),
        "path_chamfer": compute_path_chamfer(pred, ta, chamfer_type=chamfer_type),
        "chamfer_type": chamfer_type,
        "target_source": target_source,
    }
    if original_gt_root is not None:
        og = original_gt_root[:n]
        metrics["ADE_vs_original_gt"] = compute_ade(pred, og)
        metrics["FDE_vs_original_gt"] = compute_fde(pred, og)
    if motion_263 is not None:
        metrics["lateral_velocity_ratio"] = compute_lateral_velocity_ratio(
            motion_263[:n],
        )
        metrics["heading_path_error_deg"] = compute_heading_path_error_deg(
            motion_263[:n], tt,
        )
    return metrics
