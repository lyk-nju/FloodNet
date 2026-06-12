"""Unified stream benchmark metrics (Task 002).

Implements: ADE, FDE, path_arc, symmetric path_chamfer,
lateral_velocity_ratio, heading_path_error_deg, plus plan-target helpers.
"""

from __future__ import annotations

import numpy as np


def _angle_midpoints(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular midpoint for wrapped angles."""
    return np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))


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


def compute_yaw_error(pred_yaw: np.ndarray, target_yaw: np.ndarray) -> float:
    """Mean absolute wrapped yaw error in radians over the shorter sequence."""
    n = min(len(pred_yaw), len(target_yaw))
    if n == 0:
        return float("nan")
    diff = pred_yaw[:n] - target_yaw[:n]
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return float(np.mean(np.abs(wrapped)))


def compute_root_jitter(root_xyz: np.ndarray) -> float:
    """Mean XZ jerk magnitude, using third finite difference of root path."""
    if len(root_xyz) < 4:
        return float("nan")
    xz = root_xyz[:, [0, 2]] if root_xyz.shape[1] >= 3 else root_xyz
    jerk = np.diff(xz, n=3, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def build_stream_eval_summary(
    pred_root_xyz: np.ndarray,
    target_root_xyz: np.ndarray,
    *,
    pred_yaw: np.ndarray | None = None,
    target_yaw: np.ndarray | None = None,
) -> dict:
    """Minimal stream-eval summary with stable checkpoint-selection keys."""
    n = min(len(pred_root_xyz), len(target_root_xyz))
    summary = {
        "stream/root_ADE": compute_ade(pred_root_xyz, target_root_xyz),
        "stream/root_FDE": compute_fde(pred_root_xyz, target_root_xyz),
        "stream/jitter": compute_root_jitter(pred_root_xyz[:n]),
        "stream/num_frames": int(n),
    }
    if pred_yaw is not None and target_yaw is not None:
        summary["stream/yaw_error"] = compute_yaw_error(pred_yaw, target_yaw)
    return summary


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


def _as_root_7d(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 7:
        raise ValueError(f"{name} must have shape [T,7], got {arr.shape}")
    return arr


def _heading_yaw_from_7d(root_7d: np.ndarray) -> np.ndarray:
    if len(root_7d) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.arctan2(root_7d[:, 4], root_7d[:, 3]).astype(np.float32)


def _cross_track_distances(pred_xz: np.ndarray, target_xz: np.ndarray) -> np.ndarray:
    if len(pred_xz) == 0 or len(target_xz) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(target_xz) == 1:
        return np.linalg.norm(pred_xz - target_xz[0:1], axis=1).astype(np.float32)
    dists = []
    seg_start = target_xz[:-1]
    seg_end = target_xz[1:]
    seg_vec = seg_end - seg_start
    seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)
    for point in pred_xz:
        rel = point[None, :] - seg_start
        t = np.sum(rel * seg_vec, axis=1) / np.maximum(seg_len_sq, 1e-8)
        t = np.clip(t, 0.0, 1.0)
        proj = seg_start + t[:, None] * seg_vec
        dists.append(float(np.min(np.linalg.norm(proj - point[None, :], axis=1))))
    return np.asarray(dists, dtype=np.float32)


def compute_root_condition_diagnostics(
    gt_root_7d: np.ndarray,
    pred_root_7d: np.ndarray,
    *,
    gt_num_tokens: int,
    pred_num_tokens: int,
    frames_per_token: int = 4,
) -> dict:
    """Compare GT and RootRefiner 7D root conditions before LDF generation."""

    gt = _as_root_7d("gt_root_7d", gt_root_7d)
    pred = _as_root_7d("pred_root_7d", pred_root_7d)
    n = min(len(gt), len(pred))

    metrics = {
        "num_token_gt": int(gt_num_tokens),
        "num_token_pred": int(pred_num_tokens),
        "num_token_abs_error": int(abs(int(pred_num_tokens) - int(gt_num_tokens))),
        "duration_frame_abs_error": int(
            abs(int(pred_num_tokens) - int(gt_num_tokens)) * int(frames_per_token)
        ),
        "num_frames_gt": int(len(gt)),
        "num_frames_pred": int(len(pred)),
        "num_frames_compared": int(n),
    }
    if n == 0:
        metrics.update(
            {
                "xyz_ADE": float("nan"),
                "xyz_FDE": float("nan"),
                "x_AE_mean": float("nan"),
                "y_AE_mean": float("nan"),
                "y_AE_p90": float("nan"),
                "y_FDE": float("nan"),
                "z_AE_mean": float("nan"),
                "endpoint_xz_error": float("nan"),
                "heading_mae_deg": float("nan"),
                "yaw_delta_mae": float("nan"),
                "fwd_delta_mae": float("nan"),
                "path_arc_ADE": float("nan"),
                "path_chamfer": float("nan"),
                "cross_track_mean": float("nan"),
                "cross_track_max": float("nan"),
            }
        )
        return metrics

    gt_cmp = gt[:n]
    pred_cmp = pred[:n]
    gt_xz = gt_cmp[:, [0, 2]]
    pred_xz = pred_cmp[:, [0, 2]]
    xz_dist = np.linalg.norm(pred_xz - gt_xz, axis=1)
    y_abs = np.abs(pred_cmp[:, 1] - gt_cmp[:, 1])
    heading_err = np.arctan2(
        np.sin(_heading_yaw_from_7d(pred_cmp) - _heading_yaw_from_7d(gt_cmp)),
        np.cos(_heading_yaw_from_7d(pred_cmp) - _heading_yaw_from_7d(gt_cmp)),
    )
    cross_track = _cross_track_distances(pred_xz, gt_xz)

    metrics.update(
        {
            "xyz_ADE": float(np.mean(xz_dist)),
            "xyz_FDE": float(xz_dist[-1]),
            "x_AE_mean": float(np.mean(np.abs(pred_cmp[:, 0] - gt_cmp[:, 0]))),
            "y_AE_mean": float(np.mean(y_abs)),
            "y_AE_p90": float(np.percentile(y_abs, 90)),
            "y_FDE": float(y_abs[-1]),
            "z_AE_mean": float(np.mean(np.abs(pred_cmp[:, 2] - gt_cmp[:, 2]))),
            "endpoint_xz_error": float(xz_dist[-1]),
            "heading_mae_deg": float(np.mean(np.abs(heading_err)) * 180.0 / np.pi),
            "fwd_delta_mae": float(np.mean(np.abs(pred_cmp[:, 5] - gt_cmp[:, 5]))),
            "yaw_delta_mae": float(np.mean(np.abs(pred_cmp[:, 6] - gt_cmp[:, 6]))),
            "path_arc_ADE": compute_path_arc(pred_cmp[:, :3], gt_cmp[:, :3]),
            "path_chamfer": compute_path_chamfer(pred_cmp[:, :3], gt_cmp[:, :3]),
            "cross_track_mean": float(np.mean(cross_track)) if len(cross_track) else float("nan"),
            "cross_track_max": float(np.max(cross_track)) if len(cross_track) else float("nan"),
        }
    )
    return metrics


# ── motion quality metrics ─────────────────────────────────────────────


def estimate_body_yaw(motion_263: np.ndarray) -> np.ndarray:
    """Recover physical body yaw from 263D features using project convention."""
    if len(motion_263) == 0:
        return np.zeros(0, dtype=np.float32)
    import torch as _torch
    from utils.local_frame import root_quat_to_physical_yaw
    from utils.motion_process import recover_root_rot_pos

    feat = _torch.from_numpy(np.asarray(motion_263, dtype=np.float32)).unsqueeze(0)
    root_quat, _ = recover_root_rot_pos(feat)
    yaw = root_quat_to_physical_yaw(root_quat[0])
    return yaw.detach().cpu().numpy().astype(np.float32)


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
    yaw_mid = _angle_midpoints(yaw[:-1], yaw[1:])
    cos_y, sin_y = np.cos(yaw_mid), np.sin(yaw_mid)
    lateral = cos_y * vel_xz[:, 0] - sin_y * vel_xz[:, 1]
    speed = np.linalg.norm(vel_xz, axis=1)
    valid = speed > 0.02
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(lateral[valid])) / (np.mean(speed[valid]) + 1e-8))


def compute_heading_path_error_deg(
    motion_263: np.ndarray,
    target_root_xyz: np.ndarray,
    *,
    pred_yaw_offset: float = 0.0,
) -> float:
    """Mean |body-yaw - path-direction| in degrees."""
    n = min(len(motion_263), len(target_root_xyz))
    if n < 3:
        return float("nan")
    yaw = estimate_body_yaw(motion_263[:n]) + float(pred_yaw_offset)
    tvel = np.diff(target_root_xyz[:n, [0, 2]], axis=0)
    path_yaw = np.arctan2(tvel[:, 0], tvel[:, 1])
    yaw_mid = _angle_midpoints(yaw[:-1], yaw[1:])
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
    motion_yaw_offset: float = 0.0,
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
            motion_263[:n],
            tt,
            pred_yaw_offset=float(motion_yaw_offset),
        )
    return metrics
