"""Streaming trajectory geometry helpers shared by web_demo and eval.

Extracted from ``web_demo/model_manager.py`` so that metrics can construct
trajectory conditioning with the same logic as the live demo.
"""

from __future__ import annotations

import numpy as np


def project_point_to_polyline(
    point_xyz: np.ndarray, waypoints_xyz: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """Project a 3D point onto the XZ plane of a polyline.

    Args:
        point_xyz: (3,) world-space point.
        waypoints_xyz: (N, 3) polyline vertices.

    Returns:
        (projected_xyz, segment_index, t_parameter).
    """
    point_xz = point_xyz[[0, 2]]
    if len(waypoints_xyz) == 1:
        return waypoints_xyz[0].copy(), 0, 1.0

    best_dist = None
    best_proj = None
    best_seg = 0
    best_t = 0.0
    for seg_idx in range(len(waypoints_xyz) - 1):
        a = waypoints_xyz[seg_idx]
        b = waypoints_xyz[seg_idx + 1]
        a_xz = a[[0, 2]]
        b_xz = b[[0, 2]]
        ab = b_xz - a_xz
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq <= 1e-8:
            t = 0.0
            proj = a.copy()
        else:
            t = float(np.clip(np.dot(point_xz - a_xz, ab) / ab_len_sq, 0.0, 1.0))
            proj = a + (b - a) * t
        dist = float(np.linalg.norm(point_xz - proj[[0, 2]]))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_proj = proj
            best_seg = seg_idx
            best_t = t
    return best_proj, best_seg, best_t


def dedupe_polyline(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Remove consecutive XZ-duplicate vertices from a polyline."""
    if len(points) <= 1:
        return points
    keep = [0]
    for idx in range(1, len(points)):
        if np.linalg.norm(points[idx, [0, 2]] - points[keep[-1], [0, 2]]) > eps:
            keep.append(idx)
    return points[keep]


def build_remaining_polyline(
    root_xyz: np.ndarray, waypoints_xyz: np.ndarray
) -> np.ndarray:
    """Build a polyline from current root position to the end of the waypoints.

    Projects *root_xyz* onto *waypoints_xyz* and returns the concatenated path
    ``[root_xyz, projected, remaining_waypoints]`` with duplicates removed.
    """
    projected, seg_idx, seg_t = project_point_to_polyline(root_xyz, waypoints_xyz)
    suffix = [projected.astype(np.float32)]
    if len(waypoints_xyz) > 1:
        if seg_t < 1.0 - 1e-6:
            suffix.append(waypoints_xyz[seg_idx + 1].astype(np.float32))
            suffix.extend(waypoints_xyz[seg_idx + 2:].astype(np.float32))
        else:
            suffix.extend(waypoints_xyz[seg_idx + 1:].astype(np.float32))
    path = np.vstack(
        [root_xyz.astype(np.float32), np.asarray(suffix, dtype=np.float32)]
    )
    return dedupe_polyline(path)


def build_projected_suffix_polyline(
    root_xyz: np.ndarray, waypoints_xyz: np.ndarray
) -> np.ndarray:
    """Build the remaining path from the projection of ``root_xyz``.

    Unlike :func:`build_remaining_polyline`, this does not insert the current
    root as the first vertex.  It preserves along-path progress and leaves
    lateral recovery to the caller.
    """
    projected, seg_idx, seg_t = project_point_to_polyline(root_xyz, waypoints_xyz)
    suffix = [projected.astype(np.float32)]
    if len(waypoints_xyz) > 1:
        if seg_t < 1.0 - 1e-6:
            suffix.append(waypoints_xyz[seg_idx + 1].astype(np.float32))
            suffix.extend(waypoints_xyz[seg_idx + 2:].astype(np.float32))
        else:
            suffix.extend(waypoints_xyz[seg_idx + 1:].astype(np.float32))
    return dedupe_polyline(np.asarray(suffix, dtype=np.float32))


def build_recovery_future_traj(
    root_xyz: np.ndarray,
    waypoints_xyz: np.ndarray,
    num_tokens: int,
    token_step: float,
    *,
    recovery_tokens: int = 6,
) -> np.ndarray:
    """Build future targets that compensate closed-loop root drift.

    The existing web-demo path first samples ``current_root -> projected_path``.
    That can consume near-term targets on lateral return and reduce forward
    progress.  This helper instead samples along the projected suffix path, then
    blends targets from the current root to the suffix over ``recovery_tokens``.

    The first token stays at the current root, matching current streaming
    semantics where the trajectory suffix is written at ``commit_index``.
    """
    if num_tokens <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    root = np.asarray(root_xyz, dtype=np.float32).reshape(3)
    suffix = build_projected_suffix_polyline(root, waypoints_xyz)
    path_targets = resample_polyline(suffix, num_tokens, token_step)

    denom = float(max(1, int(recovery_tokens)))
    alpha = np.minimum(1.0, np.arange(num_tokens, dtype=np.float32) / denom)
    alpha = alpha[:, None]
    return ((1.0 - alpha) * root[None, :] + alpha * path_targets).astype(np.float32)


def resample_polyline(
    points_xyz: np.ndarray, num_tokens: int, token_step: float
) -> np.ndarray:
    """Resample a polyline at uniform *token_step* intervals.

    Args:
        points_xyz: (N, 3) polyline vertices.
        num_tokens: number of output tokens.
        token_step: spacing between samples (world units).

    Returns:
        (num_tokens, 3) regularly-sampled positions.
    """
    if num_tokens <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(points_xyz) == 0:
        return np.zeros((num_tokens, 3), dtype=np.float32)
    if len(points_xyz) == 1:
        return np.repeat(points_xyz.astype(np.float32), num_tokens, axis=0)

    seg_lens = np.linalg.norm(np.diff(points_xyz[:, [0, 2]], axis=0), axis=1)
    cum = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(seg_lens).astype(np.float32)]
    )
    total_len = float(cum[-1])
    if total_len <= 1e-6:
        return np.repeat(points_xyz[:1].astype(np.float32), num_tokens, axis=0)

    sample_d = np.arange(num_tokens, dtype=np.float32) * float(token_step)
    sample_d = np.clip(sample_d, 0.0, total_len)
    out = np.empty((num_tokens, 3), dtype=np.float32)
    for dim in range(3):
        out[:, dim] = np.interp(sample_d, cum, points_xyz[:, dim]).astype(np.float32)
    return out


def sample_timestamped_trajectory(
    times: np.ndarray,
    points_xyz: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Sample a time-parameterized trajectory by linear interpolation.

    Args:
        times: (N,) monotonically increasing seconds.
        points_xyz: (N, 3) world-space trajectory points.
        query_times: (M,) seconds to sample.

    Returns:
        (M, 3) world-space points. Queries outside the time range clamp to the
        endpoint values, matching ``np.interp`` semantics.
    """
    query_times = np.asarray(query_times, dtype=np.float32)
    if query_times.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    times = np.asarray(times, dtype=np.float32).reshape(-1)
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {points.shape}")
    if len(times) != len(points):
        raise ValueError(
            f"times/points length mismatch: {len(times)} vs {len(points)}"
        )
    if len(times) == 0:
        return np.zeros((len(query_times), 3), dtype=np.float32)
    if len(times) == 1:
        return np.repeat(points[:1], len(query_times), axis=0).astype(np.float32)

    order = np.argsort(times)
    times = times[order]
    points = points[order]

    # Drop duplicate timestamps; np.interp expects a strictly increasing x-axis.
    keep = np.ones(len(times), dtype=bool)
    keep[1:] = np.diff(times) > 1e-8
    times = times[keep]
    points = points[keep]
    if len(times) == 1:
        return np.repeat(points[:1], len(query_times), axis=0).astype(np.float32)

    out = np.empty((len(query_times), 3), dtype=np.float32)
    for dim in range(3):
        out[:, dim] = np.interp(query_times, times, points[:, dim]).astype(np.float32)
    return out


def assign_times_by_arclength(
    waypoints_xyz: np.ndarray,
    total_duration: float,
) -> np.ndarray:
    """Assign timestamps to waypoints by XZ arclength fraction.

    This models a user giving spatial waypoints plus total duration, but no
    per-waypoint timestamps.  Speed is constant along the waypoint polyline.
    """
    points = np.asarray(waypoints_xyz, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(points) == 1:
        return np.zeros((1,), dtype=np.float32)

    seg_lens = np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1)
    cum = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(seg_lens).astype(np.float32)]
    )
    total_len = float(cum[-1])
    if total_len <= 1e-6:
        return np.linspace(
            0.0, float(total_duration), len(points), dtype=np.float32
        )
    return (cum / total_len * float(total_duration)).astype(np.float32)


def estimate_token_step_distance(
    root_xz_history: list,
    *,
    default: float = 0.25,
    min_step: float = 0.05,
    max_step: float = 1.50,
) -> float:
    """Estimate token step distance from recent root velocity.

    Mirrors web_demo ``ModelManager._estimate_token_step_distance``: takes the
    median frame-to-frame displacement over the last 12 frames and multiplies by
    4 (causal VAE token temporal factor).

    Args:
        root_xz_history: list of (2,) or (3,) root position snapshots (XZ only used).
        default: fallback when history is too short.
        min_step, max_step: clamp bounds.

    Returns:
        Estimated world-space distance per token.
    """
    if len(root_xz_history) < 5:
        return default
    history = np.asarray(root_xz_history, dtype=np.float32)
    if history.ndim == 2 and history.shape[1] >= 2:
        history = history[:, [0, 2]] if history.shape[1] >= 3 else history
    frame_steps = np.linalg.norm(np.diff(history, axis=0), axis=1)
    frame_steps = frame_steps[np.isfinite(frame_steps)]
    if frame_steps.size == 0:
        return default
    recent = frame_steps[-min(12, frame_steps.size):]
    token_step = float(np.median(recent) * 4.0)
    return float(np.clip(token_step, min_step, max_step))
