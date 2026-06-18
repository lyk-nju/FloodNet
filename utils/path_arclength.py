"""Equal-arc-length path resampling for user-drawn paths.

Pipeline:
    raw_points_xz -> simplify_path -> arclength_resample -> ArcLengthPath

Numpy in, numpy out. No torch or project-level dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArcLengthPath:
    """Equal-arc-length-resampled path.

    Attributes:
        points_xz: [N_path, 2] resampled XZ points.
        arc_s: [N_path] cumulative arc length scaled to [0, 1].
        mask: [N_path] bool mask; all False for degenerate paths.
        total_length: path length in input units; 0 for degenerate paths.
    """

    points_xz: np.ndarray
    arc_s: np.ndarray
    mask: np.ndarray
    total_length: float


# ---------------------------------------------------------------------------
# Douglas-Peucker simplification
# ---------------------------------------------------------------------------


def _point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from `p` to the line through `a` and `b`."""
    ab = b - a
    ab_norm_sq = float(ab @ ab)
    if ab_norm_sq < 1e-20:
        return float(np.linalg.norm(p - a))
    # 2D cross product magnitude / |ab|
    cross = ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])
    return abs(cross) / np.sqrt(ab_norm_sq)


def _rdp_indices(points: np.ndarray, eps: float) -> list[int]:
    """Return sorted indices into `points` retained after RDP simplification."""
    n = len(points)
    if n <= 2:
        return list(range(n))
    keep = [False] * n
    keep[0] = True
    keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        max_d = -1.0
        max_idx = -1
        for k in range(i + 1, j):
            d = _point_line_distance(points[k], points[i], points[j])
            if d > max_d:
                max_d = d
                max_idx = k
        if max_d > eps:
            keep[max_idx] = True
            stack.append((i, max_idx))
            stack.append((max_idx, j))
    return [k for k in range(n) if keep[k]]


def simplify_path(raw_points_xz: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Douglas-Peucker simplification of an xz path.

    `raw_points_xz`: [M, 2]. Returns [K, 2] with K <= M. The returned points
    are a strict (order-preserving) subset of the input.
    """
    pts = np.asarray(raw_points_xz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"raw_points_xz must be [M, 2], got shape {pts.shape}")
    if pts.shape[0] <= 2:
        return pts.astype(np.float64).copy()
    idx = _rdp_indices(pts, eps=eps)
    return pts[idx]


# ---------------------------------------------------------------------------
# Equal-arc-length resampling
# ---------------------------------------------------------------------------


def _degenerate_result(
    n_points: int,
    fallback_xz: np.ndarray | None = None,
) -> ArcLengthPath:
    """Build a mask-all-False ArcLengthPath with a repeated fallback point."""
    if fallback_xz is None:
        fallback = np.zeros(2, dtype=np.float64)
    else:
        fallback = np.asarray(fallback_xz, dtype=np.float64).reshape(2)
    points = np.tile(fallback, (n_points, 1))
    return ArcLengthPath(
        points_xz=points,
        arc_s=np.zeros(n_points, dtype=np.float64),
        mask=np.zeros(n_points, dtype=bool),
        total_length=0.0,
    )


def arclength_resample(points_xz: np.ndarray, n_points: int = 64) -> ArcLengthPath:
    """Resample a polyline to `n_points` equal-arc-length samples.

    Degenerate paths return a repeated fallback point, all-False mask, and
    `total_length=0`.
    """
    pts = np.asarray(points_xz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_xz must be [M, 2], got shape {pts.shape}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    num_input_points = pts.shape[0]
    if num_input_points == 0:
        return _degenerate_result(n_points)
    if num_input_points == 1:
        return _degenerate_result(n_points, fallback_xz=pts[0])

    segments = pts[1:] - pts[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative_lengths[-1])
    if total_length < 1e-9:
        return _degenerate_result(n_points, fallback_xz=pts[0])

    sample_lengths = np.linspace(0.0, total_length, n_points, dtype=np.float64)

    x = np.interp(sample_lengths, cumulative_lengths, pts[:, 0])
    z = np.interp(sample_lengths, cumulative_lengths, pts[:, 1])
    sampled = np.stack([x, z], axis=-1)
    arc_s = sample_lengths / total_length

    return ArcLengthPath(
        points_xz=sampled,
        arc_s=arc_s,
        mask=np.ones(n_points, dtype=bool),
        total_length=total_length,
    )


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def build_arclength_path(
    raw_points_xz: np.ndarray,
    n_path: int = 64,
    *,
    simplify_eps: float = 0.01,
) -> ArcLengthPath:
    """Simplify and resample raw XZ points into an ArcLengthPath."""
    simplified = simplify_path(raw_points_xz, eps=simplify_eps)
    return arclength_resample(simplified, n_points=n_path)


__all__ = [
    "ArcLengthPath",
    "simplify_path",
    "arclength_resample",
    "build_arclength_path",
]
