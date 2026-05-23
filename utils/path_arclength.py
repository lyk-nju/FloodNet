"""Equal-arc-length path resampling for user-drawn raw paths.

References:
- docs/TODO.md §T_A_02 — full spec, unit tests 1-4.
- Used by T_A_04 RefinerDataset / web_demo to standardize variable-length user
  path input to a fixed `N_path=64` representation that the Refiner consumes.

Pipeline:
    raw_points_xz (Mx2) → simplify (Douglas-Peucker / RDP)
                       → arclength_resample (equal-arc-length to N points)
                       → ArcLengthPath

Numpy in / numpy out. No torch / no FloodNet-module deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArcLengthPath:
    """Equal-arc-length-resampled path.

    Attributes:
        points_xz:     [N_path, 2] resampled positions on the XZ ground plane.
        arc_s:         [N_path] cumulative arc length normalized to [0, 1]
                       (arc_s[0]=0, arc_s[-1]=1 for non-degenerate paths;
                       for degenerate paths all entries = 0).
        mask:          [N_path] 1 = valid sample, 0 = degenerate fallback.
                       Body / Refiner downstream consumers MUST honor this:
                       mask=0 ⇒ no positional/heading signal from that slot.
        total_length:  scalar, length in input units (m). 0 for degenerate inputs.
    """

    points_xz: np.ndarray
    arc_s: np.ndarray
    mask: np.ndarray
    total_length: float


# ---------------------------------------------------------------------------
# Douglas-Peucker simplification
# ---------------------------------------------------------------------------


def _point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from p to the line through a and b.

    If a == b the line is degenerate; return ||p - a||.
    """
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

    `raw_points_xz`: [M, 2]. Returns [K, 2] with K <= M; the returned points
    are a strict (order-preserving) subset of the input.

    `eps` is the perpendicular-distance tolerance in input units (m). Default
    1cm is small enough to preserve fine path detail while removing near-duplicate
    consecutive samples a user might generate by mouse-stutter.
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


def _degenerate_result(n_points: int, fallback_xz: np.ndarray | None = None) -> ArcLengthPath:
    """Build a mask-all-0 ArcLengthPath. `fallback_xz` (e.g. the single input
    point) is repeated for all output samples so downstream code sees a sane
    shape; mask=0 communicates the degeneracy.
    """
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

    `points_xz`: [M, 2] input control points (M ≥ 1).
    Returns `ArcLengthPath` with `points_xz: [n_points, 2]`, `arc_s` normalized
    to [0, 1], `mask` all True for non-degenerate inputs.

    Degenerate cases (mask all False, total_length=0):
      - M == 0 (empty)
      - M == 1
      - All input points identical (total_length below 1e-9)
    """
    pts = np.asarray(points_xz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_xz must be [M, 2], got shape {pts.shape}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    M = pts.shape[0]
    if M == 0:
        return _degenerate_result(n_points)
    if M == 1:
        return _degenerate_result(n_points, fallback_xz=pts[0])

    # Cumulative arc length along the input polyline.
    seg = pts[1:] - pts[:-1]                              # [M-1, 2]
    seg_len = np.linalg.norm(seg, axis=1)                 # [M-1]
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])     # [M]
    total = float(cum[-1])
    if total < 1e-9:
        return _degenerate_result(n_points, fallback_xz=pts[0])

    # Equal-arc-length sample positions s_i = i/(n_points-1) * total.
    s_targets = np.linspace(0.0, total, n_points, dtype=np.float64)

    # Interpolate xz at each s_target. np.interp handles per-axis linear interp.
    x = np.interp(s_targets, cum, pts[:, 0])
    z = np.interp(s_targets, cum, pts[:, 1])
    sampled = np.stack([x, z], axis=-1)                    # [n_points, 2]
    arc_s_norm = s_targets / total                          # [0, 1]

    return ArcLengthPath(
        points_xz=sampled,
        arc_s=arc_s_norm,
        mask=np.ones(n_points, dtype=bool),
        total_length=total,
    )


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def build_arclength_path(raw_points_xz: np.ndarray,
                          n_path: int = 64,
                          *,
                          simplify_eps: float = 0.01) -> ArcLengthPath:
    """Pipeline: raw → simplify (RDP) → arclength_resample → ArcLengthPath.

    Convenience wrapper. For finer control, call `simplify_path` and
    `arclength_resample` separately.
    """
    simplified = simplify_path(raw_points_xz, eps=simplify_eps)
    return arclength_resample(simplified, n_points=n_path)


__all__ = [
    "ArcLengthPath",
    "simplify_path",
    "arclength_resample",
    "build_arclength_path",
]
