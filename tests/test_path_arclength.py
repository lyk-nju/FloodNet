"""Unit tests for utils/path_arclength.py (T_A_02).

Covers the 4 tests listed in docs/TODO.md §T_A_02 plus a few edge cases.
"""

from __future__ import annotations

import numpy as np

from utils.path_arclength import (
    ArcLengthPath,
    arclength_resample,
    build_arclength_path,
    simplify_path,
)


def test_T01_straight_line_100_to_64_uniform_arc_s_and_collinear():
    """Straight line from (0,0) to (10,0) sampled at 100 points → resample to 64.
    arc_s should be evenly spaced (equal spacing), points stay collinear.
    """
    M = 100
    pts = np.zeros((M, 2), dtype=np.float64)
    pts[:, 0] = np.linspace(0.0, 10.0, M)   # vary x only, z = 0
    res = arclength_resample(pts, n_points=64)
    assert res.mask.all()
    assert abs(res.total_length - 10.0) < 1e-6
    # arc_s strictly increasing and approximately linear in index
    expected_arc_s = np.linspace(0.0, 1.0, 64)
    assert np.allclose(res.arc_s, expected_arc_s, atol=1e-9)
    # All resampled points lie on z=0 (collinear with input line)
    assert np.allclose(res.points_xz[:, 1], 0.0, atol=1e-9)
    # x coordinates are equal-spaced from 0 to 10
    expected_x = np.linspace(0.0, 10.0, 64)
    assert np.allclose(res.points_xz[:, 0], expected_x, atol=1e-6)


def test_T02_circular_arc_50_to_64_uniform_arc_length():
    """Half-circle of radius 1 sampled at 50 angles → resample to 64 points;
    consecutive samples should have roughly equal arc lengths.
    """
    angles = np.linspace(0.0, np.pi, 50, dtype=np.float64)
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=-1)   # [50, 2] on XZ
    res = arclength_resample(pts, n_points=64)
    assert res.mask.all()
    # Total length should be approximately π (half circle of radius 1).
    # Input is a polyline approximation so it's slightly less than π.
    assert abs(res.total_length - np.pi) < 1e-2, f"total_length={res.total_length}"
    # Segment lengths between consecutive resampled points should be ~equal.
    seg = np.linalg.norm(np.diff(res.points_xz, axis=0), axis=1)   # [63]
    target_seg = res.total_length / 63.0
    rel_err = np.abs(seg - target_seg) / target_seg
    assert rel_err.max() < 1e-2, f"max rel_err in segment lengths = {rel_err.max()}"


def test_T03_degenerate_single_point_input():
    """Single-point input: mask all 0, total_length 0, but output shape preserved."""
    pts = np.array([[1.5, -0.3]], dtype=np.float64)
    res = arclength_resample(pts, n_points=64)
    assert isinstance(res, ArcLengthPath)
    assert res.points_xz.shape == (64, 2)
    assert res.total_length == 0.0
    assert not res.mask.any()
    assert (res.arc_s == 0.0).all()
    # All resampled points should be the same fallback position.
    assert np.allclose(res.points_xz, np.array([1.5, -0.3]))


def test_T04_simplify_returns_monotonic_subset_of_input():
    """RDP output points should appear in the input array in the same order
    (i.e. a strict subsequence). Validates by index-search.
    """
    rng = np.random.default_rng(42)
    M = 200
    # Smooth-ish curve with noise
    t = np.linspace(0, 4 * np.pi, M)
    pts = np.stack(
        [t + 0.1 * rng.standard_normal(M), np.sin(t) + 0.1 * rng.standard_normal(M)],
        axis=-1,
    ).astype(np.float64)
    simp = simplify_path(pts, eps=0.05)
    # Each simplified point must equal some input point (RDP keeps subset).
    indices = []
    last = -1
    for q in simp:
        # find an exact match in input at index > last
        found = -1
        for i in range(last + 1, M):
            if np.array_equal(pts[i], q):
                found = i
                break
        assert found >= 0, f"simplified point {q} not found as monotonic subset of input"
        indices.append(found)
        last = found
    # indices strictly increasing
    assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))
    # Endpoints retained
    assert indices[0] == 0 and indices[-1] == M - 1
    # Simplification should reduce point count substantially for a smooth signal
    assert len(simp) < M


# ---------------------------------------------------------------------------
# Extra edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_degenerate():
    pts = np.zeros((0, 2), dtype=np.float64)
    res = arclength_resample(pts, n_points=64)
    assert not res.mask.any()
    assert res.total_length == 0.0


def test_all_identical_points_returns_degenerate():
    pts = np.tile(np.array([2.0, -1.0]), (10, 1))
    res = arclength_resample(pts, n_points=64)
    assert not res.mask.any()
    assert res.total_length == 0.0
    # Fallback to that point
    assert np.allclose(res.points_xz, np.array([2.0, -1.0]))


def test_build_arclength_path_pipeline():
    """build_arclength_path = simplify → arclength_resample."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 80)
    pts = np.stack([np.cos(t), np.sin(t)], axis=-1) + 0.005 * rng.standard_normal((80, 2))
    res = build_arclength_path(pts, n_path=64, simplify_eps=0.05)
    assert res.points_xz.shape == (64, 2)
    assert res.mask.all()
    # Total length close to 2π (full circle ~6.28)
    assert 6.0 < res.total_length < 6.5


def test_simplify_invalid_shape_raises():
    import pytest

    with pytest.raises(ValueError):
        simplify_path(np.zeros((5, 3)))


def test_resample_n_points_too_small_raises():
    import pytest

    with pytest.raises(ValueError):
        arclength_resample(np.array([[0.0, 0.0], [1.0, 0.0]]), n_points=1)
