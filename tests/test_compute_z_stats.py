"""Unit tests for scripts/compute_z_stats.py (T_B_02)."""

from __future__ import annotations

import numpy as np

from scripts.compute_z_stats import (
    compute_z_stats,
    iter_latent_files,
    save_z_stats,
)


def _write_latents(cache_dir, arrays):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(arrays):
        np.save(cache_dir / f"clip_{i:03d}.npy", a.astype(np.float32))


def test_z_stats_match_numpy_reference(tmp_path):
    rng = np.random.default_rng(0)
    D = 4
    arrays = [rng.standard_normal((rng.integers(5, 20), D)) for _ in range(6)]
    _write_latents(tmp_path, arrays)
    z_mean, z_std, n = compute_z_stats(tmp_path)

    allvec = np.concatenate([a.reshape(-1, D) for a in arrays], axis=0).astype(np.float64)
    assert z_mean.shape == (D,)
    assert n == allvec.shape[0]
    assert np.allclose(z_mean, allvec.mean(axis=0), atol=1e-5)
    assert np.allclose(z_std, allvec.std(axis=0, ddof=0), atol=1e-5)


def test_channel_axis_first(tmp_path):
    """Latents stored as [D, T] (channel first) → channel_axis=0."""
    rng = np.random.default_rng(1)
    D, T = 4, 10
    arrays = [rng.standard_normal((D, T)) for _ in range(3)]
    _write_latents(tmp_path, arrays)
    z_mean, z_std, n = compute_z_stats(tmp_path, channel_axis=0)
    # reference: move axis 0 to last then flatten
    ref = np.concatenate(
        [np.moveaxis(a, 0, -1).reshape(-1, D) for a in arrays], axis=0,
    ).astype(np.float64)
    assert z_mean.shape == (D,)
    assert np.allclose(z_mean, ref.mean(axis=0), atol=1e-5)


def test_max_files_limits(tmp_path):
    rng = np.random.default_rng(2)
    arrays = [rng.standard_normal((8, 4)) for _ in range(10)]
    _write_latents(tmp_path, arrays)
    _, _, n = compute_z_stats(tmp_path, max_files=3)
    assert n == 3 * 8


def test_save_writes_z_mean_and_z_std(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "stats"
    _write_latents(cache, [np.ones((5, 4)), np.zeros((5, 4))])
    z_mean, z_std, _ = compute_z_stats(cache)
    save_z_stats(z_mean, z_std, out)
    assert (out / "z_mean.npy").is_file()
    assert (out / "z_std.npy").is_file()
    assert np.load(out / "z_mean.npy").shape == (4,)
    assert np.load(out / "z_std.npy").shape == (4,)
    # mean of {1,0} per channel = 0.5
    assert np.allclose(np.load(out / "z_mean.npy"), 0.5, atol=1e-5)


def test_empty_cache_raises(tmp_path):
    import pytest
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        compute_z_stats(tmp_path / "empty")


def test_inconsistent_dim_raises(tmp_path):
    import pytest
    _write_latents(tmp_path, [np.zeros((5, 4)), np.zeros((5, 8))])
    with pytest.raises(ValueError):
        compute_z_stats(tmp_path)


def test_iter_latent_files_sorted(tmp_path):
    _write_latents(tmp_path, [np.zeros((2, 4)) for _ in range(3)])
    files = iter_latent_files(tmp_path)
    assert [f.name for f in files] == ["clip_000.npy", "clip_001.npy", "clip_002.npy"]
