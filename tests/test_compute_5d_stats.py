"""Unit tests for scripts/compute_5d_stats.py (T_A_06)."""

from __future__ import annotations

import numpy as np
import torch

from datasets.refiner_dataset import RefinerDataset
from scripts.compute_5d_stats import (
    CURRENT_MOTION_NORM_INDICES,
    WAYPOINT_NORM_INDICES,
    WelfordAccumulator,
    _read_all_captions,
    compute_stats,
    load_clips_from_dir,
    save_stats,
)


# ---------------------------------------------------------------------------
# Welford correctness vs numpy reference
# ---------------------------------------------------------------------------


def test_welford_single_batch_matches_numpy():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((1000, 5))
    acc = WelfordAccumulator(dim=5)
    acc.update_batch(values)
    mean, std = acc.finalize()
    assert np.allclose(mean, values.mean(axis=0), atol=1e-9)
    # WelfordAccumulator uses population std (divide by n, not n-1)
    expected_std = values.std(axis=0, ddof=0)
    assert np.allclose(std, expected_std, atol=1e-6)


def test_welford_multi_batch_equals_concatenated_single_batch():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((400, 7))
    b = rng.standard_normal((300, 7))
    c = rng.standard_normal((200, 7))

    acc_split = WelfordAccumulator(dim=7)
    acc_split.update_batch(a)
    acc_split.update_batch(b)
    acc_split.update_batch(c)

    acc_full = WelfordAccumulator(dim=7)
    acc_full.update_batch(np.concatenate([a, b, c], axis=0))

    m_split, s_split = acc_split.finalize()
    m_full, s_full = acc_full.finalize()
    assert np.allclose(m_split, m_full, atol=1e-9)
    assert np.allclose(s_split, s_full, atol=1e-6)


def test_welford_empty_batch_is_noop():
    acc = WelfordAccumulator(dim=3)
    acc.update_batch(np.empty((0, 3)))
    mean, std = acc.finalize()
    assert acc.n == 0
    assert np.allclose(mean, np.zeros(3))


def test_welford_rejects_wrong_dim_input():
    import pytest

    acc = WelfordAccumulator(dim=5)
    with pytest.raises(ValueError):
        acc.update_batch(np.zeros((10, 4)))


def test_welford_constant_channel_std_clamped_to_eps():
    """A channel with no variance should report std >= eps to avoid div-by-zero
    downstream."""
    values = np.tile([1.0, 2.0, 3.0], (50, 1))
    acc = WelfordAccumulator(dim=3)
    acc.update_batch(values)
    _, std = acc.finalize(eps=1e-6)
    assert (std >= 1e-6).all()


# ---------------------------------------------------------------------------
# End-to-end: build a small RefinerDataset, run compute_stats, save, reload.
# ---------------------------------------------------------------------------


def _make_clip(T: int, *, vx: float = 0.0, vz: float = 0.05) -> dict:
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 1] = vx
    motion[:, 2] = vz
    motion[:, 3] = 1.0    # constant root y
    return {"motion_263": motion, "text": "walk"}


def test_compute_stats_end_to_end_with_synthetic_dataset(tmp_path):
    clips = [_make_clip(T=40, vz=0.05 + 0.01 * i) for i in range(8)]
    ds = RefinerDataset(clips, full_plan_ratio=1.0, normalize=False, seed=0)
    stats = compute_stats(ds, max_samples=-1, progress=False)

    # Shape sanity.
    assert stats["current_motion_mean"].shape == (5,)
    assert stats["current_motion_std"].shape == (5,)
    assert stats["waypoint_mean"].shape == (7,)
    assert stats["waypoint_std"].shape == (7,)

    # n must be > 0 for both accumulators.
    assert stats["n_current_motion"] > 0
    assert stats["n_waypoint"] > 0

    # Health check: std > 0 on every channel (after eps clamp).
    assert (stats["current_motion_std"] > 0).all()
    assert (stats["waypoint_std"] > 0).all()


def test_save_stats_writes_six_files_and_norm_indices_have_expected_values(tmp_path):
    clips = [_make_clip(T=40) for _ in range(3)]
    ds = RefinerDataset(clips, full_plan_ratio=1.0, normalize=False, seed=0)
    stats = compute_stats(ds, progress=False)
    save_stats(stats, tmp_path)

    expected_files = {
        "current_motion_mean.npy",
        "current_motion_std.npy",
        "current_motion_norm_indices.npy",
        "waypoint_mean.npy",
        "waypoint_std.npy",
        "waypoint_norm_indices.npy",
    }
    written = {p.name for p in tmp_path.iterdir() if p.name.endswith(".npy")}
    assert expected_files.issubset(written), (
        f"missing: {expected_files - written}"
    )

    cm_norm = np.load(tmp_path / "current_motion_norm_indices.npy")
    wp_norm = np.load(tmp_path / "waypoint_norm_indices.npy")
    np.testing.assert_array_equal(cm_norm, CURRENT_MOTION_NORM_INDICES)
    np.testing.assert_array_equal(wp_norm, WAYPOINT_NORM_INDICES)

    # Shapes per spec.
    assert np.load(tmp_path / "current_motion_mean.npy").shape == (5,)
    assert np.load(tmp_path / "current_motion_std.npy").shape == (5,)
    assert np.load(tmp_path / "waypoint_mean.npy").shape == (7,)
    assert np.load(tmp_path / "waypoint_std.npy").shape == (7,)


def test_max_samples_dry_run_limits_iteration(tmp_path):
    """--max_samples 5 should accumulate stats only over 5 samples."""
    clips = [_make_clip(T=40) for _ in range(20)]
    ds = RefinerDataset(clips, full_plan_ratio=1.0, normalize=False, seed=0)
    stats = compute_stats(ds, max_samples=5, progress=False)
    # Each sample contributes 1 history slot (full mode) + variable target.
    assert stats["n_current_motion"] == 5
    assert stats["n_waypoint"] > 0


# ---------------------------------------------------------------------------
# Integration with RefinerDataset(normalize=True): cos / sin invariance
# ---------------------------------------------------------------------------


def test_loaded_stats_pass_T_A_04_T13_cos_sin_invariance(tmp_path):
    """Lock-in: after `save_stats` + RefinerDataset(normalize=True), the cos /
    sin heading channels at [3], [4] must be bit-equal to the unnormalized
    values (norm_indices excludes them). Mirrors T_A_04 T13 with the actual
    stats file format produced by this script.
    """
    clips = [_make_clip(T=60)]
    ds_compute = RefinerDataset(clips, full_plan_ratio=1.0, normalize=False, seed=0)
    stats = compute_stats(ds_compute, progress=False)
    save_stats(stats, tmp_path)

    # Now build raw + normalized datasets and compare cos/sin channels.
    ds_raw = RefinerDataset(clips, full_plan_ratio=1.0, normalize=False, seed=0)
    ds_norm = RefinerDataset(clips, full_plan_ratio=1.0, normalize=True,
                              stats_dir=tmp_path, seed=0)
    s_raw = ds_raw.get_sample(0, force_mode="full", force_num_tokens=8,
                                force_no_path_aug=True)
    s_norm = ds_norm.get_sample(0, force_mode="full", force_num_tokens=8,
                                  force_no_path_aug=True)

    # cos / sin in current_motion (channels 3, 4)
    assert torch.equal(s_raw["current_motion"][..., 3], s_norm["current_motion"][..., 3])
    assert torch.equal(s_raw["current_motion"][..., 4], s_norm["current_motion"][..., 4])
    # cos / sin in target_waypoints (channels 3, 4)
    assert torch.equal(s_raw["target_waypoints"][..., 3], s_norm["target_waypoints"][..., 3])
    assert torch.equal(s_raw["target_waypoints"][..., 4], s_norm["target_waypoints"][..., 4])
    # ⚠ Do NOT also assert "xyz differs after normalize" on this synthetic
    # fixture — local-frame x,z at the anchor are 0 by construction, and the
    # stats are computed from the same data, so normalized x,z still come out
    # as 0 (subtract mean ≈ 0, divide by std). The cos/sin invariance above
    # is the only critical lock-in for T_A_06's z-score selectivity contract.


def test_norm_indices_constants_match_spec():
    """Hard lock-in on the norm_indices constants per docs/TODO.md §T_A_06."""
    np.testing.assert_array_equal(CURRENT_MOTION_NORM_INDICES, np.array([0, 1, 2]))
    np.testing.assert_array_equal(WAYPOINT_NORM_INDICES, np.array([0, 1, 2, 5, 6]))


# ---------------------------------------------------------------------------
# Multi-caption loading (text augmentation enabler)
# ---------------------------------------------------------------------------


def _write_fake_humanml3d(root, names, captions_by_name):
    """Minimal HumanML3D layout: new_joint_vecs/<id>.npy + texts/<id>.txt."""
    ds = root / "HumanML3D"
    (ds / "new_joint_vecs").mkdir(parents=True)
    (ds / "texts").mkdir(parents=True)
    (ds / "train.txt").write_text("\n".join(names) + "\n")
    for n in names:
        arr = np.zeros((40, 263), dtype=np.float32)
        arr[:, 2] = 0.05      # small forward velocity
        arr[:, 3] = 1.0       # constant root y
        np.save(ds / "new_joint_vecs" / f"{n}.npy", arr)
        # '#'-delimited HumanML3D caption lines.
        lines = [f"{c}#tok#0.0#0.0" for c in captions_by_name[n]]
        (ds / "texts" / f"{n}.txt").write_text("\n".join(lines) + "\n")
    return ds


def test_read_all_captions_dedups_and_keeps_order(tmp_path):
    f = tmp_path / "cap.txt"
    f.write_text(
        "a person walks#x#0#0\n"
        "someone strolls forward#x#0#0\n"
        "a person walks#x#0#0\n"   # duplicate of line 1
        "\n"                        # blank line ignored
    )
    caps = _read_all_captions(f)
    assert caps == ["a person walks", "someone strolls forward"]


def test_load_clips_populates_texts_list_and_first_caption(tmp_path):
    """load_clips_from_dir must expose every distinct caption in `texts`, with
    `text` == the first one (backward compatible)."""
    caps = {
        "c1": ["a person walks forward", "someone strides ahead", "walking"],
        "c2": ["a man jumps"],
    }
    _write_fake_humanml3d(tmp_path, ["c1", "c2"], caps)
    clips = load_clips_from_dir(str(tmp_path), dataset="humanml3d",
                                split_file="train.txt",
                                feature_path="new_joint_vecs", text_path="texts")
    by_first = {c["text"]: c for c in clips}
    assert "a person walks forward" in by_first
    assert "a man jumps" in by_first
    c1 = by_first["a person walks forward"]
    assert c1["texts"] == caps["c1"]          # all captions, in order
    assert c1["text"] == caps["c1"][0]        # first caption mirrored into `text`
