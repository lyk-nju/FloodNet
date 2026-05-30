"""R2.5: path_features must be computed in PHYSICAL space, not z-score space.

The old shim read `future_xz` from `target_waypoints[..., [0,2]]` AFTER the base
dataset z-scored it (when normalize=True), so `compute_path_features` ran on
z-scored coordinates → path_length/start_distance/chord_length lived in
anisotropic z-score units and drifted in scale across samples. The duration head
then could not learn a generalizing L→N map. These tests pin the fix:

  * path_features are identical with normalize on/off (when no path-feature stats
    are present) — i.e. they are PHYSICAL, untouched by waypoint z-scoring.
  * the path GEOMETRY tokens DO get z-scored when normalize=True (they must stay
    in the same space as the z-scored waypoints for the control loss).
"""

from __future__ import annotations

import numpy as np
import torch

from datasets.humanml3d_refiner import HumanML3DRefinerDataset


def _make_clip(T: int = 80) -> dict:
    # Curved, non-trivial root path so path_length/chord differ from z-score.
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 0] = 0.05            # root_rot_velocity
    motion[:, 1] = 0.3             # root_linear_velocity x
    motion[:, 2] = 0.2             # root_linear_velocity z
    motion[:, 3] = 1.0             # root_y
    return {"motion_263": motion, "text": "walk in a curve"}


def _save_5d_stats(tmp_path) -> None:
    """Minimal current_motion + waypoint stats (anisotropic xyz) for normalize."""
    cm_mean = np.zeros(5, np.float32)
    cm_std = np.ones(5, np.float32)
    cm_idx = np.array([0, 1, 2], np.int64)              # x, y, z
    wp_mean = np.array([0.5, 0.1, -0.3, 0, 0, 0, 0], np.float32)
    wp_std = np.array([2.0, 1.0, 3.0, 1, 1, 1, 1], np.float32)  # anisotropic x vs z
    wp_idx = np.array([0, 1, 2], np.int64)              # x, y, z only (not cos/sin)
    np.save(tmp_path / "current_motion_mean.npy", cm_mean)
    np.save(tmp_path / "current_motion_std.npy", cm_std)
    np.save(tmp_path / "current_motion_norm_indices.npy", cm_idx)
    np.save(tmp_path / "waypoint_mean.npy", wp_mean)
    np.save(tmp_path / "waypoint_std.npy", wp_std)
    np.save(tmp_path / "waypoint_norm_indices.npy", wp_idx)


def _sample(ds):
    return ds.get_sample(
        0, force_mode="full", force_num_tokens=5,
        force_no_path_aug=True, force_path_mode="dense_path", force_text_idx=0,
    )


def test_path_features_are_physical_regardless_of_normalize(tmp_path):
    _save_5d_stats(tmp_path)
    common = dict(
        full_plan_ratio=1.0, n_hist=8, n_path=16,
        min_tokens=2, max_tokens=8, seed=0,
    )
    ds_raw = HumanML3DRefinerDataset([_make_clip()], normalize=False, **common)
    ds_norm = HumanML3DRefinerDataset(
        [_make_clip()], normalize=True, stats_dir=tmp_path, **common,
    )

    feat_raw = _sample(ds_raw)["path_features"]
    feat_norm = _sample(ds_norm)["path_features"]

    # No path-feature stats present → both must be the SAME physical features.
    assert torch.allclose(feat_raw, feat_norm, atol=1e-5), (
        f"path_features changed under waypoint z-score:\n raw ={feat_raw}\n norm={feat_norm}"
    )
    # physical path_length must be strictly positive for a moving clip.
    assert feat_raw[0] > 0


def test_path_geometry_tokens_are_zscored_when_normalize(tmp_path):
    _save_5d_stats(tmp_path)
    common = dict(
        full_plan_ratio=1.0, n_hist=8, n_path=16,
        min_tokens=2, max_tokens=8, seed=0,
    )
    ds_raw = HumanML3DRefinerDataset([_make_clip()], normalize=False, **common)
    ds_norm = HumanML3DRefinerDataset(
        [_make_clip()], normalize=True, stats_dir=tmp_path, **common,
    )

    path_raw = _sample(ds_raw)["path"]
    path_norm = _sample(ds_norm)["path"]

    # The geometry tokens MUST move under z-score (anisotropic wp_std=2 vs 3),
    # otherwise they would not match the z-scored waypoints in the control loss.
    assert not torch.allclose(path_raw, path_norm, atol=1e-3)


def test_path_features_match_waypoints_when_normalize_off(tmp_path):
    """Sanity: physical path_length ~ arclength of the physical future xz."""
    ds = HumanML3DRefinerDataset(
        [_make_clip()], full_plan_ratio=1.0, n_hist=8, n_path=16,
        min_tokens=2, max_tokens=8, seed=0, normalize=False,
    )
    s = _sample(ds)
    wp = s["waypoints"][s["waypoints_mask"]][:, [0, 2]]
    seg = (wp[1:] - wp[:-1]).norm(dim=-1).sum()
    # path_features[0] is the resampled arclength; close to the raw arclength.
    assert torch.allclose(s["path_features"][0], seg, rtol=0.25)
