"""RootRefiner path conditioning stays in physical frame-space."""

from __future__ import annotations

import torch

from tests.helpers.humanml3d_fixture import make_root_refiner_from_samples


def _make_clip(T: int = 80) -> dict:
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 0] = 0.05
    motion[:, 1] = 0.3
    motion[:, 2] = 0.2
    motion[:, 3] = 1.0
    return {"motion_263": motion, "text": "walk in a curve"}


def _sample(ds):
    return ds.get_sample(
        0,
        force_mode="full",
        force_num_frames=17,
        force_no_path_aug=True,
        force_path_mode="dense_path",
        force_text_idx=0,
    )


def test_path_features_match_physical_waypoints():
    ds = make_root_refiner_from_samples(
        [_make_clip()],
        full_plan_ratio=1.0,
        n_hist=8,
        n_path=16,
        min_frames=5,
        max_frames=29,
        seed=0,
    )
    s = _sample(ds)
    wp = s["waypoints"][s["waypoints_mask"]][:, [0, 2]]
    seg = (wp[1:] - wp[:-1]).norm(dim=-1).sum()

    assert torch.allclose(s["path_features"], s["path_features_raw"])
    assert torch.allclose(s["path_features"][0], seg, rtol=0.25)


def test_path_geometry_tokens_are_physical_not_zscored():
    ds = make_root_refiner_from_samples(
        [_make_clip()],
        full_plan_ratio=1.0,
        n_hist=8,
        n_path=16,
        min_frames=5,
        max_frames=29,
        seed=0,
    )
    s = _sample(ds)

    valid_path = s["path"][s["path_valid_mask"]]
    valid_waypoints = s["waypoints"][s["waypoints_mask"]][:, [0, 2]]

    assert torch.allclose(valid_path[0], valid_waypoints[0], atol=1e-5)
    assert torch.allclose(valid_path[-1], valid_waypoints[-1], atol=1e-5)
