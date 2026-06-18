from __future__ import annotations

import torch

from tests.helpers.humanml3d_fixture import make_root_refiner_from_samples
from utils.training.root_refiner import collate_fn


def _make_clip(T: int = 80) -> dict:
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 2] = 0.1
    motion[:, 3] = 1.0
    return {"motion_263": motion, "text": "walk forward"}


def test_root_refiner_sample_contract_has_new_keys_and_shapes():
    ds = make_root_refiner_from_samples(
        [_make_clip()],
        full_plan_ratio=1.0,
        n_hist=8,
        n_path=16,
        min_frames=5,
        max_frames=29,
        seed=0,
    )
    sample = ds.get_sample(
        0,
        force_mode="full",
        force_num_frames=17,
        force_no_path_aug=True,
        force_path_mode="dense_path",
    )

    required = {
        "text",
        "history_motion",
        "history_mask",
        "path",
        "path_valid_mask",
        "path_control_mask",
        "path_features",
        "path_mode",
        "waypoints",
        "waypoints_mask",
        "path_supervision_mask",
        "num_frames",
    }
    assert required.issubset(sample)
    assert "num_tokens" not in sample
    assert sample["path"].shape == (ds.n_path, 2)
    assert sample["path_valid_mask"].shape == (ds.n_path,)
    assert sample["path_control_mask"].shape == (ds.n_path,)
    assert sample["path_features"].shape == (5,)
    assert sample["history_motion"].shape == (ds.n_hist, 5)
    assert sample["waypoints"].shape == (ds.max_frames, 5)
    assert sample["waypoints_mask"].shape == (ds.max_frames,)
    assert sample["path_supervision_mask"].shape == (ds.max_frames,)
    assert sample["path_mode"] in {"dense_path", "sparse_path", "goal_point"}
    assert int(sample["num_frames"].item()) == 17
    assert int(sample["waypoints_mask"].sum()) == 17


def test_collate_fn_stacks_new_tensor_keys_and_keeps_modes_as_list():
    ds = make_root_refiner_from_samples(
        [_make_clip(), _make_clip()],
        full_plan_ratio=1.0,
        n_hist=8,
        n_path=16,
        min_frames=5,
        max_frames=29,
        seed=0,
    )
    samples = [
        ds.get_sample(i, force_mode="full", force_num_frames=9, force_no_path_aug=True)
        for i in range(2)
    ]
    batch = collate_fn(samples)

    assert isinstance(batch["text"], list)
    assert isinstance(batch["path_mode"], list)
    assert batch["path"].shape == (2, ds.n_path, 2)
    assert batch["path_features"].shape == (2, 5)
    assert batch["history_motion"].shape == (2, ds.n_hist, 5)
    assert batch["waypoints"].shape == (2, ds.max_frames, 5)
    assert torch.equal(batch["num_frames"], torch.tensor([9, 9]))
    assert "num_tokens" not in batch
