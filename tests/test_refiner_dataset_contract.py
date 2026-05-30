from __future__ import annotations

import torch

from datasets.humanml3d_refiner import HumanML3DRefinerDataset, refiner_collate
from utils.token_frame import num_frames_for_tokens


def _make_clip(T: int = 80) -> dict:
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 2] = 0.1
    motion[:, 3] = 1.0
    return {"motion_263": motion, "text": "walk forward"}


def test_humanml3d_refiner_sample_contract_has_new_keys_and_shapes():
    ds = HumanML3DRefinerDataset(
        [_make_clip()],
        full_plan_ratio=1.0,
        n_hist=8,
        n_path=16,
        min_tokens=2,
        max_tokens=8,
        normalize=False,
        seed=0,
    )
    sample = ds.get_sample(
        0,
        force_mode="full",
        force_num_tokens=4,
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
        "num_tokens",
    }
    assert required.issubset(sample)
    assert sample["path"].shape == (ds.n_path, 2)
    assert sample["path_valid_mask"].shape == (ds.n_path,)
    assert sample["path_control_mask"].shape == (ds.n_path,)
    assert sample["path_features"].shape == (5,)
    assert sample["history_motion"].shape == (ds.n_hist, 5)
    assert sample["waypoints"].shape == (ds.max_frames, 5)
    assert sample["waypoints_mask"].shape == (ds.max_frames,)
    assert sample["path_supervision_mask"].shape == (ds.max_frames,)
    assert sample["path_mode"] in {"dense_path", "sparse_path", "goal_point"}
    assert int(sample["waypoints_mask"].sum()) == num_frames_for_tokens(4)


def test_refiner_collate_stacks_new_tensor_keys_and_keeps_modes_as_list():
    ds = HumanML3DRefinerDataset(
        [_make_clip(), _make_clip()],
        full_plan_ratio=1.0,
        n_hist=8,
        n_path=16,
        min_tokens=2,
        max_tokens=8,
        normalize=False,
        seed=0,
    )
    samples = [
        ds.get_sample(i, force_mode="full", force_num_tokens=3, force_no_path_aug=True)
        for i in range(2)
    ]
    batch = refiner_collate(samples)

    assert isinstance(batch["text"], list)
    assert isinstance(batch["path_mode"], list)
    assert batch["path"].shape == (2, ds.n_path, 2)
    assert batch["path_features"].shape == (2, 5)
    assert batch["history_motion"].shape == (2, ds.n_hist, 5)
    assert batch["waypoints"].shape == (2, ds.max_frames, 5)
