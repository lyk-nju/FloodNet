from __future__ import annotations

import pytest
import torch

from utils.training.root_refiner.sample_builder import RefinerSampleBuilder
from utils.training.root_refiner.sample_creator import RefinerSampleCreator


def test_refiner_package_exports_sample_creator():
    from utils.training.root_refiner import RefinerSampleBuilder as ExportedBuilder
    from utils.training.root_refiner import RefinerSampleCreator as ExportedCreator

    assert ExportedBuilder is RefinerSampleBuilder
    assert ExportedCreator is RefinerSampleCreator


def test_full_sample_randomizes_anchor_single_history_and_max_horizon():
    creator = RefinerSampleCreator(
        n_hist=8,
        max_frames=29,
        min_frames=5,
        full_plan_ratio=1.0,
        horizon_policy="max",
        path_condition_policy="dense_path",
        seed=0,
    )

    sample = creator.create(torch.tensor([80]))

    assert sample.modes == ["full"]
    anchor = int(sample.anchor_frames[0].item())
    assert 0 < anchor <= 74
    assert sample.valid_history_frames.tolist() == [1]
    assert sample.history_mask.tolist() == [
        [False, False, False, False, False, False, False, True],
    ]
    assert sample.history_frame_indices.tolist() == [[0, 0, 0, 0, 0, 0, 0, anchor]]
    assert not hasattr(sample, "num_tokens")
    assert sample.target_frame_counts.tolist() == [min(29, 80 - anchor - 1)]
    assert sample.path_modes == ["dense_path"]
    assert sample.offset_start_frames.tolist() == [0]


def test_sliding_sample_uses_forced_anchor_and_history_indices():
    creator = RefinerSampleCreator(
        n_hist=8,
        max_frames=45,
        min_frames=5,
        full_plan_ratio=0.0,
        horizon_policy="random",
        path_condition_policy="dense_path",
        seed=0,
    )

    sample = creator.create(
        torch.tensor([80]),
        force_mode="sliding",
        force_anchor_frame=torch.tensor([30]),
        force_num_frames=torch.tensor([17]),
    )

    assert sample.modes == ["sliding"]
    assert sample.anchor_frames.tolist() == [30]
    assert sample.valid_history_frames.tolist() == [8]
    assert sample.history_mask.all()
    assert sample.history_frame_indices.tolist() == [[23, 24, 25, 26, 27, 28, 29, 30]]
    assert sample.target_frame_counts.tolist() == [17]


def test_sliding_draw_falls_back_to_full_when_clip_is_not_sliding_eligible():
    min_frames = 13
    n_hist = 20
    full_only_length = min_frames + 2
    creator = RefinerSampleCreator(
        n_hist=n_hist,
        max_frames=193,
        min_frames=min_frames,
        full_plan_ratio=0.0,
        horizon_policy="max",
        path_condition_policy="dense_path",
        seed=0,
    )

    sample = creator.create(torch.tensor([full_only_length]))

    assert sample.modes == ["full"]
    assert 0 <= int(sample.anchor_frames[0].item()) <= full_only_length - min_frames - 1
    assert sample.valid_history_frames.tolist() == [1]


def test_forced_path_mode_and_offset_gating_are_in_sample_plan():
    creator = RefinerSampleCreator(
        n_hist=8,
        max_frames=29,
        min_frames=5,
        full_plan_ratio=1.0,
        horizon_policy="max",
        path_condition_policy="mixed",
        path_condition_ratios={"dense_path": 0.0, "sparse_path": 1.0, "goal_point": 0.0},
        offset_start_enabled=True,
        offset_start_prob=1.0,
        offset_start_max_frames=40,
        offset_start_apply_to=("sparse_path",),
        seed=0,
    )

    augmented = creator.create(torch.tensor([80]), force_path_mode="sparse_path")
    no_aug = creator.create(
        torch.tensor([80]),
        force_path_mode="sparse_path",
        force_no_path_aug=True,
    )

    assert augmented.path_modes == ["sparse_path"]
    assert 0 <= int(augmented.offset_start_frames[0]) <= 27
    assert no_aug.path_modes == ["sparse_path"]
    assert no_aug.offset_start_frames.tolist() == [0]


def test_unknown_path_condition_ratio_key_raises_at_construction():
    with pytest.raises(ValueError, match="unknown path_condition_ratios"):
        RefinerSampleCreator(
            path_condition_policy="mixed",
            path_condition_ratios={
                "dense_path": 0.5,
                "sparse_path": 0.3,
                "goal_point": 0.2,
                "typo_path": 0.1,
            },
        )


@pytest.mark.parametrize("forced_frames", [4, 99])
def test_forced_num_frames_out_of_range_raises(forced_frames: int):
    creator = RefinerSampleCreator(
        n_hist=8,
        max_frames=29,
        min_frames=5,
        full_plan_ratio=1.0,
        horizon_policy="max",
        path_condition_policy="dense_path",
        seed=0,
    )

    with pytest.raises(ValueError, match="force_num_frames"):
        creator.create(
            torch.tensor([40]),
            force_mode="full",
            force_num_frames=torch.tensor([forced_frames]),
        )
