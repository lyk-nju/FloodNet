from __future__ import annotations

import random

import torch

from utils.refiner.path_condition import (
    build_dense_path_condition,
    build_goal_point_condition,
    build_path_condition,
    build_sparse_path_condition,
    compute_path_features,
    map_path_control_mask_to_frame_mask,
)


def _future_xz(num_frames: int = 17) -> torch.Tensor:
    x = torch.linspace(0.0, 4.0, num_frames)
    z = torch.sin(x) * 0.25
    return torch.stack([x, z], dim=-1)


def test_dense_path_uses_future_geometry_without_prepended_origin():
    future = _future_xz()
    result = build_dense_path_condition(
        future, n_path=8, valid_frame_count=future.shape[0],
    )

    assert result.path.shape == (8, 2)
    assert result.path_valid_mask.all()
    assert result.path_control_mask.all()
    assert result.path_mode == "dense_path"
    assert torch.allclose(result.path[0], future[0], atol=1e-5)


def test_goal_point_uses_current_to_goal_line_and_endpoint_control_only():
    future = _future_xz()
    result = build_goal_point_condition(
        future, n_path=8, valid_frame_count=future.shape[0],
    )

    assert torch.allclose(result.path[0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(result.path[-1], future[-1], atol=1e-6)
    assert int(result.path_control_mask.sum().item()) == 1
    assert result.path_control_mask[-1].item() is True
    assert result.path_mode == "goal_point"


def test_sparse_path_marks_visible_controls_and_includes_endpoint():
    future = _future_xz(25)
    result = build_sparse_path_condition(
        future,
        n_path=16,
        valid_frame_count=future.shape[0],
        point_range=(3, 5),
        rng=random.Random(0),
    )

    assert result.path.shape == (16, 2)
    assert result.path_valid_mask.all()
    assert 3 <= int(result.path_control_mask.sum().item()) <= 5
    assert result.path_control_mask[-1].item() is True
    assert torch.allclose(result.path[-1], future[-1], atol=1e-5)
    assert result.path_mode == "sparse_path"


def test_offset_start_changes_supervision_mask_but_not_valid_frame_count():
    future = _future_xz(21)
    result = build_path_condition(
        future,
        n_path=8,
        valid_frame_count=future.shape[0],
        path_mode="dense_path",
        offset_start_frames=4,
        sparse_point_range=(3, 8),
        rng=random.Random(1),
    )

    assert result.offset_start_frames == 4
    assert not result.path_supervision_mask[:4].any()
    assert result.path_supervision_mask[4:future.shape[0]].all()
    assert torch.allclose(result.path[0], future[4], atol=1e-5)


def test_path_features_are_5d_raw_geometry_features():
    path = torch.tensor([[1.0, 2.0], [4.0, 6.0], [7.0, 6.0]])
    features = compute_path_features(path)

    assert features.shape == (5,)
    assert torch.allclose(features[1:3], torch.tensor([1.0, 2.0]))
    assert torch.isclose(features[3], torch.sqrt(torch.tensor(5.0)))
    assert torch.isclose(features[4], torch.sqrt(torch.tensor(52.0)), atol=1e-5)


def test_map_path_control_mask_to_frame_mask_uses_offset_intersection_grid():
    path_control_mask = torch.zeros(8, dtype=torch.bool)
    path_control_mask[[0, 4, 7]] = True
    frame_mask = map_path_control_mask_to_frame_mask(
        path_control_mask,
        n_path=8,
        max_frames=21,
        valid_frame_count=21,
        offset_start_frames=5,
    )

    assert not frame_mask[:5].any()
    assert frame_mask[5].item() is True
    assert frame_mask[-1].item() is True
    assert int(frame_mask.sum().item()) == 3
