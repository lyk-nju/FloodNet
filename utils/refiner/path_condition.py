"""RootRefiner path-condition construction utilities.

The functions here are intentionally pure and dataset-agnostic: given a future
root XZ curve and sampling choices, they build the fixed-length path condition
and masks consumed by RootRefiner.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from utils.path_arclength import arclength_resample


@dataclass(frozen=True)
class PathConditionResult:
    path: Tensor
    path_valid_mask: Tensor
    path_control_mask: Tensor
    path_supervision_mask: Tensor
    path_features_raw: Tensor
    path_mode: str
    offset_start_frames: int


def _resample(points: Tensor, n_path: int) -> tuple[Tensor, Tensor]:
    arc = arclength_resample(
        points.detach().cpu().numpy().astype(np.float64),
        n_points=n_path,
    )
    return (
        torch.as_tensor(arc.points_xz, dtype=torch.float32),
        torch.as_tensor(arc.mask, dtype=torch.bool),
    )


def compute_path_features(path: Tensor) -> Tensor:
    """Return [path_length, start_dx, start_dz, start_distance, chord_length]."""
    path = path.to(dtype=torch.float32)
    if path.numel() == 0:
        return torch.zeros(5, dtype=torch.float32)
    if path.shape[0] < 2:
        start = path[0] if path.shape[0] else torch.zeros(2, dtype=torch.float32)
        start_dist = start.norm()
        return torch.stack([start_dist.new_tensor(0.0), start[0], start[1], start_dist, start_dist])
    seg = path[1:] - path[:-1]
    path_length = seg.norm(dim=-1).sum()
    start = path[0]
    start_distance = start.norm()
    chord_length = (path[-1] - path[0]).norm()
    return torch.stack(
        [path_length, start[0], start[1], start_distance, chord_length],
    ).to(dtype=torch.float32)


def _supervision_mask(max_frames: int, valid_frame_count: int, offset_start_frames: int) -> Tensor:
    mask = torch.zeros(max_frames, dtype=torch.bool)
    start = max(0, min(int(offset_start_frames), int(valid_frame_count)))
    mask[start:int(valid_frame_count)] = True
    return mask


def _select_arclength_indices(points: Tensor, k: int) -> list[int]:
    n = int(points.shape[0])
    if n <= 1:
        return [0]
    k = max(2, min(int(k), n))
    if k >= n:
        return list(range(n))
    seg = points[1:] - points[:-1]
    seg_len = seg.norm(dim=-1)
    cum = torch.cat([torch.zeros(1, dtype=points.dtype, device=points.device), torch.cumsum(seg_len, dim=0)])
    total = float(cum[-1].item())
    if total < 1e-9:
        return sorted(set(torch.linspace(0, n - 1, k).round().long().tolist()))
    targets = torch.linspace(0.0, total, k, dtype=points.dtype, device=points.device)
    idxs = torch.searchsorted(cum, targets).clamp(max=n - 1).tolist()
    idxs[0] = 0
    idxs[-1] = n - 1
    return sorted(set(int(i) for i in idxs))


def build_dense_path_condition(
    future_xz: Tensor,
    *,
    n_path: int,
    valid_frame_count: int,
    max_frames: int | None = None,
) -> PathConditionResult:
    max_frames = int(max_frames or valid_frame_count)
    path, valid_mask = _resample(future_xz, n_path)
    control_mask = valid_mask.clone()
    supervision_mask = _supervision_mask(max_frames, valid_frame_count, 0)
    return PathConditionResult(
        path=path,
        path_valid_mask=valid_mask,
        path_control_mask=control_mask,
        path_supervision_mask=supervision_mask,
        path_features_raw=compute_path_features(path),
        path_mode="dense_path",
        offset_start_frames=0,
    )


def build_goal_point_condition(
    future_xz: Tensor,
    *,
    n_path: int,
    valid_frame_count: int,
    max_frames: int | None = None,
) -> PathConditionResult:
    max_frames = int(max_frames or valid_frame_count)
    goal = future_xz[-1].to(dtype=torch.float32)
    alpha = torch.linspace(0.0, 1.0, n_path, dtype=torch.float32, device=goal.device)
    path = alpha[:, None] * goal[None, :]
    valid_mask = torch.ones(n_path, dtype=torch.bool)
    control_mask = torch.zeros(n_path, dtype=torch.bool)
    control_mask[-1] = True
    supervision_mask = _supervision_mask(max_frames, valid_frame_count, 0)
    return PathConditionResult(
        path=path.cpu(),
        path_valid_mask=valid_mask,
        path_control_mask=control_mask,
        path_supervision_mask=supervision_mask,
        path_features_raw=compute_path_features(path.cpu()),
        path_mode="goal_point",
        offset_start_frames=0,
    )


def build_sparse_path_condition(
    future_xz: Tensor,
    *,
    n_path: int,
    valid_frame_count: int,
    max_frames: int | None = None,
    point_range: tuple[int, int],
    rng: random.Random,
) -> PathConditionResult:
    max_frames = int(max_frames or valid_frame_count)
    lo, hi = int(point_range[0]), int(point_range[1])
    k = rng.randint(min(lo, hi), max(lo, hi))
    idxs = _select_arclength_indices(future_xz, k)
    controls = future_xz[idxs]
    path, valid_mask = _resample(controls, n_path)
    control_mask = torch.zeros(n_path, dtype=torch.bool)
    for src_idx in idxs:
        denom = max(int(future_xz.shape[0]) - 1, 1)
        pidx = round((float(src_idx) / float(denom)) * float(n_path - 1))
        control_mask[max(0, min(n_path - 1, int(pidx)))] = True
    control_mask[-1] = True
    supervision_mask = _supervision_mask(max_frames, valid_frame_count, 0)
    return PathConditionResult(
        path=path,
        path_valid_mask=valid_mask,
        path_control_mask=control_mask,
        path_supervision_mask=supervision_mask,
        path_features_raw=compute_path_features(path),
        path_mode="sparse_path",
        offset_start_frames=0,
    )


def build_path_condition(
    future_xz: Tensor,
    *,
    n_path: int,
    valid_frame_count: int,
    max_frames: int | None = None,
    path_mode: str,
    offset_start_frames: int,
    sparse_point_range: tuple[int, int],
    rng: random.Random,
) -> PathConditionResult:
    max_frames = int(max_frames or valid_frame_count)
    offset = max(0, min(int(offset_start_frames), int(valid_frame_count) - 2))
    source = future_xz[offset:] if path_mode != "goal_point" else future_xz
    if source.shape[0] < 2:
        source = future_xz
        offset = 0

    if path_mode == "dense_path":
        result = build_dense_path_condition(
            source,
            n_path=n_path,
            valid_frame_count=valid_frame_count,
            max_frames=max_frames,
        )
    elif path_mode == "sparse_path":
        result = build_sparse_path_condition(
            source,
            n_path=n_path,
            valid_frame_count=valid_frame_count,
            max_frames=max_frames,
            point_range=sparse_point_range,
            rng=rng,
        )
    elif path_mode == "goal_point":
        result = build_goal_point_condition(
            future_xz,
            n_path=n_path,
            valid_frame_count=valid_frame_count,
            max_frames=max_frames,
        )
        offset = 0
    else:
        raise ValueError(f"unknown path_mode {path_mode!r}")

    return PathConditionResult(
        path=result.path,
        path_valid_mask=result.path_valid_mask,
        path_control_mask=result.path_control_mask,
        path_supervision_mask=_supervision_mask(max_frames, valid_frame_count, offset),
        path_features_raw=result.path_features_raw,
        path_mode=result.path_mode,
        offset_start_frames=offset,
    )


def map_path_control_mask_to_frame_mask(
    path_control_mask: Tensor,
    *,
    n_path: int,
    max_frames: int,
    valid_frame_count: int,
    offset_start_frames: int = 0,
) -> Tensor:
    frame_mask = torch.zeros(max_frames, dtype=torch.bool, device=path_control_mask.device)
    controls = torch.nonzero(path_control_mask.bool(), as_tuple=False).flatten()
    if controls.numel() == 0:
        return frame_mask
    denom = max(int(n_path) - 1, 1)
    usable = max(int(valid_frame_count) - int(offset_start_frames) - 1, 0)
    for idx in controls.tolist():
        progress = float(idx) / float(denom)
        frame_idx = int(offset_start_frames) + int(round(progress * float(usable)))
        if 0 <= frame_idx < int(max_frames):
            frame_mask[frame_idx] = True
    return frame_mask


__all__ = [
    "PathConditionResult",
    "build_dense_path_condition",
    "build_sparse_path_condition",
    "build_goal_point_condition",
    "build_path_condition",
    "compute_path_features",
    "map_path_control_mask_to_frame_mask",
]
