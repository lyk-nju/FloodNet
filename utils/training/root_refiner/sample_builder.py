"""RootRefiner tensor sample construction.

RefinerSampleBuilder turns a raw HumanML3D-style motion sample and a
RefinerSample plan into the batch contract consumed by RootRefiner training.
Sampling decisions live in sample_creator.py; this module owns root recovery,
local-frame canonicalization, path-condition construction, and normalization.
"""

from __future__ import annotations

import os
import random as random_module
import numpy as np
import torch

from pathlib import Path
from typing import Any
from utils.local_frame import (
    canonicalize_5d,
    canonicalize_7d,
    root_quat_to_physical_yaw,
)
from utils.motion_process import recover_root_rot_pos, root_to_traj_feats_7d
from utils.training.root_refiner.path_condition import build_path_condition
from utils.training.root_refiner.sample_creator import RefinerSample


def _pad_or_truncate(x: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = x.shape[0]
    if cur == target_len:
        return x
    if cur > target_len:
        return x[:target_len]
    pad_shape = list(x.shape)
    pad_shape[0] = target_len - cur
    return torch.cat([x, x.new_zeros(*pad_shape)], dim=0)


class RefinerSampleBuilder:
    """Build RootRefiner training samples from raw motion and a sample plan."""

    def __init__(
        self,
        *,
        n_hist: int = 20,
        n_path: int = 64,
        max_frames: int = 193,
        min_frames: int = 13,
        normalize: bool = False,
        stats_dir: str | os.PathLike | None = None,
        sparse_path_point_range: tuple[int, int] = (3, 8),
        path_feature_stats_dir: str | None = None,
        sampling_config_hash: str | None = None,
        seed: int | None = None,
    ):
        self.n_hist = int(n_hist)
        self.n_path = int(n_path)
        self.max_frames_value = int(max_frames)
        self.min_frames = int(min_frames)
        self.normalize = bool(normalize)
        self.sparse_path_point_range = tuple(int(v) for v in sparse_path_point_range)
        self._seed = seed
        self._rng = random_module.Random(seed)

        self._cm_mean = None
        self._cm_std = None
        self._cm_norm_idx = None
        self._wp_mean = None
        self._wp_std = None
        self._wp_norm_idx = None
        if self.normalize:
            if stats_dir is None:
                raise ValueError("normalize=True requires stats_dir")
            self._load_motion_stats(Path(stats_dir))

        self._pf_mean = None
        self._pf_std = None
        if path_feature_stats_dir is not None:
            if sampling_config_hash is None:
                raise ValueError(
                    "path_feature_stats_dir was set but sampling_config_hash is None"
                )
            from utils.training.root_refiner.path_feature_stats import load_path_feature_stats

            stats = load_path_feature_stats(
                path_feature_stats_dir, expected_hash=sampling_config_hash,
            )
            self._pf_mean = stats.mean
            self._pf_std = stats.std

    @property
    def max_frames(self) -> int:
        return self.max_frames_value

    def reset_rng(self) -> None:
        """Reset sparse path sampling to the builder seed."""
        self._rng = random_module.Random(self._seed)

    def build(
        self,
        raw_sample: dict[str, Any],
        plan: RefinerSample,
        *,
        index: int = 0,
    ) -> dict[str, Any]:
        motion_263 = self._motion_of(raw_sample).unsqueeze(0)
        root_xyz, root_yaw, motion_5d_world, motion_7d_world = self.process_root(
            motion_263
        )

        anchor_frame = int(plan.anchor_frames[index].item())
        valid_history_frames = int(plan.valid_history_frames[index].item())
        history_indices = (
            plan.history_frame_indices[index][plan.history_mask[index]]
            .detach()
            .cpu()
            .tolist()
        )
        target_frame_count = int(plan.target_frame_counts[index].item())
        anchor_xz = root_xyz[anchor_frame, [0, 2]]
        anchor_yaw = root_yaw[anchor_frame]

        current_motion, history_mask = self.process_history(
            motion_5d_world,
            history_indices,
            valid_history_frames,
            anchor_xz,
            anchor_yaw,
        )
        target_waypoints, target_waypoints_physical, target_mask = self.process_target(
            motion_7d_world,
            anchor_frame,
            target_frame_count,
            anchor_xz,
            anchor_yaw,
        )

        if self.normalize:
            current_motion = self._apply_zscore(
                current_motion,
                self._cm_mean,
                self._cm_std,
                self._cm_norm_idx,
            )
            target_waypoints = self._apply_zscore(
                target_waypoints,
                self._wp_mean,
                self._wp_std,
                self._wp_norm_idx,
            )

        base_sample = {
            "text": self._text_of(raw_sample),
            "current_motion": current_motion,
            "history_mask": history_mask,
            "target_waypoints": target_waypoints,
            "target_waypoints_physical": target_waypoints_physical,
            "target_mask": target_mask,
            "num_frames": torch.tensor(
                target_frame_count,
                device=motion_263.device,
                dtype=torch.long,
            ),
            "mode": plan.modes[index],
            "anchor_frame": anchor_frame,
            "anchor_xz_world": anchor_xz.detach().clone(),
            "anchor_yaw_world": anchor_yaw.detach().clone(),
            "clip_idx": int(raw_sample.get("clip_idx", index)),
            "raw_id": raw_sample.get("raw_id", raw_sample.get("name", str(index))),
            "name": raw_sample.get("name", raw_sample.get("raw_id", str(index))),
            "split_index": raw_sample.get("split_index", index),
            "split_file": raw_sample.get("split_file"),
            "dataset": raw_sample.get("dataset"),
        }
        return self.process_output(
            base_sample,
            self.process_path(
                base_sample,
                path_mode=plan.path_modes[index],
                offset_start_frames=int(plan.offset_start_frames[index].item()),
            ),
        )

    @staticmethod
    def _motion_of(sample: dict[str, Any]) -> torch.Tensor:
        motion = sample.get("feature", sample.get("motion_263"))
        if motion is None:
            raise KeyError("raw sample must contain 'feature' or 'motion_263'")
        if not isinstance(motion, torch.Tensor):
            motion = torch.as_tensor(motion, dtype=torch.float32)
        elif motion.dtype != torch.float32:
            motion = motion.float()
        return motion

    @staticmethod
    def _text_of(sample: dict[str, Any]) -> str:
        text = sample.get("text", "")
        if isinstance(text, (list, tuple)):
            return str(text[0]) if text else ""
        return str(text)

    def process_root(self, motion_263: torch.Tensor):
        root_quat, root_xyz = recover_root_rot_pos(motion_263)
        root_quat = root_quat[0]
        root_xyz = root_xyz[0]
        root_yaw = root_quat_to_physical_yaw(root_quat)
        motion_5d_world = torch.cat(
            [
                root_xyz,
                torch.cos(root_yaw)[..., None],
                torch.sin(root_yaw)[..., None],
            ],
            dim=-1,
        )
        motion_7d_world = root_to_traj_feats_7d(
            root_quat.unsqueeze(0), root_xyz.unsqueeze(0),
        )[0]
        return root_xyz, root_yaw, motion_5d_world, motion_7d_world

    def process_history(
        self,
        motion_5d_world: torch.Tensor,
        history_frame_indices: list[int],
        valid_history_frames: int,
        anchor_xz: torch.Tensor,
        anchor_yaw: torch.Tensor,
    ):
        current_motion_world = motion_5d_world[history_frame_indices]
        current_motion_local = canonicalize_5d(
            current_motion_world, anchor_xz, anchor_yaw,
        )
        if valid_history_frames < self.n_hist:
            pad = current_motion_local.new_zeros(
                self.n_hist - valid_history_frames, 5,
            )
            current_motion = torch.cat([pad, current_motion_local], dim=0)
        else:
            current_motion = current_motion_local
        history_mask = torch.zeros(
            self.n_hist,
            device=current_motion.device,
            dtype=torch.bool,
        )
        history_mask[self.n_hist - valid_history_frames :] = True
        return current_motion, history_mask

    def process_target(
        self,
        motion_7d_world: torch.Tensor,
        anchor_frame: int,
        target_frame_count: int,
        anchor_xz: torch.Tensor,
        anchor_yaw: torch.Tensor,
    ):
        target_world = motion_7d_world[
            anchor_frame + 1 : anchor_frame + 1 + target_frame_count
        ]
        target_local = canonicalize_7d(target_world, anchor_xz, anchor_yaw)
        target_waypoints = _pad_or_truncate(target_local, self.max_frames)
        target_waypoints_physical = target_waypoints.clone()
        target_mask = torch.zeros(
            self.max_frames,
            device=target_waypoints.device,
            dtype=torch.bool,
        )
        target_mask[:target_frame_count] = True
        return target_waypoints, target_waypoints_physical, target_mask

    def process_path(
        self,
        sample: dict[str, Any],
        *,
        path_mode: str,
        offset_start_frames: int,
    ) -> dict[str, Any]:
        waypoints = sample["target_waypoints"][..., :5]
        waypoints_mask = sample["target_mask"]
        valid_frame_count = int(waypoints_mask.sum().item())
        physical_wp = sample["target_waypoints_physical"]
        future_xz = physical_wp[:valid_frame_count, [0, 2]]
        condition = build_path_condition(
            future_xz,
            n_path=self.n_path,
            valid_frame_count=valid_frame_count,
            max_frames=self.max_frames,
            path_mode=path_mode,
            offset_start_frames=offset_start_frames,
            sparse_point_range=self.sparse_path_point_range,
            rng=self._rng,
        )

        path_tokens = condition.path
        path_features_raw = condition.path_features_raw
        path_features = path_features_raw
        if self.normalize:
            path_tokens = self._zscore_path_xz(path_tokens)
            path_features = self._normalize_path_features(path_features)

        return {
            "waypoints": waypoints,
            "waypoints_mask": waypoints_mask,
            "physical_waypoints": physical_wp,
            "condition": condition,
            "path_tokens": path_tokens,
            "path_features": path_features,
            "path_features_raw": path_features_raw,
        }

    @staticmethod
    def process_output(
        sample: dict[str, Any],
        path_sample: dict[str, Any],
    ) -> dict[str, Any]:
        condition = path_sample["condition"]
        out = dict(sample)
        out.update(
            {
                "path": path_sample["path_tokens"],
                "path_valid_mask": condition.path_valid_mask,
                "path_control_mask": condition.path_control_mask,
                "path_features": path_sample["path_features"],
                "path_features_raw": path_sample["path_features_raw"],
                "path_mode": condition.path_mode,
                "history_motion": sample["current_motion"],
                "waypoints": path_sample["waypoints"],
                "waypoints_physical": path_sample["physical_waypoints"],
                "waypoints_mask": path_sample["waypoints_mask"],
                "path_supervision_mask": condition.path_supervision_mask,
                "offset_start_frames": torch.tensor(
                    condition.offset_start_frames,
                    dtype=torch.long,
                ),
            }
        )
        return out

    def _load_motion_stats(self, stats_dir: Path) -> None:
        self._cm_mean = torch.as_tensor(
            np.load(stats_dir / "current_motion_mean.npy"), dtype=torch.float32,
        )
        self._cm_std = torch.as_tensor(
            np.load(stats_dir / "current_motion_std.npy"), dtype=torch.float32,
        ).clamp(min=1e-6)
        self._cm_norm_idx = torch.as_tensor(
            np.load(stats_dir / "current_motion_norm_indices.npy"), dtype=torch.long,
        )
        self._wp_mean = torch.as_tensor(
            np.load(stats_dir / "waypoint_mean.npy"), dtype=torch.float32,
        )
        self._wp_std = torch.as_tensor(
            np.load(stats_dir / "waypoint_std.npy"), dtype=torch.float32,
        ).clamp(min=1e-6)
        self._wp_norm_idx = torch.as_tensor(
            np.load(stats_dir / "waypoint_norm_indices.npy"), dtype=torch.long,
        )
        if self._cm_mean.shape != (5,) or self._cm_std.shape != (5,):
            raise ValueError(
                f"current_motion stats must be shape (5,), got "
                f"mean={tuple(self._cm_mean.shape)} std={tuple(self._cm_std.shape)}"
            )
        if self._wp_mean.shape != (7,) or self._wp_std.shape != (7,):
            raise ValueError(
                f"waypoint stats must be shape (7,), got "
                f"mean={tuple(self._wp_mean.shape)} std={tuple(self._wp_std.shape)}"
            )
        if self._cm_norm_idx.numel() and int(self._cm_norm_idx.max()) >= 5:
            raise ValueError(
                f"current_motion_norm_indices out of range for dim 5: "
                f"{self._cm_norm_idx.tolist()}"
            )
        if set(self._cm_norm_idx.tolist()) & {3, 4}:
            raise ValueError(
                "current_motion_norm_indices must NOT include heading channels 3/4 "
                f"(cos/sin yaw are unit-vector invariant): {self._cm_norm_idx.tolist()}"
            )
        if self._wp_norm_idx.numel() and int(self._wp_norm_idx.max()) >= 7:
            raise ValueError(
                f"waypoint_norm_indices out of range for dim 7: "
                f"{self._wp_norm_idx.tolist()}"
            )
        if set(self._wp_norm_idx.tolist()) & {3, 4}:
            raise ValueError(
                "waypoint_norm_indices must NOT include heading channels 3/4 "
                f"(cos/sin yaw are unit-vector invariant): {self._wp_norm_idx.tolist()}"
            )

    @staticmethod
    def _apply_zscore(
        tensor: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        norm_idx: torch.Tensor,
    ) -> torch.Tensor:
        out = tensor.clone()
        mean = mean.to(device=tensor.device, dtype=tensor.dtype)
        std = std.to(device=tensor.device, dtype=tensor.dtype)
        for c in norm_idx.tolist():
            out[..., c] = (out[..., c] - mean[c]) / std[c]
        return out

    def _zscore_path_xz(self, path_xz: torch.Tensor) -> torch.Tensor:
        if self._wp_mean is None or self._wp_std is None or self._wp_norm_idx is None:
            return path_xz
        out = path_xz.clone()
        idx_set = set(self._wp_norm_idx.tolist())
        mean = self._wp_mean.to(device=path_xz.device, dtype=path_xz.dtype)
        std = self._wp_std.to(device=path_xz.device, dtype=path_xz.dtype)
        if 0 in idx_set:
            out[..., 0] = (out[..., 0] - mean[0]) / std[0]
        if 2 in idx_set:
            out[..., 1] = (out[..., 1] - mean[2]) / std[2]
        return out

    def _normalize_path_features(self, features: torch.Tensor) -> torch.Tensor:
        if self._pf_mean is None or self._pf_std is None:
            return features
        mean = self._pf_mean.to(device=features.device, dtype=features.dtype)
        std = self._pf_std.to(device=features.device, dtype=features.dtype)
        return (features - mean) / std


__all__ = ["RefinerSampleBuilder"]
