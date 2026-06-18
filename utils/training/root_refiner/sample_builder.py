"""RootRefiner tensor sample construction."""

from __future__ import annotations

import random as random_module
from typing import Any

import torch

from utils.local_frame import (
    canonicalize_5d,
    canonicalize_7d,
    root_quat_to_physical_yaw,
)
from utils.motion_process import recover_root_rot_pos, root_to_traj_feats_7d
from utils.conditions.root_refiner import build_root_refiner_path_condition
from utils.training.root_refiner.sample_creator import RefinerSample


def _pad_or_truncate(values: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = values.shape[0]
    if current_len == target_len:
        return values
    if current_len > target_len:
        return values[:target_len]
    pad_shape = list(values.shape)
    pad_shape[0] = target_len - current_len
    return torch.cat([values, values.new_zeros(*pad_shape)], dim=0)


class RefinerSampleBuilder:
    """Build RootRefiner training samples from raw motion and a sample plan."""

    def __init__(
        self,
        *,
        n_hist: int = 20,
        n_path: int = 64,
        max_frames: int = 193,
        min_frames: int = 13,
        sparse_path_point_range: tuple[int, int] = (3, 8),
        seed: int | None = None,
    ):
        self.n_hist = int(n_hist)
        self.n_path = int(n_path)
        self._max_frames = int(max_frames)
        self.max_frames_value = self._max_frames
        self.min_frames = int(min_frames)
        self.sparse_path_point_range = tuple(int(v) for v in sparse_path_point_range)
        self._seed = seed
        self._rng = random_module.Random(seed)

    @property
    def max_frames(self) -> int:
        return self._max_frames

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

        sample = {
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
            sample,
            self.process_path(
                sample,
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
        history_motion_world = motion_5d_world[history_frame_indices]
        history_motion_local = canonicalize_5d(
            history_motion_world, anchor_xz, anchor_yaw,
        )
        if valid_history_frames < self.n_hist:
            pad = history_motion_local.new_zeros(
                self.n_hist - valid_history_frames, 5,
            )
            history_motion = torch.cat([pad, history_motion_local], dim=0)
        else:
            history_motion = history_motion_local
        history_mask = torch.zeros(
            self.n_hist,
            device=history_motion.device,
            dtype=torch.bool,
        )
        history_mask[self.n_hist - valid_history_frames :] = True
        return history_motion, history_mask

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
        target_waypoints = sample["target_waypoints"][..., :5]
        target_mask = sample["target_mask"]
        valid_frame_count = int(target_mask.sum().item())
        physical_waypoints = sample["target_waypoints_physical"]
        future_xz = physical_waypoints[:valid_frame_count, [0, 2]]
        condition = build_root_refiner_path_condition(
            future_xz,
            n_path=self.n_path,
            valid_frame_count=valid_frame_count,
            max_frames=self.max_frames,
            path_mode=path_mode,
            offset_start_frames=offset_start_frames,
            sparse_point_range=self.sparse_path_point_range,
            rng=self._rng,
        )

        return {
            "waypoints": target_waypoints,
            "waypoints_mask": target_mask,
            "physical_waypoints": physical_waypoints,
            "condition": condition,
            "path_tokens": condition.path,
            "path_features": condition.path_features,
            "path_features_raw": condition.path_features_raw,
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


__all__ = ["RefinerSampleBuilder"]
