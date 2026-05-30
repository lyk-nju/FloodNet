"""HumanML3D-backed RootRefiner datasets."""

from __future__ import annotations

from typing import Any

import torch

from datasets._humanml3d_refiner_base import (
    RefinerDataset as _BaseHumanML3DRefinerDataset,
    refiner_worker_init_fn,
)
from datasets.refiner_fixed import (
    FixedRefinerSampleDataset as FixedHumanML3DRefinerDataset,
    build_fixed_refiner_samples,
    clone_refiner_sample,
)


class HumanML3DRefinerDataset(_BaseHumanML3DRefinerDataset):
    """RootRefiner dataset using the new batch contract.

    This class initially reuses the legacy HumanML3D-to-refiner motion pipeline
    and exposes the redesigned field names and masks. Path-mode construction is
    implemented in `utils.refiner.path_condition` and can be integrated without
    changing the public contract.
    """

    def __init__(
        self,
        *args,
        horizon_policy: str | None = None,
        path_condition_policy: str = "dense_path",
        path_condition_ratios: dict[str, float] | None = None,
        offset_start_enabled: bool = False,
        offset_start_prob: float = 0.0,
        offset_start_max_frames: int = 40,
        offset_start_apply_to: tuple[str, ...] | list[str] = ("dense_path", "sparse_path"),
        sparse_path_point_range: tuple[int, int] = (3, 8),
        **kwargs,
    ):
        if horizon_policy is not None and "num_token_policy" not in kwargs:
            kwargs["num_token_policy"] = horizon_policy
        super().__init__(*args, **kwargs)
        self.path_condition_policy = str(path_condition_policy)
        self.path_condition_ratios = dict(
            path_condition_ratios
            or {"dense_path": 0.5, "sparse_path": 0.3, "goal_point": 0.2}
        )
        self.offset_start_enabled = bool(offset_start_enabled)
        self.offset_start_prob = float(offset_start_prob)
        self.offset_start_max_frames = int(offset_start_max_frames)
        self.offset_start_apply_to = tuple(offset_start_apply_to)
        self.sparse_path_point_range = tuple(int(v) for v in sparse_path_point_range)
        self.max_frames = self.max_frames if hasattr(self, "max_frames") else None

    @property
    def max_frames(self) -> int:
        from utils.token_frame import num_frames_for_tokens

        return num_frames_for_tokens(self.max_tokens, self.frames_per_token)

    @max_frames.setter
    def max_frames(self, value: int | None) -> None:
        # Compatibility no-op; the legacy class computes this ad hoc.
        pass

    def get_sample(
        self,
        idx_in_valid: int,
        *,
        force_path_mode: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        force_no_path_aug = bool(kwargs.get("force_no_path_aug", False))
        sample = super().get_sample(idx_in_valid, **kwargs)
        return self._convert_legacy_sample(
            sample,
            force_path_mode=force_path_mode,
            force_no_path_aug=force_no_path_aug,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_idx = self.valid_indices[idx]
        sample = self._make_sample(clip_idx)
        return self._convert_legacy_sample(
            sample,
            force_path_mode=None,
            force_no_path_aug=False,
        )

    def _convert_legacy_sample(
        self,
        sample: dict[str, Any],
        *,
        force_path_mode: str | None,
        force_no_path_aug: bool,
    ) -> dict[str, Any]:
        from utils.refiner.path_condition import build_path_condition

        waypoints = sample["target_waypoints"][..., :5]
        waypoints_mask = sample["target_mask"]
        valid_frame_count = int(waypoints_mask.sum().item())
        future_xz = waypoints[:valid_frame_count, [0, 2]]
        path_mode = force_path_mode or self._sample_path_mode()
        offset_start_frames = 0
        if (
            not force_no_path_aug
            and self.offset_start_enabled
            and path_mode in self.offset_start_apply_to
            and self._rng.random() < self.offset_start_prob
        ):
            max_offset = min(self.offset_start_max_frames, max(0, valid_frame_count - 2))
            offset_start_frames = self._rng.randint(0, max_offset) if max_offset > 0 else 0

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

        out = dict(sample)
        out.update(
            {
                "path": condition.path,
                "path_valid_mask": condition.path_valid_mask,
                "path_control_mask": condition.path_control_mask,
                "path_features": condition.path_features_raw,
                "path_mode": condition.path_mode,
                "history_motion": sample["current_motion"],
                "waypoints": waypoints,
                "waypoints_mask": waypoints_mask,
                "path_supervision_mask": condition.path_supervision_mask,
                "offset_start_frames": torch.tensor(
                    condition.offset_start_frames,
                    dtype=torch.long,
                ),
            }
        )
        return out

    def _sample_path_mode(self) -> str:
        policy = self.path_condition_policy
        if policy in {"dense_path", "sparse_path", "goal_point"}:
            return policy
        if policy != "mixed":
            raise ValueError(f"unknown path_condition_policy {policy!r}")
        total = sum(max(0.0, float(v)) for v in self.path_condition_ratios.values())
        if total <= 0:
            return "dense_path"
        pick = self._rng.random() * total
        acc = 0.0
        for mode in ("dense_path", "sparse_path", "goal_point"):
            acc += max(0.0, float(self.path_condition_ratios.get(mode, 0.0)))
            if pick <= acc:
                return mode
        return "goal_point"

    @staticmethod
    def _path_features(path: torch.Tensor) -> torch.Tensor:
        from utils.refiner.path_condition import compute_path_features

        return compute_path_features(path)


def refiner_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "text": [sample["text"] for sample in batch],
        "path_mode": [sample["path_mode"] for sample in batch],
    }
    if "mode" in batch[0]:
        out["mode"] = [sample["mode"] for sample in batch]
    for key in (
        "path",
        "path_valid_mask",
        "path_control_mask",
        "path_features",
        "history_motion",
        "history_mask",
        "waypoints",
        "waypoints_mask",
        "path_supervision_mask",
        "offset_start_frames",
        "num_tokens",
    ):
        out[key] = torch.stack([sample[key] for sample in batch])
    return out


__all__ = [
    "HumanML3DRefinerDataset",
    "FixedHumanML3DRefinerDataset",
    "build_fixed_refiner_samples",
    "clone_refiner_sample",
    "refiner_collate",
    "refiner_worker_init_fn",
]
