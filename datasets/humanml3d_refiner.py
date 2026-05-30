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
        path_feature_stats_dir: str | None = None,
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
        # R2.5: optional PHYSICAL path-feature normalization stats (own mean/std,
        # NOT the waypoint stats). When normalize=True and a stats dir is given,
        # path_features are z-scored by these; otherwise they stay raw-physical.
        self._pf_mean = None
        self._pf_std = None
        if path_feature_stats_dir is not None:
            from utils.refiner.path_feature_stats import load_path_feature_stats

            stats = load_path_feature_stats(path_feature_stats_dir)
            self._pf_mean = stats.mean
            self._pf_std = stats.std

    def _zscore_path_xz(self, path_xz: torch.Tensor) -> torch.Tensor:
        """Z-score path geometry [N, 2] with the WAYPOINT x/z stats (indices 0,2),
        mirroring the base dataset's xz_path normalization so path tokens match
        the z-scored waypoints used by the control loss."""
        wp_mean, wp_std = self._wp_mean, self._wp_std
        wp_idx = self._wp_norm_idx
        if wp_mean is None or wp_std is None or wp_idx is None:
            return path_xz
        out = path_xz.clone()
        idx_set = set(wp_idx.tolist())
        if 0 in idx_set:
            out[..., 0] = (out[..., 0] - wp_mean[0]) / wp_std[0]
        if 2 in idx_set:
            out[..., 1] = (out[..., 1] - wp_mean[2]) / wp_std[2]
        return out

    def _normalize_path_features(self, features: torch.Tensor) -> torch.Tensor:
        """Z-score physical path_features by their OWN stats, if loaded."""
        if self._pf_mean is None or self._pf_std is None:
            return features
        return (features - self._pf_mean) / self._pf_std

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
        # R2.5: build the path condition (and its physical path_features) from the
        # PHYSICAL pre-z-score waypoints, not the z-scored `target_waypoints`.
        # Reading z-scored xz here made compute_path_features run in anisotropic
        # z-units → non-physical path_length and the duration-head overfit. The
        # geometry `path` tokens are re-z-scored below to stay in the waypoint
        # space the control loss compares against; only path_features stay physical.
        if "target_waypoints_physical" in sample:
            physical_wp = sample["target_waypoints_physical"]
        elif getattr(self, "normalize", False):
            # When normalizing, the base MUST hand us the pre-z-score waypoints.
            # Silently falling back to the z-scored `target_waypoints` would
            # rebuild path_features in z-score space — the exact R2.5 overfit
            # bug — so fail loudly (e.g. a sample cache built before R2.5).
            raise KeyError(
                "target_waypoints_physical missing while normalize=True; the "
                "base dataset must emit pre-z-score waypoints. Rebuild any stale "
                "sample cache — a z-scored fallback would silently reintroduce "
                "the R2.5 path-feature-space bug."
            )
        else:
            # normalize=False: target_waypoints is already physical (never z-scored).
            physical_wp = sample["target_waypoints"]
        future_xz = physical_wp[:valid_frame_count, [0, 2]]
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
        # R2.5: path was built in physical space. Re-apply the waypoint xz z-score
        # to the GEOMETRY tokens (only) so they live in the same space as the
        # z-scored `waypoints` the dense/sparse control loss compares them against.
        # path_features stay PHYSICAL (optionally normalized by their OWN stats).
        path_tokens = condition.path
        path_features = condition.path_features_raw
        if getattr(self, "normalize", False):
            path_tokens = self._zscore_path_xz(path_tokens)
            path_features = self._normalize_path_features(path_features)
        out.update(
            {
                "path": path_tokens,
                "path_valid_mask": condition.path_valid_mask,
                "path_control_mask": condition.path_control_mask,
                "path_features": path_features,
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
