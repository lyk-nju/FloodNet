from __future__ import annotations

import copy
import random as random_module
import torch

from pathlib import Path
from typing import Any
from torch.utils.data import Dataset
from utils.token_frame import num_frames_for_tokens
from utils.training.root_refiner.sample_builder import RefinerSampleBuilder
from utils.training.root_refiner.sample_creator import RefinerSampleCreator


class RootRefinerBatchBuilder:
    """Build RootRefiner training samples from raw HumanML3D samples."""

    def __init__(
        self,
        *,
        n_hist: int = 20,
        n_path: int = 64,
        max_tokens: int = 49,
        min_tokens: int = 4,
        frames_per_token: int = 4,
        full_plan_ratio: float = 0.5,
        num_token_policy: str = "random",
        horizon_policy: str | None = None,
        path_condition_policy: str = "dense_path",
        path_condition_ratios: dict[str, float] | None = None,
        offset_start_enabled: bool = False,
        offset_start_prob: float = 0.0,
        offset_start_max_frames: int = 40,
        offset_start_apply_to: tuple[str, ...] | list[str] = (
            "dense_path",
            "sparse_path",
        ),
        sparse_path_point_range: tuple[int, int] = (3, 8),
        normalize: bool = False,
        stats_dir=None,
        path_feature_stats_dir: str | None = None,
        sampling_config_hash: str | None = None,
        seed: int | None = None,
        randomize_caption: bool = True,
    ):
        self.n_hist = int(n_hist)
        self.n_path = int(n_path)
        self.max_tokens = int(max_tokens)
        self.min_tokens = int(min_tokens)
        self.frames_per_token = int(frames_per_token)
        self.full_plan_ratio = float(full_plan_ratio)
        self.num_token_policy = str(
            num_token_policy if horizon_policy is None else horizon_policy
        )
        self.path_condition_policy = str(path_condition_policy)
        self.path_condition_ratios = path_condition_ratios
        self.offset_start_enabled = bool(offset_start_enabled)
        self.offset_start_prob = float(offset_start_prob)
        self.offset_start_max_frames = int(offset_start_max_frames)
        self.offset_start_apply_to = tuple(offset_start_apply_to)
        self.sparse_path_point_range = tuple(int(v) for v in sparse_path_point_range)
        self.normalize = bool(normalize)
        self.randomize_caption = bool(randomize_caption)
        self._seed = seed
        self._rng = random_module.Random(seed)

        self.sample_creator = RefinerSampleCreator(
            n_hist=self.n_hist,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            frames_per_token=self.frames_per_token,
            full_plan_ratio=self.full_plan_ratio,
            num_token_policy=self.num_token_policy,
            path_condition_policy=self.path_condition_policy,
            path_condition_ratios=self.path_condition_ratios,
            offset_start_enabled=self.offset_start_enabled,
            offset_start_prob=self.offset_start_prob,
            offset_start_max_frames=self.offset_start_max_frames,
            offset_start_apply_to=self.offset_start_apply_to,
            seed=seed,
        )
        self.sample_builder = RefinerSampleBuilder(
            n_hist=self.n_hist,
            n_path=self.n_path,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            frames_per_token=self.frames_per_token,
            normalize=self.normalize,
            stats_dir=stats_dir if self.normalize else None,
            sparse_path_point_range=self.sparse_path_point_range,
            path_feature_stats_dir=path_feature_stats_dir,
            sampling_config_hash=sampling_config_hash,
            seed=seed,
        )
        self._copy_builder_stats()

    @property
    def max_frames(self) -> int:
        return num_frames_for_tokens(self.max_tokens, self.frames_per_token)

    def reset_rng(self) -> None:
        self._rng = random_module.Random(self._seed)
        self.sample_creator.reset_rng()
        self.sample_builder.reset_rng()

    def set_worker_seed(self) -> None:
        base = int(torch.initial_seed()) % (2 ** 31)
        if self._seed is not None:
            base = (base + int(self._seed)) % (2 ** 31)
        self._rng = random_module.Random(base)
        self.sample_creator._rng = random_module.Random(base)
        self.sample_builder._rng = random_module.Random(base)

    def build(
        self,
        raw_sample: dict[str, Any],
        *,
        index: int = 0,
        force_mode: str | None = None,
        force_num_tokens: int | None = None,
        force_anchor_frame: int | None = None,
        force_path_mode: str | None = None,
        force_no_path_aug: bool = False,
        force_text_idx: int | None = None,
    ) -> dict[str, Any]:
        motion = self._motion_of(raw_sample)
        text = self._sample_text(raw_sample, force_text_idx)
        plan = self.sample_creator.create(
            torch.tensor([int(motion.shape[0])], dtype=torch.long),
            force_mode=force_mode,
            force_num_tokens=force_num_tokens,
            force_anchor_frame=force_anchor_frame,
            force_path_mode=force_path_mode,
            force_no_path_aug=force_no_path_aug,
        )
        builder_sample = self._builder_sample(raw_sample, motion, text, index)
        return self.sample_builder.build(builder_sample, plan)

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

    def _sample_text(self, sample: dict[str, Any], force_text_idx: int | None) -> str:
        text_options = sample.get("text_all") or sample.get("texts")
        if text_options:
            if force_text_idx is not None:
                return str(text_options[int(force_text_idx) % len(text_options)])
            if self.randomize_caption:
                return str(self._rng.choice(text_options))
            return str(text_options[0])
        text = sample.get("text", "")
        if isinstance(text, (list, tuple)):
            return str(text[0]) if text else ""
        return str(text)

    @staticmethod
    def _builder_sample(
        sample: dict[str, Any],
        motion: torch.Tensor,
        text: str,
        index: int,
    ) -> dict[str, Any]:
        out = dict(sample)
        out["motion_263"] = motion
        out["text"] = text
        out.setdefault("texts", [text])
        out.setdefault("clip_idx", int(index))
        out.setdefault("raw_id", out.get("name", str(index)))
        out.setdefault("name", out.get("raw_id", str(index)))
        out.setdefault("split_index", int(index))
        out.setdefault("split_file", out.get("split_file"))
        out.setdefault("dataset", out.get("dataset"))
        return out

    def _copy_builder_stats(self) -> None:
        self._cm_mean = self.sample_builder._cm_mean
        self._cm_std = self.sample_builder._cm_std
        self._cm_norm_idx = self.sample_builder._cm_norm_idx
        self._wp_mean = self.sample_builder._wp_mean
        self._wp_std = self.sample_builder._wp_std
        self._wp_norm_idx = self.sample_builder._wp_norm_idx
        self._pf_mean = self.sample_builder._pf_mean
        self._pf_std = self.sample_builder._pf_std


class RootRefinerDataset(Dataset):
    """RootRefiner training adapter over a raw HumanML3D dataset."""

    def __init__(self, raw_dataset, **builder_kwargs):
        self.raw_dataset = raw_dataset
        self.batch_builder = RootRefinerBatchBuilder(**builder_kwargs)
        self.n_hist = self.batch_builder.n_hist
        self.n_path = self.batch_builder.n_path
        self.max_tokens = self.batch_builder.max_tokens
        self.min_tokens = self.batch_builder.min_tokens
        self.frames_per_token = self.batch_builder.frames_per_token
        self.full_plan_ratio = self.batch_builder.full_plan_ratio
        self.num_token_policy = self.batch_builder.num_token_policy
        self.path_condition_policy = self.batch_builder.path_condition_policy
        self.path_condition_ratios = self.batch_builder.path_condition_ratios
        self.offset_start_enabled = self.batch_builder.offset_start_enabled
        self.offset_start_prob = self.batch_builder.offset_start_prob
        self.offset_start_max_frames = self.batch_builder.offset_start_max_frames
        self.offset_start_apply_to = self.batch_builder.offset_start_apply_to
        self.sparse_path_point_range = self.batch_builder.sparse_path_point_range
        self.normalize = self.batch_builder.normalize
        self._copy_builder_stats()

        self._sample_lengths = [
            self._effective_motion_length(i) for i in range(len(raw_dataset))
        ]
        min_full = num_frames_for_tokens(self.min_tokens, self.frames_per_token)
        min_sliding = (self.n_hist - 1) + min_full
        self.full_eligible_indices = [
            i for i, length in enumerate(self._sample_lengths) if length >= min_full
        ]
        self.sliding_eligible_indices = {
            i for i, length in enumerate(self._sample_lengths) if length >= min_sliding
        }
        self.valid_indices = self.full_eligible_indices

    @property
    def max_frames(self) -> int:
        return self.batch_builder.max_frames

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._process(self.valid_indices[int(idx)])

    def get_sample(
        self,
        idx_in_valid: int,
        *,
        force_mode: str | None = None,
        force_num_tokens: int | None = None,
        force_anchor_frame: int | None = None,
        force_path_mode: str | None = None,
        force_no_path_aug: bool = False,
        force_text_idx: int | None = None,
    ) -> dict[str, Any]:
        return self._process(
            self.valid_indices[int(idx_in_valid)],
            force_mode=force_mode,
            force_num_tokens=force_num_tokens,
            force_anchor_frame=force_anchor_frame,
            force_path_mode=force_path_mode,
            force_no_path_aug=force_no_path_aug,
            force_text_idx=force_text_idx,
        )

    def set_worker_seed(self) -> None:
        self.batch_builder.set_worker_seed()

    def reset_rng(self) -> None:
        self.batch_builder.reset_rng()

    def _process(
        self,
        raw_idx: int,
        *,
        force_mode: str | None = None,
        force_num_tokens: int | None = None,
        force_anchor_frame: int | None = None,
        force_path_mode: str | None = None,
        force_no_path_aug: bool = False,
        force_text_idx: int | None = None,
    ) -> dict[str, Any]:
        raw_sample = dict(self.raw_dataset[int(raw_idx)])
        raw_sample.setdefault("raw_id", raw_sample.get("name", str(raw_idx)))
        raw_sample.setdefault("split_index", int(raw_idx))
        if "split_file" not in raw_sample:
            file_list = getattr(self.raw_dataset, "file_list", None)
            if file_list:
                raw_sample["split_file"] = Path(file_list[0]).name
        return self.batch_builder.build(
            raw_sample,
            index=int(raw_idx),
            force_mode=force_mode,
            force_num_tokens=force_num_tokens,
            force_anchor_frame=force_anchor_frame,
            force_path_mode=force_path_mode,
            force_no_path_aug=force_no_path_aug,
            force_text_idx=force_text_idx,
        )

    def _effective_motion_length(self, raw_idx: int) -> int:
        records = getattr(self.raw_dataset, "dataset", None)
        if records is not None and raw_idx < len(records):
            record = records[int(raw_idx)]
            if "feature_length" in record:
                length = int(record["feature_length"])
            elif "feature" in record:
                length = int(record["feature"].shape[0])
            elif "motion_263" in record:
                length = int(record["motion_263"].shape[0])
            else:
                length = self._sample_motion_length(raw_idx)
        else:
            length = self._sample_motion_length(raw_idx)
        window_length = getattr(self.raw_dataset, "window_length", None)
        if window_length is not None:
            length = min(length, int(window_length))
        return length

    def _sample_motion_length(self, raw_idx: int) -> int:
        sample = self.raw_dataset[int(raw_idx)]
        return int(RootRefinerBatchBuilder._motion_of(sample).shape[0])

    def _copy_builder_stats(self) -> None:
        self._cm_mean = self.batch_builder._cm_mean
        self._cm_std = self.batch_builder._cm_std
        self._cm_norm_idx = self.batch_builder._cm_norm_idx
        self._wp_mean = self.batch_builder._wp_mean
        self._wp_std = self.batch_builder._wp_std
        self._wp_norm_idx = self.batch_builder._wp_norm_idx
        self._pf_mean = self.batch_builder._pf_mean
        self._pf_std = self.batch_builder._pf_std


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
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
        "path_features_raw",
        "history_motion",
        "history_mask",
        "waypoints",
        "waypoints_mask",
        "path_supervision_mask",
        "offset_start_frames",
        "num_tokens",
    ):
        out[key] = torch.stack([sample[key] for sample in batch])
    if "waypoints_physical" in batch[0]:
        out["waypoints_physical"] = torch.stack(
            [sample["waypoints_physical"] for sample in batch],
        )
    return out


def worker_init_fn(worker_id: int) -> None:
    """Set a distinct dataset RNG for each DataLoader worker."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "set_worker_seed"):
        info.dataset.set_worker_seed()


def copy_refiner_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Copy a sample, cloning tensors."""
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


class FixedRefinerSampleDataset(Dataset):
    """Fixed RootRefiner samples."""

    def __init__(self, samples: list[dict[str, Any]]):
        if not samples:
            raise ValueError("FixedRefinerSampleDataset requires at least one sample")
        self._samples = [copy_refiner_sample(sample) for sample in samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return copy_refiner_sample(self._samples[int(idx)])


def _mode_for_index(mode_policy: str, idx: int) -> str | None:
    mode_policy = str(mode_policy).lower()
    if mode_policy in ("random", "dataset", "none"):
        return None
    if mode_policy in ("full", "sliding"):
        return mode_policy
    if mode_policy in ("mixed", "alternate", "alternating"):
        return "full" if idx % 2 == 0 else "sliding"
    raise ValueError(
        "mode_policy must be one of full, sliding, mixed, or random; "
        f"got {mode_policy!r}"
    )


def build_fixed_samples(
    source,
    *,
    num_samples: int,
    mode_policy: str = "mixed",
    force_no_path_aug: bool = True,
    force_text_idx: int | None = 0,
    force_anchor_frame: int | None = None,
) -> list[dict[str, Any]]:
    """Freeze deterministic samples from a source dataset."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if len(source) == 0:
        raise ValueError("source dataset is empty")
    if hasattr(source, "reset_rng"):
        source.reset_rng()

    samples: list[dict[str, Any]] = []
    for out_idx in range(int(num_samples)):
        src_idx = out_idx % len(source)
        if hasattr(source, "get_sample"):
            sample = source.get_sample(
                src_idx,
                force_mode=_mode_for_index(mode_policy, out_idx),
                force_no_path_aug=bool(force_no_path_aug),
                force_text_idx=force_text_idx,
                force_anchor_frame=force_anchor_frame,
            )
        else:
            sample = source[src_idx]
        samples.append(copy_refiner_sample(sample))
    return samples


__all__ = [
    "FixedRefinerSampleDataset",
    "RootRefinerBatchBuilder",
    "RootRefinerDataset",
    "build_fixed_samples",
    "collate_fn",
    "copy_refiner_sample",
    "worker_init_fn",
]
