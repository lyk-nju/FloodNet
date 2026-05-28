"""Deterministic fixed-sample wrapper for RootRefiner diagnostics.

The normal RefinerDataset intentionally samples a new task on every __getitem__
(mode, anchor, horizon, path augmentation). That is good for training, but bad
for overfit diagnostics. This module freezes those sampled tasks into a small
list so the same index is bitwise stable across epochs and validation passes.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch.utils.data import Dataset


def clone_refiner_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a Refiner sample, cloning tensors explicitly."""
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


class FixedRefinerSampleDataset(Dataset):
    """Dataset backed by pre-sampled RefinerDataset outputs."""

    def __init__(self, samples: list[dict[str, Any]]):
        if not samples:
            raise ValueError("FixedRefinerSampleDataset requires at least one sample")
        self._samples = [clone_refiner_sample(sample) for sample in samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return clone_refiner_sample(self._samples[int(idx)])


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


def build_fixed_refiner_samples(
    source,
    *,
    num_samples: int,
    mode_policy: str = "mixed",
    force_no_path_aug: bool = True,
    force_text_idx: int | None = 0,
) -> list[dict[str, Any]]:
    """Pre-sample deterministic Refiner tasks from *source*.

    Anchor frame and num_tokens are intentionally not forced here: they are drawn
    once from the source dataset RNG and then frozen. This preserves realistic
    task diversity while making overfit/validation deterministic.
    """
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
            )
        else:
            sample = source[src_idx]
        samples.append(clone_refiner_sample(sample))
    return samples


__all__ = [
    "FixedRefinerSampleDataset",
    "build_fixed_refiner_samples",
    "clone_refiner_sample",
]
