"""Immutable decision snapshots and weighted replay sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class NoiseInitializerReplaySnapshot:
    seed: int
    commit_index: int
    source: str
    valid_affected_frames: int
    requested_affected_frames: int
    model_state: dict[str, Any]
    vae_state: Any
    context: Any
    batch: dict[str, Any]
    conditioner: Any
    recovery: Any
    first_chunk: bool


@dataclass(frozen=True)
class ReplaySamplingConfig:
    late_progress_start: float = 0.55
    late_weight_multiplier: float = 3.0
    initializer_probability: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.late_progress_start) <= 1.0:
            raise ValueError("late_progress_start must be in [0,1]")
        if float(self.late_weight_multiplier) <= 0.0:
            raise ValueError("late_weight_multiplier must be positive")
        if not 0.0 <= float(self.initializer_probability) <= 1.0:
            raise ValueError("initializer_probability must be in [0,1]")


def snapshot_sampling_weight(
    snapshot: NoiseInitializerReplaySnapshot,
    *,
    max_commit: int,
    cfg: ReplaySamplingConfig,
) -> float:
    requested = max(1, int(snapshot.requested_affected_frames))
    valid_ratio = max(0.0, min(1.0, int(snapshot.valid_affected_frames) / requested))
    progress = int(snapshot.commit_index) / max(1, int(max_commit))
    progress_weight = (
        float(cfg.late_weight_multiplier)
        if progress >= float(cfg.late_progress_start)
        else 1.0
    )
    return float(progress_weight * valid_ratio)


class WeightedSnapshotSampler:
    """Sample Gaussian or initializer-rollin snapshots with stable source ratios."""

    def __init__(
        self,
        gaussian_snapshots: Sequence[NoiseInitializerReplaySnapshot],
        initializer_snapshots: Sequence[NoiseInitializerReplaySnapshot],
        *,
        cfg: ReplaySamplingConfig,
        seed: int,
    ):
        self.cfg = cfg
        self._rng = random.Random(int(seed))
        self.gaussian_snapshots = tuple(gaussian_snapshots)
        self.initializer_snapshots = tuple(initializer_snapshots)
        all_snapshots = self.gaussian_snapshots + self.initializer_snapshots
        self.max_commit = max((int(item.commit_index) for item in all_snapshots), default=0)
        self._gaussian_weights = self._weights_for(self.gaussian_snapshots)
        self._initializer_weights = self._weights_for(self.initializer_snapshots)
        if not self.gaussian_snapshots or sum(self._gaussian_weights) <= 0.0:
            raise ValueError("Gaussian replay pool must contain valid affected snapshots")
        if self.initializer_snapshots and sum(self._initializer_weights) <= 0.0:
            raise ValueError("initializer replay pool must contain valid affected snapshots")

    def _weights_for(
        self,
        snapshots: Sequence[NoiseInitializerReplaySnapshot],
    ) -> tuple[float, ...]:
        return tuple(
            snapshot_sampling_weight(item, max_commit=self.max_commit, cfg=self.cfg)
            for item in snapshots
        )

    def _choose(
        self,
        snapshots: Sequence[NoiseInitializerReplaySnapshot],
        weights: Sequence[float],
    ) -> NoiseInitializerReplaySnapshot:
        return self._rng.choices(snapshots, weights=weights, k=1)[0]

    def sample(self, *, stage: int) -> NoiseInitializerReplaySnapshot:
        if int(stage) == 1 or not self.initializer_snapshots:
            return self._choose(self.gaussian_snapshots, self._gaussian_weights)
        if int(stage) != 2:
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        if self._rng.random() < float(self.cfg.initializer_probability):
            return self._choose(self.initializer_snapshots, self._initializer_weights)
        return self._choose(self.gaussian_snapshots, self._gaussian_weights)


__all__ = [
    "NoiseInitializerReplaySnapshot",
    "ReplaySamplingConfig",
    "WeightedSnapshotSampler",
    "snapshot_sampling_weight",
]
