"""RootRefiner sampling decisions.

This module chooses history mode, anchor frame, target horizon, path condition
mode, and path offset. Tensor construction lives in `sample_builder.py`.
"""

from __future__ import annotations

import random as random_module
from dataclasses import dataclass
from typing import Iterable

import torch


_PATH_MODES = ("dense_path", "sparse_path", "goal_point")
_DEFAULT_PATH_RATIOS = {"dense_path": 0.5, "sparse_path": 0.3, "goal_point": 0.2}


@dataclass(frozen=True)
class RefinerSample:
    """RootRefiner sampling decisions, before tensor feature construction."""

    modes: list[str]
    anchor_frames: torch.Tensor
    valid_history_frames: torch.Tensor
    history_frame_indices: torch.Tensor
    history_mask: torch.Tensor
    target_frame_counts: torch.Tensor
    path_modes: list[str]
    offset_start_frames: torch.Tensor


class RefinerSampleCreator:
    """Create one RootRefiner sample plan from motion lengths."""

    def __init__(
        self,
        *,
        n_hist: int = 20,
        max_frames: int = 193,
        min_frames: int = 13,
        full_plan_ratio: float = 0.5,
        horizon_policy: str = "random",
        path_condition_policy: str = "dense_path",
        path_condition_ratios: dict[str, float] | None = None,
        offset_start_enabled: bool = False,
        offset_start_prob: float = 0.0,
        offset_start_max_frames: int = 40,
        offset_start_apply_to: tuple[str, ...] | list[str] = (
            "dense_path",
            "sparse_path",
        ),
        seed: int | None = None,
    ):
        self.n_hist = int(n_hist)
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.full_plan_ratio = float(full_plan_ratio)
        self.horizon_policy = str(horizon_policy)
        self.path_condition_policy = str(path_condition_policy)
        self.path_condition_ratios = dict(
            path_condition_ratios or _DEFAULT_PATH_RATIOS
        )
        unknown_keys = set(self.path_condition_ratios) - set(_PATH_MODES)
        if unknown_keys:
            raise ValueError(
                "unknown path_condition_ratios keys: "
                f"{sorted(unknown_keys)}; expected keys from {_PATH_MODES}"
            )
        self.offset_start_enabled = bool(offset_start_enabled)
        self.offset_start_prob = float(offset_start_prob)
        self.offset_start_max_frames = int(offset_start_max_frames)
        self.offset_start_apply_to = tuple(offset_start_apply_to)
        self._seed = seed
        self._rng = random_module.Random(seed)

    def reset_rng(self) -> None:
        """Reset sampling to the creator seed for deterministic repeated passes."""
        self._rng = random_module.Random(self._seed)

    def create(
        self,
        motion_lengths,
        *,
        force_mode: str | Iterable[str] | None = None,
        force_num_frames=None,
        force_anchor_frame=None,
        force_path_mode: str | Iterable[str] | None = None,
        force_no_path_aug: bool = False,
    ) -> RefinerSample:
        if torch.is_tensor(motion_lengths):
            device = motion_lengths.device
            lengths = motion_lengths.to(device=device, dtype=torch.long).view(-1)
        else:
            lengths = torch.as_tensor(motion_lengths, dtype=torch.long).view(-1)
            device = lengths.device
        batch_size = int(lengths.numel())
        if batch_size <= 0:
            raise ValueError("motion_lengths must contain at least one sample")
        if self.n_hist <= 0:
            raise ValueError(f"n_hist must be > 0, got {self.n_hist}")
        if self.min_frames <= 0:
            raise ValueError(f"min_frames must be > 0, got {self.min_frames}")
        if self.max_frames < self.min_frames:
            raise ValueError(
                f"max_frames must be >= min_frames; got "
                f"max_frames={self.max_frames}, min_frames={self.min_frames}"
            )

        force_modes = _to_str_batch(
            force_mode, batch_size=batch_size, name="force_mode",
        )
        force_path_modes = _to_str_batch(
            force_path_mode, batch_size=batch_size, name="force_path_mode",
        )
        forced_frames = (
            None if force_num_frames is None else _to_long_batch(
                force_num_frames,
                batch_size=batch_size,
                device=device,
                name="force_num_frames",
            )
        )
        forced_anchors = (
            None if force_anchor_frame is None else _to_long_batch(
                force_anchor_frame,
                batch_size=batch_size,
                device=device,
                name="force_anchor_frame",
            )
        )

        modes: list[str] = []
        path_modes: list[str] = []
        anchor_frames = torch.zeros(batch_size, device=device, dtype=torch.long)
        valid_history_frames = torch.zeros(batch_size, device=device, dtype=torch.long)
        history_frame_indices = torch.zeros(
            batch_size, self.n_hist, device=device, dtype=torch.long,
        )
        history_mask = torch.zeros(
            batch_size, self.n_hist, device=device, dtype=torch.bool,
        )
        target_frame_counts = torch.zeros(batch_size, device=device, dtype=torch.long)

        min_full_motion_length = self.min_frames + 1
        min_sliding_motion_length = self.n_hist + self.min_frames
        if bool((lengths < min_full_motion_length).any()):
            raise ValueError(
                "motion length is too short for min_frames; "
                f"motion_lengths={lengths.tolist()}, "
                f"min_required_frames={min_full_motion_length}"
            )

        for batch_idx in range(batch_size):
            motion_length = int(lengths[batch_idx].item())
            mode = self._sample_mode(
                motion_length,
                force_mode=None if force_modes is None else force_modes[batch_idx],
                min_sliding=min_sliding_motion_length,
            )
            anchor = self._sample_anchor(
                motion_length,
                mode=mode,
                min_future_frames=self.min_frames,
                forced_anchor=(
                    None
                    if forced_anchors is None
                    else int(forced_anchors[batch_idx].item())
                ),
            )
            max_valid_frames = min(self.max_frames, motion_length - anchor - 1)
            target_frames = self._sample_num_frames(
                max_valid_frames,
                forced_frames=(
                    None
                    if forced_frames is None
                    else int(forced_frames[batch_idx].item())
                ),
            )
            if anchor + 1 + target_frames > motion_length:
                raise ValueError(
                    "sampled target window exceeds motion length; "
                    f"sample={batch_idx}, anchor_frame={anchor}, "
                    f"target_frame_count={target_frames}, "
                    f"motion_length={motion_length}"
                )

            anchor_frames[batch_idx] = anchor
            target_frame_counts[batch_idx] = target_frames
            modes.append(mode)
            if mode == "full":
                valid_history_frames[batch_idx] = 1
                history_frame_indices[batch_idx, -1] = anchor
                history_mask[batch_idx, -1] = True
            else:
                valid_history_frames[batch_idx] = self.n_hist
                indices = torch.arange(
                    anchor - self.n_hist + 1,
                    anchor + 1,
                    device=device,
                    dtype=torch.long,
                )
                history_frame_indices[batch_idx] = indices
                history_mask[batch_idx] = True

            path_mode = (
                self._validate_path_mode(force_path_modes[batch_idx])
                if force_path_modes is not None
                else self._sample_path_mode()
            )
            path_modes.append(path_mode)

        offset_start_frames = self._sample_offset_start_frames(
            target_frame_counts,
            path_modes,
            force_no_path_aug=force_no_path_aug,
        )
        return RefinerSample(
            modes=modes,
            anchor_frames=anchor_frames,
            valid_history_frames=valid_history_frames,
            history_frame_indices=history_frame_indices,
            history_mask=history_mask,
            target_frame_counts=target_frame_counts,
            path_modes=path_modes,
            offset_start_frames=offset_start_frames,
        )

    def _sample_mode(
        self,
        motion_length: int,
        *,
        force_mode: str | None,
        min_sliding: int,
    ) -> str:
        if force_mode is not None:
            mode = str(force_mode)
        elif self._rng.random() < self.full_plan_ratio:
            mode = "full"
        else:
            mode = "sliding"
        if mode not in {"full", "sliding"}:
            raise ValueError(f"mode must be 'full' or 'sliding', got {mode!r}")
        if mode == "sliding" and motion_length < min_sliding:
            return "full"
        return mode

    def _sample_anchor(
        self,
        motion_length: int,
        *,
        mode: str,
        min_future_frames: int,
        forced_anchor: int | None,
    ) -> int:
        if mode == "full":
            min_anchor = 0
            max_anchor = motion_length - int(min_future_frames) - 1
            if max_anchor < min_anchor:
                raise ValueError(
                    "full anchor range invalid; "
                    f"lo={min_anchor}, hi={max_anchor}, "
                    f"motion_length={motion_length}"
                )
            anchor = (
                int(forced_anchor)
                if forced_anchor is not None
                else self._rng.randint(min_anchor, max_anchor)
            )
        else:
            min_anchor = self.n_hist - 1
            max_anchor = motion_length - int(min_future_frames) - 1
            if max_anchor < min_anchor:
                raise ValueError(
                    "sliding anchor range invalid; "
                    f"lo={min_anchor}, hi={max_anchor}, "
                    f"motion_length={motion_length}"
                )
            anchor = (
                int(forced_anchor)
                if forced_anchor is not None
                else self._rng.randint(min_anchor, max_anchor)
            )
        if anchor < min_anchor or anchor > max_anchor:
            raise ValueError(
                "anchor_frame must be within the valid anchor range for "
                f"{mode} mode; anchor_frame={anchor}, "
                f"valid_range=[{min_anchor}, {max_anchor}]"
            )
        if anchor < 0:
            raise ValueError(f"anchor_frame must be >= 0, got {anchor}")
        if anchor >= motion_length:
            raise ValueError(
                f"anchor_frame must be < motion_length; "
                f"anchor_frame={anchor}, motion_length={motion_length}"
            )
        return anchor

    def _sample_num_frames(
        self,
        max_valid_frames: int,
        *,
        forced_frames: int | None,
    ) -> int:
        if max_valid_frames < self.min_frames:
            raise ValueError(
                "not enough future frames for min_frames after anchor; "
                f"max_valid_frames={max_valid_frames}, min_frames={self.min_frames}"
            )
        if forced_frames is not None:
            frames = int(forced_frames)
            if frames < self.min_frames or frames > max_valid_frames:
                raise ValueError(
                    "force_num_frames must be within the valid horizon range; "
                    f"got {frames}, valid=[{self.min_frames}, {max_valid_frames}]"
                )
        elif self.horizon_policy == "max":
            frames = max_valid_frames
        elif self.horizon_policy == "random":
            frames = self._rng.randint(self.min_frames, max_valid_frames)
        else:
            raise ValueError(
                "horizon_policy must be 'random' or 'max', "
                f"got {self.horizon_policy!r}"
            )
        return frames

    def _sample_path_mode(self) -> str:
        policy = self.path_condition_policy
        if policy in _PATH_MODES:
            return policy
        if policy != "mixed":
            raise ValueError(f"unknown path_condition_policy {policy!r}")
        total = sum(max(0.0, float(v)) for v in self.path_condition_ratios.values())
        if total <= 0:
            return "dense_path"
        pick = self._rng.random() * total
        acc = 0.0
        for mode in _PATH_MODES:
            acc += max(0.0, float(self.path_condition_ratios.get(mode, 0.0)))
            if pick <= acc:
                return mode
        return "goal_point"

    @staticmethod
    def _validate_path_mode(path_mode: str) -> str:
        path_mode = str(path_mode)
        if path_mode not in _PATH_MODES:
            raise ValueError(
                "path_mode must be one of dense_path, sparse_path, goal_point; "
                f"got {path_mode!r}"
            )
        return path_mode

    def _sample_offset_start_frames(
        self,
        target_frame_counts: torch.Tensor,
        path_modes: list[str],
        *,
        force_no_path_aug: bool,
    ) -> torch.Tensor:
        offsets = torch.zeros_like(target_frame_counts)
        if force_no_path_aug or not self.offset_start_enabled:
            return offsets
        for batch_idx, path_mode in enumerate(path_modes):
            if path_mode not in self.offset_start_apply_to:
                continue
            if self._rng.random() >= self.offset_start_prob:
                continue
            valid_frame_count = int(target_frame_counts[batch_idx].item())
            max_offset = min(
                self.offset_start_max_frames,
                max(0, valid_frame_count - 2),
            )
            offsets[batch_idx] = (
                self._rng.randint(0, max_offset)
                if max_offset > 0
                else 0
            )
        return offsets


def _to_long_batch(value, *, batch_size: int, device, name: str) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.long).view(-1)
    else:
        out = torch.as_tensor(value, device=device, dtype=torch.long).view(-1)
    if out.numel() == 1 and batch_size > 1:
        out = out.expand(batch_size)
    if out.numel() != batch_size:
        raise ValueError(
            f"{name} must be scalar or length {batch_size}; "
            f"got shape {tuple(out.shape)}"
        )
    return out


def _to_str_batch(
    value: str | Iterable[str] | None,
    *,
    batch_size: int,
    name: str,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value] * batch_size
    out = [str(item) for item in value]
    if len(out) == 1 and batch_size > 1:
        out = out * batch_size
    if len(out) != batch_size:
        raise ValueError(f"{name} must be scalar or length {batch_size}; got {out!r}")
    return out


__all__ = ["RefinerSample", "RefinerSampleCreator"]
