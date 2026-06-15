"""RootRefiner sample creation.

RefinerSampleCreator owns motion-window and path-mode sampling only. Root
recovery, local-frame canonicalization, path tensor construction, and
normalization stay outside this file so dataset and online training paths can
share the same sample contract.
"""

from __future__ import annotations

import random as random_module
import torch

from dataclasses import dataclass
from typing import Iterable
from utils.token_frame import num_frames_for_tokens


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
    num_tokens: torch.Tensor
    target_frame_counts: torch.Tensor
    path_modes: list[str]
    offset_start_frames: torch.Tensor


class RefinerSampleCreator:
    """Create one RootRefiner sample plan from motion lengths."""

    def __init__(
        self,
        *,
        n_hist: int = 20,
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
        seed: int | None = None,
    ):
        self.n_hist = int(n_hist)
        self.max_tokens = int(max_tokens)
        self.min_tokens = int(min_tokens)
        self.frames_per_token = int(frames_per_token)
        self.full_plan_ratio = float(full_plan_ratio)
        self.num_token_policy = str(
            num_token_policy if horizon_policy is None else horizon_policy
        )
        self.path_condition_policy = str(path_condition_policy)
        self.path_condition_ratios = dict(
            path_condition_ratios or _DEFAULT_PATH_RATIOS
        )
        unknown_ratio_keys = set(self.path_condition_ratios) - set(_PATH_MODES)
        if unknown_ratio_keys:
            raise ValueError(
                "unknown path_condition_ratios keys: "
                f"{sorted(unknown_ratio_keys)}; expected keys from {_PATH_MODES}"
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
        force_num_tokens=None,
        force_anchor_frame=None,
        force_path_mode: str | Iterable[str] | None = None,
        force_no_path_aug: bool = False,
    ) -> RefinerSample:
        ##############################
        # inputs
        ##############################
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
        if self.min_tokens <= 0:
            raise ValueError(f"min_tokens must be > 0, got {self.min_tokens}")
        if self.max_tokens < self.min_tokens:
            raise ValueError(
                f"max_tokens must be >= min_tokens; got "
                f"max_tokens={self.max_tokens}, min_tokens={self.min_tokens}"
            )

        ##############################
        # forced decisions
        ##############################
        force_modes = _to_str_batch(
            force_mode, batch_size=batch_size, name="force_mode",
        )
        force_path_modes = _to_str_batch(
            force_path_mode, batch_size=batch_size, name="force_path_mode",
        )
        forced_tokens = (
            None if force_num_tokens is None else _to_long_batch(
                force_num_tokens,
                batch_size=batch_size,
                device=device,
                name="force_num_tokens",
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

        ##############################
        # buffers
        ##############################
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
        num_tokens = torch.zeros(batch_size, device=device, dtype=torch.long)

        min_full = num_frames_for_tokens(self.min_tokens, self.frames_per_token)
        min_sliding = (self.n_hist - 1) + min_full
        if bool((lengths < min_full).any()):
            raise ValueError(
                "motion length is too short for min_tokens; "
                f"motion_lengths={lengths.tolist()}, min_required_frames={min_full}"
            )

        ##############################
        # per-sample decisions
        ##############################
        for b in range(batch_size):
            T = int(lengths[b].item())
            mode = self._sample_mode(
                T,
                force_mode=None if force_modes is None else force_modes[b],
                min_sliding=min_sliding,
            )
            anchor = self._sample_anchor(
                T,
                mode=mode,
                min_full=min_full,
                forced_anchor=(
                    None
                    if forced_anchors is None
                    else int(forced_anchors[b].item())
                ),
            )
            max_valid_tokens = min(
                self.max_tokens,
                self._max_tokens_in_frames(T - anchor),
            )
            tokens = self._sample_num_tokens(
                max_valid_tokens,
                forced_tokens=(
                    None
                    if forced_tokens is None
                    else int(forced_tokens[b].item())
                ),
            )
            target_frames = num_frames_for_tokens(tokens, self.frames_per_token)
            if anchor + target_frames > T:
                raise ValueError(
                    "sampled target window exceeds motion length; "
                    f"sample={b}, anchor_frame={anchor}, "
                    f"target_frame_count={target_frames}, motion_length={T}"
                )

            anchor_frames[b] = anchor
            num_tokens[b] = tokens
            modes.append(mode)
            if mode == "full":
                valid_history_frames[b] = 1
                history_frame_indices[b, -1] = anchor
                history_mask[b, -1] = True
            else:
                valid_history_frames[b] = self.n_hist
                indices = torch.arange(
                    anchor - self.n_hist + 1,
                    anchor + 1,
                    device=device,
                    dtype=torch.long,
                )
                history_frame_indices[b] = indices
                history_mask[b] = True

            path_mode = (
                self._validate_path_mode(force_path_modes[b])
                if force_path_modes is not None
                else self._sample_path_mode()
            )
            path_modes.append(path_mode)

        ##############################
        # path augmentation
        ##############################
        target_frame_counts = _frames_from_tokens(
            num_tokens, frames_per_token=self.frames_per_token,
        )
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
            num_tokens=num_tokens,
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
            mode_drawn = str(force_mode)
        elif self._rng.random() < self.full_plan_ratio:
            mode_drawn = "full"
        else:
            mode_drawn = "sliding"
        if mode_drawn not in {"full", "sliding"}:
            raise ValueError(f"mode must be 'full' or 'sliding', got {mode_drawn!r}")
        if mode_drawn == "sliding" and motion_length < min_sliding:
            return "full"
        return mode_drawn

    def _sample_anchor(
        self,
        motion_length: int,
        *,
        mode: str,
        min_full: int,
        forced_anchor: int | None,
    ) -> int:
        if mode == "full":
            anchor = 0 if forced_anchor is None else int(forced_anchor)
        else:
            lo = self.n_hist - 1
            hi = motion_length - min_full
            if hi < lo:
                raise ValueError(
                    "sliding anchor range invalid; "
                    f"lo={lo}, hi={hi}, motion_length={motion_length}"
                )
            anchor = (
                int(forced_anchor)
                if forced_anchor is not None
                else self._rng.randint(lo, hi)
            )
        if anchor < 0:
            raise ValueError(f"anchor_frame must be >= 0, got {anchor}")
        if anchor >= motion_length:
            raise ValueError(
                f"anchor_frame must be < motion_length; "
                f"anchor_frame={anchor}, motion_length={motion_length}"
            )
        return anchor

    def _sample_num_tokens(
        self,
        max_valid_tokens: int,
        *,
        forced_tokens: int | None,
    ) -> int:
        if max_valid_tokens < self.min_tokens:
            raise ValueError(
                "not enough frames for min_tokens after anchor; "
                f"max_valid_tokens={max_valid_tokens}, min_tokens={self.min_tokens}"
            )
        if forced_tokens is not None:
            tokens = int(forced_tokens)
            if tokens < self.min_tokens or tokens > max_valid_tokens:
                raise ValueError(
                    "force_num_tokens must be within the valid horizon range; "
                    f"got {tokens}, valid=[{self.min_tokens}, {max_valid_tokens}]"
                )
        elif self.num_token_policy == "max":
            tokens = max_valid_tokens
        elif self.num_token_policy == "random":
            tokens = self._rng.randint(self.min_tokens, max_valid_tokens)
        else:
            raise ValueError(
                "num_token_policy must be 'random' or 'max', "
                f"got {self.num_token_policy!r}"
            )
        return tokens

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
        for b, path_mode in enumerate(path_modes):
            if path_mode not in self.offset_start_apply_to:
                continue
            if self._rng.random() >= self.offset_start_prob:
                continue
            valid_frame_count = int(target_frame_counts[b].item())
            max_offset = min(
                self.offset_start_max_frames,
                max(0, valid_frame_count - 2),
            )
            offsets[b] = (
                self._rng.randint(0, max_offset)
                if max_offset > 0
                else 0
            )
        return offsets

    def _max_tokens_in_frames(self, remaining_frames: int) -> int:
        if remaining_frames <= 0:
            return 0
        return (
            remaining_frames + self.frames_per_token - 1
        ) // self.frames_per_token


def _to_long_batch(value, *, batch_size: int, device, name: str) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.long).view(-1)
    else:
        out = torch.as_tensor(value, device=device, dtype=torch.long).view(-1)
    if out.numel() == 1 and batch_size > 1:
        out = out.expand(batch_size)
    if out.numel() != batch_size:
        raise ValueError(
            f"{name} must be scalar or length {batch_size}; got shape {tuple(out.shape)}"
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


def _frames_from_tokens(
    tokens: torch.Tensor,
    *,
    frames_per_token: int,
) -> torch.Tensor:
    values = [
        num_frames_for_tokens(int(v.item()), frames_per_token)
        for v in tokens.view(-1)
    ]
    return torch.as_tensor(values, device=tokens.device, dtype=torch.long).view_as(tokens)


__all__ = ["RefinerSample", "RefinerSampleCreator"]
