"""Absolute-coordinate history for generated root frames."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class GeneratedRootHistory:
    """Bounded generated root frames with an absolute frame origin."""

    base_frame_abs: int
    frames_7d: Tensor

    def __post_init__(self) -> None:
        self.base_frame_abs = int(self.base_frame_abs)
        if not isinstance(self.frames_7d, torch.Tensor):
            raise TypeError("frames_7d must be a torch.Tensor")
        if self.frames_7d.ndim != 2 or self.frames_7d.shape[-1] != 7:
            raise ValueError("frames_7d must have shape [num_frames, 7]")

    @classmethod
    def empty(
        cls,
        base_frame_abs: int = 0,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "GeneratedRootHistory":
        """Create an empty history beginning at ``base_frame_abs``."""
        return cls(
            base_frame_abs=int(base_frame_abs),
            frames_7d=torch.empty((0, 7), dtype=dtype, device=device),
        )

    @property
    def next_frame_abs(self) -> int:
        """Absolute frame index immediately after the generated history."""
        return self.base_frame_abs + int(self.frames_7d.shape[0])

    def append(self, frames_7d: Tensor, *, start_frame_abs: int) -> None:
        """Append contiguous frames beginning at the current next frame."""
        if not isinstance(frames_7d, torch.Tensor):
            raise TypeError("frames_7d must be a torch.Tensor")
        if frames_7d.ndim != 2 or frames_7d.shape[-1] != 7:
            raise ValueError("frames_7d must have shape [num_frames, 7]")
        if int(start_frame_abs) != self.next_frame_abs:
            raise ValueError(
                "append must start at next_frame_abs: "
                f"got {start_frame_abs}, expected {self.next_frame_abs}"
            )
        if frames_7d.dtype != self.frames_7d.dtype or frames_7d.device != self.frames_7d.device:
            raise ValueError("appended frames must match history dtype and device")
        self.frames_7d = torch.cat((self.frames_7d, frames_7d), dim=0)

    def slice_abs(self, start_frame_abs: int, stop_frame_abs: int) -> Tensor:
        """Return generated frames in absolute half-open range ``[start, stop)``."""
        start = int(start_frame_abs)
        stop = int(stop_frame_abs)
        if start > stop:
            raise ValueError("slice start must not exceed stop")
        if start < self.base_frame_abs or stop > self.next_frame_abs:
            raise ValueError(
                "requested range is trimmed or not generated: "
                f"[{start}, {stop}) outside "
                f"[{self.base_frame_abs}, {self.next_frame_abs})"
            )
        return self.frames_7d[
            start - self.base_frame_abs : stop - self.base_frame_abs
        ]

    def trim_before(self, frame_abs: int) -> None:
        """Discard frames before ``frame_abs`` while preserving absolute indices."""
        target = int(frame_abs)
        if target <= self.base_frame_abs:
            return
        if target > self.next_frame_abs:
            raise ValueError(
                f"cannot trim beyond next_frame_abs={self.next_frame_abs}: {target}"
            )
        self.frames_7d = self.frames_7d[target - self.base_frame_abs :]
        self.base_frame_abs = target

    def reset_to(self, base_frame_abs: int = 0) -> None:
        """Clear retained frames in place while preserving object identity."""
        self.base_frame_abs = int(base_frame_abs)
        self.frames_7d = self.frames_7d.new_empty((0, 7))


__all__ = ["GeneratedRootHistory"]
