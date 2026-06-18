"""Runtime state enums shared by web demo services and workers."""

from __future__ import annotations

from enum import Enum


class GenerationState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    RUNNING = "running"
    PAUSED = "paused"
    RESETTING = "resetting"
    ERROR = "error"


__all__ = ["GenerationState"]

