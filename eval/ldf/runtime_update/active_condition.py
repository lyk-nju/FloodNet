"""Compatibility wrapper for active-window world condition composition."""

from utils.inference.runtime_update.active_condition import ActiveWindowSegment
from utils.inference.runtime_update.active_condition import compose_active_window_segment
from utils.inference.runtime_update.active_condition import (
    compose_active_window_world_condition,
)

__all__ = [
    "ActiveWindowSegment",
    "compose_active_window_segment",
    "compose_active_window_world_condition",
]
