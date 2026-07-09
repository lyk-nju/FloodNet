"""Compatibility package for eval runtime-update imports.

Core runtime-update logic lives in :mod:`utils.inference.runtime_update`.
Eval-only diagnostics stay in this package.
"""

from utils.inference.runtime_update import ActiveWindowSegment
from utils.inference.runtime_update import RootSourceProposal
from utils.inference.runtime_update import RouteProgress
from utils.inference.runtime_update import RouteProgressTracker
from utils.inference.runtime_update import build_active_window_root_plan
from utils.inference.runtime_update import build_active_window_stream_payload
from utils.inference.runtime_update import build_world_condition_stream_payload
from utils.inference.runtime_update import compose_active_window_segment
from utils.inference.runtime_update import compose_active_window_world_condition

__all__ = [
    "ActiveWindowSegment",
    "RouteProgress",
    "RouteProgressTracker",
    "RootSourceProposal",
    "build_active_window_root_plan",
    "build_active_window_stream_payload",
    "build_world_condition_stream_payload",
    "compose_active_window_segment",
    "compose_active_window_world_condition",
]
