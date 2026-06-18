"""Inference runtime API."""

from .condition_manager import ConditionManager, RuntimeRootRefinerCondition
from .root_plan import (
    RootPlan,
    build_root_plan_stream_payload,
    plan_local_to_body_window_local,
    root_plan_to_body_condition,
    slice_plan_with_mask,
)
from .route_condition import (
    RouteConditionState,
    RoutePlan,
    RouteReferenceMode,
    RouteUpdate,
    reanchor_route_to_xz,
    sample_route_future,
)
from .stream_generator import StreamGenerator, StreamStepInput
from .text_condition import TextConditionBundle, TextConditionState, TextSegment
from .timeline import (
    RootFrameState,
    RootTimeline,
    advance_head_from_body_window,
    append_timeline_state_at_token_start_frame,
    body_window_start_commit_idx,
    committed_frame_slice,
    recovery_root_state_to_world,
)

__all__ = [
    "ConditionManager",
    "RootFrameState",
    "RootPlan",
    "RuntimeRootRefinerCondition",
    "RootTimeline",
    "RouteConditionState",
    "RoutePlan",
    "RouteReferenceMode",
    "RouteUpdate",
    "StreamGenerator",
    "StreamStepInput",
    "TextConditionBundle",
    "TextConditionState",
    "TextSegment",
    "advance_head_from_body_window",
    "append_timeline_state_at_token_start_frame",
    "body_window_start_commit_idx",
    "build_root_plan_stream_payload",
    "committed_frame_slice",
    "plan_local_to_body_window_local",
    "reanchor_route_to_xz",
    "recovery_root_state_to_world",
    "root_plan_to_body_condition",
    "sample_route_future",
    "slice_plan_with_mask",
]
