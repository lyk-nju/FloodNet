"""State primitives for authoritative stream runtime execution."""

from .contracts import (
    ActivatedRootSource,
    ComposeResult,
    KernelStepResult,
    RootSourceCommand,
    RootSourceProposal,
    RouteProgressState,
    RouteStatus,
    RuntimeEvent,
    RuntimeStepConfig,
    SegmentLabel,
    SessionResetEvent,
    SpaceContract,
    StreamCommitEvent,
)
from .commands import (
    ClearRootSource,
    PreparedCommandBatch,
    PreparedRuntimeTransition,
    ResetSession,
    RuntimeCommand,
    RuntimeCommandQueue,
    SetGuidance,
    SetRootFeedback,
    SetRootSource,
    SetRuntimeControls,
    SetText,
    reduce_commands,
)
from .history import GeneratedRootHistory
from .composer import ConditionComposer
from .payload_builder import PayloadBuilder
from .snapshots import RNGStreamState, restore_rng_state, snapshot_rng_state
from .progress import (
    RelativeRouteProgressPolicy,
    RouteProjection,
    WorldRouteProgressPolicy,
)

__all__ = [
    "ActivatedRootSource",
    "ComposeResult",
    "ConditionComposer",
    "PayloadBuilder",
    "RNGStreamState",
    "ClearRootSource",
    "GeneratedRootHistory",
    "KernelStepResult",
    "PreparedCommandBatch",
    "PreparedRuntimeTransition",
    "ResetSession",
    "RootSourceCommand",
    "RootSourceProposal",
    "RouteProgressState",
    "RouteProjection",
    "RouteStatus",
    "RuntimeEvent",
    "RuntimeCommand",
    "RuntimeCommandQueue",
    "RuntimeStepConfig",
    "SegmentLabel",
    "SessionResetEvent",
    "SetGuidance",
    "SetRootFeedback",
    "SetRootSource",
    "SetRuntimeControls",
    "SetText",
    "SpaceContract",
    "StreamCommitEvent",
    "RelativeRouteProgressPolicy",
    "WorldRouteProgressPolicy",
    "reduce_commands",
    "restore_rng_state",
    "snapshot_rng_state",
]
