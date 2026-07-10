"""Immutable data contracts for authoritative stream-runtime execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np
import torch
from torch import Tensor

from utils.inference.timeline import RootFrameState
from utils.token_frame import first_future_frame_abs


class SpaceContract(str, Enum):
    """How an activated route is interpreted during composition."""

    WORLD_ROUTE = "world_route"
    RELATIVE_ROUTE = "relative_route"


class RouteStatus(str, Enum):
    """Committed lifecycle state of the active route."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    EXHAUSTED = "exhausted"


class SegmentLabel(IntEnum):
    """Frame provenance within a composed world-space condition."""

    HISTORY = 0
    GENERATED_HISTORY = 0
    BOUNDARY = 1
    BRIDGE = 2
    ROUTE = 3
    PADDING = 4


def _clone_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return MappingProxyType({key: _clone_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_clone_value(item) for item in value)
    return copy.deepcopy(value)


def _clone_tensor(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.detach().cpu().clone()


def _clone_root_frame_state(value: RootFrameState, *, name: str) -> RootFrameState:
    if not isinstance(value, RootFrameState):
        raise TypeError(f"{name} must be RootFrameState")
    return RootFrameState(
        commit_idx=int(value.commit_idx),
        world_xz=_clone_tensor(value.world_xz, name=f"{name}.world_xz"),
        world_yaw=_clone_tensor(value.world_yaw, name=f"{name}.world_yaw"),
        source=str(value.source),
    )


@dataclass(frozen=True)
class RouteProgressState:
    """Committed route-local progress for a single active source."""

    route_index: int = 0
    route_arc_length: float = 0.0

    def __post_init__(self) -> None:
        route_index = int(self.route_index)
        route_arc_length = float(self.route_arc_length)
        if route_index < 0:
            raise ValueError("route_index must be >= 0")
        if route_arc_length < 0.0:
            raise ValueError("route_arc_length must be >= 0")
        object.__setattr__(self, "route_index", route_index)
        object.__setattr__(self, "route_arc_length", route_arc_length)

    @classmethod
    def initial(cls) -> "RouteProgressState":
        return cls()


@dataclass(frozen=True)
class RootSourceProposal:
    """Immutable, future-only authored world-route frames."""

    future_traj7: Tensor
    future_frame_mask: Tensor
    source_id: str
    version: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        future = _clone_tensor(self.future_traj7, name="future_traj7")
        mask = _clone_tensor(self.future_frame_mask, name="future_frame_mask")
        if future.ndim != 2 or int(future.shape[-1]) != 7:
            raise ValueError("future_traj7 must have shape [num_frames, 7]")
        if int(future.shape[0]) <= 0:
            raise ValueError("future_traj7 must contain at least one future frame")
        if mask.ndim != 1:
            raise ValueError("future_frame_mask must have shape [num_frames]")
        if mask.dtype != torch.bool:
            raise TypeError("future_frame_mask dtype must be torch.bool")
        if int(mask.shape[0]) != int(future.shape[0]):
            raise ValueError("future_frame_mask must have the same length as future_traj7")
        source_id = str(self.source_id)
        version = int(self.version)
        if not source_id:
            raise ValueError("source_id must be non-empty")
        if version < 0:
            raise ValueError("version must be >= 0")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "future_traj7", future)
        object.__setattr__(self, "future_frame_mask", mask)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "metadata", _clone_value(self.metadata))

    @property
    def num_future_frames(self) -> int:
        return int(self.future_traj7.shape[0])


@dataclass(frozen=True)
class RootSourceCommand:
    """A replace or clear request applied at a worker commit boundary."""

    proposal: RootSourceProposal | None
    command_version: int
    requested_activation_commit: int
    space_contract: SpaceContract | None
    kind: Literal["replace", "clear"]

    def __post_init__(self) -> None:
        command_version = int(self.command_version)
        requested_activation_commit = int(self.requested_activation_commit)
        if command_version < 0:
            raise ValueError("command_version must be >= 0")
        if requested_activation_commit < 0:
            raise ValueError("requested_activation_commit must be >= 0")
        if self.kind not in {"replace", "clear"}:
            raise ValueError("kind must be 'replace' or 'clear'")
        if self.kind == "clear":
            if self.proposal is not None or self.space_contract is not None:
                raise ValueError("clear command requires proposal=None and space_contract=None")
        else:
            if not isinstance(self.proposal, RootSourceProposal):
                raise ValueError("replace command requires a RootSourceProposal")
            if not isinstance(self.space_contract, SpaceContract):
                raise ValueError("replace command requires a SpaceContract")
        object.__setattr__(self, "command_version", command_version)
        object.__setattr__(self, "requested_activation_commit", requested_activation_commit)

    @classmethod
    def replace(
        cls,
        *,
        proposal: RootSourceProposal,
        command_version: int,
        requested_activation_commit: int,
        space_contract: SpaceContract,
    ) -> "RootSourceCommand":
        return cls(
            proposal=proposal,
            command_version=command_version,
            requested_activation_commit=requested_activation_commit,
            space_contract=space_contract,
            kind="replace",
        )

    @classmethod
    def clear(
        cls,
        *,
        command_version: int,
        requested_activation_commit: int,
    ) -> "RootSourceCommand":
        return cls(
            proposal=None,
            command_version=command_version,
            requested_activation_commit=requested_activation_commit,
            space_contract=None,
            kind="clear",
        )


@dataclass(frozen=True)
class ActivatedRootSource:
    """A proposal committed against its actual worker-boundary state."""

    proposal: RootSourceProposal
    requested_activation_commit: int
    actual_activation_commit: int
    boundary_state: RootFrameState
    first_future_frame_abs: int
    space_contract: SpaceContract
    progress: RouteProgressState

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, RootSourceProposal):
            raise TypeError("proposal must be RootSourceProposal")
        requested = int(self.requested_activation_commit)
        actual = int(self.actual_activation_commit)
        first_future = int(self.first_future_frame_abs)
        if requested < 0 or actual < 0:
            raise ValueError("activation commits must be >= 0")
        if actual < requested:
            raise ValueError("actual_activation_commit must be >= requested_activation_commit")
        if first_future != first_future_frame_abs(actual):
            raise ValueError(
                "first_future_frame_abs must match actual_activation_commit under "
                "the shared token/frame mapping"
            )
        boundary = _clone_root_frame_state(self.boundary_state, name="boundary_state")
        if boundary.commit_idx != actual:
            raise ValueError("boundary_state.commit_idx must equal actual_activation_commit")
        if not isinstance(self.space_contract, SpaceContract):
            raise TypeError("space_contract must be SpaceContract")
        if not isinstance(self.progress, RouteProgressState):
            raise TypeError("progress must be RouteProgressState")
        object.__setattr__(self, "requested_activation_commit", requested)
        object.__setattr__(self, "actual_activation_commit", actual)
        object.__setattr__(self, "first_future_frame_abs", first_future)
        object.__setattr__(self, "boundary_state", boundary)

    def future_local_index(self, absolute_frame: int) -> int:
        """Convert an absolute future frame to a checked proposal-local index."""
        local_index = int(absolute_frame) - self.first_future_frame_abs
        if local_index < 0:
            raise ValueError("absolute frame is before the activated future range")
        if local_index >= self.proposal.num_future_frames:
            raise ValueError("absolute frame is outside the activated future range")
        return local_index

    def future_frame_abs(self, local_index: int) -> int:
        """Convert a checked proposal-local future index to an absolute frame."""
        local_index = int(local_index)
        if local_index < 0 or local_index >= self.proposal.num_future_frames:
            raise ValueError("future local index is outside the activated future range")
        return self.first_future_frame_abs + local_index


@dataclass(frozen=True)
class ComposeResult:
    """Pure composition output awaiting transactional runtime commit."""

    frame_start_abs: int
    world_condition_7d: Tensor
    frame_mask: Tensor
    segment_labels: Tensor
    proposed_route_progress: RouteProgressState
    route_status: RouteStatus
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        frame_start_abs = int(self.frame_start_abs)
        world = _clone_tensor(self.world_condition_7d, name="world_condition_7d")
        mask = _clone_tensor(self.frame_mask, name="frame_mask")
        labels = _clone_tensor(self.segment_labels, name="segment_labels")
        if frame_start_abs < 0:
            raise ValueError("frame_start_abs must be >= 0")
        if world.ndim != 2 or int(world.shape[-1]) != 7:
            raise ValueError("world_condition_7d must have shape [num_frames, 7]")
        if mask.ndim != 1 or labels.ndim != 1:
            raise ValueError("frame_mask and segment_labels must be rank-1")
        if mask.dtype != torch.bool:
            raise TypeError("frame_mask dtype must be torch.bool")
        if int(mask.shape[0]) != int(world.shape[0]) or int(labels.shape[0]) != int(world.shape[0]):
            raise ValueError("condition, mask, and segment labels must have the same length")
        if not isinstance(self.proposed_route_progress, RouteProgressState):
            raise TypeError("proposed_route_progress must be RouteProgressState")
        if not isinstance(self.route_status, RouteStatus):
            raise TypeError("route_status must be RouteStatus")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(self, "frame_start_abs", frame_start_abs)
        object.__setattr__(self, "world_condition_7d", world)
        object.__setattr__(self, "frame_mask", mask)
        object.__setattr__(self, "segment_labels", labels)
        object.__setattr__(self, "diagnostics", _clone_value(self.diagnostics))


@dataclass(frozen=True)
class RuntimeStepConfig:
    """All generation-affecting controls frozen for one runtime step."""

    text: str = ""
    text_guidance_scale: float = 1.0
    trajectory_guidance_scale: float = 1.0
    root_feedback_enabled: bool = False
    root_feedback_xz_blend_alpha: float = 0.5
    history_tokens: int = 30
    horizon_tokens: int = 20
    num_denoise_steps: int | None = None

    def __post_init__(self) -> None:
        history_tokens = int(self.history_tokens)
        horizon_tokens = int(self.horizon_tokens)
        num_denoise_steps = (
            None if self.num_denoise_steps is None else int(self.num_denoise_steps)
        )
        alpha = float(self.root_feedback_xz_blend_alpha)
        if history_tokens < 0 or horizon_tokens < 0:
            raise ValueError("history_tokens and horizon_tokens must be >= 0")
        if num_denoise_steps is not None and num_denoise_steps <= 0:
            raise ValueError("num_denoise_steps must be > 0 when set")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("root_feedback_xz_blend_alpha must be in [0, 1]")
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "text_guidance_scale", float(self.text_guidance_scale))
        object.__setattr__(self, "trajectory_guidance_scale", float(self.trajectory_guidance_scale))
        object.__setattr__(self, "root_feedback_enabled", bool(self.root_feedback_enabled))
        object.__setattr__(self, "root_feedback_xz_blend_alpha", alpha)
        object.__setattr__(self, "history_tokens", history_tokens)
        object.__setattr__(self, "horizon_tokens", horizon_tokens)
        object.__setattr__(self, "num_denoise_steps", num_denoise_steps)

    @classmethod
    def default(cls) -> "RuntimeStepConfig":
        return cls()


@dataclass(frozen=True)
class KernelStepResult:
    """LDF-kernel output without decoded, recovered, or timeline state."""

    raw_latent: Tensor
    actual_payload: Mapping[str, Any] | None
    local_commit_before: int
    local_commit_after: int
    latent_buffer_start_commit_abs: int
    latent_buffer_epoch: int

    def __post_init__(self) -> None:
        local_before = int(self.local_commit_before)
        local_after = int(self.local_commit_after)
        buffer_start = int(self.latent_buffer_start_commit_abs)
        buffer_epoch = int(self.latent_buffer_epoch)
        if local_after != local_before + 1:
            raise ValueError("local_commit_after must equal local_commit_before + 1")
        if buffer_start < 0 or buffer_epoch < 0:
            raise ValueError("rolling-buffer metadata must be >= 0")
        if self.actual_payload is not None and not isinstance(self.actual_payload, Mapping):
            raise TypeError("actual_payload must be a mapping or None")
        object.__setattr__(self, "raw_latent", _clone_tensor(self.raw_latent, name="raw_latent"))
        object.__setattr__(self, "actual_payload", _clone_value(self.actual_payload))
        object.__setattr__(self, "local_commit_before", local_before)
        object.__setattr__(self, "local_commit_after", local_after)
        object.__setattr__(self, "latent_buffer_start_commit_abs", buffer_start)
        object.__setattr__(self, "latent_buffer_epoch", buffer_epoch)


@dataclass(frozen=True)
class StreamCommitEvent:
    """Immutable record published after one successful token transaction."""

    absolute_commit_before: int
    absolute_commit_after: int
    local_commit_before: int
    local_commit_after: int
    latent_buffer_start_commit_abs: int
    latent_buffer_epoch: int
    committed_latent: Tensor
    decoded_chunk: Tensor
    joint_frames: Tensor
    root_frames_start_abs: int
    root_frames: Tensor
    timeline_state: RootFrameState
    actual_payload: Mapping[str, Any] | None
    source_id: str | None
    source_version: int | None
    actual_activation_commit: int | None
    lifecycle_events: tuple[Any, ...]
    route_status: RouteStatus = RouteStatus.INACTIVE
    root_feedback_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        absolute_before = int(self.absolute_commit_before)
        absolute_after = int(self.absolute_commit_after)
        local_before = int(self.local_commit_before)
        local_after = int(self.local_commit_after)
        buffer_start = int(self.latent_buffer_start_commit_abs)
        buffer_epoch = int(self.latent_buffer_epoch)
        root_start = int(self.root_frames_start_abs)
        if absolute_after != absolute_before + 1:
            raise ValueError("absolute_commit_after must equal absolute_commit_before + 1")
        if local_after != local_before + 1:
            raise ValueError("local_commit_after must equal local_commit_before + 1")
        if min(absolute_before, local_before, buffer_start, buffer_epoch, root_start) < 0:
            raise ValueError("commit and frame indices must be >= 0")
        if self.actual_payload is not None and not isinstance(self.actual_payload, Mapping):
            raise TypeError("actual_payload must be a mapping or None")
        if self.source_version is not None and int(self.source_version) < 0:
            raise ValueError("source_version must be >= 0 when set")
        if self.actual_activation_commit is not None and int(self.actual_activation_commit) < 0:
            raise ValueError("actual_activation_commit must be >= 0 when set")
        if not isinstance(self.route_status, RouteStatus):
            raise TypeError("route_status must be RouteStatus")
        if not isinstance(self.root_feedback_diagnostics, Mapping):
            raise TypeError("root_feedback_diagnostics must be a mapping")
        committed_latent = _clone_tensor(self.committed_latent, name="committed_latent")
        decoded_chunk = _clone_tensor(self.decoded_chunk, name="decoded_chunk")
        joint_frames = _clone_tensor(self.joint_frames, name="joint_frames")
        root_frames = _clone_tensor(self.root_frames, name="root_frames")
        if root_frames.ndim != 2 or int(root_frames.shape[-1]) != 7:
            raise ValueError("root_frames must have shape [num_frames, 7]")
        if int(root_frames.shape[0]) != int(joint_frames.shape[0]):
            raise ValueError("root_frames and joint_frames must have the same frame count")
        timeline_state = _clone_root_frame_state(self.timeline_state, name="timeline_state")
        if timeline_state.commit_idx != absolute_after:
            raise ValueError("timeline_state.commit_idx must equal absolute_commit_after")
        object.__setattr__(self, "absolute_commit_before", absolute_before)
        object.__setattr__(self, "absolute_commit_after", absolute_after)
        object.__setattr__(self, "local_commit_before", local_before)
        object.__setattr__(self, "local_commit_after", local_after)
        object.__setattr__(self, "latent_buffer_start_commit_abs", buffer_start)
        object.__setattr__(self, "latent_buffer_epoch", buffer_epoch)
        object.__setattr__(self, "committed_latent", committed_latent)
        object.__setattr__(self, "decoded_chunk", decoded_chunk)
        object.__setattr__(self, "joint_frames", joint_frames)
        object.__setattr__(self, "root_frames_start_abs", root_start)
        object.__setattr__(self, "root_frames", root_frames)
        object.__setattr__(self, "timeline_state", timeline_state)
        object.__setattr__(self, "actual_payload", _clone_value(self.actual_payload))
        object.__setattr__(self, "source_id", None if self.source_id is None else str(self.source_id))
        object.__setattr__(self, "source_version", None if self.source_version is None else int(self.source_version))
        object.__setattr__(
            self,
            "actual_activation_commit",
            None if self.actual_activation_commit is None else int(self.actual_activation_commit),
        )
        object.__setattr__(self, "lifecycle_events", tuple(_clone_value(event) for event in self.lifecycle_events))
        object.__setattr__(self, "root_feedback_diagnostics", _clone_value(self.root_feedback_diagnostics))


@dataclass(frozen=True)
class SessionResetEvent:
    """Immutable record of an exclusive session reset boundary."""

    previous_session_epoch: int
    session_epoch: int
    applied_command_version: int

    def __post_init__(self) -> None:
        previous_epoch = int(self.previous_session_epoch)
        session_epoch = int(self.session_epoch)
        command_version = int(self.applied_command_version)
        if previous_epoch < 0 or session_epoch < 0 or command_version < 0:
            raise ValueError("session epochs and command version must be >= 0")
        if session_epoch != previous_epoch + 1:
            raise ValueError("session_epoch must equal previous_session_epoch + 1")
        object.__setattr__(self, "previous_session_epoch", previous_epoch)
        object.__setattr__(self, "session_epoch", session_epoch)
        object.__setattr__(self, "applied_command_version", command_version)


RuntimeEvent: TypeAlias = StreamCommitEvent | SessionResetEvent


__all__ = [
    "ActivatedRootSource",
    "ComposeResult",
    "KernelStepResult",
    "RootSourceCommand",
    "RootSourceProposal",
    "RouteProgressState",
    "RouteStatus",
    "RuntimeEvent",
    "RuntimeStepConfig",
    "SegmentLabel",
    "SessionResetEvent",
    "SpaceContract",
    "StreamCommitEvent",
]
