"""Versioned, boundary-applied stream-runtime command handling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock
from typing import TypeAlias

from utils.inference.timeline import RootFrameState

from .contracts import (
    PreparedRuntimeTransition,
    RootSourceCommand,
    RootSourceProposal,
    RuntimeStepConfig,
    SpaceContract,
)


class _Unset:
    __slots__ = ()


UNSET = _Unset()


@dataclass(frozen=True)
class RuntimeCommand:
    """Common envelope shared by every queued runtime command."""

    version: int
    requested_commit_abs: int

    def __post_init__(self) -> None:
        version = int(self.version)
        requested_commit_abs = int(self.requested_commit_abs)
        if version < 0:
            raise ValueError("version must be >= 0")
        if requested_commit_abs < 0:
            raise ValueError("requested_commit_abs must be >= 0")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "requested_commit_abs", requested_commit_abs)


@dataclass(frozen=True)
class SetRootSource(RuntimeCommand):
    """Request activation of an immutable root-source proposal."""

    proposal: RootSourceProposal
    space_contract: SpaceContract

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.proposal, RootSourceProposal):
            raise TypeError("proposal must be RootSourceProposal")
        if not isinstance(self.space_contract, SpaceContract):
            raise TypeError("space_contract must be SpaceContract")


@dataclass(frozen=True)
class ClearRootSource(RuntimeCommand):
    """Request removal of the active root-source proposal."""


@dataclass(frozen=True)
class SetText(RuntimeCommand):
    """Replace the generation text prompt."""

    text: str

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "text", str(self.text))


@dataclass(frozen=True)
class SetGuidance(RuntimeCommand):
    """Replace one or both independent classifier-free guidance scales."""

    text_guidance_scale: float | None = None
    trajectory_guidance_scale: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.text_guidance_scale is None and self.trajectory_guidance_scale is None:
            raise ValueError("SetGuidance requires at least one guidance scale")
        if self.text_guidance_scale is not None:
            object.__setattr__(self, "text_guidance_scale", float(self.text_guidance_scale))
        if self.trajectory_guidance_scale is not None:
            object.__setattr__(
                self,
                "trajectory_guidance_scale",
                float(self.trajectory_guidance_scale),
            )


@dataclass(frozen=True)
class SetRootFeedback(RuntimeCommand):
    """Replace one or both root-feedback controls."""

    enabled: bool | None = None
    xz_blend_alpha: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.enabled is None and self.xz_blend_alpha is None:
            raise ValueError("SetRootFeedback requires at least one control")
        if self.enabled is not None:
            object.__setattr__(self, "enabled", bool(self.enabled))
        if self.xz_blend_alpha is not None:
            object.__setattr__(self, "xz_blend_alpha", float(self.xz_blend_alpha))


@dataclass(frozen=True)
class SetRuntimeControls(RuntimeCommand):
    """Replace one or more non-guidance RuntimeStepConfig controls."""

    history_tokens: int | _Unset = UNSET
    horizon_tokens: int | _Unset = UNSET
    num_denoise_steps: int | None | _Unset = UNSET

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.history_tokens is UNSET
            and self.horizon_tokens is UNSET
            and self.num_denoise_steps is UNSET
        ):
            raise ValueError("SetRuntimeControls requires at least one control")
        if self.history_tokens is not UNSET:
            object.__setattr__(self, "history_tokens", int(self.history_tokens))
        if self.horizon_tokens is not UNSET:
            object.__setattr__(self, "horizon_tokens", int(self.horizon_tokens))
        if self.num_denoise_steps is not UNSET and self.num_denoise_steps is not None:
            object.__setattr__(self, "num_denoise_steps", int(self.num_denoise_steps))


@dataclass(frozen=True)
class ResetSession(RuntimeCommand):
    """Request a new runtime session epoch at a worker boundary."""


RuntimeCommandEnvelope: TypeAlias = (
    SetRootSource
    | ClearRootSource
    | SetText
    | SetGuidance
    | SetRootFeedback
    | SetRuntimeControls
    | ResetSession
)


@dataclass(frozen=True)
class PreparedCommandBatch:
    """An immutable due-command snapshot that can be acknowledged exactly."""

    commands: tuple[RuntimeCommandEnvelope, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.commands, tuple):
            raise TypeError("commands must be a tuple")
        if not all(isinstance(command, RuntimeCommand) for command in self.commands):
            raise TypeError("commands must contain RuntimeCommand instances")
        versions = tuple(command.version for command in self.commands)
        if any(current <= previous for previous, current in zip(versions, versions[1:])):
            raise ValueError("commands must be strictly increasing by version")

    @property
    def versions(self) -> tuple[int, ...]:
        return tuple(command.version for command in self.commands)


class RuntimeCommandQueue:
    """Thread-safe global command queue with transactional acknowledgement."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: list[RuntimeCommandEnvelope] = []
        self._last_submitted_version = -1

    @property
    def pending_versions(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(command.version for command in self._pending)

    def snapshot(self) -> tuple[RuntimeCommandEnvelope, ...]:
        with self._lock:
            return tuple(self._pending)

    def submit(self, command: RuntimeCommandEnvelope) -> int:
        if not isinstance(command, RuntimeCommand):
            raise TypeError("command must be a RuntimeCommand")
        with self._lock:
            if command.version <= self._last_submitted_version:
                raise ValueError("command versions must be strictly increasing globally")
            self._pending.append(command)
            self._last_submitted_version = command.version
        return command.version

    def prepare_due(self, commit_abs: int) -> PreparedCommandBatch:
        commit_abs = int(commit_abs)
        if commit_abs < 0:
            raise ValueError("commit_abs must be >= 0")
        with self._lock:
            commands = tuple(
                command
                for command in self._pending
                if command.requested_commit_abs <= commit_abs
            )
        return PreparedCommandBatch(commands=commands)

    def ack(self, batch: PreparedCommandBatch) -> None:
        if not isinstance(batch, PreparedCommandBatch):
            raise TypeError("batch must be PreparedCommandBatch")
        versions = set(batch.versions)
        with self._lock:
            self._pending = [
                command for command in self._pending if command.version not in versions
            ]


def reduce_commands(
    base_config: RuntimeStepConfig,
    batch: PreparedCommandBatch,
    boundary_state: RootFrameState,
) -> PreparedRuntimeTransition:
    """Purely reduce a command snapshot into the next commit-boundary intent."""

    if not isinstance(base_config, RuntimeStepConfig):
        raise TypeError("base_config must be RuntimeStepConfig")
    if not isinstance(batch, PreparedCommandBatch):
        raise TypeError("batch must be PreparedCommandBatch")
    if not isinstance(boundary_state, RootFrameState):
        raise TypeError("boundary_state must be RootFrameState")

    commands = batch.commands
    reset_index = next(
        (
            index
            for index in range(len(commands) - 1, -1, -1)
            if isinstance(commands[index], ResetSession)
        ),
        None,
    )
    reset_intent = None if reset_index is None else commands[reset_index]
    start_index = 0 if reset_index is None else reset_index + 1
    config = RuntimeStepConfig.default() if reset_intent is not None else base_config
    root_source_command: RootSourceCommand | None = None
    winning_versions: set[int] = set()
    field_versions: dict[str, int] = {}

    if reset_intent is not None:
        winning_versions.add(reset_intent.version)

    for command in commands[start_index:]:
        if isinstance(command, SetRootSource):
            root_source_command = RootSourceCommand.replace(
                proposal=command.proposal,
                command_version=command.version,
                requested_activation_commit=command.requested_commit_abs,
                space_contract=command.space_contract,
            )
            field_versions["root_source"] = command.version
        elif isinstance(command, ClearRootSource):
            root_source_command = RootSourceCommand.clear(
                command_version=command.version,
                requested_activation_commit=command.requested_commit_abs,
            )
            field_versions["root_source"] = command.version
        elif isinstance(command, SetText):
            config = replace(config, text=command.text)
            field_versions["text"] = command.version
        elif isinstance(command, SetGuidance):
            updates: dict[str, float] = {}
            if command.text_guidance_scale is not None:
                updates["text_guidance_scale"] = command.text_guidance_scale
                field_versions["text_guidance_scale"] = command.version
            if command.trajectory_guidance_scale is not None:
                updates["trajectory_guidance_scale"] = command.trajectory_guidance_scale
                field_versions["trajectory_guidance_scale"] = command.version
            config = replace(config, **updates)
        elif isinstance(command, SetRootFeedback):
            updates = {}
            if command.enabled is not None:
                updates["root_feedback_enabled"] = command.enabled
                field_versions["root_feedback_enabled"] = command.version
            if command.xz_blend_alpha is not None:
                updates["root_feedback_xz_blend_alpha"] = command.xz_blend_alpha
                field_versions["root_feedback_xz_blend_alpha"] = command.version
            config = replace(config, **updates)
        elif isinstance(command, SetRuntimeControls):
            updates = {}
            if command.history_tokens is not UNSET:
                updates["history_tokens"] = command.history_tokens
                field_versions["history_tokens"] = command.version
            if command.horizon_tokens is not UNSET:
                updates["horizon_tokens"] = command.horizon_tokens
                field_versions["horizon_tokens"] = command.version
            if command.num_denoise_steps is not UNSET:
                updates["num_denoise_steps"] = command.num_denoise_steps
                field_versions["num_denoise_steps"] = command.version
            config = replace(config, **updates)
        else:
            raise TypeError(f"unsupported RuntimeCommand type: {type(command)!r}")

    winning_versions.update(field_versions.values())
    superseded_versions = tuple(
        command.version for command in commands if command.version not in winning_versions
    )
    reset_is_exclusive = reset_index is not None and reset_index == len(commands) - 1
    return PreparedRuntimeTransition(
        proposed_config=config,
        root_source_command=root_source_command,
        superseded_versions=superseded_versions,
        diagnostics={
            "boundary_commit_abs": int(boundary_state.commit_idx),
            "applied_versions": tuple(command.version for command in commands[start_index:]),
            "reset_is_exclusive": reset_is_exclusive,
        },
        reset_intent=reset_intent,
    )


__all__ = [
    "ClearRootSource",
    "PreparedCommandBatch",
    "ResetSession",
    "RuntimeCommand",
    "RuntimeCommandEnvelope",
    "RuntimeCommandQueue",
    "SetGuidance",
    "SetRootFeedback",
    "SetRootSource",
    "SetRuntimeControls",
    "SetText",
    "UNSET",
    "reduce_commands",
]
