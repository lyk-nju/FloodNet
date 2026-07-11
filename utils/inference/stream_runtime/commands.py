"""Versioned, boundary-applied stream-runtime command handling."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, Mapping, TypeAlias

from utils.inference.timeline import RootFrameState

from .contracts import (
    RootSourceCommand,
    RootSourceProposal,
    RuntimeStepConfig,
    SpaceContract,
    _clone_value,
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
            alpha = float(self.xz_blend_alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("xz_blend_alpha must be in [0, 1]")
            object.__setattr__(self, "xz_blend_alpha", alpha)


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
            history = int(self.history_tokens)
            if history < 1:
                raise ValueError("history_tokens must be >= 1")
            object.__setattr__(self, "history_tokens", history)
        if self.horizon_tokens is not UNSET:
            horizon = int(self.horizon_tokens)
            if horizon < 0:
                raise ValueError("horizon_tokens must be >= 0")
            object.__setattr__(self, "horizon_tokens", horizon)
        if self.num_denoise_steps is not UNSET and self.num_denoise_steps is not None:
            steps = int(self.num_denoise_steps)
            if steps <= 0:
                raise ValueError("num_denoise_steps must be > 0")
            object.__setattr__(self, "num_denoise_steps", steps)


@dataclass(frozen=True)
class ResetSession(RuntimeCommand):
    """Request a new runtime session epoch at a worker boundary."""


@dataclass(frozen=True)
class PreparedRuntimeTransition:
    """Pure command reduction result awaiting a successful runtime commit."""

    proposed_config: RuntimeStepConfig
    root_source_command: RootSourceCommand | None
    superseded_versions: tuple[int, ...]
    diagnostics: Mapping[str, Any]
    reset_intent: ResetSession | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposed_config, RuntimeStepConfig):
            raise TypeError("proposed_config must be RuntimeStepConfig")
        if self.root_source_command is not None and not isinstance(
            self.root_source_command, RootSourceCommand
        ):
            raise TypeError("root_source_command must be RootSourceCommand or None")
        versions = tuple(int(version) for version in self.superseded_versions)
        if any(version < 0 for version in versions):
            raise ValueError("superseded_versions must be >= 0")
        if len(set(versions)) != len(versions):
            raise ValueError("superseded_versions must not contain duplicates")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        if self.reset_intent is not None and not isinstance(
            self.reset_intent, ResetSession
        ):
            raise TypeError("reset_intent must be ResetSession or None")
        object.__setattr__(self, "superseded_versions", versions)
        object.__setattr__(self, "diagnostics", _clone_value(self.diagnostics))

    @property
    def reset_command(self) -> ResetSession | None:
        """Compatibility name for consumers that treat reset as a command."""
        return self.reset_intent


_RUNTIME_COMMAND_TYPES = (
    SetRootSource,
    ClearRootSource,
    SetText,
    SetGuidance,
    SetRootFeedback,
    SetRuntimeControls,
    ResetSession,
)


RuntimeCommandEnvelope: TypeAlias = (
    SetRootSource
    | ClearRootSource
    | SetText
    | SetGuidance
    | SetRootFeedback
    | SetRuntimeControls
    | ResetSession
)


def _is_runtime_command_envelope(command: object) -> bool:
    return type(command) in _RUNTIME_COMMAND_TYPES


@dataclass(frozen=True)
class PreparedCommandBatch:
    """An immutable due-command snapshot that can be acknowledged exactly."""

    commands: tuple[RuntimeCommandEnvelope, ...]
    _ack_token: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.commands, tuple):
            raise TypeError("commands must be a tuple")
        if not all(_is_runtime_command_envelope(command) for command in self.commands):
            raise TypeError("commands must contain supported runtime commands")
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
        self._issued_batches: dict[object, PreparedCommandBatch] = {}

    @property
    def pending_versions(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(command.version for command in self._pending)

    def snapshot(self) -> tuple[RuntimeCommandEnvelope, ...]:
        with self._lock:
            return tuple(self._pending)

    def submit(self, command: RuntimeCommandEnvelope) -> int:
        if not _is_runtime_command_envelope(command):
            raise TypeError("command must be a supported runtime command")
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
            batch = PreparedCommandBatch(commands=commands)
            token = object()
            object.__setattr__(batch, "_ack_token", token)
            self._issued_batches[token] = batch
        return batch

    def ack(
        self,
        batch: PreparedCommandBatch,
        *,
        discard_through_version: int | None = None,
    ) -> None:
        if not isinstance(batch, PreparedCommandBatch):
            raise TypeError("batch must be PreparedCommandBatch")
        with self._lock:
            token = batch._ack_token
            issued = self._issued_batches.get(token)
            if issued is not batch:
                raise ValueError("batch was not issued by this queue or is no longer valid")
            versions = set(batch.versions)
            if discard_through_version is not None:
                cutoff = int(discard_through_version)
                versions.update(
                    command.version
                    for command in self._pending
                    if command.version <= cutoff
                )
            pending_versions = {command.version for command in self._pending}
            if not versions.issubset(pending_versions):
                self._issued_batches.pop(token, None)
                raise ValueError("batch was not issued by this queue or is no longer valid")
            self._pending = [
                command for command in self._pending if command.version not in versions
            ]
            for issued_token, issued_batch in tuple(self._issued_batches.items()):
                if issued_token is token or versions.intersection(issued_batch.versions):
                    del self._issued_batches[issued_token]

    def release(self, batch: PreparedCommandBatch) -> None:
        """Discard one prepared handle without acknowledging its commands."""
        if not isinstance(batch, PreparedCommandBatch):
            raise TypeError("batch must be PreparedCommandBatch")
        with self._lock:
            token = batch._ack_token
            issued = self._issued_batches.get(token)
            if issued is not batch:
                raise ValueError("batch was not issued by this queue or is no longer valid")
            del self._issued_batches[token]


def reduce_commands(
    base_config: RuntimeStepConfig,
    batch: PreparedCommandBatch,
    boundary_state: RootFrameState,
    *,
    reset_base_config: RuntimeStepConfig | None = None,
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
    config = (
        (reset_base_config or RuntimeStepConfig.default())
        if reset_intent is not None
        else base_config
    )
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
    "PreparedRuntimeTransition",
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
