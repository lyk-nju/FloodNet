"""Transactional ownership of activated root sources and route progress."""

from __future__ import annotations

from dataclasses import dataclass, replace

from utils.inference.timeline import RootFrameState
from utils.token_frame import first_future_frame_abs

from .contracts import (
    ActivatedRootSource,
    RootSourceCommand,
    RouteProgressState,
    RouteStatus,
)


@dataclass(frozen=True)
class PreparedSourceTransition:
    active: ActivatedRootSource | None
    command_version: int | None
    lifecycle_events: tuple[str, ...]
    source_changed: bool


class RootSourceManager:
    """Commit route lifecycle only after the token transaction succeeds."""

    def __init__(self) -> None:
        self.active: ActivatedRootSource | None = None
        self.route_status = RouteStatus.INACTIVE
        self._exhaustion_emitted = False

    def snapshot_state(self) -> dict:
        return {
            # ActivatedRootSource and all nested proposal fields are immutable
            # defensive copies, so retaining the reference is snapshot-safe.
            "active": self.active,
            "route_status": self.route_status,
            "exhaustion_emitted": bool(self._exhaustion_emitted),
        }

    def restore_state(self, state: dict) -> None:
        self.active = state["active"]
        self.route_status = state["route_status"]
        self._exhaustion_emitted = bool(state["exhaustion_emitted"])

    def reset(self) -> None:
        self.active = None
        self.route_status = RouteStatus.INACTIVE
        self._exhaustion_emitted = False

    def prepare_transition(
        self,
        command: RootSourceCommand | None,
        boundary_state: RootFrameState,
        commit_abs: int,
        *,
        reset_epoch: bool = False,
    ) -> PreparedSourceTransition:
        """Purely materialize a command against the real worker boundary."""
        commit = int(commit_abs)
        if boundary_state.commit_idx != commit:
            raise ValueError("boundary_state.commit_idx must equal commit_abs")
        if command is None:
            return PreparedSourceTransition(self.active, None, (), False)
        if command.kind == "clear":
            events = ("route_cleared",) if self.active is not None else ()
            return PreparedSourceTransition(
                None,
                command.command_version,
                events,
                self.active is not None,
            )

        requested = 0 if reset_epoch else int(command.requested_activation_commit)
        activated = ActivatedRootSource(
            proposal=command.proposal,
            requested_activation_commit=min(requested, commit),
            actual_activation_commit=commit,
            boundary_state=boundary_state,
            first_future_frame_abs=first_future_frame_abs(commit),
            space_contract=command.space_contract,
            progress=RouteProgressState.initial(),
        )
        return PreparedSourceTransition(
            activated,
            command.command_version,
            ("route_active",),
            True,
        )

    def commit_transition(
        self,
        prepared: PreparedSourceTransition,
        *,
        proposed_progress: RouteProgressState | None,
        route_status: RouteStatus,
    ) -> tuple[str, ...]:
        if not isinstance(prepared, PreparedSourceTransition):
            raise TypeError("prepared must be PreparedSourceTransition")
        active = prepared.active
        if active is not None and proposed_progress is not None:
            active = replace(active, progress=proposed_progress)
        self.active = active
        self.route_status = RouteStatus.INACTIVE if active is None else route_status
        if prepared.source_changed:
            self._exhaustion_emitted = False
        events = list(prepared.lifecycle_events)
        if (
            active is not None
            and self.route_status is RouteStatus.EXHAUSTED
            and not self._exhaustion_emitted
        ):
            events.append("route_exhausted")
            self._exhaustion_emitted = True
        return tuple(events)


__all__ = ["PreparedSourceTransition", "RootSourceManager"]
