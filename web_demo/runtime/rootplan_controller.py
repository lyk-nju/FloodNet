"""Deprecated RootPlan compatibility state with no execution ownership."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _ControllerState:
    plan: object = None
    source: object = None
    source_contract: str | None = None
    progress: int | None = None
    model_plan_version: object = None


class RootPlanController:
    """Presentation/debug compatibility state; never mutates runtime execution."""

    def __init__(self, stream_generator=None):
        self.stream_generator = stream_generator
        self._state = _ControllerState()

    @property
    def active_plan(self):
        return self._state.plan

    @property
    def active_source(self):
        return self._state.source

    @property
    def source_contract(self):
        return self._state.source_contract

    @property
    def progress(self):
        return self._state.progress

    @property
    def model_plan_version(self):
        return self._state.model_plan_version

    @model_plan_version.setter
    def model_plan_version(self, value):
        self._state.model_plan_version = value

    def clear(self):
        self._state = _ControllerState()

    def set_active(self, root_plan, *, model_plan_version=None):
        self._state = _ControllerState(
            plan=root_plan,
            model_plan_version=model_plan_version,
        )

    def set_active_source(
        self,
        root_source_proposal,
        *,
        contract: str = "world_route",
        model_plan_version=None,
        progress: int = 0,
    ):
        self._state = _ControllerState(
            source=root_source_proposal,
            source_contract=str(contract),
            progress=int(progress),
            model_plan_version=model_plan_version,
        )

    def snapshot_state(self):
        state = self._state
        return _ControllerState(
            plan=state.plan,
            source=state.source,
            source_contract=state.source_contract,
            progress=state.progress,
            model_plan_version=state.model_plan_version,
        )

    def restore_state(self, state) -> None:
        self._state = state

    @contextmanager
    def temporarily_active(self, root_plan, *, model_plan_version=None):
        previous = self.snapshot_state()
        self.set_active(root_plan, model_plan_version=model_plan_version)
        try:
            yield root_plan
        finally:
            self.restore_state(previous)

    @contextmanager
    def temporarily_active_source(
        self,
        root_source_proposal,
        *,
        contract: str = "world_route",
        model_plan_version=None,
    ):
        previous = self.snapshot_state()
        self.set_active_source(
            root_source_proposal,
            contract=contract,
            model_plan_version=model_plan_version,
        )
        try:
            yield root_source_proposal
        finally:
            self.restore_state(previous)


__all__ = ["RootPlanController"]
