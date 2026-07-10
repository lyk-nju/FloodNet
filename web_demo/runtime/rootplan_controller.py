"""RootPlan state controller for the web runtime."""

from __future__ import annotations


class RootPlanController:
    """Owns writes to `StreamGenerator.active_root_plan` for the web runtime."""

    def __init__(self, stream_generator):
        self.stream_generator = stream_generator
        self.model_plan_version = None

    @property
    def active_plan(self):
        return self.stream_generator.active_root_plan

    @property
    def active_source(self):
        return getattr(self.stream_generator, "active_root_source_proposal", None)

    def clear(self):
        self.stream_generator.active_root_plan = None
        if hasattr(self.stream_generator, "clear_active_root_source"):
            self.stream_generator.clear_active_root_source()
        self.model_plan_version = None

    def set_active(self, root_plan, *, model_plan_version=None):
        if hasattr(self.stream_generator, "clear_active_root_source"):
            self.stream_generator.clear_active_root_source()
        self.stream_generator.active_root_plan = root_plan
        self.model_plan_version = model_plan_version

    def set_active_source(
        self,
        root_source_proposal,
        *,
        contract: str = "absolute_route",
        model_plan_version=None,
    ):
        self.stream_generator.active_root_plan = None
        self.stream_generator.set_active_root_source_proposal(
            root_source_proposal,
            contract=contract,
        )
        self.model_plan_version = model_plan_version

    def temporarily_active(self, root_plan, *, model_plan_version=None):
        return _TemporaryRootPlan(self, root_plan, model_plan_version)

    def temporarily_active_source(
        self,
        root_source_proposal,
        *,
        contract: str = "absolute_route",
        model_plan_version=None,
    ):
        return _TemporaryRootSource(
            self,
            root_source_proposal,
            contract,
            model_plan_version,
        )

    def snapshot_state(self):
        stream = self.stream_generator
        tracker = getattr(stream, "_active_root_source_tracker", None)
        return {
            "plan": stream.active_root_plan,
            "source": getattr(stream, "active_root_source_proposal", None),
            "contract": getattr(
                stream, "active_root_source_contract", "absolute_route"
            ),
            "progress": None if tracker is None else int(tracker.last_index),
            "version": self.model_plan_version,
        }

    def restore_state(self, state) -> None:
        stream = self.stream_generator
        stream.active_root_plan = state["plan"]
        source = state["source"]
        if source is None:
            stream.clear_active_root_source()
        else:
            stream.set_active_root_source_proposal(
                source,
                contract=state["contract"],
            )
            tracker = getattr(stream, "_active_root_source_tracker", None)
            if tracker is not None and state["progress"] is not None:
                tracker._last_index = int(state["progress"])
        self.model_plan_version = state["version"]


class _TemporaryRootPlan:
    def __init__(self, controller: RootPlanController, root_plan, model_plan_version):
        self.controller = controller
        self.root_plan = root_plan
        self.model_plan_version = model_plan_version
        self.previous_state = None

    def __enter__(self):
        self.previous_state = self.controller.snapshot_state()
        self.controller.set_active(
            self.root_plan,
            model_plan_version=self.model_plan_version,
        )
        return self.root_plan

    def __exit__(self, exc_type, exc, tb):
        self.controller.restore_state(self.previous_state)
        return False


class _TemporaryRootSource:
    def __init__(
        self,
        controller: RootPlanController,
        root_source_proposal,
        contract: str,
        model_plan_version,
    ):
        self.controller = controller
        self.root_source_proposal = root_source_proposal
        self.contract = str(contract)
        self.model_plan_version = model_plan_version
        self.previous_state = None

    def __enter__(self):
        self.previous_state = self.controller.snapshot_state()
        self.controller.set_active_source(
            self.root_source_proposal,
            contract=self.contract,
            model_plan_version=self.model_plan_version,
        )
        return self.root_source_proposal

    def __exit__(self, exc_type, exc, tb):
        self.controller.restore_state(self.previous_state)
        return False


__all__ = ["RootPlanController"]
