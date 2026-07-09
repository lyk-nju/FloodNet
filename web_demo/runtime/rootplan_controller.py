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


class _TemporaryRootPlan:
    def __init__(self, controller: RootPlanController, root_plan, model_plan_version):
        self.controller = controller
        self.root_plan = root_plan
        self.model_plan_version = model_plan_version
        self.previous = None
        self.previous_version = None

    def __enter__(self):
        self.previous = self.controller.active_plan
        self.previous_version = self.controller.model_plan_version
        self.controller.set_active(
            self.root_plan,
            model_plan_version=self.model_plan_version,
        )
        return self.root_plan

    def __exit__(self, exc_type, exc, tb):
        self.controller.set_active(
            self.previous,
            model_plan_version=self.previous_version,
        )
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
        self.previous_plan = None
        self.previous_source = None
        self.previous_contract = None
        self.previous_version = None

    def __enter__(self):
        stream = self.controller.stream_generator
        self.previous_plan = stream.active_root_plan
        self.previous_source = getattr(stream, "active_root_source_proposal", None)
        self.previous_contract = getattr(
            stream,
            "active_root_source_contract",
            "absolute_route",
        )
        self.previous_version = self.controller.model_plan_version
        self.controller.set_active_source(
            self.root_source_proposal,
            contract=self.contract,
            model_plan_version=self.model_plan_version,
        )
        return self.root_source_proposal

    def __exit__(self, exc_type, exc, tb):
        stream = self.controller.stream_generator
        stream.active_root_plan = self.previous_plan
        if self.previous_source is None:
            stream.clear_active_root_source()
        else:
            stream.set_active_root_source_proposal(
                self.previous_source,
                contract=self.previous_contract,
            )
        self.controller.model_plan_version = self.previous_version
        return False


__all__ = ["RootPlanController"]
