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

    def clear(self):
        self.stream_generator.active_root_plan = None
        self.model_plan_version = None

    def set_active(self, root_plan, *, model_plan_version=None):
        self.stream_generator.active_root_plan = root_plan
        self.model_plan_version = model_plan_version

    def temporarily_active(self, root_plan, *, model_plan_version=None):
        return _TemporaryRootPlan(self, root_plan, model_plan_version)


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


__all__ = ["RootPlanController"]
