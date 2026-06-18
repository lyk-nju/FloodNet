"""RootPlan state controller for the web runtime."""

from __future__ import annotations


class RootPlanController:
    """Owns writes to `StreamGenerator.active_root_plan` for the web runtime."""

    def __init__(self, stream_generator):
        self.stream_generator = stream_generator

    @property
    def active_plan(self):
        return self.stream_generator.active_root_plan

    def clear(self):
        self.stream_generator.active_root_plan = None

    def set_active(self, root_plan):
        self.stream_generator.active_root_plan = root_plan

    def temporarily_active(self, root_plan):
        return _TemporaryRootPlan(self, root_plan)


class _TemporaryRootPlan:
    def __init__(self, controller: RootPlanController, root_plan):
        self.controller = controller
        self.root_plan = root_plan
        self.previous = None

    def __enter__(self):
        self.previous = self.controller.active_plan
        self.controller.set_active(self.root_plan)
        return self.root_plan

    def __exit__(self, exc_type, exc, tb):
        self.controller.set_active(self.previous)
        return False


__all__ = ["RootPlanController"]
