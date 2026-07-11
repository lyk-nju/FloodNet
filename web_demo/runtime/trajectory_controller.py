"""Trajectory state controller for the web runtime."""

from __future__ import annotations

import threading
from numbers import Integral

from .contracts import TrajectoryRuntimeControls


class TrajectoryController:
    """Owns web trajectory state, pending route updates, and runtime controls."""

    def __init__(self, controls: TrajectoryRuntimeControls):
        self.controls = controls
        self.lock = threading.RLock()
        self.active_route = None
        self.pending_update = None
        self.current_waypoints = None
        self.current_times = None
        self.current_mode = "replace_future"
        self.state = "none"
        self.plan_version_counter = 0
        self._display_traj = None
        self.display_lock = threading.Lock()

    def update_controls(
        self,
        *,
        route_mode: str | None = None,
        horizon_tokens=None,
        delay_enabled=None,
        delay_tokens=None,
        blend_enabled=None,
        blend_tokens=None,
    ) -> TrajectoryRuntimeControls:
        current = self.controls
        if blend_enabled is not None and self._coerce_bool(
            blend_enabled,
            default=False,
            name="blend_enabled",
        ):
            raise ValueError(
                "route blending is not supported by the authoritative runtime"
            )
        if blend_tokens is not None and self._coerce_int(
            blend_tokens,
            default=0,
            min_value=0,
            name="blend_tokens",
        ) > 0:
            raise ValueError(
                "route blending is not supported by the authoritative runtime"
            )
        controls = TrajectoryRuntimeControls(
            route_mode=str(route_mode or current.route_mode),
            horizon_tokens=self._coerce_int(
                horizon_tokens,
                default=current.horizon_tokens,
                min_value=1,
                name="horizon_tokens",
            ),
            delay_enabled=self._coerce_bool(
                delay_enabled,
                default=current.delay_enabled,
                name="delay_enabled",
            ),
            delay_tokens=self._coerce_int(
                delay_tokens,
                default=current.delay_tokens,
                min_value=0,
                name="delay_tokens",
            ),
            blend_enabled=False,
            blend_tokens=0,
        )
        self.controls = controls
        return controls

    def clear(self) -> None:
        with self.lock:
            self.active_route = None
            self.pending_update = None
            self.current_waypoints = None
            self.current_times = None
            self.current_mode = "replace_future"
        self.state = "none"
        self.set_display(None)

    def next_plan_version(self) -> int:
        with self.lock:
            self.plan_version_counter += 1
            return self.plan_version_counter

    def set_active_route(self, route, *, waypoints=None, times=None, mode="replace_future"):
        with self.lock:
            self.active_route = route
            self.pending_update = None
            self.current_waypoints = waypoints
            self.current_times = times
            self.current_mode = mode

    def set_pending_update(self, update, *, waypoints=None, times=None, mode="replace_future"):
        with self.lock:
            self.pending_update = update
            self.current_waypoints = waypoints
            self.current_times = times
            self.current_mode = mode

    def replace_with_pending(self, update) -> None:
        with self.lock:
            self.active_route = update.new_route
            self.pending_update = None

    def snapshot(self):
        with self.lock:
            return self.pending_update, self.active_route

    def set_display(self, trajectory) -> None:
        with self.display_lock:
            self._display_traj = None if trajectory is None else trajectory.copy()

    def get_display(self):
        with self.display_lock:
            if self._display_traj is None:
                return None
            return self._display_traj.copy()

    @staticmethod
    def _coerce_int(value, *, default: int, min_value: int, name: str) -> int:
        if value is None:
            value = default
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
        if result < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {result}")
        return result

    @staticmethod
    def _coerce_bool(value, *, default: bool, name: str) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, Integral):
            return bool(value)
        raise ValueError(f"{name} must be a boolean, got {value!r}")


__all__ = ["TrajectoryController"]
