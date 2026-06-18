"""Trajectory state controller for the web runtime."""

from __future__ import annotations

from numbers import Integral

from .contracts import TrajectoryRuntimeControls


class TrajectoryController:
    """Owns trajectory runtime controls during the staged refactor."""

    def __init__(self, controls: TrajectoryRuntimeControls):
        self.controls = controls

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
            blend_enabled=self._coerce_bool(
                blend_enabled,
                default=current.blend_enabled,
                name="blend_enabled",
            ),
            blend_tokens=self._coerce_int(
                blend_tokens,
                default=current.blend_tokens,
                min_value=0,
                name="blend_tokens",
            ),
        )
        self.controls = controls
        return controls

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
