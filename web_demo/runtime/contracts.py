"""Internal runtime DTOs for the web demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryRuntimeControls:
    route_mode: str
    horizon_tokens: int
    delay_enabled: bool
    delay_tokens: int
    blend_enabled: bool
    blend_tokens: int

    def to_status_dict(self) -> dict:
        return {
            "trajectory_route_mode": self.route_mode,
            "trajectory_horizon_tokens": self.horizon_tokens,
            "trajectory_delay_enabled": self.delay_enabled,
            "trajectory_delay_tokens": self.delay_tokens,
            "trajectory_blend_enabled": self.blend_enabled,
            "trajectory_blend_tokens": self.blend_tokens,
            "trajectory_blend_supported": False,
        }


__all__ = ["TrajectoryRuntimeControls"]
