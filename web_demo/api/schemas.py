"""Request schema boundaries for the staged web demo API refactor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UpdateTrajectoryRequest:
    session_id: str
    waypoints: list | None
    mode: str = "replace_future"
    source: str = "manual"
    duration_seconds: float | None = None
    route_mode: str | None = None
    horizon_tokens: int | None = None
    delay_enabled: bool | None = None
    delay_tokens: int | None = None

    @classmethod
    def from_payload(cls, payload: dict):
        payload = payload or {}
        return cls(
            session_id=payload.get("session_id"),
            waypoints=payload.get("waypoints"),
            mode=payload.get("mode", "replace_future"),
            source=payload.get("source", "manual"),
            duration_seconds=payload.get("duration_seconds"),
            route_mode=payload.get("route_mode"),
            horizon_tokens=payload.get("horizon_tokens"),
            delay_enabled=payload.get("delay_enabled"),
            delay_tokens=payload.get("delay_tokens"),
        )


__all__ = ["UpdateTrajectoryRequest"]
