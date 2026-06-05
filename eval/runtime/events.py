"""Runtime eval event schema and acceptance policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    event_type: str
    submit_commit: int
    effective_commit: int
    route_version: int | None = None
    text_version: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "event_type": str(self.event_type),
            "submit_commit": int(self.submit_commit),
            "effective_commit": int(self.effective_commit),
            "route_version": (
                None if self.route_version is None else int(self.route_version)
            ),
            "text_version": (
                None if self.text_version is None else int(self.text_version)
            ),
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RootRefinerResponse:
    request_id: str
    request_route_version: int
    request_text_version: int
    request_submit_commit: int
    request_base_commit: int
    response_commit: int
    plan: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "request_id": str(self.request_id),
            "request_route_version": int(self.request_route_version),
            "request_text_version": int(self.request_text_version),
            "request_submit_commit": int(self.request_submit_commit),
            "request_base_commit": int(self.request_base_commit),
            "response_commit": int(self.response_commit),
            "has_plan": self.plan is not None,
            "metadata": dict(self.metadata),
        }


def should_accept_root_refiner_response(
    response: RootRefinerResponse,
    *,
    active_route_version: int,
    active_text_version: int,
    min_response_commit: int,
    min_base_commit: int | None = None,
) -> dict[str, Any]:
    if response.plan is None:
        return {"accepted": False, "reject_reason": "invalid_plan"}
    if int(response.request_route_version) != int(active_route_version):
        return {"accepted": False, "reject_reason": "stale_route"}
    if int(response.request_text_version) != int(active_text_version):
        return {"accepted": False, "reject_reason": "stale_text"}
    if (
        min_base_commit is not None
        and int(response.request_base_commit) < int(min_base_commit)
    ):
        return {"accepted": False, "reject_reason": "stale_commit"}
    if int(response.response_commit) < int(min_response_commit):
        return {"accepted": False, "reject_reason": "stale_commit"}
    return {"accepted": True, "reject_reason": None}


__all__ = [
    "RootRefinerResponse",
    "RuntimeEvent",
    "should_accept_root_refiner_response",
]
