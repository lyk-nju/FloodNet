"""Route-edit runtime suite case builders."""

from __future__ import annotations

from typing import Any, Sequence

from eval.runtime.events import RuntimeEvent
from eval.runtime.state_machine import RuntimeCase
from eval.runtime.suites import ROUTE_EDIT


def build_route_replace_case(
    *,
    case_id: str,
    initial_text: str,
    initial_route: Sequence[Any],
    updated_route: Sequence[Any],
    submit_commit: int,
    delay_tokens: int = 0,
    route_version: int = 2,
    text_version: int = 1,
    mode: str = "route_replace_future",
) -> RuntimeCase:
    effective_commit = int(submit_commit) + int(delay_tokens)
    event = RuntimeEvent(
        event_type=str(mode),
        submit_commit=int(submit_commit),
        effective_commit=effective_commit,
        route_version=int(route_version),
        text_version=int(text_version),
        payload={"route_points": list(updated_route)},
        metadata={"delay_tokens": int(delay_tokens)},
    )
    return RuntimeCase(
        suite_name=ROUTE_EDIT,
        sample_id=str(case_id),
        initial_text=str(initial_text),
        initial_route=tuple(initial_route),
        event_schedule=(event,),
        metadata={
            "case_type": str(mode),
            "initial_route_version": 1,
            "updated_route_version": int(route_version),
        },
    )


__all__ = ["build_route_replace_case"]
