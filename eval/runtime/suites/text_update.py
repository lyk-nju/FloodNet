"""Text-update runtime suite case builders."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from eval.runtime.events import RuntimeEvent
from eval.runtime.state_machine import RuntimeCase
from eval.runtime.suites import TEXT_UPDATE


def build_text_update_case(
    *,
    case_id: str,
    route_points: Sequence[Any],
    text_segments: Sequence[Mapping[str, Any]],
) -> RuntimeCase:
    if not text_segments:
        raise ValueError("text_segments must contain at least one segment")

    events: list[RuntimeEvent] = []
    for idx, segment in enumerate(text_segments):
        text_version = idx + 1
        events.append(
            RuntimeEvent(
                event_type="text_replace",
                submit_commit=int(segment.get("start_commit", 0)),
                effective_commit=int(segment.get("effective_commit", segment.get("start_commit", 0))),
                route_version=1,
                text_version=text_version,
                payload={"text": str(segment["text"])},
            )
        )

    return RuntimeCase(
        suite_name=TEXT_UPDATE,
        sample_id=str(case_id),
        initial_text=str(text_segments[0]["text"]),
        initial_route=tuple(route_points),
        event_schedule=tuple(events),
        metadata={
            "num_text_segments": len(text_segments),
            "route_version": 1,
        },
    )


__all__ = ["build_text_update_case"]
