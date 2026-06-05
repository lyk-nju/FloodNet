"""BABEL long-session runtime suite contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eval.runtime.events import RuntimeEvent
from eval.runtime.state_machine import RuntimeCase
from eval.runtime.suites import BABEL_LONG_SESSION


@dataclass(frozen=True, slots=True)
class BabelAssemblyMetadata:
    babel_sequence_id: str
    segment_ids: tuple[str, ...]
    segment_texts: tuple[str, ...]
    segment_start_commits: tuple[int, ...]
    segment_end_commits: tuple[int, ...]
    is_valid_runtime_sample: bool = True
    invalid_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "babel_sequence_id": str(self.babel_sequence_id),
            "segment_ids": [str(item) for item in self.segment_ids],
            "segment_texts": [str(item) for item in self.segment_texts],
            "segment_start_commits": [int(item) for item in self.segment_start_commits],
            "segment_end_commits": [int(item) for item in self.segment_end_commits],
            "is_valid_runtime_sample": bool(self.is_valid_runtime_sample),
            "invalid_reason": self.invalid_reason,
            "metadata": dict(self.metadata),
        }


def _validate_assembly_metadata(assembly: BabelAssemblyMetadata) -> None:
    lengths = {
        "segment_ids": len(assembly.segment_ids),
        "segment_texts": len(assembly.segment_texts),
        "segment_start_commits": len(assembly.segment_start_commits),
        "segment_end_commits": len(assembly.segment_end_commits),
    }
    expected = lengths["segment_texts"]
    if any(length != expected for length in lengths.values()):
        details = ", ".join(f"{key}={value}" for key, value in sorted(lengths.items()))
        raise ValueError(
            f"{assembly.babel_sequence_id}: segment metadata length mismatch "
            f"({details})"
        )


def build_babel_runtime_case(
    *,
    assembly: BabelAssemblyMetadata,
    route_schedule: Sequence[Any],
) -> RuntimeCase:
    _validate_assembly_metadata(assembly)
    events: list[RuntimeEvent] = []
    for idx, text in enumerate(assembly.segment_texts):
        start_commit = int(assembly.segment_start_commits[idx])
        events.append(
            RuntimeEvent(
                event_type="text_replace",
                submit_commit=start_commit,
                effective_commit=start_commit,
                route_version=idx + 1,
                text_version=idx + 1,
                payload={"text": text},
            )
        )
        if idx < len(route_schedule):
            events.append(
                RuntimeEvent(
                    event_type="route_replace_future",
                    submit_commit=start_commit,
                    effective_commit=start_commit,
                    route_version=idx + 1,
                    text_version=idx + 1,
                    payload={"route": route_schedule[idx]},
                )
            )

    return RuntimeCase(
        suite_name=BABEL_LONG_SESSION,
        sample_id=assembly.babel_sequence_id,
        initial_text=assembly.segment_texts[0] if assembly.segment_texts else "",
        initial_route=tuple(route_schedule[0]) if route_schedule else (),
        event_schedule=tuple(events),
        metadata=assembly.to_metadata(),
    )


__all__ = ["BabelAssemblyMetadata", "build_babel_runtime_case"]
