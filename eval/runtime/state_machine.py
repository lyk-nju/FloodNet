"""Runtime eval state-machine boundary types.

The real rollout implementation should live behind this boundary. These
dataclasses keep suite builders and metrics from depending on ad hoc script
globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.runtime.events import RuntimeEvent


@dataclass(frozen=True, slots=True)
class RuntimeCase:
    suite_name: str
    sample_id: str
    initial_text: str
    initial_route: Sequence[Any] = ()
    event_schedule: tuple[RuntimeEvent, ...] = ()
    expected_duration_tokens: int | None = None
    path_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "suite_name": str(self.suite_name),
            "sample_id": str(self.sample_id),
            "initial_text": str(self.initial_text),
            "initial_route": list(self.initial_route),
            "event_schedule": [event.to_metadata() for event in self.event_schedule],
            "expected_duration_tokens": (
                None
                if self.expected_duration_tokens is None
                else int(self.expected_duration_tokens)
            ),
            "path_names": [str(name) for name in self.path_names],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeRunConfig:
    seed: int = 0
    context_tokens: int = 30
    horizon_tokens: int = 20
    frames_per_token: int = 4
    text_cfg_scale: float = 1.0
    traj_cfg_scale: float = 1.0
    route_update_delay_tokens: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "context_tokens": int(self.context_tokens),
            "horizon_tokens": int(self.horizon_tokens),
            "frames_per_token": int(self.frames_per_token),
            "text_cfg_scale": float(self.text_cfg_scale),
            "traj_cfg_scale": float(self.traj_cfg_scale),
            "route_update_delay_tokens": int(self.route_update_delay_tokens),
        }


@dataclass(slots=True)
class RuntimeRunResult:
    generated_motion: Any = None
    recovered_root: Any = None
    timeline: Any = None
    event_log: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str | Path] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "event_log": list(self.event_log),
            "metrics": dict(self.metrics),
            "artifacts": {
                key: str(value) for key, value in self.artifacts.items()
            },
        }


__all__ = [
    "RuntimeCase",
    "RuntimeRunConfig",
    "RuntimeRunResult",
]
