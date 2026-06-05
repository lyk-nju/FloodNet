from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class EvalContext:
    config: Any = None
    device: str = "cpu"
    seed: int = 0
    output_dir: str | Path = "outputs/eval"
    logger: Any = None
    rank: int = 0

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_metadata(self) -> dict:
        return {
            "device": str(self.device),
            "seed": int(self.seed),
            "output_dir": str(self.output_path),
        }


@dataclass(slots=True)
class ModelBundle:
    model: Any = None
    vae: Any = None
    root_refiner: Any = None
    text_encoder: Any = None
    tokenizer: Any = None
    stats: Mapping[str, Any] = field(default_factory=dict)

    def require(self, *names: str) -> "ModelBundle":
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "missing required model components: " + ", ".join(missing)
            )
        return self


@dataclass(slots=True)
class EvalSample:
    text: str
    gt_motion: Any = None
    gt_root: Any = None
    user_path: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalPrediction:
    motion: Any = None
    root: Any = None
    latent: Any = None
    root_plan: Any = None
    debug_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricResult:
    summary: dict[str, Any] = field(default_factory=dict)
    per_sample: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str | Path] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "summary": dict(self.summary),
            "per_sample": list(self.per_sample),
            "artifacts": {
                key: str(value) for key, value in self.artifacts.items()
            },
        }


@dataclass(frozen=True, slots=True)
class EvalPathSpec:
    """Stable description of one eval generation path."""

    name: str
    description: str = ""
    enabled_by_default: bool = True
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "description": str(self.description),
            "enabled_by_default": bool(self.enabled_by_default),
            "tags": [str(tag) for tag in self.tags],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvalEvent:
    """Dataset-independent eval event used by runtime-style suites."""

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
class EvalCase:
    """Serializable case boundary shared by RootRefiner, LDF, and runtime eval."""

    suite_name: str
    case_id: str
    dataset: str = ""
    sample_id: str = ""
    path_names: tuple[str, ...] = ()
    events: tuple[EvalEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "suite_name": str(self.suite_name),
            "case_id": str(self.case_id),
            "dataset": str(self.dataset),
            "sample_id": str(self.sample_id),
            "path_names": [str(name) for name in self.path_names],
            "events": [event.to_metadata() for event in self.events],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    """Serializable result boundary for one evaluated case."""

    suite_name: str
    case_id: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str | Path] = field(default_factory=dict)
    events: tuple[EvalEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "suite_name": str(self.suite_name),
            "case_id": str(self.case_id),
            "metrics": dict(self.metrics),
            "artifacts": {
                key: str(value) for key, value in self.artifacts.items()
            },
            "events": [event.to_metadata() for event in self.events],
            "metadata": dict(self.metadata),
        }
