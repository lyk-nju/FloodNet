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
