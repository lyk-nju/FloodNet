"""Training schedule utilities for RootRefiner."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from omegaconf import OmegaConf


@dataclass(frozen=True)
class TrainingSchedulePhase:
    name: str
    steps: int
    start_step: int
    end_step: int
    sampling: dict[str, Any]
    freeze_refiner_modules: tuple[str, ...]


class TrainingSchedule:
    """Resolved RootRefiner training phases."""

    def __init__(self, phases: list[TrainingSchedulePhase]):
        if not phases:
            raise ValueError("training_schedule requires at least one phase")
        self.phases = tuple(phases)
        self.total_steps = int(sum(phase.steps for phase in self.phases))

    @classmethod
    def from_config(cls, cfg: Mapping) -> "TrainingSchedule | None":
        cfg = _plain(cfg)
        schedule_cfg = cfg.get("training_schedule") or {}
        if not schedule_cfg or not bool(schedule_cfg.get("enabled", False)):
            return None
        base_sampling = cfg.get("sampling") or {}
        phases_cfg = schedule_cfg.get("phases") or []
        phases: list[TrainingSchedulePhase] = []
        cursor = 0
        for phase_idx, raw_phase in enumerate(phases_cfg):
            phase = dict(raw_phase or {})
            steps = int(phase.get("steps", 0))
            if steps <= 0:
                raise ValueError(
                    "training_schedule.phases[].steps must be > 0; "
                    f"phase={phase_idx}, steps={steps}"
                )
            sampling = _deep_merge(base_sampling, phase.get("sampling") or {})
            freeze_cfg = phase.get("freeze") or {}
            freeze_modules = _as_tuple(freeze_cfg.get("refiner_modules") or ())
            name = str(phase.get("name") or f"phase_{phase_idx}")
            phases.append(
                TrainingSchedulePhase(
                    name=name,
                    steps=steps,
                    start_step=cursor,
                    end_step=cursor + steps,
                    sampling=sampling,
                    freeze_refiner_modules=freeze_modules,
                )
            )
            cursor += steps
        return cls(phases)

    def phase_index_for_step(self, step: int) -> int:
        step = max(0, int(step))
        for idx, phase in enumerate(self.phases):
            if step < phase.end_step:
                return idx
        return len(self.phases) - 1

    def phase_for_step(self, step: int) -> TrainingSchedulePhase:
        return self.phases[self.phase_index_for_step(step)]

    def phase(self, index: int) -> TrainingSchedulePhase:
        return self.phases[int(index)]

    def sampling_for_phase_index(self, index: int) -> dict[str, Any]:
        return copy.deepcopy(self.phase(index).sampling)

    def freeze_modules_for_phase_index(self, index: int) -> tuple[str, ...]:
        return self.phase(index).freeze_refiner_modules

    def all_schedule_freeze_modules(self) -> tuple[str, ...]:
        names: list[str] = []
        for phase in self.phases:
            for name in phase.freeze_refiner_modules:
                if name not in names:
                    names.append(name)
        return tuple(names)


def apply_training_schedule_to_cfg(cfg: Mapping) -> TrainingSchedule | None:
    """Resolve schedule and set trainer.max_steps to the phase-step sum."""
    schedule = TrainingSchedule.from_config(cfg)
    if schedule is None:
        return None
    if OmegaConf.is_config(cfg):
        OmegaConf.update(cfg, "trainer.max_steps", schedule.total_steps, merge=True)
    else:
        trainer = cfg.setdefault("trainer", {})
        trainer["max_steps"] = schedule.total_steps
    return schedule


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _deep_merge(base: Mapping, override: Mapping) -> dict[str, Any]:
    result = copy.deepcopy(_plain(base) or {})
    override = _plain(override) or {}
    for key, value in override.items():
        if (
            isinstance(result.get(key), Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _as_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


__all__ = [
    "TrainingSchedule",
    "TrainingSchedulePhase",
    "apply_training_schedule_to_cfg",
]
