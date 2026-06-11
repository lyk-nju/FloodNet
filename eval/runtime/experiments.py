"""Runtime debug experiment matrix definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from eval.runtime.root_sources import normalize_root_source


DEFAULT_ROOT_SOURCES: tuple[str, ...] = ("gtroot", "rootrefiner", "notraj")
DEFAULT_ROTATION_DEGREES: tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90)
DEFAULT_TURN_DELAY_TOKENS: tuple[int, ...] = (5, 10, 20)
DEFAULT_TURN_BLEND_TOKENS: tuple[int, ...] = (2, 4, 8)
DEFAULT_TURN_ANGLE_DEG = 30.0
DEFAULT_TURN_BLEND_DELAY_TOKENS = 20


@dataclass(frozen=True)
class RuntimeExperimentSpec:
    root_source: str
    family: str
    name: str
    sample_id: str
    parts: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)


def _rot_name(angle: int | float) -> str:
    return f"rot_{int(round(float(angle))):03d}"


def _delay_name(delay_tokens: int) -> str:
    return f"delay_{int(delay_tokens):03d}"


def _blend_name(blend_tokens: int) -> str:
    return f"blend_{int(blend_tokens):03d}"


def build_default_runtime_experiments(
    *,
    sample_id: str,
    root_sources: tuple[str, ...] = DEFAULT_ROOT_SOURCES,
    rotation_degrees: tuple[int, ...] = DEFAULT_ROTATION_DEGREES,
    turn_delay_tokens: tuple[int, ...] = DEFAULT_TURN_DELAY_TOKENS,
    turn_blend_tokens: tuple[int, ...] = DEFAULT_TURN_BLEND_TOKENS,
) -> list[RuntimeExperimentSpec]:
    """Build the default web-demo-equivalent runtime debug matrix."""

    specs: list[RuntimeExperimentSpec] = []
    for root_source in root_sources:
        specs.append(
            RuntimeExperimentSpec(
                root_source=root_source,
                family="web_stream",
                name="web_stream",
                sample_id=sample_id,
            )
        )

        if root_source == "notraj":
            continue

        for angle in rotation_degrees:
            name = _rot_name(angle)
            specs.append(
                RuntimeExperimentSpec(
                    root_source=root_source,
                    family="rotation",
                    name=name,
                    sample_id=sample_id,
                    parts=(name,),
                    params={"rotate_plan_deg": float(angle)},
                )
            )

        specs.append(
            RuntimeExperimentSpec(
                root_source=root_source,
                family="turn",
                name="base",
                sample_id=sample_id,
                parts=("base",),
                params={
                    "update_angle": DEFAULT_TURN_ANGLE_DEG,
                    "mid_update_delay_tokens": 0,
                    "mid_update_blend_tokens": 0,
                },
            )
        )
        for delay_tokens in turn_delay_tokens:
            name = _delay_name(delay_tokens)
            specs.append(
                RuntimeExperimentSpec(
                    root_source=root_source,
                    family="turn",
                    name=name,
                    sample_id=sample_id,
                    parts=("delay", name),
                    params={
                        "update_angle": DEFAULT_TURN_ANGLE_DEG,
                        "mid_update_delay_tokens": int(delay_tokens),
                        "mid_update_blend_tokens": 0,
                    },
                )
            )
        for blend_tokens in turn_blend_tokens:
            name = _blend_name(blend_tokens)
            specs.append(
                RuntimeExperimentSpec(
                    root_source=root_source,
                    family="turn",
                    name=name,
                    sample_id=sample_id,
                    parts=("blend", name),
                    params={
                        "update_angle": DEFAULT_TURN_ANGLE_DEG,
                        "mid_update_delay_tokens": DEFAULT_TURN_BLEND_DELAY_TOKENS,
                        "mid_update_blend_tokens": int(blend_tokens),
                    },
                )
            )

    return specs


def filter_runtime_experiments(
    specs: list[RuntimeExperimentSpec],
    *,
    families: tuple[str, ...] | None = None,
) -> list[RuntimeExperimentSpec]:
    if families is None:
        return list(specs)
    wanted = {str(family) for family in families}
    return [spec for spec in specs if spec.family in wanted]


def runtime_debug_mode_for_source(root_source: str) -> tuple[str, bool, bool]:
    source = normalize_root_source(root_source)
    if source == "gtroot":
        return "real_gtroot", True, False
    if source == "rootrefiner":
        return "real_route", False, False
    if source == "notraj":
        return "real_no_traj", False, True
    raise AssertionError(source)


def parse_csv_strings(raw: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return tuple(default)
    items = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    return items if items else tuple()


def parse_csv_ints(raw: str | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return tuple(default)
    items = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        items.append(int(item))
    return tuple(items)


def summarize_numeric_records(records: list[dict]) -> dict:
    out = {"num_records": int(len(records))}
    keys = sorted({key for rec in records for key in rec})
    for key in keys:
        vals = []
        for rec in records:
            value = rec.get(key)
            if isinstance(value, (bool, str)) or value is None:
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fval):
                vals.append(fval)
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{key}_mean"] = float(arr.mean())
            out[f"{key}_std"] = float(arr.std())
            out[f"{key}_count"] = int(arr.size)
    return out


__all__ = [
    "DEFAULT_ROOT_SOURCES",
    "DEFAULT_ROTATION_DEGREES",
    "DEFAULT_TURN_ANGLE_DEG",
    "DEFAULT_TURN_BLEND_TOKENS",
    "DEFAULT_TURN_DELAY_TOKENS",
    "RuntimeExperimentSpec",
    "build_default_runtime_experiments",
    "filter_runtime_experiments",
    "parse_csv_ints",
    "parse_csv_strings",
    "runtime_debug_mode_for_source",
    "summarize_numeric_records",
]
