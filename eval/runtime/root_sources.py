"""Root source naming and metadata for runtime debug evaluation."""

from __future__ import annotations

from typing import Any


ROOT_SOURCE_ALIASES = {
    "gtroot": "gtroot",
    "gt_root": "gtroot",
    "gt_7d": "gtroot",
    "gt_7d_ldf": "gtroot",
    "route_7d": "gtroot",
    "route_7d_ldf": "gtroot",
    "rootrefiner": "rootrefiner",
    "refineroot": "rootrefiner",
    "root_refiner": "rootrefiner",
    "rootrefiner_7d": "rootrefiner",
    "root_refiner_7d": "rootrefiner",
    "rootrefiner_7d_ldf": "rootrefiner",
    "root_refiner_7d_ldf": "rootrefiner",
    "notraj": "notraj",
    "no_traj": "notraj",
    "no_traj_ldf": "notraj",
}


ROOT_SOURCE_METADATA = {
    "gtroot": {
        "root_source": "gtroot",
        "condition_source": "gt_motion_7d",
        "description": "Dataset motion 7D extracted from the ground-truth motion.",
        "requires_root_refiner": False,
        "passes_root_condition": True,
    },
    "rootrefiner": {
        "root_source": "rootrefiner",
        "condition_source": "rootrefiner_7d",
        "description": "RootRefiner-predicted 7D root condition passed to LDF.",
        "requires_root_refiner": True,
        "passes_root_condition": True,
    },
    "notraj": {
        "root_source": "notraj",
        "condition_source": "none",
        "description": "No trajectory/root condition is passed to LDF.",
        "requires_root_refiner": False,
        "passes_root_condition": False,
    },
}


def normalize_root_source(name: str) -> str:
    key = str(name).strip().lower()
    if key in ROOT_SOURCE_ALIASES:
        return ROOT_SOURCE_ALIASES[key]
    valid = ", ".join(sorted(ROOT_SOURCE_ALIASES))
    raise ValueError(f"unknown root source {name!r}; expected one of: {valid}")


def root_source_metadata(name: str) -> dict[str, Any]:
    source = normalize_root_source(name)
    return dict(ROOT_SOURCE_METADATA[source])


def runtime_debug_condition_source(name: str, *, family: str) -> str:
    """Per-experiment condition source label for runtime debug records."""

    source = normalize_root_source(name)
    if source == "gtroot" and str(family) == "turn":
        return "route_derived_7d"
    return str(ROOT_SOURCE_METADATA[source]["condition_source"])


__all__ = [
    "ROOT_SOURCE_ALIASES",
    "ROOT_SOURCE_METADATA",
    "normalize_root_source",
    "root_source_metadata",
    "runtime_debug_condition_source",
]
