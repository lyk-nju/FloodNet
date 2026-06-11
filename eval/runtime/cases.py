"""Benchmark case and suite definitions (Task 002).

Four suites: step / real / turn / babel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eval.runtime.suites import RUNTIME_SUITE_NAMES


@dataclass
class StreamBenchmarkCase:
    name: str
    suite: str
    sample_id: str
    dataset: str                     # "humanml3d" or "babel"
    mode: str
    sample_ids: list[str] = field(default_factory=list)
    mode_kwargs: dict[str, Any] = field(default_factory=dict)
    expected_fields: list[str] = field(
        default_factory=lambda: [
            "ADE", "FDE", "path_arc", "path_chamfer",
            "chamfer_type", "target_source",
        ]
    )


# ── smoke preset ───────────────────────────────────────────────────────

SMOKE_CASES: list[StreamBenchmarkCase] = [
    StreamBenchmarkCase(
        name="step_metric_001168", suite="step",
        sample_id="001168", dataset="humanml3d", mode="step_gtroot",
    ),
    StreamBenchmarkCase(
        name="real_route_metric_001168", suite="real",
        sample_id="001168", dataset="humanml3d", mode="real_route",
    ),
    StreamBenchmarkCase(
        name="turn_metric_001168_rot30", suite="turn",
        sample_id="001168", dataset="humanml3d", mode="turn_delay20_rot30",
        mode_kwargs={"update_angle": 30.0},
    ),
    StreamBenchmarkCase(
        name="babel_metric_9797", suite="babel",
        sample_id="9797", dataset="babel", mode="babel_real",
        sample_ids=["9797_1", "9797_2", "9797_3"],
    ),
]

# ── full suites ────────────────────────────────────────────────────────

STEP_CASES = [
    StreamBenchmarkCase(
        name="step_gtroot_001168", suite="step",
        sample_id="001168", dataset="humanml3d", mode="step_gtroot",
    ),
    StreamBenchmarkCase(
        name="step_predroot_001168", suite="step",
        sample_id="001168", dataset="humanml3d", mode="step_predroot",
    ),
    StreamBenchmarkCase(
        name="step_no_traj_001168", suite="step",
        sample_id="001168", dataset="humanml3d", mode="step_no_traj",
    ),
]

REAL_CASES = [
    StreamBenchmarkCase(
        name="real_route_001168", suite="real",
        sample_id="001168", dataset="humanml3d", mode="real_route",
    ),
    StreamBenchmarkCase(
        name="real_route_rot90_001168", suite="real",
        sample_id="001168", dataset="humanml3d", mode="real_route",
        mode_kwargs={"rotate_plan_deg": 90.0},
    ),
]

TURN_CASES = [
    StreamBenchmarkCase(
        name="turn_immediate_rot30_001168", suite="turn",
        sample_id="001168", dataset="humanml3d",
        mode="turn_immediate_rot30",
        mode_kwargs={"update_angle": 30.0, "mid_update_delay_tokens": "0", "mid_update_blend_tokens": 0},
    ),
    StreamBenchmarkCase(
        name="turn_delay20_rot30_001168", suite="turn",
        sample_id="001168", dataset="humanml3d",
        mode="turn_delay20_rot30",
        mode_kwargs={"update_angle": 30.0, "mid_update_delay_tokens": "20", "mid_update_blend_tokens": 0},
    ),
    StreamBenchmarkCase(
        name="turn_delay20_blend4_rot30_001168", suite="turn",
        sample_id="001168", dataset="humanml3d",
        mode="turn_delay20_blend4_rot30",
        mode_kwargs={
            "update_angle": 30.0,
            "mid_update_delay_tokens": "20",
            "mid_update_blend_tokens": 4,
        },
    ),
]

BABEL_CASES = [
    StreamBenchmarkCase(
        name="babel_real_9797", suite="babel",
        sample_id="9797", dataset="babel", mode="babel_real",
        sample_ids=["9797_1", "9797_2", "9797_3"],
    ),
    StreamBenchmarkCase(
        name="babel_timestamped_9797", suite="babel",
        sample_id="9797", dataset="babel", mode="babel_timestamped",
        sample_ids=["9797_1", "9797_2", "9797_3"],
    ),
    StreamBenchmarkCase(
        name="babel_no_traj_9797", suite="babel",
        sample_id="9797", dataset="babel", mode="babel_no_traj",
        sample_ids=["9797_1", "9797_2", "9797_3"],
    ),
]

FULL_CASES = STEP_CASES + REAL_CASES + TURN_CASES + BABEL_CASES
ALL_SUITES = ["step", "real", "turn", "babel"]
DESIGN_SUITES = list(RUNTIME_SUITE_NAMES)


def get_cases(suites=None, preset=None):
    if suites and "all" in suites:
        return list(FULL_CASES)
    if suites:
        return [c for c in FULL_CASES if c.suite in suites]
    if preset == "smoke":
        return list(SMOKE_CASES)
    if preset == "full":
        return list(FULL_CASES)
    return list(SMOKE_CASES)


def get_runtime_suite_names() -> list[str]:
    """Return the design-aligned runtime suite names.

    ``get_cases`` remains the legacy benchmark case selector. New suite modules
    should use these names when producing web-demo-equivalent runtime cases.
    """
    return list(RUNTIME_SUITE_NAMES)
