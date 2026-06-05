"""HumanML3D runtime control suite case builders."""

from __future__ import annotations

from typing import Any, Sequence

from eval.runtime.state_machine import RuntimeCase
from eval.runtime.suites import HUMANML3D_CONTROL

GT_TRAJ = "gt_traj"
REFINER_TRAJ = "refiner_traj"
HUMANML3D_CONTROL_PATHS = (GT_TRAJ, REFINER_TRAJ)


def build_humanml3d_control_case(
    *,
    sample_id: str,
    text: str,
    route_points: Sequence[Any],
    duration_tokens: int,
    route_mode: str = "dense",
    metadata: dict[str, Any] | None = None,
) -> RuntimeCase:
    extra = dict(metadata or {})
    extra.update(
        {
            "dataset": "humanml3d",
            "route_mode": str(route_mode),
            "duration_tokens": int(duration_tokens),
        }
    )
    return RuntimeCase(
        suite_name=HUMANML3D_CONTROL,
        sample_id=str(sample_id),
        initial_text=str(text),
        initial_route=tuple(route_points),
        expected_duration_tokens=int(duration_tokens),
        path_names=HUMANML3D_CONTROL_PATHS,
        metadata=extra,
    )


__all__ = [
    "GT_TRAJ",
    "HUMANML3D_CONTROL_PATHS",
    "REFINER_TRAJ",
    "build_humanml3d_control_case",
]
