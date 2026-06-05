"""RootRefiner eval adapter contracts.

This module keeps naming and metadata decisions close to the RootRefiner eval
package so benchmark code does not invent local variants.
"""

from __future__ import annotations

from typing import Any, Sequence


DURATION_PRED = "pred_duration"
DURATION_GROUNDTRUTH = "groundtruth_duration"
DURATION_MODES = (DURATION_PRED, DURATION_GROUNDTRUTH)

ROUTE_MODES = (
    "dense_gt",
    "sparse_gt",
    "user_polyline",
    "noisy_sparse",
    "offset_sparse",
)

ROOT_REFINER_ARTIFACT_NAMES = {
    DURATION_PRED: {
        "pred_root_7d": "pred_root_7d_pred_duration.npy",
        "rootplan": "rootplan_pred_duration.json",
    },
    DURATION_GROUNDTRUTH: {
        "pred_root_7d": "pred_root_7d_groundtruth_duration.npy",
        "rootplan": "rootplan_groundtruth_duration.json",
    },
    "shared": {
        "metadata": "metadata.json",
        "metrics": "metrics.json",
        "plot_xz": "plot_xz.png",
        "plot_yaw": "plot_yaw.png",
    },
}


def normalize_duration_mode(duration_mode: str) -> str:
    mode = str(duration_mode)
    if mode not in DURATION_MODES:
        raise ValueError(
            f"unknown duration_mode {duration_mode!r}; expected one of {DURATION_MODES}"
        )
    return mode


def normalize_route_mode(route_mode: str) -> str:
    mode = str(route_mode)
    if mode not in ROUTE_MODES:
        raise ValueError(f"unknown route_mode {route_mode!r}; expected one of {ROUTE_MODES}")
    return mode


def build_root_refiner_sample_metadata(
    *,
    sample_id: str,
    route_mode: str,
    duration_mode: str,
    offset_frame: int | None = None,
    offset_token: int | None = None,
    anchor_world_xz: Sequence[float] | None = None,
    anchor_world_yaw: float | None = None,
    gt_slice_start: int | None = None,
    gt_slice_end: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "sample_id": str(sample_id),
        "route_mode": normalize_route_mode(route_mode),
        "duration_mode": normalize_duration_mode(duration_mode),
        "offset_frame": None if offset_frame is None else int(offset_frame),
        "offset_token": None if offset_token is None else int(offset_token),
        "anchor_world_xz": (
            None
            if anchor_world_xz is None
            else [float(anchor_world_xz[0]), float(anchor_world_xz[1])]
        ),
        "anchor_world_yaw": None if anchor_world_yaw is None else float(anchor_world_yaw),
        "gt_slice": {
            "start": None if gt_slice_start is None else int(gt_slice_start),
            "end": None if gt_slice_end is None else int(gt_slice_end),
        },
    }
    if extra:
        metadata.update(extra)
    return metadata


__all__ = [
    "DURATION_GROUNDTRUTH",
    "DURATION_MODES",
    "DURATION_PRED",
    "ROOT_REFINER_ARTIFACT_NAMES",
    "ROUTE_MODES",
    "build_root_refiner_sample_metadata",
    "normalize_duration_mode",
    "normalize_route_mode",
]
