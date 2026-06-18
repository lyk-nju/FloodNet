"""Facade for text and route conditions used by streaming inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.inference.route_condition import (
    RouteConditionState,
    RoutePlan,
    RouteReferenceMode,
    RouteUpdate,
)
from utils.inference.text_condition import (
    TextConditionBundle,
    TextConditionState,
)
from utils.conditions.root_refiner import RootRefinerPathCondition


@dataclass(frozen=True)
class RuntimeRootRefinerCondition:
    """Resolved RootRefiner inputs owned by the runtime condition layer."""

    text: str
    route: RoutePlan | None
    path_condition: RootRefinerPathCondition | None


class ConditionManager:
    """Owns external text and route state without running any model."""

    def __init__(
        self,
        *,
        initial_text: str = "",
        route_mode: RouteReferenceMode | str = RouteReferenceMode.ABSOLUTE,
        sparse_point_range: tuple[int, int] = (3, 8),
    ):
        self.text = TextConditionState(initial_text)
        self.route = RouteConditionState(
            mode=route_mode,
            sparse_point_range=sparse_point_range,
        )

    def reset(self, *, text: str = "") -> None:
        self.text.reset(text)
        self.route.clear()

    def update_text(self, text: str, commit_idx: int | None = None) -> None:
        self.text.update_text(text, commit_idx=commit_idx)

    def update_route(
        self,
        route,
        edit_commit_idx: int,
        delay_tokens: int = 0,
        blend_tokens: int = 0,
        *,
        source: str = "manual",
        version: int | None = None,
    ) -> RouteUpdate | None:
        if route is None:
            self.route.clear()
            return None
        plan = _coerce_route_plan(
            route,
            start_commit_index=int(edit_commit_idx) + int(delay_tokens),
            source=source,
            version=0 if version is None else int(version),
        )
        return self.route.update_route(
            plan,
            edit_commit_idx=edit_commit_idx,
            delay_tokens=delay_tokens,
            blend_tokens=blend_tokens,
        )

    def set_route_mode(self, mode: RouteReferenceMode | str) -> None:
        self.route.set_mode(mode)

    def build_root_refiner_condition(
        self,
        *,
        anchor_state,
        history_motion_world_5d=None,
        n_path: int,
        max_frames: int,
        current_commit_idx: int | None = None,
    ) -> RuntimeRootRefinerCondition:
        commit = (
            int(anchor_state.commit_idx)
            if current_commit_idx is None
            else int(current_commit_idx)
        )
        text = self.text.text_at(commit)
        path_condition = self.route.build_root_refiner_path_condition(
            anchor_state=anchor_state,
            n_path=n_path,
            max_frames=max_frames,
            current_commit_idx=commit,
        )
        del history_motion_world_5d
        return RuntimeRootRefinerCondition(
            text=text,
            route=self.route.active_route(commit),
            path_condition=path_condition,
        )

    def build_ldf_text_condition(
        self,
        commit_idx: int,
        model_sl: int | None = None,
        batch_size: int | None = None,
        device=None,
        dtype=None,
    ) -> TextConditionBundle:
        del model_sl, batch_size, device, dtype
        return self.text.build_bundle(commit_idx)


def _coerce_route_plan(
    route,
    *,
    start_commit_index: int,
    source: str,
    version: int,
) -> RoutePlan:
    if isinstance(route, RoutePlan):
        return route
    points = np.asarray(route, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[-1] == 2:
        zeros = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points[:, :1], zeros, points[:, 1:2]], axis=-1)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"route must be [N,2] or [N,3], got {points.shape}")
    times = np.arange(points.shape[0], dtype=np.float32)
    return RoutePlan(
        times=times,
        points_xyz=points,
        start_commit_index=int(start_commit_index),
        version=int(version),
        source=str(source),
    )


__all__ = [
    "ConditionManager",
    "RuntimeRootRefinerCondition",
]
