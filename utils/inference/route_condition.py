"""User route state and RootRefiner route-condition presentation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from utils.conditions.root_refiner import (
    RootRefinerPathCondition,
    build_root_refiner_path_condition,
)
from utils.inference.geometry import (
    build_projected_suffix_polyline,
    sample_plan_by_time,
)
from utils.local_frame import transform_xz_world_to_local


class RouteReferenceMode(str, Enum):
    """How a world-space route is presented to RootRefiner."""

    ABSOLUTE = "absolute"
    RELATIVE_TO_ACTOR = "relative_to_actor"
    GOAL_POINT = "goal_point"
    SPARSE = "sparse"


@dataclass
class RoutePlan:
    """World-space user route."""

    times: np.ndarray
    points_xyz: np.ndarray
    start_commit_index: int
    version: int
    source: str


@dataclass
class RouteUpdate:
    """Pending mid-session route update."""

    old_route: RoutePlan | None
    new_route: RoutePlan
    edit_commit_index: int
    effective_commit_index: int
    delay_tokens: int
    blend_tokens: int
    version: int


@dataclass(frozen=True)
class RootRefinerConditionBundle:
    """Inputs derived from external route state for RootRefiner."""

    text: str
    route: RoutePlan | None
    path_condition: RootRefinerPathCondition | None


class RouteConditionState:
    """Manage the current user route and pending route edits."""

    def __init__(
        self,
        route: RoutePlan | None = None,
        *,
        mode: RouteReferenceMode | str = RouteReferenceMode.ABSOLUTE,
        sparse_point_range: tuple[int, int] = (3, 8),
    ):
        self.route = route
        self.pending_update: RouteUpdate | None = None
        self.mode = RouteReferenceMode(mode)
        self.sparse_point_range = tuple(int(v) for v in sparse_point_range)

    def set_mode(self, mode: RouteReferenceMode | str) -> None:
        self.mode = RouteReferenceMode(mode)

    def clear(self) -> None:
        self.route = None
        self.pending_update = None

    def update_route(
        self,
        route: RoutePlan,
        *,
        edit_commit_idx: int,
        delay_tokens: int = 0,
        blend_tokens: int = 0,
    ) -> RouteUpdate:
        effective_commit = int(edit_commit_idx) + max(0, int(delay_tokens))
        update = RouteUpdate(
            old_route=self.route,
            new_route=route,
            edit_commit_index=int(edit_commit_idx),
            effective_commit_index=effective_commit,
            delay_tokens=max(0, int(delay_tokens)),
            blend_tokens=max(0, int(blend_tokens)),
            version=int(route.version),
        )
        self.pending_update = update
        if update.delay_tokens == 0:
            self.route = route
            self.pending_update = None
        return update

    def active_route(self, commit_idx: int) -> RoutePlan | None:
        if (
            self.pending_update is not None
            and int(commit_idx) >= self.pending_update.effective_commit_index
        ):
            self.route = self.pending_update.new_route
            self.pending_update = None
        return self.route

    def build_root_refiner_path_condition(
        self,
        *,
        anchor_state,
        n_path: int,
        max_frames: int,
        valid_frame_count: int | None = None,
        current_commit_idx: int | None = None,
        rng: random.Random | None = None,
    ) -> RootRefinerPathCondition | None:
        commit = anchor_state.commit_idx if current_commit_idx is None else current_commit_idx
        route = self.active_route(commit)
        if route is None:
            return None
        return self.build_root_refiner_path_condition_for_route(
            route,
            anchor_state=anchor_state,
            n_path=n_path,
            max_frames=max_frames,
            valid_frame_count=valid_frame_count,
            rng=rng,
        )

    def build_root_refiner_path_condition_for_route(
        self,
        route: RoutePlan,
        *,
        anchor_state,
        n_path: int,
        max_frames: int,
        valid_frame_count: int | None = None,
        rng: random.Random | None = None,
    ) -> RootRefinerPathCondition | None:
        """Build RootRefiner path inputs for an explicit route and anchor."""
        future_xz = self._route_future_xz(route, anchor_state)
        if future_xz.shape[0] <= 0:
            return None
        frame_count = int(valid_frame_count or max_frames)
        mode = "goal_point" if self.mode == RouteReferenceMode.GOAL_POINT else "dense_path"
        if self.mode == RouteReferenceMode.SPARSE:
            mode = "sparse_path"
        return build_root_refiner_path_condition(
            future_xz,
            n_path=int(n_path),
            valid_frame_count=frame_count,
            max_frames=int(max_frames),
            path_mode=mode,
            offset_start_frames=0,
            sparse_point_range=self.sparse_point_range,
            rng=rng or random.Random(0),
        )

    def _route_future_xz(self, route: RoutePlan, anchor_state) -> torch.Tensor:
        points = np.asarray(route.points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"RoutePlan.points_xyz must be [N,3], got {points.shape}")
        if self.mode == RouteReferenceMode.RELATIVE_TO_ACTOR:
            anchor_xyz = np.zeros(3, dtype=np.float32)
            anchor_xyz[[0, 2]] = (
                anchor_state.world_xz.detach().cpu().numpy().astype(np.float32)
            )
            points = build_projected_suffix_polyline(anchor_xyz, points)
        elif self.mode == RouteReferenceMode.GOAL_POINT:
            points = points[-1:, :]
        anchor_xz = anchor_state.world_xz.to(dtype=torch.float32)
        anchor_yaw = anchor_state.world_yaw.to(dtype=torch.float32)
        world_xz = torch.as_tensor(
            points[:, [0, 2]],
            device=anchor_xz.device,
            dtype=torch.float32,
        )
        return transform_xz_world_to_local(world_xz, anchor_xz, anchor_yaw).detach().cpu()


def sample_route_future(
    route: RoutePlan,
    *,
    current_commit: int,
    current_root_xyz: np.ndarray,
    horizon_tokens: int,
    token_dt: float,
    reanchor_to_current_root: bool,
) -> np.ndarray:
    """Sample a route's future positions in world space."""
    elapsed_tokens = max(0, int(current_commit) - int(route.start_commit_index))
    query_times = (
        float(elapsed_tokens) * float(token_dt)
        + np.arange(int(horizon_tokens), dtype=np.float32) * float(token_dt)
    )
    future = sample_plan_by_time(route.times, route.points_xyz, query_times)
    if reanchor_to_current_root and len(future) > 0:
        root = np.asarray(current_root_xyz, dtype=np.float32).reshape(3)
        anchor = sample_plan_by_time(
            route.times,
            route.points_xyz,
            np.asarray([query_times[0]], dtype=np.float32),
        )[0]
        future = root[None, :] + (future - anchor[None, :])
    return future.astype(np.float32)


def reanchor_route_to_xz(route: RoutePlan, anchor_xz) -> RoutePlan:
    """Translate a route so its local t=0 point matches ``anchor_xz``."""
    points = np.asarray(route.points_xyz, dtype=np.float32)
    if points.size == 0:
        return route
    anchor = np.asarray(anchor_xz, dtype=np.float32).reshape(2)
    route_zero = sample_plan_by_time(
        np.asarray(route.times, dtype=np.float32),
        points,
        np.asarray([0.0], dtype=np.float32),
    )[0]
    offset = anchor - route_zero[[0, 2]]
    if float(np.linalg.norm(offset)) <= 1e-7:
        return route
    shifted = points.copy()
    shifted[:, [0, 2]] += offset[None, :]
    return RoutePlan(
        times=np.asarray(route.times, dtype=np.float32).copy(),
        points_xyz=shifted.astype(np.float32),
        start_commit_index=int(route.start_commit_index),
        version=int(route.version),
        source=str(route.source),
    )


__all__ = [
    "RootRefinerConditionBundle",
    "RouteConditionState",
    "RoutePlan",
    "RouteReferenceMode",
    "RouteUpdate",
    "reanchor_route_to_xz",
    "sample_route_future",
]
