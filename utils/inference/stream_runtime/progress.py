"""Pure route-local progress policies for authoritative stream runtime state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from utils.local_frame import heading_dir_xz

from .contracts import ActivatedRootSource, RouteProgressState


@dataclass(frozen=True)
class RouteProjection:
    """A policy result that can be committed as immutable route progress."""

    route_index: int
    future_index: int
    distance: float
    heading_dot: float
    proposed_progress: RouteProgressState

    def __post_init__(self) -> None:
        route_index = int(self.route_index)
        future_index = int(self.future_index)
        if route_index < 0 or future_index < route_index:
            raise ValueError("route and future indices must satisfy 0 <= route <= future")
        if not isinstance(self.proposed_progress, RouteProgressState):
            raise TypeError("proposed_progress must be RouteProgressState")
        object.__setattr__(self, "route_index", route_index)
        object.__setattr__(self, "future_index", future_index)
        object.__setattr__(self, "distance", float(self.distance))
        object.__setattr__(self, "heading_dot", float(self.heading_dot))

    @property
    def selected_route_index(self) -> int:
        """Explicit alias for diagnostics consumers."""
        return self.route_index

    @property
    def selected_future_index(self) -> int:
        """Explicit alias for diagnostics consumers."""
        return self.future_index


def _route_geometry(route_traj7: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    route = torch.as_tensor(route_traj7).detach().cpu().float()
    if route.ndim != 2 or int(route.shape[-1]) < 5 or int(route.shape[0]) < 1:
        raise ValueError(f"route_traj7 must be [T,>=5] with T>=1, got {tuple(route.shape)}")
    xz = route[:, [0, 2]]
    yaw = torch.atan2(route[:, 4], route[:, 3])
    if int(route.shape[0]) == 1:
        arc = xz.new_zeros(1)
    else:
        segment_lengths = torch.linalg.norm(xz[1:] - xz[:-1], dim=-1)
        arc = torch.cat([segment_lengths.new_zeros(1), torch.cumsum(segment_lengths, dim=0)])
    return xz, yaw, arc


def _lookahead_index(arc: Tensor, route_index: int, lookahead_m: float) -> int:
    last_index = int(arc.shape[0]) - 1
    target_arc = float(arc[route_index].item()) + max(0.0, float(lookahead_m))
    candidate = int(torch.searchsorted(arc, arc.new_tensor(target_arc)).item())
    return min(last_index, max(route_index + 1, candidate))


def _proposed_progress(
    previous_progress: RouteProgressState,
    route_index: int,
    arc: Tensor,
) -> RouteProgressState:
    return RouteProgressState(
        route_index=route_index,
        route_arc_length=max(previous_progress.route_arc_length, float(arc[route_index].item())),
    )


@dataclass(frozen=True)
class WorldRouteProgressPolicy:
    """Project an actor pose against authored world-space route coordinates."""

    lookahead_m: float = 0.25
    heading_weight: float = 0.15
    search_forward: int = 96

    def __post_init__(self) -> None:
        if float(self.lookahead_m) < 0.0:
            raise ValueError("lookahead_m must be >= 0")
        if float(self.heading_weight) < 0.0:
            raise ValueError("heading_weight must be >= 0")
        if int(self.search_forward) < 1:
            raise ValueError("search_forward must be >= 1")
        object.__setattr__(self, "lookahead_m", float(self.lookahead_m))
        object.__setattr__(self, "heading_weight", float(self.heading_weight))
        object.__setattr__(self, "search_forward", int(self.search_forward))

    def project(
        self,
        route_traj7: Tensor,
        actor_xz: Tensor,
        actor_yaw: Tensor | float,
        previous_progress: RouteProgressState,
    ) -> RouteProjection:
        """Select a monotonic route point using world position and heading."""
        if not isinstance(previous_progress, RouteProgressState):
            raise TypeError("previous_progress must be RouteProgressState")
        xz, yaw, arc = _route_geometry(route_traj7)
        current_xz = torch.as_tensor(actor_xz, dtype=xz.dtype).detach().cpu().reshape(-1)
        if int(current_xz.numel()) != 2:
            raise ValueError("actor_xz must contain exactly two coordinates")
        current_yaw = torch.as_tensor(actor_yaw, dtype=xz.dtype).detach().cpu().reshape(())

        last_index = int(xz.shape[0]) - 1
        lower_bound = min(last_index, max(0, previous_progress.route_index))
        upper_bound = min(last_index + 1, lower_bound + self.search_forward + 1)
        candidates = xz[lower_bound:upper_bound]
        distance = torch.linalg.norm(candidates - current_xz[None, :], dim=-1)
        route_heading = heading_dir_xz(yaw[lower_bound:upper_bound])
        actor_heading = heading_dir_xz(current_yaw).reshape(1, 2)
        heading_dot = (route_heading * actor_heading).sum(dim=-1).clamp(-1.0, 1.0)
        cost = distance + self.heading_weight * (1.0 - heading_dot)
        selected_local = int(torch.argmin(cost).item())
        route_index = lower_bound + selected_local
        future_index = _lookahead_index(arc, route_index, self.lookahead_m)

        return RouteProjection(
            route_index=route_index,
            future_index=future_index,
            distance=float(distance[selected_local].item()),
            heading_dot=float(heading_dot[selected_local].item()),
            proposed_progress=_proposed_progress(previous_progress, route_index, arc),
        )


@dataclass(frozen=True)
class RelativeRouteProgressPolicy:
    """Advance an anchor-relative route by immutable absolute future phase."""

    lookahead_m: float = 0.25

    def __post_init__(self) -> None:
        if float(self.lookahead_m) < 0.0:
            raise ValueError("lookahead_m must be >= 0")
        object.__setattr__(self, "lookahead_m", float(self.lookahead_m))

    def project(
        self,
        activated: ActivatedRootSource,
        *,
        current_first_future_frame_abs: int,
        previous_progress: RouteProgressState,
    ) -> RouteProjection:
        """Advance from the active source's absolute future-frame phase only."""
        if not isinstance(activated, ActivatedRootSource):
            raise TypeError("activated must be ActivatedRootSource")
        if not isinstance(previous_progress, RouteProgressState):
            raise TypeError("previous_progress must be RouteProgressState")
        _, _, arc = _route_geometry(activated.proposal.future_traj7)
        last_index = int(arc.shape[0]) - 1
        consumed_phase = int(current_first_future_frame_abs) - activated.first_future_frame_abs
        phase_index = min(last_index, max(0, consumed_phase))
        route_index = min(last_index, max(previous_progress.route_index, phase_index))
        future_index = _lookahead_index(arc, route_index, self.lookahead_m)

        return RouteProjection(
            route_index=route_index,
            future_index=future_index,
            distance=0.0,
            heading_dot=1.0,
            proposed_progress=_proposed_progress(previous_progress, route_index, arc),
        )


__all__ = [
    "RelativeRouteProgressPolicy",
    "RouteProjection",
    "WorldRouteProgressPolicy",
]
