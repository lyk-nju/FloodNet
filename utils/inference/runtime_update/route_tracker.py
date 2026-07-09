"""Route progress tracking for active-window runtime updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.local_frame import heading_dir_xz


@dataclass(frozen=True)
class RouteProgress:
    route_index: int
    future_index: int
    distance: float
    heading_dot: float


def _yaw_from_7d(traj7: torch.Tensor) -> torch.Tensor:
    return torch.atan2(traj7[:, 4], traj7[:, 3])


class RouteProgressTracker:
    """Monotonic, heading-aware projection onto a world-frame route."""

    def __init__(
        self,
        route_traj7: torch.Tensor,
        *,
        lookahead_m: float = 0.25,
        heading_weight: float = 0.15,
        search_back: int = 8,
        search_forward: int = 96,
    ):
        route = route_traj7.detach().cpu().float()
        if route.dim() != 2 or route.shape[-1] < 5 or route.shape[0] < 2:
            raise ValueError(
                f"route_traj7 must be [T,>=5] with T>=2, got {tuple(route.shape)}"
            )
        self.route = route
        self.xz = route[:, [0, 2]]
        self.yaw = _yaw_from_7d(route)
        self.lookahead_m = float(lookahead_m)
        self.heading_weight = float(heading_weight)
        self.search_back = max(0, int(search_back))
        self.search_forward = max(1, int(search_forward))
        self._last_index = 0
        diffs = self.xz[1:] - self.xz[:-1]
        seg_len = torch.linalg.norm(diffs, dim=-1)
        self._arc = torch.cat([seg_len.new_zeros(1), torch.cumsum(seg_len, dim=0)])

    @property
    def last_index(self) -> int:
        return int(self._last_index)

    def project(
        self,
        *,
        current_xz: torch.Tensor,
        current_yaw: torch.Tensor,
        min_index: int | None = None,
    ) -> RouteProgress:
        current = torch.as_tensor(current_xz, dtype=self.xz.dtype).detach().cpu().view(2)
        yaw = torch.as_tensor(current_yaw, dtype=self.xz.dtype).detach().cpu().reshape(())
        lower_bound = self._last_index if min_index is None else max(self._last_index, int(min_index))
        lower_bound = min(max(0, int(lower_bound)), int(self.xz.shape[0]) - 1)
        # Runtime route progress is a hard monotonic contract.  Do not let
        # behind-route candidates participate in the argmin and then clamp the
        # result afterward, because that makes debug metrics describe a point
        # different from the returned route_index.
        lo = lower_bound
        hi = min(int(self.xz.shape[0]), lower_bound + self.search_forward + 1)
        candidates = self.xz[lo:hi]
        dist = torch.linalg.norm(candidates - current[None, :], dim=-1)
        route_heading = heading_dir_xz(self.yaw[lo:hi])
        actor_heading = heading_dir_xz(yaw).view(1, 2)
        heading_dot = (route_heading * actor_heading).sum(-1).clamp(-1.0, 1.0)
        cost = dist + float(self.heading_weight) * (1.0 - heading_dot)
        best_local = int(torch.argmin(cost).item())
        route_index = min(lo + best_local, int(self.xz.shape[0]) - 1)
        self._last_index = route_index

        target_arc = float(self._arc[route_index].item()) + max(0.0, self.lookahead_m)
        future_index = int(torch.searchsorted(self._arc, torch.tensor(target_arc)).item())
        future_index = max(route_index + 1, min(future_index, int(self.xz.shape[0]) - 1))
        return RouteProgress(
            route_index=route_index,
            future_index=future_index,
            distance=float(dist[best_local].item()),
            heading_dot=float(heading_dot[best_local].item()),
        )


__all__ = ["RouteProgress", "RouteProgressTracker"]
