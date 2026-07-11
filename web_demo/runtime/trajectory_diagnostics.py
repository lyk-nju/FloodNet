"""Web-only trajectory diagnostics derived from committed runtime state."""

from __future__ import annotations

import copy
import threading
from collections import deque
from collections.abc import Mapping

import torch

from utils.local_frame import uncanonicalize_7d
from utils.token_frame import first_future_frame_abs, token_start_frame


def _valid_prefix(mask: torch.Tensor) -> int:
    invalid = torch.nonzero(~mask.bool(), as_tuple=False)
    return int(mask.shape[0]) if int(invalid.shape[0]) == 0 else int(invalid[0, 0])


def _xyz_list(traj7: torch.Tensor, mask: torch.Tensor) -> list[list[float]]:
    traj = torch.as_tensor(traj7).detach().cpu().float()
    valid_mask = torch.as_tensor(mask).detach().cpu().bool().reshape(-1)
    if traj.ndim != 2 or int(traj.shape[-1]) != 7:
        raise ValueError(f"trajectory must have shape [T,7], got {tuple(traj.shape)}")
    if int(valid_mask.shape[0]) != int(traj.shape[0]):
        raise ValueError("trajectory mask length does not match trajectory")
    return traj[: _valid_prefix(valid_mask), :3].tolist()


class TrajectoryDiagnosticsStore:
    """Keep bounded presentation diagnostics without owning runtime state."""

    def __init__(self, *, max_snapshots: int = 8):
        if int(max_snapshots) <= 0:
            raise ValueError("max_snapshots must be > 0")
        self.max_snapshots = int(max_snapshots)
        self._lock = threading.RLock()
        self._snapshots = deque(maxlen=self.max_snapshots)
        self._snapshot_versions: set[int] = set()
        self._snapshot_revision = 0
        self._current = self._empty_current()
        self._last_error = None

    @staticmethod
    def _empty_current() -> dict:
        return {
            "authored_route": [],
            "root_source_proposal": [],
            "actual_payload": [],
            "actual_payload_future": [],
            "source_id": None,
            "source_version": None,
            "activation_commit": None,
            "payload_commit": None,
            "route_status": "inactive",
        }

    def clear(self) -> None:
        with self._lock:
            self._current = self._empty_current()
            self._snapshots.clear()
            self._snapshot_versions.clear()
            self._snapshot_revision += 1
            self._last_error = None

    def set_authored_route(self, points) -> None:
        with self._lock:
            if points is None:
                values_list = []
            else:
                values = torch.as_tensor(points).detach().cpu().float()
                if values.ndim != 2 or int(values.shape[-1]) != 3:
                    raise ValueError("authored route must have shape [T,3]")
                values_list = values.tolist()
            if values_list == self._current["authored_route"]:
                return
            self._current["authored_route"] = values_list
            self._snapshot_revision += 1

    @staticmethod
    def _proposal_xyz(source_manager) -> list[list[float]]:
        active = getattr(source_manager, "active", None)
        if active is None:
            return []
        proposal = active.proposal
        return _xyz_list(proposal.future_traj7, proposal.future_frame_mask)

    @staticmethod
    def _payload_xyz(payload, timeline, *, commit_before: int):
        if not isinstance(payload, Mapping):
            if payload is None:
                return []
            raise TypeError("actual payload must be a mapping")
        traj = torch.as_tensor(payload["traj_cond_7d_frame"]).detach().cpu().float()
        if traj.ndim == 3:
            traj = traj[0]
        mask = torch.as_tensor(payload["traj_cond_frame_mask"]).detach().cpu()
        if mask.ndim == 2:
            mask = mask[0]
        anchor_commit = int(payload["body_anchor_abs_token"])
        if not timeline.has_exact_state(anchor_commit):
            raise ValueError(f"timeline has no payload anchor commit {anchor_commit}")
        anchor = timeline.at_commit(anchor_commit)
        world = uncanonicalize_7d(
            traj.unsqueeze(0),
            anchor.world_xz.detach().cpu().float().unsqueeze(0),
            anchor.world_yaw.detach().cpu().float().reshape(1),
        )[0]
        valid = _valid_prefix(mask.bool().reshape(-1))
        all_xyz = world[:valid, :3].tolist()
        payload_start_abs = token_start_frame(anchor_commit)
        future_offset = max(
            0,
            first_future_frame_abs(int(commit_before)) - payload_start_abs,
        )
        return all_xyz, world[min(future_offset, valid) : valid, :3].tolist()

    def update_from_commit(self, event, source_manager, timeline) -> None:
        with self._lock:
            if "route_cleared" in event.lifecycle_events:
                self.clear()
                return
            errors = []
            if event.source_version != self._current["source_version"]:
                try:
                    self._current["root_source_proposal"] = self._proposal_xyz(
                        source_manager
                    )
                except Exception as exc:
                    errors.append(f"proposal: {exc}")
            try:
                payload_xyz, future_xyz = self._payload_xyz(
                    event.actual_payload,
                    timeline,
                    commit_before=event.absolute_commit_before,
                )
                self._current["actual_payload"] = payload_xyz
                self._current["actual_payload_future"] = future_xyz
            except Exception as exc:
                errors.append(f"payload: {exc}")

            self._current.update(
                {
                    "source_id": event.source_id,
                    "source_version": event.source_version,
                    "activation_commit": event.actual_activation_commit,
                    "payload_commit": int(event.absolute_commit_before),
                    "route_status": getattr(
                        event.route_status,
                        "value",
                        str(event.route_status),
                    ),
                }
            )
            version = event.source_version
            if (
                "route_active" in event.lifecycle_events
                and version is not None
                and int(version) not in self._snapshot_versions
            ):
                version = int(version)
                if len(self._snapshots) == self.max_snapshots:
                    removed = self._snapshots[0]["source_version"]
                    self._snapshot_versions.discard(int(removed))
                self._snapshots.append(
                    {
                        "source_id": event.source_id,
                        "source_version": version,
                        "activation_commit": event.actual_activation_commit,
                        "proposal": copy.deepcopy(
                            self._current["root_source_proposal"]
                        ),
                        "payload": copy.deepcopy(self._current["actual_payload"]),
                    }
                )
                self._snapshot_versions.add(version)
                self._snapshot_revision += 1
            self._last_error = "; ".join(errors) if errors else None

    def to_payload(self, *, client_snapshot_revision: int | None = None) -> dict:
        with self._lock:
            include_static = (
                client_snapshot_revision is None
                or int(client_snapshot_revision) != self._snapshot_revision
            )
            if include_static:
                current = copy.deepcopy(self._current)
            else:
                current = copy.deepcopy(
                    {
                        key: value
                        for key, value in self._current.items()
                        if key not in {"authored_route", "root_source_proposal"}
                    }
                )
            payload = {
                "current": current,
                "snapshot_revision": int(self._snapshot_revision),
                "last_error": self._last_error,
            }
            if include_static:
                payload["snapshots"] = copy.deepcopy(list(self._snapshots))
            return payload


__all__ = ["TrajectoryDiagnosticsStore"]
