"""Root timeline state and dual-anchor commit helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from utils.local_frame import (
    transform_xz_local_delta_to_world,
    wrap_angle,
)
from utils.token_frame import (
    frame_idx_to_token_idx,
    token_range_to_frame_slice,
    token_start_frame,
)

log = logging.getLogger(__name__)


@dataclass
class RootFrameState:
    """A world-space root snapshot at a commit index."""

    commit_idx: int
    world_xz: Tensor
    world_yaw: Tensor
    source: str = "init"

    @classmethod
    def initial(
        cls,
        xz=(0.0, 0.0),
        yaw=0.0,
        device=None,
        dtype=None,
    ) -> "RootFrameState":
        return cls(
            commit_idx=0,
            world_xz=torch.tensor(xz, device=device, dtype=dtype),
            world_yaw=torch.tensor(yaw, device=device, dtype=dtype),
            source="init",
        )

    def to(self, device=None, dtype=None) -> "RootFrameState":
        return RootFrameState(
            commit_idx=int(self.commit_idx),
            world_xz=self.world_xz.to(device=device, dtype=dtype),
            world_yaw=self.world_yaw.to(device=device, dtype=dtype),
            source=self.source,
        )


def advance_head_from_body_window(
    head_state: RootFrameState,
    body_anchor_state: RootFrameState,
    body_output_local_xz_first: Tensor,
    body_output_local_xz_last: Tensor,
    body_output_local_yaw_first: Tensor,
    body_output_local_yaw_last: Tensor,
    committed_tokens: int,
) -> RootFrameState:
    """Advance head state from a body-window-local output span."""
    delta_xz_local = body_output_local_xz_last - body_output_local_xz_first
    delta_yaw_local = wrap_angle(
        body_output_local_yaw_last - body_output_local_yaw_first
    )
    if not (
        torch.isfinite(delta_xz_local).all().item()
        and torch.isfinite(delta_yaw_local).all().item()
    ):
        log.warning(
            "advance_head_from_body_window: non-finite body-local delta; "
            "keeping head_state commit_idx=%d",
            head_state.commit_idx,
        )
        return head_state

    delta_xz_world = transform_xz_local_delta_to_world(
        delta_xz_local,
        body_anchor_state.world_yaw,
    )
    return RootFrameState(
        commit_idx=int(head_state.commit_idx) + int(committed_tokens),
        world_xz=head_state.world_xz + delta_xz_world,
        world_yaw=wrap_angle(head_state.world_yaw + delta_yaw_local),
        source="commit",
    )


class RootTimeline:
    """Append-only root state timeline indexed by commit index."""

    def __init__(self, initial: RootFrameState):
        if not isinstance(initial, RootFrameState):
            raise TypeError(f"initial must be RootFrameState, got {type(initial)}")
        self._states: list[RootFrameState] = [initial]

    @property
    def head(self) -> RootFrameState:
        return self._states[-1]

    @property
    def earliest(self) -> RootFrameState:
        return self._states[0]

    def __len__(self) -> int:
        return len(self._states)

    def append(self, state: RootFrameState) -> None:
        if not isinstance(state, RootFrameState):
            raise TypeError(f"state must be RootFrameState, got {type(state)}")
        if state.commit_idx <= self.head.commit_idx:
            raise ValueError(
                f"new state commit_idx={state.commit_idx} must be > "
                f"head commit_idx={self.head.commit_idx}"
            )
        self._states.append(state)

    def reset_to(self, state: RootFrameState) -> None:
        """Reset this timeline in place so external references stay valid."""
        if not isinstance(state, RootFrameState):
            raise TypeError(f"state must be RootFrameState, got {type(state)}")
        self._states[:] = [state]

    def _binary_search_exact(self, commit_idx: int) -> int | None:
        lo, hi = 0, len(self._states) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_commit = self._states[mid].commit_idx
            if mid_commit == commit_idx:
                return mid
            if mid_commit < commit_idx:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def has_reached(self, commit_idx: int) -> bool:
        return int(commit_idx) <= self.head.commit_idx

    def has_exact_state(self, commit_idx: int) -> bool:
        return self._binary_search_exact(int(commit_idx)) is not None

    def at_commit(self, commit_idx: int) -> RootFrameState:
        commit_idx = int(commit_idx)
        index = self._binary_search_exact(commit_idx)
        if index is not None:
            return self._states[index]
        if commit_idx > self.head.commit_idx:
            log.warning(
                "RootTimeline.at_commit(%d): query past head commit_idx=%d; "
                "returning head",
                commit_idx,
                self.head.commit_idx,
            )
            return self.head
        if commit_idx < self.earliest.commit_idx:
            log.warning(
                "RootTimeline.at_commit(%d): query before earliest commit_idx=%d; "
                "state may have been trimmed; returning earliest",
                commit_idx,
                self.earliest.commit_idx,
            )
            return self.earliest
        lo, hi = 0, len(self._states) - 1
        candidate = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._states[mid].commit_idx < commit_idx:
                candidate = mid
                lo = mid + 1
            else:
                hi = mid - 1
        log.warning(
            "RootTimeline.at_commit(%d): no exact snapshot; returning "
            "nearest preceding %d",
            commit_idx,
            self._states[candidate].commit_idx,
        )
        return self._states[candidate]

    def trim_before(self, commit_idx: int) -> None:
        cutoff = int(commit_idx)
        first_keep = 0
        while (
            first_keep < len(self._states)
            and self._states[first_keep].commit_idx < cutoff
        ):
            first_keep += 1
        if first_keep == len(self._states):
            self._states = self._states[-1:]
        elif first_keep > 0:
            self._states = self._states[first_keep:]

    def advance_from_body_window(
        self,
        *,
        body_anchor_state: RootFrameState,
        body_output_local_xz_first: Tensor,
        body_output_local_xz_last: Tensor,
        body_output_local_yaw_first: Tensor,
        body_output_local_yaw_last: Tensor,
        committed_tokens: int,
    ) -> RootFrameState:
        state = advance_head_from_body_window(
            self.head,
            body_anchor_state,
            body_output_local_xz_first,
            body_output_local_xz_last,
            body_output_local_yaw_first,
            body_output_local_yaw_last,
            committed_tokens,
        )
        if state.commit_idx > self.head.commit_idx:
            self.append(state)
        return self.head


def body_window_start_commit_idx(
    head_commit_idx: int,
    body_history_tokens: int,
    explicit_start: int | None = None,
) -> int:
    """Return the body window history0 commit index."""
    if explicit_start is not None:
        return int(explicit_start)
    return max(0, int(head_commit_idx) - int(body_history_tokens))


def committed_frame_slice(
    head_commit_idx: int,
    body_anchor_commit_idx: int,
    committed_tokens: int,
    frames_per_token: int = 4,
) -> slice:
    """Committed body-output frame slice in body-window-local coordinates."""
    relative_start_token = int(head_commit_idx) - int(body_anchor_commit_idx)
    if relative_start_token < 0:
        raise ValueError(
            f"head_commit_idx ({head_commit_idx}) < body_anchor_commit_idx "
            f"({body_anchor_commit_idx})"
        )
    if committed_tokens <= 0:
        raise ValueError(f"committed_tokens must be > 0, got {committed_tokens}")
    return token_range_to_frame_slice(
        relative_start_token,
        int(committed_tokens),
        int(frames_per_token),
    )


def recovery_root_state_to_world(
    recovery: Any,
    anchor_state: RootFrameState,
) -> tuple[np.ndarray, float]:
    """Convert a full-stream local recovery state back to session-world root."""
    local_root = np.asarray(recovery.r_pos_accum, dtype=np.float32)
    anchor_xz = anchor_state.world_xz.to(dtype=torch.float32)
    anchor_yaw = anchor_state.world_yaw.to(dtype=torch.float32)
    local_xz = torch.as_tensor(
        local_root[[0, 2]],
        dtype=torch.float32,
        device=anchor_xz.device,
    )
    world_xz = anchor_xz + transform_xz_local_delta_to_world(local_xz, anchor_yaw)
    local_yaw = torch.as_tensor(
        -2.0 * float(recovery.r_rot_ang_accum),
        dtype=torch.float32,
        device=anchor_xz.device,
    )
    world_yaw = wrap_angle(anchor_yaw + local_yaw)
    root = local_root.copy()
    root[[0, 2]] = world_xz.detach().cpu().numpy()
    return root.astype(np.float32), float(world_yaw.detach().cpu().item())


def append_timeline_state_at_token_start_frame(
    timeline: RootTimeline,
    *,
    frame_idx: int,
    recovery: Any,
    session_anchor_state: RootFrameState | None = None,
    source: str = "stream_recovery",
    frames_per_token: int = 4,
) -> bool:
    """Append recovered root state when ``frame_idx`` is a token start frame."""
    frame_idx = int(frame_idx)
    if frame_idx <= 0:
        return False
    commit_idx = frame_idx_to_token_idx(frame_idx, frames_per_token)
    if token_start_frame(commit_idx, frames_per_token) != frame_idx:
        return False
    if commit_idx <= timeline.head.commit_idx:
        return False
    if session_anchor_state is None:
        session_anchor_state = timeline.at_commit(0)
    root, yaw = recovery_root_state_to_world(recovery, session_anchor_state)
    device = session_anchor_state.world_xz.device
    timeline.append(
        RootFrameState(
            commit_idx=commit_idx,
            world_xz=torch.as_tensor(root[[0, 2]], device=device, dtype=torch.float32),
            world_yaw=torch.as_tensor(yaw, device=device, dtype=torch.float32),
            source=source,
        )
    )
    return True


__all__ = [
    "RootFrameState",
    "RootTimeline",
    "advance_head_from_body_window",
    "append_timeline_state_at_token_start_frame",
    "body_window_start_commit_idx",
    "committed_frame_slice",
    "recovery_root_state_to_world",
]
