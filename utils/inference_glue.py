"""Inference glue state + timeline: world ↔ local pose snapshots and history.

References:
- docs/TODO.md §T_A_03 lines 634-762 — InferenceGlueState / advance helper /
  InferenceGlueTimeline spec.
- docs/design.md §0.3 (Anchor Convention v1) — dual-anchor rules.

This module is **pure geometry + bookkeeping** (no model state, no torch nn).
It owns:
  - `InferenceGlueState`: a single (commit_idx, world_xz, world_yaw) snapshot,
    with `.to(device, dtype)` and a device/dtype-aware `initial(...)` classmethod.
  - `advance_head_from_body_window`: the canonical head-advance formula using
    body-window-local first→last delta and `body_anchor.world_yaw` (NOT head)
    for the position rotation back to world.
  - `InferenceGlueTimeline`: an append-only sequence of states indexed by
    `commit_idx`, supporting exact-state lookup (`has_exact_state` /
    `at_commit`) and a coarser "have we passed this point" probe
    (`has_reached`). Mid-session edit / delay-effective-commit logic depends
    on both, and the two are intentionally distinct (see §3.7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

log = logging.getLogger(__name__)


@dataclass
class InferenceGlueState:
    """Single world ↔ local conversion state.

    commit_idx semantics (NORMATIVE, locked by G04):
        = next first-token index after committed tokens
        = total committed token count = next window start index
        NOT "the last committed token's index".

    Examples:
        initial → commit_idx = 0
        after committing 5 tokens → commit_idx = 5
        after committing 5 more → commit_idx = 10

    world_xz: [2] xz position on the ground plane (Y-up convention).
    world_yaw: scalar physical yaw (rad).
    source: provenance tag for debug / event logging.
    """

    commit_idx: int
    world_xz: Tensor
    world_yaw: Tensor
    source: str = "init"

    @classmethod
    def initial(cls, xz=(0.0, 0.0), yaw=0.0, device=None, dtype=None) -> "InferenceGlueState":
        """Construct the session-start state with explicit device/dtype injection.

        ⚠ device/dtype MUST be passed by the caller (typically body model's
        device/dtype). Defaulting to CPU+fp32 here would cause downstream
        device mismatch with traj tensors that live on CUDA (round 8 P0-1).
        """
        return cls(
            commit_idx=0,
            world_xz=torch.tensor(xz, device=device, dtype=dtype),
            world_yaw=torch.tensor(yaw, device=device, dtype=dtype),
            source="init",
        )

    def to(self, device=None, dtype=None) -> "InferenceGlueState":
        """Cast state to a new device/dtype, returning a fresh state (not in-place).

        Only `world_xz` / `world_yaw` are cast; `commit_idx` (int) and `source`
        (str) are immutable Python scalars.

        Runtime usage (round 8 P0-1):
          ModelManager._step_body_window must `state.to(body_device, body_dtype)`
          on both head_state and body_anchor_state before passing them into
          TrajStreamBuffer / canonicalize_*, since traj tensors may live on CUDA
          while the timeline state objects may have been constructed on CPU.
        """
        return InferenceGlueState(
            commit_idx=self.commit_idx,
            world_xz=self.world_xz.to(device=device, dtype=dtype),
            world_yaw=self.world_yaw.to(device=device, dtype=dtype),
            source=self.source,
        )

    # ⚠ The single-state .advance() method is intentionally NOT defined.
    # Earlier prototype used `head.world_yaw` to rotate position delta, which is
    # WRONG under dual anchor — local frame is body_anchor-anchored, so the
    # position-delta rotation must use body_anchor.world_yaw, not head.world_yaw.
    # Use `advance_head_from_body_window` (below) instead, which forces the
    # caller to supply both states.


def advance_head_from_body_window(
    head_state: InferenceGlueState,
    body_anchor_state: InferenceGlueState,
    body_output_local_xz_first: Tensor,    # [2] body output first-frame xz in body-window-local
    body_output_local_xz_last: Tensor,     # [2] body output last-frame xz in body-window-local
    body_output_local_yaw_first: Tensor,   # scalar — HumanML3D recovery forces first yaw = 0
    body_output_local_yaw_last: Tensor,    # scalar
    committed_tokens: int,
) -> InferenceGlueState:
    """Advance `head_state` by a body-window-local first→last delta.

    ⚠ Dual anchor rules (§0.3):
      1. Position delta is rotated by `body_anchor_state.world_yaw` (NOT
         head_state.world_yaw). The body output lives in body-window-local
         frame whose +Z axis points along body_anchor.world_yaw, so the
         delta-to-world rotation uses body_anchor's yaw.
      2. Yaw delta is added to `head_state.world_yaw` (head advances its own
         orientation; body_anchor is fixed history0).
      3. The body output's first-frame yaw is forced to 0 by HumanML3D recovery,
         so we do NOT read it to reconstruct world yaw — we accumulate yaw
         deltas onto head_state instead.

    NaN/Inf protection: if any input delta is non-finite, log a warning and
    return `head_state` unchanged (commit_idx does NOT advance). This avoids
    poisoning the timeline with NaN poses on the first divergent body step.
    """
    from utils.local_frame import transform_xz_local_delta_to_world, wrap_angle

    delta_xz_local = body_output_local_xz_last - body_output_local_xz_first
    delta_yaw_local = wrap_angle(body_output_local_yaw_last - body_output_local_yaw_first)

    # NaN/Inf guard — preserve old state.
    if not (torch.isfinite(delta_xz_local).all().item()
            and torch.isfinite(delta_yaw_local).all().item()):
        log.warning(
            "advance_head_from_body_window: NaN/Inf in body-local delta; "
            "keeping old head_state (commit_idx=%d)", head_state.commit_idx,
        )
        return head_state

    # Rotate delta to world frame using body_anchor's yaw — NOT head's yaw.
    delta_xz_world = transform_xz_local_delta_to_world(
        delta_xz_local, body_anchor_state.world_yaw,
    )

    new_world_xz = head_state.world_xz + delta_xz_world
    new_world_yaw = wrap_angle(head_state.world_yaw + delta_yaw_local)

    return InferenceGlueState(
        commit_idx=head_state.commit_idx + committed_tokens,
        world_xz=new_world_xz,
        world_yaw=new_world_yaw,
        source="commit",
    )


class InferenceGlueTimeline:
    """Append-only sequence of `InferenceGlueState` snapshots, indexed by
    `commit_idx` (monotonically increasing).

    Supports mid-session edit / delayed-Refiner logic:
      - `head`: latest state.
      - `at_commit(commit_idx)`: exact lookup for delay-anchor or stale check;
         non-exact lookups return a fallback + log warning (NOT for stale check).
      - `has_reached(commit_idx)`: cheap "did we pass this point" test
         (effective_commit gating).
      - `has_exact_state(commit_idx)`: stale-check primitive — True only when an
         exact snapshot exists (trim invalidates older points; see `trim_before`).
      - `trim_before(commit_idx)`: drop snapshots older than `commit_idx` to
         cap memory; always keeps at least the head so the timeline is never empty.

    The timeline does NOT own model state, plans, or pending edits — it just
    bookkeeps geometry-with-commit-index snapshots.
    """

    def __init__(self, initial: InferenceGlueState):
        if not isinstance(initial, InferenceGlueState):
            raise TypeError(f"initial must be InferenceGlueState, got {type(initial)}")
        self._states: list[InferenceGlueState] = [initial]

    @property
    def head(self) -> InferenceGlueState:
        return self._states[-1]

    @property
    def earliest(self) -> InferenceGlueState:
        """The oldest state still in memory (after any trim_before)."""
        return self._states[0]

    def __len__(self) -> int:
        return len(self._states)

    def append(self, state: InferenceGlueState) -> None:
        """Append a state. `state.commit_idx` MUST be > current head's
        commit_idx (timeline is strictly monotone). Otherwise raises.
        """
        if not isinstance(state, InferenceGlueState):
            raise TypeError(f"state must be InferenceGlueState, got {type(state)}")
        if state.commit_idx <= self.head.commit_idx:
            raise ValueError(
                f"new state commit_idx={state.commit_idx} must be > head "
                f"commit_idx={self.head.commit_idx} (timeline is strictly "
                f"monotone)"
            )
        self._states.append(state)

    # ------------------------------------------------------------------
    # Lookup primitives
    # ------------------------------------------------------------------

    def _binary_search_exact(self, commit_idx: int) -> int | None:
        """Return list index of the state matching `commit_idx`, or None."""
        lo, hi = 0, len(self._states) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            m_ci = self._states[mid].commit_idx
            if m_ci == commit_idx:
                return mid
            if m_ci < commit_idx:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def has_reached(self, commit_idx: int) -> bool:
        """Has the timeline advanced past or to `commit_idx`?

        Used by: effective_commit gating — "has head reached the point where
        we should dispatch the delayed Refiner request?"
        Note: returning True does NOT guarantee an exact snapshot exists at
        `commit_idx` (especially after `trim_before`); for that use
        `has_exact_state` instead.
        """
        return commit_idx <= self.head.commit_idx

    def has_exact_state(self, commit_idx: int) -> bool:
        """Does an exact snapshot exist at `commit_idx`?

        Used by: stale-Refiner-result validation, delayed-edit anchor precision.
        After `trim_before(k)`, snapshots with `commit_idx < k` are gone and
        this returns False for them even though `has_reached` would still
        return True.
        """
        return self._binary_search_exact(commit_idx) is not None

    def at_commit(self, commit_idx: int) -> InferenceGlueState:
        """Exact snapshot at `commit_idx`, else a fallback + log warning.

        v1 does NOT predict future anchors: when `commit_idx > head.commit_idx`,
        the fallback is `head` itself (you should typically wait until
        `has_reached(commit_idx)` becomes True before calling this).

        When `commit_idx < earliest.commit_idx` (trimmed), the fallback is the
        earliest available snapshot. In that case `has_exact_state` returns
        False, and you MUST NOT use this fallback for stale-result validation.

        For gaps inside `[earliest, head]` (which shouldn't happen if every
        commit appends a snapshot, but is possible if append was skipped), the
        fallback is the nearest preceding snapshot.
        """
        idx = self._binary_search_exact(commit_idx)
        if idx is not None:
            return self._states[idx]

        # Not an exact match — pick a fallback and warn.
        if commit_idx > self.head.commit_idx:
            log.warning(
                "InferenceGlueTimeline.at_commit(%d): query past head "
                "(commit_idx=%d); returning head as fallback (v1 does not "
                "predict future anchors)",
                commit_idx, self.head.commit_idx,
            )
            return self.head

        if commit_idx < self.earliest.commit_idx:
            log.warning(
                "InferenceGlueTimeline.at_commit(%d): query before earliest "
                "(commit_idx=%d) — likely trimmed; returning earliest as "
                "fallback. has_exact_state is False; do NOT use for stale check.",
                commit_idx, self.earliest.commit_idx,
            )
            return self.earliest

        # Gap inside [earliest, head]: nearest preceding snapshot.
        # Find largest index whose commit_idx < commit_idx.
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
            "InferenceGlueTimeline.at_commit(%d): no exact snapshot; "
            "returning nearest preceding at commit_idx=%d",
            commit_idx, self._states[candidate].commit_idx,
        )
        return self._states[candidate]

    # ------------------------------------------------------------------
    # Memory cap
    # ------------------------------------------------------------------

    def trim_before(self, commit_idx: int) -> None:
        """Drop snapshots with `commit_idx < `commit_idx`. Always keeps the
        latest state (head) so the timeline is never empty — even if all
        snapshots are older than the cutoff, only the head survives.
        """
        # Find the first index whose commit_idx >= commit_idx.
        cut = 0
        while cut < len(self._states) and self._states[cut].commit_idx < commit_idx:
            cut += 1
        if cut == len(self._states):
            # All states are older than cutoff — keep only the head.
            self._states = self._states[-1:]
        elif cut > 0:
            self._states = self._states[cut:]


__all__ = [
    "InferenceGlueState",
    "advance_head_from_body_window",
    "InferenceGlueTimeline",
]
