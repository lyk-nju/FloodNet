"""Inference glue state: world ↔ local pose snapshot + commit-delta advance helper.

References:
- docs/TODO.md §T_A_03 lines 634-722 — InferenceGlueState spec, advance helper.
- docs/design.md §0.3 (Anchor Convention v1) — dual-anchor rules.

This module is **pure geometry + bookkeeping** (no model state, no torch nn).
It owns:
  - `InferenceGlueState`: a single (commit_idx, world_xz, world_yaw) snapshot,
    with `.to(device, dtype)` and a device/dtype-aware `initial(...)` classmethod.
  - `advance_head_from_body_window`: the canonical head-advance formula using
    body-window-local first→last delta and `body_anchor.world_yaw` (NOT head)
    for the position rotation back to world.

`InferenceGlueTimeline` (the sequence-of-states class) is added separately in
T_A_03c.
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


__all__ = [
    "InferenceGlueState",
    "advance_head_from_body_window",
]
