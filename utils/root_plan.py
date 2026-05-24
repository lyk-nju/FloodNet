"""RootPlan dataclass + plan-local slicing + plan-local→body-window-local helper.

References:
- docs/TODO.md §T_A_03 lines 535-630 — RootPlan / slice_plan_with_mask /
  plan_local_to_body_window_local spec.
- docs/design.md §0.3 (Anchor Convention v1) and §3.6.1 (plan-local two-step
  conversion).

Two anchors (HARD CONSTRAINT, dual anchor §0.3):
    plan anchor:  Refiner anchor at plan-creation time
                  (= timeline.head when Refiner ran)
    body anchor:  body window history0 (leftmost frame of the body window)
                  — NOT head_state. Body diffusion training distribution is
                  history0-anchored, never current-root-anchored.

`plan_local_to_body_window_local` is the canonical helper that bridges them
(uncanonicalize from plan anchor → canonicalize to body anchor).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RootPlan:
    """Refiner output in **plan-anchor-local frame** (B-full convention).

    A single plan can span multiple body windows; runtime stores only the
    `valid_frames = num_frames_for_tokens(num_tokens_pred)` prefix of
    waypoints (padding beyond that is irrelevant since `slice_plan_with_mask`
    treats it as overflow + mask=0).
    """

    num_tokens_pred: int                   # token-level duration (includes anchor token)
    valid_frames: int                      # = num_frames_for_tokens(num_tokens_pred) = 4N-3 (N>=1)
    waypoints_local_7d: Tensor             # [valid_frames, 7] plan-anchor-local

    frame_dt: float                        # = 1 / fps
    frames_per_token: int = 4              # causal VAE token width

    # Plan anchor (used when switching frame across body windows).
    # IMPORTANT: anchor_commit_idx is the global token index where this plan's
    # local origin sits — typically == timeline.head.commit_idx at plan creation
    # (the Refiner anchor / effective_commit for delayed edits).
    anchor_commit_idx: int = 0
    anchor_world_xz: Tensor = None         # [2] world xz of the plan anchor
    anchor_world_yaw: Tensor = None        # scalar physical yaw

    source: str = "refiner"                # "refiner" | "gt" | "debug"

    def __post_init__(self):
        if self.waypoints_local_7d.ndim != 2 or self.waypoints_local_7d.shape[-1] != 7:
            raise ValueError(
                f"waypoints_local_7d must be [T, 7], got shape "
                f"{tuple(self.waypoints_local_7d.shape)}"
            )
        if self.waypoints_local_7d.shape[0] < self.valid_frames:
            raise ValueError(
                f"waypoints_local_7d has {self.waypoints_local_7d.shape[0]} frames "
                f"but valid_frames={self.valid_frames}"
            )
        if self.anchor_world_xz is None or self.anchor_world_yaw is None:
            raise ValueError("anchor_world_xz and anchor_world_yaw must be provided")


def slice_plan_with_mask(plan: RootPlan,
                          *,
                          frame_slice: slice | None = None,
                          current_plan_token: int | None = None,
                          horizon_tokens: int | None = None,
                          hold_last_on_overflow: bool = True,
                          ) -> tuple[Tensor, Tensor]:
    """Slice a plan-local frame range and return (traj, mask) with overflow handling.

    Either `frame_slice` (preferred, plan-local frame index) or the pair
    `(current_plan_token, horizon_tokens)` must be provided.

    ⚠ frame_slice indexes `plan.waypoints_local_7d`. It lives in **plan-local
    frame space**, derived by the caller from `current_plan_token + H_frame`.
    Do not pass body runtime's `expected_horizon_frame_slice` here — that one
    lives in body-window space and the two have different semantics even if
    they happen to share length.

    Returns:
      traj_plan_local: [H, 7] in plan-anchor-local frame; valid range is real
                        waypoints, overflow region is hold-last (or zero if
                        `hold_last_on_overflow=False`).
      traj_mask_frame: [H] bool; True on valid frames, False on overflow.
                        Downstream consumers MUST honor this — otherwise
                        hold-last will be misread as "stopped at end".
    """
    from utils.token_frame import token_range_to_frame_slice

    if frame_slice is None:
        if current_plan_token is None or horizon_tokens is None:
            raise ValueError(
                "Either frame_slice or (current_plan_token, horizon_tokens) "
                "must be provided"
            )
        # P1-4: a negative current_plan_token means the plan has not taken effect
        # yet; token_start_frame clamps token_idx<=0 to frame 0, which would
        # SILENTLY slice a pending plan from the start. Reject it instead.
        if current_plan_token < 0:
            raise ValueError(
                f"current_plan_token must be >= 0, got {current_plan_token} "
                "(a pending / not-yet-active plan must not be sliced)"
            )
        frame_slice = token_range_to_frame_slice(
            current_plan_token, horizon_tokens, plan.frames_per_token,
        )

    f_start = frame_slice.start
    f_stop = frame_slice.stop
    H = f_stop - f_start
    if H < 0:
        raise ValueError(f"frame_slice has negative length: {frame_slice}")

    # ⚠ device/dtype must match plan.waypoints_local_7d (possibly GPU / fp16).
    # `new_zeros` preserves both; never write `torch.zeros(...)` which
    # silently defaults to CPU + fp32 and breaks downstream body forward.
    out = plan.waypoints_local_7d.new_zeros(H, 7)
    mask = torch.zeros(H, device=plan.waypoints_local_7d.device, dtype=torch.bool)

    if H == 0:
        return out, mask

    if f_start >= plan.valid_frames:
        # Slice fully past the valid region: full hold-last (if enabled), mask all False.
        if hold_last_on_overflow and plan.valid_frames > 0:
            out[:] = plan.waypoints_local_7d[plan.valid_frames - 1]
        return out, mask

    # Partial overflow possible.
    valid_end = min(plan.valid_frames, f_stop)
    n_valid = valid_end - f_start
    out[:n_valid] = plan.waypoints_local_7d[f_start:valid_end]
    mask[:n_valid] = True
    if hold_last_on_overflow and n_valid < H:
        out[n_valid:] = plan.waypoints_local_7d[valid_end - 1]   # hold-last
        # mask[n_valid:] remains False — overflow region is unmasked.

    return out, mask


def plan_local_to_body_window_local(
    traj_plan_local: Tensor,           # [..., H, 7] in plan-anchor-local frame
    plan_anchor_xz: Tensor,            # [2] plan's anchor world xz
    plan_anchor_yaw: Tensor,           # scalar plan's anchor world yaw
    body_anchor_world_xz: Tensor,      # [2] body window history0 world xz  (NOT head)
    body_anchor_world_yaw: Tensor,     # scalar body window history0 world yaw (NOT head)
) -> Tensor:
    """Two-step conversion: plan-anchor-local → world → body-window-local.

    ⚠ Dual anchor (§0.3):
       - plan anchor: Refiner anchor at plan-creation time (= effective_commit head)
       - body anchor: body window history0 (leftmost frame of body window)
       The two are almost always different. Body diffusion training distribution
       is history0-anchored — do NOT pass head_state as body anchor here.

    Pure composition of `canonicalize_7d` / `uncanonicalize_7d` from local_frame —
    no hand-rolled rotation matrix.
    """
    from utils.local_frame import canonicalize_7d, uncanonicalize_7d

    traj_world = uncanonicalize_7d(traj_plan_local, plan_anchor_xz, plan_anchor_yaw)
    traj_body_local = canonicalize_7d(
        traj_world, body_anchor_world_xz, body_anchor_world_yaw,
    )
    return traj_body_local


__all__ = [
    "RootPlan",
    "slice_plan_with_mask",
    "plan_local_to_body_window_local",
]
