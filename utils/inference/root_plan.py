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

import torch

from dataclasses import dataclass
from torch import Tensor


@dataclass
class RootPlan:
    """Refiner output in **plan-anchor-local frame** (B-full convention).

    A single plan can span multiple body windows. `valid_frames` is the real
    plan length in frame units; `num_tokens_pred` is only the LDF/VAE backend
    allocation derived from that frame length.
    """

    num_tokens_pred: int                   # LDF token allocation derived from valid_frames
    valid_frames: int                      # real plan length in frames, including anchor
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


def _build_single_rootplan_payload(
    traj_buf,
    timeline,
    *,
    local_start_token: int,
    absolute_start_token: int,
    absolute_final_right_token: int,
    horizon_tokens: int,
) -> dict | None:
    from utils.token_frame import token_range_to_frame_slice

    num_tokens = max(0, absolute_final_right_token + horizon_tokens - absolute_start_token)
    if (
        num_tokens <= 0
        or not timeline.has_exact_state(absolute_start_token)
    ):
        return None

    body_anchor_state = timeline.at_commit(absolute_start_token)
    frame_slice = token_range_to_frame_slice(absolute_start_token, num_tokens)
    traj_cond, traj_mask = traj_buf.get_body_traj_cond(
        head_state=body_anchor_state,
        body_anchor_state=body_anchor_state,
        horizon_tokens=num_tokens,
        expected_horizon_frame_slice=frame_slice,
    )
    return {
        "traj_cond_7d_frame": traj_cond.unsqueeze(0),
        "traj_cond_frame_mask": traj_mask.unsqueeze(0).to(dtype=torch.float32),
        "traj_start_token": int(local_start_token),
        "traj_abs_start_token": int(absolute_start_token),
        "traj_num_tokens": int(num_tokens),
        "body_anchor_token": int(local_start_token),
        "body_anchor_abs_token": int(absolute_start_token),
    }


def build_rootplan_stream_payload_from_buffer(
    traj_buf,
    timeline,
    *,
    local_commit_index: int,
    absolute_commit_index: int,
    chunk_size: int,
    history_length: int,
    traj_horizon_tokens: int,
) -> dict | None:
    """Build the direct 7D RootPlan payload consumed by stream_generate_step.

    ``local_commit_index`` indexes the model's rolling latent cache.
    ``absolute_commit_index`` indexes the world-space inference timeline and
    RootPlan anchor state. Keeping them separate is required after the model
    rolls its internal generated buffer.
    """
    if (
        traj_buf is None
        or timeline is None
        or not hasattr(traj_buf, "has_active_plan")
        or not traj_buf.has_active_plan()
    ):
        return None

    local_commit = int(local_commit_index)
    absolute_commit = int(absolute_commit_index)
    chunk = int(chunk_size)
    local_earliest_right_token = local_commit + 1
    absolute_earliest_right_token = absolute_commit + 1
    local_final_right_token = local_commit + chunk
    absolute_final_right_token = absolute_commit + chunk
    earliest_model_sl = min(local_earliest_right_token, int(history_length))
    local_start_token = max(0, local_earliest_right_token - earliest_model_sl)
    absolute_start_token = max(0, absolute_earliest_right_token - earliest_model_sl)
    horizon = max(0, int(traj_horizon_tokens))
    num_tokens = max(0, absolute_final_right_token + horizon - absolute_start_token)
    if (
        num_tokens <= 0
        or local_final_right_token <= local_start_token
        or not timeline.has_exact_state(absolute_start_token)
    ):
        return None

    subpayloads = []
    seen_starts: set[int] = set()
    for local_right_token in range(local_earliest_right_token, local_final_right_token + 1):
        model_sl = min(local_right_token, int(history_length))
        sub_local_start = max(0, local_right_token - model_sl)
        if sub_local_start in seen_starts:
            continue
        seen_starts.add(sub_local_start)
        absolute_right_token = absolute_commit + (local_right_token - local_commit)
        sub_abs_start = max(0, absolute_right_token - model_sl)
        subpayload = _build_single_rootplan_payload(
            traj_buf,
            timeline,
            local_start_token=sub_local_start,
            absolute_start_token=sub_abs_start,
            absolute_final_right_token=absolute_final_right_token,
            horizon_tokens=horizon,
        )
        if subpayload is None:
            return None
        subpayloads.append(subpayload)

    if not subpayloads or not any(
        bool(subpayload["traj_cond_frame_mask"].any())
        for subpayload in subpayloads
    ):
        return None

    payload = dict(subpayloads[0])
    payload["traj_substep_payloads"] = subpayloads
    return payload


__all__ = [
    "RootPlan",
    "build_rootplan_stream_payload_from_buffer",
    "slice_plan_with_mask",
    "plan_local_to_body_window_local",
]
