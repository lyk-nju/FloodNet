"""Shared RootPlan runtime payload helpers."""

from __future__ import annotations

import torch

from utils.token_frame import token_range_to_frame_slice


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
    local_right_token = local_commit + chunk
    absolute_right_token = absolute_commit + chunk
    model_sl = min(local_right_token, int(history_length))
    local_start_token = max(0, local_right_token - model_sl)
    absolute_start_token = max(0, absolute_right_token - model_sl)
    horizon = max(0, int(traj_horizon_tokens))
    traj_local_start_token = local_start_token
    traj_abs_start_token = absolute_start_token
    num_tokens = max(0, model_sl + horizon)
    if (
        num_tokens <= 0
        or not timeline.has_exact_state(absolute_start_token)
        or not timeline.has_exact_state(traj_abs_start_token)
    ):
        return None

    # The direct 7D stream payload is aligned to the latent attention window:
    # latent keys [S, E), trajectory keys [S, E + H).  get_body_traj_cond names
    # this argument head_state because it computes the plan-local slice from
    # commit_idx; here that slice starts at S, while body_anchor_state also
    # anchors canonicalization at S.
    head_state = timeline.at_commit(traj_abs_start_token)
    body_anchor_state = timeline.at_commit(absolute_start_token)
    frame_slice = token_range_to_frame_slice(traj_abs_start_token, num_tokens)
    traj_cond, traj_mask = traj_buf.get_body_traj_cond(
        head_state=head_state,
        body_anchor_state=body_anchor_state,
        horizon_tokens=num_tokens,
        expected_horizon_frame_slice=frame_slice,
    )
    if not bool(traj_mask.any()):
        return None
    return {
        "traj_cond_7d_frame": traj_cond.unsqueeze(0),
        "traj_cond_frame_mask": traj_mask.unsqueeze(0).to(dtype=torch.float32),
        "traj_start_token": traj_local_start_token,
        "traj_abs_start_token": traj_abs_start_token,
        "traj_num_tokens": num_tokens,
        "body_anchor_token": local_start_token,
        "body_anchor_abs_token": absolute_start_token,
    }


__all__ = ["build_rootplan_stream_payload_from_buffer"]
