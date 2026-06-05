"""Shared RootPlan runtime payload helpers."""

from __future__ import annotations

import torch

from utils.token_frame import token_range_to_frame_slice


def _build_single_rootplan_payload(
    traj_buf,
    timeline,
    *,
    local_start_token: int,
    absolute_start_token: int,
    absolute_final_right_token: int,
    horizon_tokens: int,
) -> dict | None:
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


__all__ = ["build_rootplan_stream_payload_from_buffer"]
