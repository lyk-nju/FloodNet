"""Payload helpers for active-window runtime updates."""

from __future__ import annotations

import torch

from utils.inference.root_plan import RootPlan, build_root_plan_stream_payload
from utils.inference.timeline import RootFrameState
from utils.local_frame import canonicalize_7d
from utils.token_frame import num_tokens_for_frame_len


def build_active_window_root_plan(
    world_segment_traj7: torch.Tensor,
    *,
    anchor_state: RootFrameState,
    anchor_commit_idx: int,
    token_dt: float,
    frames_per_token: int = 4,
    source: str = "active_window",
) -> RootPlan:
    segment = world_segment_traj7.detach().to(
        device=anchor_state.world_xz.device,
        dtype=torch.float32,
    )
    if segment.dim() != 2 or segment.shape[-1] != 7:
        raise ValueError(
            f"world_segment_traj7 must be [T,7], got {tuple(segment.shape)}"
        )
    anchor_xz = anchor_state.world_xz.detach().to(
        device=segment.device,
        dtype=segment.dtype,
    )
    anchor_yaw = anchor_state.world_yaw.detach().to(
        device=segment.device,
        dtype=segment.dtype,
    )
    local = canonicalize_7d(segment.unsqueeze(0), anchor_xz.unsqueeze(0), anchor_yaw.reshape(1))[0]
    return RootPlan(
        num_tokens_pred=num_tokens_for_frame_len(int(segment.shape[0]), int(frames_per_token)),
        valid_frames=int(segment.shape[0]),
        waypoints_local_7d=local,
        frame_dt=float(token_dt) / float(frames_per_token),
        frames_per_token=int(frames_per_token),
        anchor_commit_idx=int(anchor_commit_idx),
        anchor_world_xz=anchor_xz,
        anchor_world_yaw=anchor_yaw,
        source=str(source),
    )


def _slice_world_condition(
    world_condition_traj7: torch.Tensor,
    *,
    frame_start: int,
    frame_count: int,
    generated_history_traj7: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    world = world_condition_traj7.detach().float()
    if world.dim() != 2 or world.shape[-1] != 7:
        raise ValueError(
            f"world_condition_traj7 must be [T,7], got {tuple(world.shape)}"
        )
    frame_count = max(0, int(frame_count))
    out = world.new_zeros(frame_count, 7)
    mask = torch.zeros(frame_count, device=world.device, dtype=torch.bool)
    if frame_count == 0 or int(world.shape[0]) == 0:
        return out, mask
    start = max(0, int(frame_start))
    if start >= int(world.shape[0]):
        out[:] = world[-1]
        return out, mask
    end = min(int(world.shape[0]), start + frame_count)
    n = max(0, end - start)
    if n > 0:
        out[:n] = world[start:end]
        mask[:n] = True
    if n < frame_count:
        out[n:] = world[end - 1]
    if generated_history_traj7 is not None:
        history = generated_history_traj7.detach().to(device=world.device, dtype=world.dtype)
        if history.dim() != 2 or history.shape[-1] != 7:
            raise ValueError(
                f"generated_history_traj7 must be [T,7], got {tuple(history.shape)}"
            )
        hist_start = max(0, int(frame_start))
        hist_end = min(int(history.shape[0]), int(frame_start) + frame_count)
        if hist_end > hist_start:
            dst0 = hist_start - int(frame_start)
            count = hist_end - hist_start
            out[dst0 : dst0 + count] = history[hist_start:hist_end]
            mask[dst0 : dst0 + count] = True
    return out, mask


def _build_single_world_condition_payload(
    world_condition_traj7: torch.Tensor,
    timeline,
    *,
    local_start_token: int,
    absolute_start_token: int,
    absolute_final_right_token: int,
    horizon_tokens: int,
    frames_per_token: int,
    generated_history_traj7: torch.Tensor | None = None,
) -> dict | None:
    from utils.token_frame import token_start_frame

    num_tokens = max(0, int(absolute_final_right_token) + int(horizon_tokens) - int(absolute_start_token))
    if num_tokens <= 0 or not timeline.has_exact_state(int(absolute_start_token)):
        return None
    body_anchor_state = timeline.at_commit(int(absolute_start_token))
    frame_start = token_start_frame(int(absolute_start_token), int(frames_per_token))
    frame_count = int(num_tokens) * int(frames_per_token)
    world_slice, mask = _slice_world_condition(
        world_condition_traj7.to(
            device=body_anchor_state.world_xz.device,
            dtype=body_anchor_state.world_xz.dtype,
        ),
        frame_start=frame_start,
        frame_count=frame_count,
        generated_history_traj7=(
            None
            if generated_history_traj7 is None
            else generated_history_traj7.to(
                device=body_anchor_state.world_xz.device,
                dtype=body_anchor_state.world_xz.dtype,
            )
        ),
    )
    if world_slice.numel() == 0:
        return None
    local = canonicalize_7d(
        world_slice.unsqueeze(0),
        body_anchor_state.world_xz.detach().unsqueeze(0),
        body_anchor_state.world_yaw.detach().reshape(1),
    )[0]
    return {
        "traj_cond_7d_frame": local.unsqueeze(0),
        "traj_cond_frame_mask": mask.unsqueeze(0).to(dtype=torch.float32),
        "traj_start_token": int(local_start_token),
        "traj_abs_start_token": int(absolute_start_token),
        "traj_num_tokens": int(num_tokens),
        "body_anchor_token": int(local_start_token),
        "body_anchor_abs_token": int(absolute_start_token),
    }


def build_world_condition_stream_payload(
    world_condition_traj7: torch.Tensor,
    timeline,
    *,
    local_commit_index: int,
    absolute_commit_index: int,
    chunk_size: int,
    history_length: int,
    traj_horizon_tokens: int,
    frames_per_token: int = 4,
    generated_history_traj7: torch.Tensor | None = None,
) -> dict | None:
    local_commit = int(local_commit_index)
    absolute_commit = int(absolute_commit_index)
    chunk = int(chunk_size)
    local_earliest_right_token = local_commit + 1
    absolute_earliest_right_token = absolute_commit + 1
    local_final_right_token = local_commit + chunk
    absolute_final_right_token = absolute_commit + chunk
    local_start_token = max(0, local_earliest_right_token - min(local_earliest_right_token, int(history_length)))
    absolute_start_token = max(0, absolute_earliest_right_token - min(absolute_earliest_right_token, int(history_length)))
    num_tokens = max(0, absolute_final_right_token + int(traj_horizon_tokens) - absolute_start_token)
    if num_tokens <= 0 or local_final_right_token <= local_start_token:
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
        subpayload = _build_single_world_condition_payload(
            world_condition_traj7,
            timeline,
            local_start_token=sub_local_start,
            absolute_start_token=sub_abs_start,
            absolute_final_right_token=absolute_final_right_token,
            horizon_tokens=int(traj_horizon_tokens),
            frames_per_token=int(frames_per_token),
            generated_history_traj7=generated_history_traj7,
        )
        if subpayload is None:
            return None
        subpayloads.append(subpayload)
    if not subpayloads:
        return None
    payload = dict(subpayloads[0])
    payload["traj_substep_payloads"] = subpayloads
    return payload


def build_active_window_stream_payload(
    root_plan: RootPlan,
    timeline,
    *,
    local_commit_index: int,
    absolute_commit_index: int,
    chunk_size: int,
    history_length: int,
    traj_horizon_tokens: int,
) -> dict | None:
    return build_root_plan_stream_payload(
        root_plan,
        timeline,
        local_commit_index=int(local_commit_index),
        absolute_commit_index=int(absolute_commit_index),
        chunk_size=int(chunk_size),
        history_length=int(history_length),
        traj_horizon_tokens=int(traj_horizon_tokens),
    )


__all__ = [
    "build_active_window_root_plan",
    "build_active_window_stream_payload",
    "build_world_condition_stream_payload",
]
