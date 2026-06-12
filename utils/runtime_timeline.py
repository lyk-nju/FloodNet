"""Frame-aware runtime timeline helpers shared by eval and web demo."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from utils.inference_glue import InferenceGlueState, InferenceGlueTimeline
from utils.local_frame import transform_xz_local_delta_to_world, wrap_angle
from utils.token_frame import frame_idx_to_token_idx, token_start_frame


def recovery_root_state_to_world(
    recovery: Any,
    anchor_state: InferenceGlueState,
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
    timeline: InferenceGlueTimeline,
    *,
    frame_idx: int,
    recovery: Any,
    session_anchor_state: InferenceGlueState | None = None,
    source: str = "stream_recovery",
    frames_per_token: int = 4,
) -> bool:
    """Append a timeline state only when ``frame_idx`` is a token start frame.

    ``InferenceGlueState.commit_idx`` is the token whose first frame has just
    been recovered. Frame 0 is already represented by the initial state.
    """

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
        InferenceGlueState(
            commit_idx=commit_idx,
            world_xz=torch.as_tensor(root[[0, 2]], device=device, dtype=torch.float32),
            world_yaw=torch.as_tensor(yaw, device=device, dtype=torch.float32),
            source=source,
        )
    )
    return True


__all__ = [
    "append_timeline_state_at_token_start_frame",
    "recovery_root_state_to_world",
]
