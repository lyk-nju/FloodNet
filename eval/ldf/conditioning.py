"""Shared LDF eval conditioning helpers.

The 7D body model is trained on body-window-local trajectory conditions, while
dataset batches expose ``traj_cond_7d`` in world/clip coordinates. The helpers
here keep LDF eval entrypoints on the same 7D contract instead of each script
hand-rolling trajectory routing.
"""

from __future__ import annotations

import numpy as np
import torch

from utils.inference_glue import InferenceGlueState, InferenceGlueTimeline
from utils.local_frame import (
    canonicalize_7d,
    transform_xz_local_delta_to_world,
    wrap_angle,
)
from utils.root_plan import RootPlan
from utils.runtime_rootplan import build_rootplan_stream_payload_from_buffer
from utils.token_frame import num_tokens_for_frame_len
from utils.traj_stream_buffer import TrajStreamBuffer
from utils.training.model_batch import prepare_model_input


def _as_tensor(value, *, device=None, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value
    elif isinstance(value, np.ndarray):
        out = torch.from_numpy(value)
    else:
        out = torch.as_tensor(value)
    if dtype is not None and torch.is_floating_point(out):
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device)
    return out


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(value, device) for value in obj)
    return obj


def _has_7d_traj(batch: dict) -> bool:
    value = batch.get("traj_cond_7d", batch.get("traj_features"))
    if value is None:
        return False
    shape = value.shape if hasattr(value, "shape") else None
    return shape is not None and len(shape) >= 2 and int(shape[-1]) == 7


def _canonicalize_7d_clip_start(traj_7d) -> torch.Tensor:
    traj = _as_tensor(traj_7d, dtype=torch.float32)
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if traj.dim() != 3 or traj.shape[-1] != 7:
        raise ValueError(f"traj_cond_7d must be [B,T,7], got {tuple(traj.shape)}")
    if traj.shape[1] <= 0:
        return traj
    anchor_xz = traj[:, 0, [0, 2]]
    anchor_yaw = torch.atan2(traj[:, 0, 4], traj[:, 0, 3])
    return canonicalize_7d(traj, anchor_xz, anchor_yaw)


def prepare_ldf_eval_model_batch(batch: dict, device, model=None) -> dict:
    """Prepare a model batch for LDF eval with 7D clip-start-local conditioning.

    ``prepare_model_input`` preserves the training field routing, but it does not
    canonicalize 7D world-frame trajectory conditions. Offline eval has no
    rolling body window, so the stable eval convention is clip-start-local.
    """
    model_batch = prepare_model_input(batch)
    if _has_7d_traj(model_batch):
        source = model_batch.get("traj_features", model_batch.get("traj_cond_7d"))
        canon = _canonicalize_7d_clip_start(source)
        model_batch["traj_features"] = canon
        if "traj_cond_7d" in model_batch:
            model_batch["traj_cond_7d"] = canon
    return _to_device(model_batch, device)


def _first_scalar(value, default: int) -> int:
    if value is None:
        return int(default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return int(default)
        return int(value.reshape(-1)[0].detach().cpu().item())
    arr = np.asarray(value)
    if arr.size == 0:
        return int(default)
    return int(arr.reshape(-1)[0])


def build_gt_rootplan_from_batch(
    sample_batch: dict,
    *,
    token_dt: float,
    frames_per_token: int = 4,
    device=None,
) -> RootPlan:
    """Build a single-sample GT RootPlan from frame-level world 7D condition."""
    if "traj_cond_7d" not in sample_batch:
        raise KeyError("sample_batch must contain traj_cond_7d to build a GT RootPlan")
    traj = _as_tensor(sample_batch["traj_cond_7d"], device=device, dtype=torch.float32)
    if traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if traj.dim() != 3 or traj.shape[0] != 1 or traj.shape[-1] != 7:
        raise ValueError(
            "build_gt_rootplan_from_batch expects single-sample traj_cond_7d "
            f"[1,T,7], got {tuple(traj.shape)}"
        )
    valid_frames = _first_scalar(sample_batch.get("traj_length"), traj.shape[1])
    valid_frames = max(0, min(int(valid_frames), int(traj.shape[1])))
    if valid_frames <= 0:
        raise ValueError("traj_cond_7d has no valid frames")

    traj = traj[:, :valid_frames, :]
    anchor_xz = traj[0, 0, [0, 2]].clone()
    anchor_yaw = torch.atan2(traj[0, 0, 4], traj[0, 0, 3]).clone()
    waypoints_local = canonicalize_7d(
        traj,
        anchor_xz.unsqueeze(0),
        anchor_yaw.reshape(1),
    )[0]
    return RootPlan(
        num_tokens_pred=num_tokens_for_frame_len(valid_frames, frames_per_token),
        valid_frames=valid_frames,
        waypoints_local_7d=waypoints_local,
        frame_dt=float(token_dt) / float(frames_per_token),
        frames_per_token=int(frames_per_token),
        anchor_commit_idx=0,
        anchor_world_xz=anchor_xz,
        anchor_world_yaw=anchor_yaw,
        source="ldf_eval_gt",
    )


class LdfEvalStreamConditioner:
    """Stateful GT-RootPlan conditioner for LDF ``stream_generate_step`` eval."""

    def __init__(
        self,
        sample_batch: dict,
        *,
        history_length: int,
        traj_horizon_tokens: int,
        token_dt: float,
        device,
        frames_per_token: int = 4,
    ):
        self.device = torch.device(device)
        self.history_length = int(history_length)
        self.traj_horizon_tokens = int(traj_horizon_tokens)
        self.frames_per_token = int(frames_per_token)
        self.root_plan = build_gt_rootplan_from_batch(
            sample_batch,
            token_dt=token_dt,
            frames_per_token=frames_per_token,
            device=self.device,
        )
        token_length = _first_scalar(sample_batch.get("token_length"), self.root_plan.num_tokens_pred)
        buf_len = max(
            self.history_length * 2 + 1,
            token_length + self.history_length + self.traj_horizon_tokens + 4,
        )
        self.traj_buf = TrajStreamBuffer(
            batch_size=1,
            buf_len=buf_len,
            device=self.device,
            dtype=self.root_plan.waypoints_local_7d.dtype,
        )
        self.traj_buf.set_root_plan(self.root_plan)
        self.timeline = InferenceGlueTimeline(
            InferenceGlueState(
                commit_idx=0,
                world_xz=self.root_plan.anchor_world_xz.clone(),
                world_yaw=self.root_plan.anchor_world_yaw.clone(),
                source="ldf_eval_gt_anchor",
            )
        )
        self._anchor_xz = self.root_plan.anchor_world_xz.clone()
        self._anchor_yaw = self.root_plan.anchor_world_yaw.clone()

    def build_step_payload(
        self,
        *,
        local_commit_index: int,
        absolute_commit_index: int | None = None,
        chunk_size: int = 1,
    ) -> dict | None:
        absolute_commit = (
            int(local_commit_index)
            if absolute_commit_index is None
            else int(absolute_commit_index)
        )
        return build_rootplan_stream_payload_from_buffer(
            self.traj_buf,
            self.timeline,
            local_commit_index=int(local_commit_index),
            absolute_commit_index=absolute_commit,
            chunk_size=int(chunk_size),
            history_length=self.history_length,
            traj_horizon_tokens=self.traj_horizon_tokens,
        )

    def append_decoded(self, decoded_chunk, *, commit_idx: int, recovery) -> None:
        frames = decoded_chunk
        if torch.is_tensor(frames):
            frames = frames.detach().cpu().numpy()
        frames = np.asarray(frames, dtype=np.float32)
        for frame in frames:
            recovery.process_frame(frame)
        local_xz = torch.as_tensor(
            recovery.r_pos_accum[[0, 2]],
            device=self.device,
            dtype=self._anchor_xz.dtype,
        )
        local_yaw = torch.as_tensor(
            -2.0 * float(recovery.r_rot_ang_accum),
            device=self.device,
            dtype=self._anchor_yaw.dtype,
        )
        world_xz = self._anchor_xz + transform_xz_local_delta_to_world(
            local_xz,
            self._anchor_yaw,
        )
        world_yaw = wrap_angle(self._anchor_yaw + local_yaw)
        if int(commit_idx) > self.timeline.head.commit_idx:
            self.timeline.append(
                InferenceGlueState(
                    commit_idx=int(commit_idx),
                    world_xz=world_xz,
                    world_yaw=world_yaw,
                    source="ldf_eval_stream",
                )
            )


__all__ = [
    "LdfEvalStreamConditioner",
    "build_gt_rootplan_from_batch",
    "prepare_ldf_eval_model_batch",
]
