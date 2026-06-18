"""Validation batch conditioning helpers for LDF training/eval."""

from __future__ import annotations

import numpy as np
import torch

from utils.local_frame import canonicalize_7d
from utils.token_frame import token_range_to_frame_slice
from utils.training.ldf.conditioning import prepare_generate_condition
from utils.training.ldf.sample_creator import SampleCreator


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


def build_windowed_metric_ground_truth(batch: dict, model_batch: dict):
    """Return GT tensors cropped to the same latent window as generation."""
    gt_token = model_batch["token"]
    gt_token_length = model_batch["token_length"]
    raw_feature = batch["feature"]
    raw_feature_length = batch["feature_length"]
    latent_lengths = model_batch.get("feature_length", gt_token_length)
    starts = model_batch.get("_window_global_start_token")
    batch_size = int(gt_token.shape[0])
    device = gt_token.device
    if starts is None:
        starts = torch.zeros(batch_size, device=device, dtype=torch.long)
    else:
        starts = starts.to(device=device, dtype=torch.long).view(-1)
        if starts.numel() == 1 and batch_size > 1:
            starts = starts.expand(batch_size)
    latent_lengths = latent_lengths.to(device=device, dtype=torch.long).view(-1)
    raw_feature_length = raw_feature_length.to(device=device, dtype=torch.long).view(-1)

    gt_feature = []
    gt_feature_length = []
    for i in range(batch_size):
        start_token = int(starts[i].item())
        num_tokens = int(latent_lengths[i].item())
        frame_slice = token_range_to_frame_slice(start_token, num_tokens)
        raw_len = int(raw_feature_length[i].item())
        start_frame = min(int(frame_slice.start), raw_len)
        stop_frame = min(int(frame_slice.stop), raw_len)
        gt_feature.append(raw_feature[i, start_frame:stop_frame])
        gt_feature_length.append(max(0, stop_frame - start_frame))
    return gt_token, gt_token_length, gt_feature, gt_feature_length


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
    """Prepare an LDF validation/eval batch with clip-start-local 7D control."""
    if "token_length" not in batch:
        raise ValueError("prepare_ldf_eval_model_batch requires batch['token_length']")
    model_batch = SampleCreator(
        window_policy="prefix",
        sample_policy="fixed_window",
        end_tokens=batch["token_length"],
    ).create(batch)
    if _has_7d_traj(model_batch):
        source = model_batch.get("traj_features", model_batch.get("traj_cond_7d"))
        canon = _canonicalize_7d_clip_start(source)
        model_batch["traj_features"] = canon
        if "traj_cond_7d" in model_batch:
            model_batch["traj_cond_7d"] = canon
    model_batch = _to_device(model_batch, device)
    if model is not None:
        model_batch["ldf_condition"] = prepare_generate_condition(
            model,
            model_batch,
            device,
        )
    return model_batch


__all__ = [
    "build_windowed_metric_ground_truth",
    "prepare_ldf_eval_model_batch",
]
