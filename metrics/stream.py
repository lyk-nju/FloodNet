from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

try:
    from FloodNet.eval.eval_summary import build_eval_summary
    from FloodNet.utils.motion_process import (
        extract_root_trajectory_263_torch,
        recover_joint_positions_263,
    )
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from eval.eval_summary import build_eval_summary
    from utils.motion_process import (
        extract_root_trajectory_263_torch,
        recover_joint_positions_263,
    )


def _to_feature_tensor(pred_feature: torch.Tensor | np.ndarray) -> torch.Tensor:
    if torch.is_tensor(pred_feature):
        feat = pred_feature.detach().float().cpu()
    else:
        feat = torch.from_numpy(np.asarray(pred_feature)).float()
    if feat.ndim != 2:
        raise ValueError(f"Expected feature tensor with shape (T, C), got {tuple(feat.shape)}")
    return feat


def decode_stream_chunks(
    vae,
    latent_chunks: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    decoded_chunks: List[torch.Tensor] = []
    chunk_frame_ends: List[int] = []
    total_frames = 0
    first_chunk = True

    vae.clear_cache()
    try:
        for latent_chunk in latent_chunks:
            if latent_chunk is None:
                continue
            chunk = latent_chunk.detach()
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(0)
            decoded_chunk = vae.stream_decode(chunk, first_chunk=first_chunk)[0]
            decoded_chunk = decoded_chunk.float().detach().cpu()
            first_chunk = False
            decoded_chunks.append(decoded_chunk)
            total_frames += int(decoded_chunk.shape[0])
            chunk_frame_ends.append(total_frames)
    finally:
        vae.clear_cache()

    if decoded_chunks:
        decoded_feature = torch.cat(decoded_chunks, dim=0)
    else:
        decoded_feature = torch.zeros((0, 0), dtype=torch.float32)
    return decoded_feature, decoded_chunks, chunk_frame_ends


def compute_stream_boundary_metrics(
    pred_feature: torch.Tensor | np.ndarray,
    chunk_frame_ends: Sequence[int],
    joints_num: int = 22,
) -> Dict:
    feat = _to_feature_tensor(pred_feature)
    if feat.numel() == 0 or feat.shape[-1] != 263:
        return {
            "root_jump_per_boundary": [],
            "joint_jump_per_boundary": [],
            "root_jump_mean": float("nan"),
            "root_jump_max": float("nan"),
            "joint_jump_mean": float("nan"),
            "n_boundaries": 0,
        }

    valid_boundaries = [
        int(boundary)
        for boundary in chunk_frame_ends[:-1]
        if 0 < int(boundary) < int(feat.shape[0])
    ]
    if not valid_boundaries:
        return {
            "root_jump_per_boundary": [],
            "joint_jump_per_boundary": [],
            "root_jump_mean": float("nan"),
            "root_jump_max": float("nan"),
            "joint_jump_mean": float("nan"),
            "n_boundaries": 0,
        }

    root_xyz = extract_root_trajectory_263_torch(feat.unsqueeze(0))[0].cpu().numpy()
    joints_xyz = recover_joint_positions_263(feat.numpy(), joints_num=joints_num)

    root_jumps: List[float] = []
    joint_jumps: List[float] = []
    for boundary in valid_boundaries:
        root_prev = root_xyz[boundary - 1, [0, 2]]
        root_next = root_xyz[boundary, [0, 2]]
        root_jumps.append(float(np.linalg.norm(root_next - root_prev)))

        joint_prev = joints_xyz[boundary - 1]
        joint_next = joints_xyz[boundary]
        joint_jumps.append(
            float(np.linalg.norm(joint_next - joint_prev, axis=-1).mean())
        )

    return {
        "root_jump_per_boundary": root_jumps,
        "joint_jump_per_boundary": joint_jumps,
        "root_jump_mean": float(np.mean(root_jumps)),
        "root_jump_max": float(np.max(root_jumps)),
        "joint_jump_mean": float(np.mean(joint_jumps)),
        "n_boundaries": int(len(valid_boundaries)),
    }


def compute_stream_vs_offline_metrics(
    pred_stream_feature: torch.Tensor | np.ndarray,
    pred_offline_feature: torch.Tensor | np.ndarray,
) -> Dict:
    stream_feat = _to_feature_tensor(pred_stream_feature)
    offline_feat = _to_feature_tensor(pred_offline_feature)
    if stream_feat.numel() == 0 or offline_feat.numel() == 0:
        return {
            "feature_l2_mean": float("nan"),
            "feature_l2_max": float("nan"),
            "root_ade": float("nan"),
            "length_delta": abs(int(stream_feat.shape[0]) - int(offline_feat.shape[0])),
        }

    aligned_len = min(int(stream_feat.shape[0]), int(offline_feat.shape[0]))
    diff = stream_feat[:aligned_len] - offline_feat[:aligned_len]
    feature_l2 = diff.norm(dim=-1)

    result = {
        "feature_l2_mean": float(feature_l2.mean().item()),
        "feature_l2_max": float(feature_l2.max().item()),
        "length_delta": abs(int(stream_feat.shape[0]) - int(offline_feat.shape[0])),
    }

    if stream_feat.shape[-1] == 263 and offline_feat.shape[-1] == 263:
        stream_root = extract_root_trajectory_263_torch(stream_feat[:aligned_len].unsqueeze(0))[0]
        offline_root = extract_root_trajectory_263_torch(offline_feat[:aligned_len].unsqueeze(0))[0]
        root_diff = stream_root[:, [0, 2]] - offline_root[:, [0, 2]]
        result["root_ade"] = float(root_diff.norm(dim=-1).mean().item())
    else:
        result["root_ade"] = float("nan")
    return result


def _yaw_from_root_path(root_xyz: np.ndarray) -> np.ndarray:
    root = np.asarray(root_xyz, dtype=np.float32)
    if root.ndim != 2 or root.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    xz = root[:, [0, 2]] if root.shape[1] >= 3 else root
    if xz.shape[0] < 2:
        return np.zeros((xz.shape[0],), dtype=np.float32)
    vel = np.gradient(xz, axis=0)
    yaw = np.arctan2(vel[:, 0], vel[:, 1])
    if yaw.shape[0] > 1:
        yaw[0] = yaw[1]
    return yaw.astype(np.float32)


def compute_root_path_yaw_error(
    pred_root_xyz: torch.Tensor | np.ndarray,
    target_root_xyz: torch.Tensor | np.ndarray,
) -> float:
    """Mean wrapped yaw error from XZ root-path heading."""
    pred = pred_root_xyz.detach().cpu().numpy() if torch.is_tensor(pred_root_xyz) else np.asarray(pred_root_xyz)
    target = target_root_xyz.detach().cpu().numpy() if torch.is_tensor(target_root_xyz) else np.asarray(target_root_xyz)
    pred_yaw = _yaw_from_root_path(pred)
    target_yaw = _yaw_from_root_path(target)
    n = min(int(pred_yaw.shape[0]), int(target_yaw.shape[0]))
    if n == 0:
        return float("nan")
    diff = pred_yaw[:n] - target_yaw[:n]
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return float(np.mean(np.abs(wrapped)))


def summarize_stream_records(records: Sequence[Dict]) -> Dict:
    if not records:
        return {}
    summary = build_eval_summary(records)

    def _append_stats(record_key: str, summary_prefix: str):
        vals = [record[record_key] for record in records if record_key in record and record[record_key] == record[record_key]]
        if vals:
            summary[f"{summary_prefix}_mean"] = round(float(np.mean(vals)), 10)
            summary[f"{summary_prefix}_std"] = round(float(np.std(vals)), 10)

    def _append_delta_stats(
        base_key: str,
        ablation_key: str,
        summary_prefix: str,
    ):
        vals = []
        for record in records:
            if base_key not in record or ablation_key not in record:
                continue
            base = record[base_key]
            ablation = record[ablation_key]
            if base == base and ablation == ablation:
                vals.append(float(ablation) - float(base))
        if vals:
            summary[f"{summary_prefix}_mean"] = round(float(np.mean(vals)), 10)
            summary[f"{summary_prefix}_std"] = round(float(np.std(vals)), 10)

    _append_stats("stream_root_jump_mean", "stream_boundary/root_jump")
    _append_stats("stream_root_jump_max", "stream_boundary/root_jump_max")
    _append_stats("stream_joint_jump_mean", "stream_boundary/joint_jump")
    _append_stats("stream_num_boundaries", "stream_boundary/n_boundaries")
    _append_stats("stream_offline_feature_l2_mean", "stream_vs_offline/feature_l2")
    _append_stats("stream_offline_feature_l2_max", "stream_vs_offline/feature_l2_max")
    _append_stats("stream_offline_root_ade", "stream_vs_offline/root_ade")
    _append_stats("stream_offline_length_delta", "stream_vs_offline/length_delta")
    _append_stats("stream_yaw_error", "stream_gt/yaw_error")
    _append_stats("stream_no_traj/ade", "stream_no_traj/root_ADE")
    _append_stats("stream_no_traj/fde", "stream_no_traj/root_FDE")
    _append_stats("stream_no_traj/path_arc_ade", "stream_no_traj/path_arc_ADE")
    _append_delta_stats("ade", "stream_no_traj/ade", "control_gain/root_ADE_delta")
    _append_delta_stats("fde", "stream_no_traj/fde", "control_gain/root_FDE_delta")

    def _copy_alias(src: str, dst: str):
        value = summary.get(src)
        if value is None:
            return
        if value == value:
            summary[dst] = round(float(value), 10)

    for src, dst in (
        ("traj/ADE_mean", "stream_gt/root_ADE"),
        ("traj/FDE_mean", "stream_gt/root_FDE"),
        ("path/arc_ADE_mean", "stream_gt/path_arc_ADE"),
        ("traj/jitter_mean", "stream_gt/jitter"),
        ("stream_gt/yaw_error_mean", "stream_gt/yaw_error"),
        ("control/Control_L2_dist_mean", "stream_gt/control_L2"),
        ("control/Skating_Ratio_mean", "stream_gt/foot_skating"),
        ("stream_boundary/root_jump_mean", "stream_gt/chunk_boundary_root_jump"),
        ("stream_vs_offline/root_ade_mean", "stream_vs_offline/root_ADE"),
        ("stream_vs_offline/feature_l2_mean", "stream_vs_offline/feature_L2"),
        ("stream_no_traj/root_ADE_mean", "stream_no_traj/root_ADE"),
        ("stream_no_traj/root_FDE_mean", "stream_no_traj/root_FDE"),
        ("stream_no_traj/path_arc_ADE_mean", "stream_no_traj/path_arc_ADE"),
        ("control_gain/root_ADE_delta_mean", "control_gain/root_ADE_delta"),
        ("control_gain/root_FDE_delta_mean", "control_gain/root_FDE_delta"),
    ):
        _copy_alias(src, dst)
    return summary
