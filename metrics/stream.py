from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

try:
    from FloodNet.eval.inline_eval_summary import build_inline_eval_summary
    from FloodNet.utils.motion_process import (
        extract_root_trajectory_263_torch,
        recover_joint_positions_263,
    )
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from eval.inline_eval_summary import build_inline_eval_summary
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


def summarize_stream_records(records: Sequence[Dict]) -> Dict:
    if not records:
        return {}
    summary = build_inline_eval_summary(records)

    def _append_stats(record_key: str, summary_prefix: str):
        vals = [record[record_key] for record in records if record_key in record and record[record_key] == record[record_key]]
        if vals:
            summary[f"{summary_prefix}_mean"] = float(np.mean(vals))
            summary[f"{summary_prefix}_std"] = float(np.std(vals))

    _append_stats("stream_root_jump_mean", "stream_boundary/root_jump")
    _append_stats("stream_root_jump_max", "stream_boundary/root_jump_max")
    _append_stats("stream_joint_jump_mean", "stream_boundary/joint_jump")
    _append_stats("stream_num_boundaries", "stream_boundary/n_boundaries")
    _append_stats("stream_offline_feature_l2_mean", "stream_vs_offline/feature_l2")
    _append_stats("stream_offline_feature_l2_max", "stream_vs_offline/feature_l2_max")
    _append_stats("stream_offline_root_ade", "stream_vs_offline/root_ade")
    _append_stats("stream_offline_length_delta", "stream_vs_offline/length_delta")
    return summary
