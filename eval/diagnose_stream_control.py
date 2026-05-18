"""Single-sample stream control diagnostic matrix.

Runs 6 modes on a fixed sample to isolate which dimension (inference path,
horizon, pred-root closed loop) causes trajectory control degradation.

Usage::

    cd FloodNet
    PYTHONPATH=. python eval/diagnose_stream_control.py \\
        --config configs/stream.yaml \\
        --ckpt /path/to/checkpoint.ckpt \\
        --vae_ckpt /path/to/vae.ckpt \\
        --sample_id 000021 \\
        --out_dir outputs/diagnose_stream/000021
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from any directory (e.g. ``python eval/diagnose_stream_control.py``).
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
from lightning import seed_everything
from torch_ema import ExponentialMovingAverage

from omegaconf import OmegaConf

from utils.initialize import (
    check_state_dict,
    instantiate,
    load_config,
)
from utils.motion_process import (
    StreamJointRecovery263,
    extract_root_trajectory_263,
)
from utils.stream_rollout import (
    StreamTextSegment,
    StreamTextRolloutController,
    build_stream_step_model_input,
    build_stream_suffix_conditioning,
    clip_traj_input_to_horizon,
)
from utils.stream_traj import (
    build_remaining_polyline,
    build_recovery_future_traj,
    assign_times_by_arclength,
    estimate_token_step_distance,
    project_point_to_polyline,
    resample_polyline,
    sample_timestamped_trajectory,
)
from utils.traj_batch import root_to_traj_feats
from utils.visualize import render_single_video

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# helpers: model loading
# ---------------------------------------------------------------------------

def _load_vae(cfg, device):
    vae = instantiate(
        target=cfg.test_vae.target, cfg=None, hfstyle=False, **cfg.test_vae.params
    )
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(
            vae.parameters(), decay=cfg.test_vae.ema_decay
        )
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
        print(f"Loaded VAE from {cfg.test_vae_ckpt} with EMA")
    else:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        print(f"Loaded VAE from {cfg.test_vae_ckpt} w/o EMA")
    check_state_dict(
        state_dict=vae.state_dict(),
        named_parameters=vae.named_parameters(),
        named_buffers=vae.named_buffers(),
    )
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _load_model(cfg, ckpt_path, device):
    model = instantiate(
        target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(checkpoint["state_dict"].keys())
    controlnet_missing = not any(k.startswith("controlnet.") for k in ckpt_keys)
    strict = not controlnet_missing
    result = model.load_state_dict(checkpoint["state_dict"], strict=strict)
    if not strict and result.missing_keys:
        print(
            f"Loaded LDF with strict=False (no ControlNet in ckpt). "
            f"Missing keys: {result.missing_keys}"
        )
        if any("controlnet." in k for k in result.missing_keys):
            model.controlnet.init_from_backbone(model.model)
            print("Re-initialized ControlNet from backbone weights")
    if "ema_state" in checkpoint:
        n_shadow = len(checkpoint["ema_state"]["shadow_params"])
        ema_params = [p for p in model.parameters() if p.requires_grad]
        if len(ema_params) != n_shadow:
            ema_params = list(model.parameters())
        assert len(ema_params) == n_shadow, (
            f"EMA shadow count mismatch: {len(ema_params)} vs {n_shadow}"
        )
        ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(ema_params)
        print(f"Loaded model from {ckpt_path} with EMA ({n_shadow} params)")
    else:
        print(f"Loaded model from {ckpt_path} w/o EMA")
    check_state_dict(
        state_dict=model.state_dict(),
        named_parameters=model.named_parameters(),
        named_buffers=model.named_buffers(),
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# helpers: sample loading
# ---------------------------------------------------------------------------

def _load_sample(raw_data_dir: str, sample_id: str, dataset: str = "humanml3d"):
    """Load a single sample, returning a flat batch dict.

    Args:
        raw_data_dir: root raw_data directory.
        sample_id: sample name without extension.
        dataset: ``"humanml3d"`` or ``"babel"``.
    """
    if dataset == "babel":
        return _load_babel_sample(raw_data_dir, sample_id)
    return _load_humanml3d_sample(raw_data_dir, sample_id)


def _load_humanml3d_sample(raw_data_dir: str, sample_id: str):
    """Load a single HumanML3D sample."""
    data_dir = os.path.join(raw_data_dir, "HumanML3D")

    feat_path = os.path.join(data_dir, "new_joint_vecs", f"{sample_id}.npy")
    feature = np.load(feat_path).astype(np.float32)
    feature_length = feature.shape[0]

    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            caption = parts[0]
            tokens = parts[1].split(" ") if len(parts) > 1 else []
            f_tag = float(parts[2]) if len(parts) > 2 else 0.0
            to_tag = float(parts[3]) if len(parts) > 3 else 0.0
            text_data.append(
                {"caption": caption, "tokens": tokens, "f_tag": f_tag, "to_tag": to_tag}
            )

    traj_xyz = extract_root_trajectory_263(feature)
    traj_features = root_to_traj_feats(traj_xyz)

    token_dir = os.path.join(data_dir, "TOKENS_20251030_085836_vae_wan_z4")
    token_path = os.path.join(token_dir, f"{sample_id}.npy")
    if os.path.exists(token_path):
        token = np.load(token_path).astype(np.float32)
        token_length = token.shape[0]
    else:
        downsample_factor = 4
        token_length = (feature_length + downsample_factor - 1) // downsample_factor + 1
        token = np.zeros((token_length, 4), dtype=np.float32)

    text_dict = text_data[0]
    text = text_dict["caption"]

    sample = {
        "name": sample_id,
        "dataset": "HumanML3D",
        "feature": torch.from_numpy(feature).float(),
        "feature_length": feature_length,
        "token": torch.from_numpy(token).float(),
        "token_length": token_length,
        "text": text,
        "text_all": [td["caption"] for td in text_data],
        "text_data": text_data,
        "text_tokens": text_dict["tokens"],
        "traj": torch.from_numpy(traj_xyz).float(),
        "traj_length": len(traj_xyz),
        "traj_features": torch.from_numpy(traj_features).float(),
        "token_text_end": [token_length],
        "feature_text_end": [feature_length],
        "token_mask": torch.ones(token_length, dtype=torch.float32),
        "traj_mask": torch.ones(len(traj_xyz), dtype=torch.float32),
    }
    return sample


def _load_babel_sample(raw_data_dir: str, sample_id: str):
    """Load a single BABEL sample with multi-segment text."""
    data_dir = os.path.join(raw_data_dir, "BABEL_streamed")

    feat_path = os.path.join(data_dir, "motions", f"{sample_id}.npy")
    feature = np.load(feat_path).astype(np.float32)
    feature_length = feature.shape[0]

    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            caption = parts[0].strip()
            tokens = parts[1].split(" ") if len(parts) > 1 else []
            f_tag = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
            to_tag = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0.0
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            text_data.append(
                {"caption": caption, "tokens": tokens, "f_tag": f_tag, "to_tag": to_tag}
            )

    traj_xyz = extract_root_trajectory_263(feature)
    traj_features = root_to_traj_feats(traj_xyz)

    token_dir = os.path.join(data_dir, "TOKENS_20251030_085836_vae_wan_z4")
    token_path = os.path.join(token_dir, f"{sample_id}.npy")
    if os.path.exists(token_path):
        token = np.load(token_path).astype(np.float32)
        token_length = token.shape[0]
    else:
        downsample_factor = 4
        token_length = (feature_length + downsample_factor - 1) // downsample_factor + 1
        token = np.zeros((token_length, 4), dtype=np.float32)

    # Build multi-segment text with feature_text_end / token_text_end.
    texts: list[str] = []
    feature_text_end: list[int] = []
    cursor = 0
    for td in text_data:
        f_tag, to_tag = td["f_tag"], td["to_tag"]
        if f_tag == 0.0 and to_tag == 0.0:
            # A full-duration caption — place after any existing gaps.
            if cursor < feature_length:
                texts.append(td["caption"])
                feature_text_end.append(feature_length)
            continue
        abs_start = max(0, int(f_tag * 20.0 + 0.5))
        abs_end = int(to_tag * 20.0 + 0.5) if to_tag > 0 else feature_length
        if abs_end <= abs_start:
            continue
        if abs_start > cursor:
            texts.append("")
            feature_text_end.append(abs_start)
            cursor = abs_start
        if abs_end <= cursor:
            continue
        texts.append(td["caption"])
        feature_text_end.append(abs_end)
        cursor = abs_end
    if cursor < feature_length:
        texts.append("")
        feature_text_end.append(feature_length)
    if not texts:
        texts = [td["caption"] for td in text_data] or [""]
        feature_text_end = [feature_length]

    # Convert frame ends → token ends (causal VAE convention).
    token_text_end: list[int] = []
    for ef in feature_text_end:
        last_frame = ef - 1
        tok_end = (last_frame + 3) // 4 + 1
        token_text_end.append(max(0, min(token_length, tok_end)))

    sample = {
        "name": sample_id,
        "dataset": "BABEL_streamed",
        "feature": torch.from_numpy(feature).float(),
        "feature_length": feature_length,
        "token": torch.from_numpy(token).float(),
        "token_length": token_length,
        "text": texts,
        "text_all": [td["caption"] for td in text_data],
        "text_data": text_data,
        "text_tokens": text_data[0]["tokens"] if text_data else [],
        "traj": torch.from_numpy(traj_xyz).float(),
        "traj_length": len(traj_xyz),
        "traj_features": torch.from_numpy(traj_features).float(),
        "token_text_end": token_text_end,
        "feature_text_end": feature_text_end,
        "token_mask": torch.ones(token_length, dtype=torch.float32),
        "traj_mask": torch.ones(len(traj_xyz), dtype=torch.float32),
    }
    return sample


# ---------------------------------------------------------------------------
# BABEL multi-segment merge (long-session eval)
# ---------------------------------------------------------------------------

def _merge_babel_samples(raw_data_dir: str, sample_ids_str: str) -> dict:
    """Load and merge multiple BABEL present segments into one long session.

    Returns a flat sample dict with concatenated features, merged text schedule,
    and segment-boundary-smoothed trajectory.
    """
    sample_ids = [s.strip() for s in sample_ids_str.split(",")]
    data_dir = os.path.join(raw_data_dir, "BABEL_streamed")

    features, tokens_list = [], []
    feat_offset, token_offset = 0, 0
    feat_fps = 20.0

    # Merge text_data with cumulative time offsets.
    text_data = []
    boundary_frames = []  # frame indices of segment boundaries
    sample_spans = []
    for i, sid in enumerate(sample_ids):
        feat = np.load(os.path.join(data_dir, "motions", f"{sid}.npy")).astype(np.float32)
        tok = np.load(os.path.join(data_dir, "TOKENS_20251030_085836_vae_wan_z4",
                                   f"{sid}.npy")).astype(np.float32)
        feat_start = feat_offset
        token_start = token_offset
        features.append(feat)
        tokens_list.append(tok)

        txt_path = os.path.join(data_dir, "texts", f"{sid}.txt")
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("#")
                caption = parts[0].strip()
                tokens = parts[1].split(" ") if len(parts) > 1 else []
                f_tag = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
                to_tag = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0.0
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                if f_tag == 0.0 and to_tag == 0.0:
                    abs_f = feat_offset / feat_fps
                    abs_t = (feat_offset + len(feat)) / feat_fps
                else:
                    abs_f = feat_offset / feat_fps + f_tag
                    abs_t = feat_offset / feat_fps + to_tag
                text_data.append({
                    "caption": caption, "tokens": tokens,
                    "f_tag": abs_f, "to_tag": abs_t,
                })

        if i > 0:
            boundary_frames.append(feat_offset)
        feat_offset += len(feat)
        token_offset += len(tok)
        sample_spans.append({
            "sample_id": sid,
            "frame_start": int(feat_start),
            "frame_end": int(feat_offset),
            "token_start": int(token_start),
            "token_end": int(token_offset),
        })

    merged_feat = np.concatenate(features, axis=0)
    merged_token = np.concatenate(tokens_list, axis=0)
    total_frames = len(merged_feat)
    total_tokens = len(merged_token)

    # Build feature_text_end / token_text_end.
    texts: list[str] = []
    feature_text_end: list[int] = []
    cursor = 0
    for td in text_data:
        abs_start = max(0, int(td["f_tag"] * feat_fps + 0.5))
        abs_end = int(td["to_tag"] * feat_fps + 0.5) if td["to_tag"] > 0 else total_frames
        if abs_end <= abs_start:
            continue
        if abs_start > cursor:
            texts.append("")
            feature_text_end.append(min(abs_start, total_frames))
            cursor = abs_start
        texts.append(td["caption"])
        feature_text_end.append(min(abs_end, total_frames))
        cursor = abs_end
    if cursor < total_frames:
        texts.append("")
        feature_text_end.append(total_frames)
    if not texts:
        texts = [td["caption"] or "" for td in text_data] or [""]
        feature_text_end = [total_frames]

    token_text_end = []
    for ef in feature_text_end:
        tok_end = max(0, min(total_tokens, (ef - 1 + 3) // 4 + 1))
        token_text_end.append(tok_end)

    # Extract root, apply boundary smoothing.
    traj_xyz = extract_root_trajectory_263(merged_feat)
    for bf in boundary_frames:
        if 0 < bf < len(traj_xyz) - 1:
            _pre = traj_xyz[max(0, bf - 1)]
            _post = traj_xyz[min(len(traj_xyz) - 1, bf)]
            # Simple linear mid-point: averages the 2 frames around the boundary.
            # More sophisticated smoothing can use scipy.ndimage.gaussian_filter1d.
            traj_xyz[bf] = (_pre + _post) / 2.0

    traj_features = root_to_traj_feats(traj_xyz)

    sample = {
        "name": sample_ids[0].rsplit("_", 1)[0],
        "dataset": "BABEL_streamed",
        "feature": torch.from_numpy(merged_feat).float(),
        "feature_length": total_frames,
        "token": torch.from_numpy(merged_token).float(),
        "token_length": total_tokens,
        "text": texts,
        "text_data": text_data,
        "traj": torch.from_numpy(traj_xyz).float(),
        "traj_length": len(traj_xyz),
        "traj_features": torch.from_numpy(traj_features).float(),
        "token_text_end": token_text_end,
        "feature_text_end": feature_text_end,
        "token_mask": torch.ones(total_tokens, dtype=torch.float32),
        "traj_mask": torch.ones(len(traj_xyz), dtype=torch.float32),
        "_sample_ids": sample_ids,
        "_boundary_frames": boundary_frames,
        "_sample_spans": sample_spans,
    }
    return sample


def _build_timestamped_traj_input(sample: dict, commit_idx: int,
                                  horizon_tokens: int, token_dt: float,
                                  *, times=None, waypoints=None,
                                  query_time_offset: float = 0.0):
    """Sample future H tokens from a timestamped trajectory plan."""
    token_length = sample["token_length"]
    if commit_idx >= token_length:
        return None
    if times is None:
        gt_root = sample["traj"].numpy()
        total_frames = len(gt_root)
        times = np.arange(total_frames, dtype=np.float32) / 20.0
        waypoints = gt_root
    n = min(horizon_tokens, token_length - commit_idx)
    query_times = (
        np.arange(n, dtype=np.float32) * token_dt
        + commit_idx * token_dt
        - float(query_time_offset)
    )
    future_traj = sample_timestamped_trajectory(times, waypoints, query_times)
    return {
        "traj": torch.from_numpy(future_traj).float().unsqueeze(0),
        "token_mask": torch.ones(1, n),
    }


def _build_babel_segment_waypoint_plans(sample: dict, waypoint_dt: float,
                                        motion_fps: float):
    """Build one timestamped waypoint plan per original BABEL segment.

    This mirrors web-demo usage where each present segment can be treated as a
    separate trajectory update.  Time therefore resets inside each source
    sample instead of running over the merged global path.
    """
    gt_root = sample["traj"].numpy()
    stride_frames = max(1, int(float(waypoint_dt) * float(motion_fps)))
    plans = []
    for span in sample.get("_sample_spans", []):
        fs = int(span["frame_start"])
        fe = int(span["frame_end"])
        local_root = gt_root[fs:fe]
        if len(local_root) == 0:
            continue
        local_idx = np.arange(0, len(local_root), stride_frames)
        if len(local_idx) == 0 or local_idx[-1] != len(local_root) - 1:
            local_idx = np.concatenate(
                [local_idx, np.array([len(local_root) - 1], dtype=local_idx.dtype)]
            )
        waypoints = local_root[local_idx].astype(np.float32)
        times = np.arange(len(waypoints), dtype=np.float32) * float(waypoint_dt)
        plans.append({
            "sample_id": span["sample_id"],
            "token_start": int(span["token_start"]),
            "token_end": int(span["token_end"]),
            "frame_start": fs,
            "frame_end": fe,
            "times": times,
            "waypoints": waypoints,
        })
    return plans


def _select_babel_segment_plan(plans: list[dict], commit_idx: int):
    for plan in plans:
        if plan["token_start"] <= commit_idx < plan["token_end"]:
            return plan
    if not plans:
        return None
    return plans[-1]


def _wrap_flat_sample_for_suffix(sample: dict) -> dict:
    """Wrap flat-sample scalar fields into batch-style dict for \
    ``build_stream_suffix_conditioning``."""
    out = dict(sample)
    for key in ("token_length", "traj_length", "feature_length"):
        v = out.get(key)
        if v is not None and not hasattr(v, "shape"):
            out[key] = torch.tensor([v])
    for key in ("traj", "traj_features", "token_mask", "traj_mask"):
        v = out.get(key)
        if v is not None and torch.is_tensor(v) and v.ndim in (1, 2):
            out[key] = v.unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# helpers: metrics
# ---------------------------------------------------------------------------

def _compute_ade(pred_root: np.ndarray, gt_root: np.ndarray) -> float:
    n = min(len(pred_root), len(gt_root))
    if n == 0:
        return float("nan")
    diff = pred_root[:n, [0, 2]] - gt_root[:n, [0, 2]]
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _compute_fde(pred_root: np.ndarray, gt_root: np.ndarray) -> float:
    n = min(len(pred_root), len(gt_root))
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(pred_root[n - 1, [0, 2]] - gt_root[n - 1, [0, 2]]))


def _path_arc_ade(pred_root: np.ndarray, gt_root: np.ndarray,
                  num_samples: int = 100) -> float:
    """Arc-length reparameterized ADE between two paths."""
    n = min(len(pred_root), len(gt_root))
    if n < 2:
        return _compute_ade(pred_root, gt_root)
    xz_p = pred_root[:n, [0, 2]] if pred_root.shape[1] >= 3 else pred_root[:n]
    xz_g = gt_root[:n, [0, 2]] if gt_root.shape[1] >= 3 else gt_root[:n]
    def _resample(path, ns):
        segs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([np.zeros(1), np.cumsum(segs)])
        total = cum[-1]
        if total < 1e-8:
            return np.tile(path[:1], (ns, 1))
        q = np.linspace(0, total, ns)
        return np.column_stack([np.interp(q, cum, path[:, d]) for d in range(path.shape[1])])
    rp = _resample(xz_p, num_samples)
    rg = _resample(xz_g, num_samples)
    return float(np.mean(np.linalg.norm(rp - rg, axis=1)))


def _path_chamfer(pred_root: np.ndarray, gt_root: np.ndarray) -> float:
    """One-sided Chamfer distance: mean min-dist from pred to gt path."""
    n = min(len(pred_root), len(gt_root))
    if n < 2:
        return _compute_ade(pred_root, gt_root)
    xz_p = pred_root[:n, [0, 2]] if pred_root.shape[1] >= 3 else pred_root[:n]
    xz_g = gt_root[:n, [0, 2]] if gt_root.shape[1] >= 3 else gt_root[:n]
    from scipy.spatial import cKDTree
    tree = cKDTree(xz_g)
    dists, _ = tree.query(xz_p)
    return float(np.mean(dists))


def _compute_root_path_length(root: np.ndarray) -> float:
    if len(root) < 2:
        return 0.0
    if root.shape[1] == 2:
        return float(np.sum(np.linalg.norm(np.diff(root, axis=0), axis=1)))
    return float(np.sum(np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1)))


def _compute_oracle_token_step(sample: dict) -> float:
    """Estimate token-step distance from GT token-level root motion.

    This is diagnostic-only.  If oracle step improves pred-root behaviour, the
    web path's online velocity estimate is part of the issue.
    """
    traj = sample["traj"].numpy().astype(np.float32)
    token_length = int(sample["token_length"])
    if token_length <= 1 or len(traj) <= 1:
        return 0.25
    frame_indices = [
        0,
        *[min(4 * token_idx, len(traj) - 1) for token_idx in range(1, token_length)],
    ]
    token_root = traj[frame_indices, :]
    steps = np.linalg.norm(np.diff(token_root[:, [0, 2]], axis=0), axis=1)
    steps = steps[np.isfinite(steps) & (steps > 1e-6)]
    if steps.size == 0:
        return 0.25
    return float(np.clip(np.median(steps), 0.05, 1.50))


def _make_gt_timestamped_traj(sample: dict, motion_fps: float = 20.0):
    """Build a timestamped trajectory from dataset GT root frames."""
    traj = sample["traj"].numpy().astype(np.float32)
    times = np.arange(len(traj), dtype=np.float32) / float(motion_fps)
    return times, traj


def _make_duration_waypoint_traj(
    sample: dict,
    *,
    motion_fps: float = 20.0,
    waypoint_stride_seconds: float = 1.0,
):
    """Approximate GT with spatial waypoints plus only total duration.

    Waypoints are sampled from the GT root at a coarse fixed time stride, then
    timestamps are reassigned by arclength over the total clip duration.  This
    emulates a user providing spatial waypoints and total duration, without
    manually timestamping each point.
    """
    gt_times, gt_traj = _make_gt_timestamped_traj(sample, motion_fps=motion_fps)
    if len(gt_traj) == 0:
        return gt_times, gt_traj, gt_traj

    total_duration = float(gt_times[-1]) if len(gt_times) > 1 else 0.0
    stride_frames = max(1, int(round(float(waypoint_stride_seconds) * motion_fps)))
    idx = list(range(0, len(gt_traj), stride_frames))
    if idx[-1] != len(gt_traj) - 1:
        idx.append(len(gt_traj) - 1)
    waypoints = gt_traj[idx].astype(np.float32)
    times = assign_times_by_arclength(waypoints, total_duration)
    return times, waypoints, waypoints


# ---------------------------------------------------------------------------
# generation modes
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_generate_full(model, vae, sample, device):
    """Offline generate() baseline."""
    model_batch = _make_model_batch(sample, device)
    output = model.generate(model_batch, num_denoise_steps=10)
    generated = output["generated"][0]
    decoded = vae.decode(generated[None, :].to(device))[0].float().cpu().numpy()
    pred_root = extract_root_trajectory_263(decoded)
    return decoded, pred_root


@torch.no_grad()
def run_stream_generate_full(model, vae, sample, device):
    """Offline stream_generate() sanity check — full sequence."""
    model_batch = _make_model_batch(sample, device)
    vae.clear_cache()
    all_chunks = []
    for output in model.stream_generate(model_batch, num_denoise_steps=10):
        generated = output["generated"]
        if generated[0] is not None:
            decoded = (
                vae.stream_decode(
                    generated[0][None, :].to(device),
                    first_chunk=(len(all_chunks) == 0),
                )[0]
                .float()
                .cpu()
                .numpy()
            )
            all_chunks.append(decoded)
    vae.clear_cache()
    decoded_full = (
        np.concatenate(all_chunks, axis=0) if all_chunks else np.zeros((0, 263))
    )
    pred_root = extract_root_trajectory_263(decoded_full)
    return decoded_full, pred_root


@torch.no_grad()
def run_stream_step(
    model,
    vae,
    sample,
    device,
    *,
    history_length: int,
    num_denoise_steps: int,
    horizon_tokens: int | None,
    use_pred_root: bool,
    use_features_path: bool = False,
    no_traj: bool = False,
    collect_latents: bool = False,
    predroot_mode: str = "current_polyline",
    recovery_tokens: int = 6,
    oracle_token_step: float | None = None,
    timestamped_traj: tuple[np.ndarray, np.ndarray] | None = None,
    token_dt: float = 0.2,
    trace_sink: list | None = None,
):
    """Run stream_generate_step() with configurable horizon and root source.

    When *use_pred_root* is True, replicates web_demo closed-loop behaviour.
    When *use_features_path* is True, uses ``traj_features`` (bypasses
    xyz→anchor-subtract→LocalTrajEncoder path).
    When *no_traj* is True, skips trajectory conditioning entirely.
    When *collect_latents* is True, returns latent tokens instead of decoded
    motion (for offline-decode comparison).
    *predroot_mode* selects how the closed-loop future trajectory is rebuilt
    from the predicted root.  ``current_polyline`` mirrors web_demo; ``recovery``
    preserves path progress while blending lateral error back to the path;
    ``timestamped`` samples a time-parameterized trajectory at token times.
    """
    token_length = sample["token_length"]
    total_frames = 1 + 4 * (token_length - 1) if token_length > 1 else 1

    # Build text controller: handle both single-text (HumanML3D) and
    # multi-segment (BABEL) formats.
    if isinstance(sample["text"], list):
        segments = [
            StreamTextSegment(text=t, token_end=te)
            for t, te in zip(sample["text"], sample["token_text_end"])
        ]
    else:
        segments = [
            StreamTextSegment(text=sample["text"], token_end=token_length)
        ]
    text_ctrl = StreamTextRolloutController(segments)

    # build_stream_suffix_conditioning expects batch-style fields (tensor
    # with batch dim).  Wrap the flat sample once.
    _batch_sample = _wrap_flat_sample_for_suffix(sample)

    # Match web_demo root semantics: stream output frames are incremental 263D
    # features, so root position must be accumulated across decoded chunks.
    # Calling extract_root_trajectory_263() per chunk would reset the root at
    # every token and corrupt step-path ADE/FDE.
    stream_recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    if use_pred_root:
        gt_polyline = sample["traj"].numpy()  # (T, 3) world-space GT polyline

    vae.clear_cache()
    model.init_generated(
        history_length, batch_size=1, num_denoise_steps=num_denoise_steps
    )

    all_decoded = []
    all_pred_root = []
    first_chunk = True

    # Pred-root closed-loop state (matches web_demo ModelManager).
    root_xz_history: list[np.ndarray] = []

    for commit_idx in range(token_length):
        current_text = text_ctrl.get_text_for_commit_index(commit_idx)

        if use_pred_root:
            # Replicate web_demo's _build_stream_traj_input pipeline:
            #   1. current_root = stream_recovery.r_pos_accum
            #   2. project onto GT polyline → build_remaining_polyline
            #   3. estimate token_step from predicted velocity history
            #   4. resample H future tokens at that step
            # H shrinks with remaining tokens so that step_full_xyz_predroot
            # and step_full_xyz_gtroot differ ONLY in root source, not horizon.
            if horizon_tokens is not None:
                h = horizon_tokens
            else:
                h = max(1, token_length - commit_idx)
            current_root = np.zeros(3, dtype=np.float32)
            current_root[[0, 2]] = stream_recovery.r_pos_accum[[0, 2]].astype(np.float32)
            token_step = (
                float(oracle_token_step)
                if oracle_token_step is not None
                else estimate_token_step_distance(root_xz_history)
            )
            if predroot_mode == "current_polyline":
                polyline = build_remaining_polyline(current_root, gt_polyline)
                future_traj = resample_polyline(polyline, h, token_step)
            elif predroot_mode == "recovery":
                future_traj = build_recovery_future_traj(
                    current_root,
                    gt_polyline,
                    h,
                    token_step,
                    recovery_tokens=recovery_tokens,
                )
            elif predroot_mode == "timestamped":
                if timestamped_traj is None:
                    raise ValueError("timestamped_traj is required for timestamped mode")
                traj_times, traj_xyz = timestamped_traj
                query_times = (
                    float(commit_idx) * float(token_dt)
                    + np.arange(h, dtype=np.float32) * float(token_dt)
                )
                future_traj = sample_timestamped_trajectory(
                    traj_times, traj_xyz, query_times
                )
            else:
                raise ValueError(f"Unknown predroot_mode: {predroot_mode}")
            if trace_sink is not None:
                projected, seg_idx, seg_t = project_point_to_polyline(
                    current_root, gt_polyline
                )
                if len(future_traj) > 1:
                    _future_steps = np.linalg.norm(
                        np.diff(future_traj[:, [0, 2]], axis=0), axis=1
                    )
                    _future_token_step = float(np.median(_future_steps))
                else:
                    _future_token_step = 0.0
                trace_sink.append(
                    {
                        "commit_index": int(commit_idx),
                        "mode": predroot_mode,
                        "token_step": float(token_step),
                        "future_token_step": _future_token_step,
                        "recovery_tokens": int(recovery_tokens),
                        "current_root": current_root.tolist(),
                        "projected_root": projected.astype(np.float32).tolist(),
                        "query_time": float(commit_idx) * float(token_dt),
                        "segment_index": int(seg_idx),
                        "segment_t": float(seg_t),
                        "cross_track_error": float(
                            np.linalg.norm(
                                current_root[[0, 2]] - projected[[0, 2]]
                            )
                        ),
                        "future_first": future_traj[0].tolist()
                        if len(future_traj)
                        else None,
                        "future_last": future_traj[-1].tolist()
                        if len(future_traj)
                        else None,
                    }
                )
            traj_input = {
                "traj": torch.from_numpy(future_traj).float().unsqueeze(0),
                "token_mask": torch.ones(1, future_traj.shape[0]),
            }
        elif no_traj:
            traj_input = None
        else:
            # GT-root path: use suffix from dataset.
            _prefer_xyz = not use_features_path
            traj_input = build_stream_suffix_conditioning(
                _batch_sample, commit_idx, prefer_xyz=_prefer_xyz
            )
            if horizon_tokens is not None:
                traj_input = clip_traj_input_to_horizon(traj_input, horizon_tokens)

        step_payload = build_stream_step_model_input(current_text, traj_input=traj_input)

        output = model.stream_generate_step(step_payload, first_chunk=first_chunk)
        generated = output["generated"]
        latent_token = generated[0].float().cpu().numpy()  # (C_latent,) or (,C_latent)

        if collect_latents:
            all_decoded.append(latent_token)
        else:
            decoded = (
                vae.stream_decode(
                    generated[0][None, :].to(device), first_chunk=first_chunk
                )[0]
                .float()
                .cpu()
                .numpy()
            )
            all_decoded.append(decoded)

            pred_root_chunk = []
            for frame in decoded:
                stream_recovery.process_frame(frame)
                root_pos = stream_recovery.r_pos_accum.astype(np.float32).copy()
                pred_root_chunk.append(root_pos)
                if use_pred_root:
                    root_xz_history.append(root_pos[[0, 2]].copy())
            if pred_root_chunk:
                all_pred_root.append(np.stack(pred_root_chunk, axis=0))

        first_chunk = False

    vae.clear_cache()

    if collect_latents:
        latent_full = np.concatenate(all_decoded, axis=0)[:token_length]
        return latent_full, None  # caller decodes offline

    decoded_full = (
        np.concatenate(all_decoded, axis=0)[:total_frames]
        if all_decoded
        else np.zeros((0, 263))
    )
    pred_root_full = (
        np.concatenate(all_pred_root, axis=0)[:total_frames]
        if all_pred_root
        else np.zeros((0, 3))
    )
    return decoded_full, pred_root_full


def _make_model_batch(sample, device):
    mb = {
        "feature": sample["token"].unsqueeze(0).to(device),
        "feature_length": torch.tensor([sample["token_length"]], device=device),
        "text": [sample["text"]],  # str → [str]; List[str] → [List[str]]
    }
    # Multi-segment text fields (used by BABEL).
    if isinstance(sample["text"], list):
        mb["feature_text_end"] = [sample["feature_text_end"]]
        mb["token_text_end"] = [sample["token_text_end"]]
    if sample.get("traj") is not None:
        mb["traj"] = sample["traj"].unsqueeze(0).to(device)
        mb["traj_features"] = sample["traj_features"].unsqueeze(0).to(device)
        mb["traj_length"] = torch.tensor([sample["traj_length"]])
        mb["token_mask"] = sample["token_mask"].unsqueeze(0).to(device)
        mb["traj_mask"] = sample["traj_mask"].unsqueeze(0).to(device)
    return mb


# ---------------------------------------------------------------------------
# artifact saving
# ---------------------------------------------------------------------------

def _save_artifacts(out_dir, pred_motion, pred_root, gt_root, target_traj, metrics):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pred_motion.npy"), pred_motion)
    np.save(os.path.join(out_dir, "pred_root.npy"), pred_root)
    np.save(os.path.join(out_dir, "target_root.npy"), gt_root)
    np.save(os.path.join(out_dir, "target_traj.npy"), target_traj)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def _build_mode_metrics(pred_root, gt_root, target_traj, mode_name, horizon, root_source,
                        traj_encoder_path=""):
    n = min(len(pred_root), len(gt_root))
    return {
        "mode": mode_name,
        "horizon": "full" if horizon is None else int(horizon),
        "root_source": root_source,
        "ADE": _compute_ade(pred_root[:n], gt_root[:n]),
        "FDE": _compute_fde(pred_root[:n], gt_root[:n]),
        "pred_root_path_length": _compute_root_path_length(pred_root[:n]),
        "gt_root_path_length": _compute_root_path_length(gt_root[:n]),
        "target_traj_path_length": _compute_root_path_length(target_traj[:n]),
        "pred_frames": int(len(pred_root)),
        "gt_frames": int(len(gt_root)),
        "traj_encoder_path": traj_encoder_path,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _run_babel_long_session(args, model, vae, device, out_root):
    """BABEL multi-sample long-session eval with timestamped trajectory modes."""
    _babel_modes = (
        ["gt_suffix", "timestamped_gt_plan", "duration_waypoints", "no_traj"]
        if args.babel_mode == "all" else [args.babel_mode]
    )

    print(f"\nMerging BABEL samples: {args.sample_ids}")
    sample = _merge_babel_samples(args.raw_data_dir, args.sample_ids)
    print(f"  frames={sample['feature_length']}, tokens={sample['token_length']}, "
          f"texts={len(sample['text'])}")
    for i, t in enumerate(sample["text"]):
        print(f"    [{i}] '{t[:60]}' -> frame {sample['feature_text_end'][i]}")
    if sample.get("_boundary_frames"):
        print(f"  boundaries smoothed: {sample['_boundary_frames']}")

    gt_root = extract_root_trajectory_263(sample["feature"].numpy())
    target_traj = sample["traj"].numpy()
    all_records = []

    for mode in _babel_modes:
        print(f"\n--- {mode} ---")
        tl = sample["token_length"]
        total_frames = 1 + 4 * (tl - 1) if tl > 1 else 1
        segments = [StreamTextSegment(text=t, token_end=te)
                    for t, te in zip(sample["text"], sample["token_text_end"])]
        text_ctrl = StreamTextRolloutController(segments)

        vae.clear_cache()
        model.init_generated(args.history_length, batch_size=1,
                            num_denoise_steps=args.num_denoise_steps)
        model.generated = model.generated.to(device)
        stream_recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
        all_dec, all_pr = [], []
        first_chunk = True

        # BABEL long-session mode historically used --waypoint_dt, while the
        # single-sample timestamped ablation used --duration_waypoint_stride_seconds.
        # Accept both so old commands do not silently keep the wrong stride.
        babel_waypoint_dt = (
            args.duration_waypoint_stride_seconds
            if args.duration_waypoint_stride_seconds != 1.0
            else args.waypoint_dt
        )
        segment_waypoint_plans = _build_babel_segment_waypoint_plans(
            sample, babel_waypoint_dt, args.motion_fps
        )

        for ci in range(tl):
            cur_text = text_ctrl.get_text_for_commit_index(ci)
            if mode == "no_traj":
                ti = None
            elif mode == "gt_suffix":
                _bs = _wrap_flat_sample_for_suffix(sample)
                ti = build_stream_suffix_conditioning(_bs, ci, prefer_xyz=True)
                if args.traj_horizon_tokens is not None and args.traj_horizon_tokens > 0:
                    ti = clip_traj_input_to_horizon(ti, args.traj_horizon_tokens)
            elif mode == "timestamped_gt_plan":
                ti = _build_timestamped_traj_input(
                    sample, ci, args.traj_horizon_tokens, args.token_dt)
            elif mode == "duration_waypoints":
                plan = _select_babel_segment_plan(segment_waypoint_plans, ci)
                if plan is None:
                    ti = None
                else:
                    # Simulate a trajectory update at each original BABEL
                    # present-segment boundary: local plan time starts at 0.
                    ti = _build_timestamped_traj_input(
                        sample,
                        ci,
                        args.traj_horizon_tokens,
                        args.token_dt,
                        times=plan["times"],
                        waypoints=plan["waypoints"],
                        query_time_offset=plan["token_start"] * args.token_dt,
                    )
            sp = build_stream_step_model_input(cur_text, traj_input=ti)
            out = model.stream_generate_step(sp, first_chunk=first_chunk)
            dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                     first_chunk=first_chunk)[0].float().cpu().numpy())
            first_chunk = False
            pred_root_chunk = []
            for frm in dec:
                stream_recovery.process_frame(frm)
                pred_root_chunk.append(
                    stream_recovery.r_pos_accum.astype(np.float32).copy()
                )
            all_dec.append(dec)
            if pred_root_chunk:
                all_pr.append(np.stack(pred_root_chunk, axis=0))

        vae.clear_cache()
        pred_motion = (np.concatenate(all_dec, axis=0)[:total_frames]
                       if all_dec else np.zeros((0, 263)))
        pred_root = (np.concatenate(all_pr, axis=0)[:total_frames]
                     if all_pr else np.zeros((0, 3)))

        seg_pairs = []
        for i, t in enumerate(sample["text"]):
            sf = sample["feature_text_end"][i - 1] if i > 0 else 0
            ef = sample["feature_text_end"][i]
            seg_pairs.append({"text": t, "start_frame": sf, "end_frame": ef})
        seg_met = []
        for sp in seg_pairs:
            sf2, ef2 = sp["start_frame"], sp["end_frame"]
            ef2 = min(ef2, len(pred_root), len(gt_root))
            if ef2 <= sf2:
                seg_met.append(None)
                continue
            pr, gr = pred_root[sf2:ef2], gt_root[sf2:ef2]
            seg_met.append({"text": sp["text"], "start_frame": sf2, "end_frame": ef2,
                            "ADE": _compute_ade(pr, gr), "FDE": _compute_fde(pr, gr),
                            "pred_path_length": _compute_root_path_length(pr),
                            "gt_path_length": _compute_root_path_length(gr)})

        _mode_label = mode
        if mode == "duration_waypoints":
            _mode_label = f"{mode}_wp{babel_waypoint_dt:.3f}"
        elif mode == "timestamped_gt_plan":
            _mode_label = f"{mode}_tdt{args.token_dt:.3f}"
        metrics = _build_mode_metrics(
            pred_root, gt_root, target_traj, _mode_label, None, "gt",
            traj_encoder_path=f"babel-{mode}")
        metrics["segments"] = seg_met
        metrics["pred_path_length"] = float(_compute_root_path_length(pred_root))
        metrics["gt_path_length"] = float(_compute_root_path_length(gt_root))
        metrics["path_length_ratio"] = (
            metrics["pred_path_length"] / max(metrics["gt_path_length"], 1e-6))
        if mode == "duration_waypoints":
            metrics["duration_waypoint_scope"] = "per_babel_sample"
            metrics["duration_waypoint_dt"] = float(babel_waypoint_dt)
            metrics["duration_waypoint_count"] = int(
                sum(len(plan["waypoints"]) for plan in segment_waypoint_plans)
            )
            metrics["duration_waypoint_plans"] = [
                {
                    "sample_id": plan["sample_id"],
                    "frame_start": int(plan["frame_start"]),
                    "frame_end": int(plan["frame_end"]),
                    "token_start": int(plan["token_start"]),
                    "token_end": int(plan["token_end"]),
                    "num_waypoints": int(len(plan["waypoints"])),
                }
                for plan in segment_waypoint_plans
            ]
        print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}  "
              f"path_ratio={metrics['path_length_ratio']:.4f}")
        for sm in seg_met:
            if sm:
                print(f"    [{sm['text'][:40]}] ADE={sm['ADE']:.4f}")

        mode_dir = os.path.join(out_root, _mode_label)
        _save_artifacts(mode_dir, pred_motion, pred_root,
                        gt_root[:len(pred_root)], target_traj[:len(pred_root)], metrics)
        np.save(os.path.join(mode_dir, "gt_motion.npy"),
                sample["feature"].numpy()[:len(pred_motion)])
        all_records.append(metrics)

        if args.render_video and pred_motion.size > 0:
            _mp4 = os.path.join(mode_dir, "pred_motion.mp4")
            render_single_video(motion=pred_motion, save_path=_mp4,
                                dim=263, render_setting={})
            render_single_video(
                motion=sample["feature"].numpy()[:len(pred_motion)],
                save_path=os.path.join(mode_dir, "gt_motion.mp4"),
                dim=263, render_setting={})
            _n = min(len(pred_root), len(gt_root))
            if _n > 1:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from matplotlib.animation import FFMpegWriter
                _t_mp4 = os.path.join(mode_dir, "traj_compare.mp4")
                _f2, _a2 = plt.subplots(figsize=(7, 7))
                _axa = [gt_root[:_n, 0], pred_root[:_n, 0]]
                _aza = [gt_root[:_n, 2], pred_root[:_n, 2]]
                _xl = (min(a.min() for a in _axa) - 0.5, max(a.max() for a in _axa) + 0.5)
                _zl = (min(a.min() for a in _aza) - 0.5, max(a.max() for a in _aza) + 0.5)
                _wr = FFMpegWriter(fps=20)
                _sf = max(1, _n // 150)
                with _wr.saving(_f2, _t_mp4, dpi=100):
                    for _f in range(1, _n + 1, _sf):
                        _a2.clear()
                        _a2.plot(gt_root[:min(_f, _n), 0], gt_root[:min(_f, _n), 2],
                                 "g-", lw=1.5, alpha=0.7, label="target")
                        _a2.plot(pred_root[:min(_f, _n), 0], pred_root[:min(_f, _n), 2],
                                 "r-", lw=1.5, alpha=0.7, label="pred")
                        _a2.plot(gt_root[0, 0], gt_root[0, 2], "go", ms=6)
                        _a2.plot(pred_root[min(_f - 1, _n - 1), 0],
                                 pred_root[min(_f - 1, _n - 1), 2], "r.", ms=8)
                        for sp2 in seg_pairs:
                            if 0 < sp2["start_frame"] < _n:
                                _a2.axvline(x=gt_root[sp2["start_frame"], 0],
                                            color="gray", ls=":", alpha=0.3)
                        _a2.set_xlim(_xl)
                        _a2.set_ylim(_zl)
                        _a2.set_aspect("equal")
                        _a2.legend(loc="upper right")
                        _a2.set_title(f"{mode}  frame {min(_f, _n)}/{_n}")
                        _wr.grab_frame()
                plt.close(_f2)

    summary = {
        "base_name": sample["name"], "sample_ids_str": args.sample_ids,
        "config": args.config, "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt,
        "history_length": args.history_length,
        "traj_horizon_tokens": args.traj_horizon_tokens,
        "waypoint_dt": babel_waypoint_dt, "token_dt": args.token_dt,
        "modes": {m["mode"]: {k: v for k, v in m.items()
                              if k not in ("segments",)}
                  for m in all_records},
        "text_schedule": [
            {"text": t, "end_frame": int(fe), "end_token": int(te)}
            for t, fe, te in zip(sample["text"], sample["feature_text_end"],
                                 sample["token_text_end"])
        ],
    }
    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    print(f"\n{'Mode':<25} {'ADE':>8} {'FDE':>8} {'path_ratio':>10}")
    print("-" * 55)
    for m in all_records:
        print(f"{m['mode']:<25} {m['ADE']:>8.4f} {m['FDE']:>8.4f} "
              f"{m.get('path_length_ratio', float('nan')):>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Stream control diagnostic matrix on a single sample"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--sample_id", default="000021")
    parser.add_argument("--out_dir", default="outputs/diagnose_stream")
    parser.add_argument("--history_length", type=int, default=30)
    parser.add_argument(
        "--ablate_history",
        type=str,
        default=None,
        help="Comma-separated history_length values for context-window ablation. "
        "When set, runs step_full_xyz_gtroot at each value (bypasses the full matrix).",
    )
    parser.add_argument(
        "--diagnose_step_path",
        action="store_true",
        default=False,
        help="Run step-path diagnosis: features-path, no-traj, offline-decode. "
        "Bypasses the full diagnostic matrix.",
    )
    parser.add_argument(
        "--debug_traj_emb",
        action="store_true",
        default=False,
        help="Compare traj_emb statistics between generate and stream_generate_step paths.",
    )
    parser.add_argument(
        "--debug_control_effect",
        action="store_true",
        default=False,
        help="Trace ControlNet residual and CFG delta magnitudes "
        "inside _denoise_with_cfg during stream_generate_step.",
    )
    parser.add_argument(
        "--step_global_text",
        action="store_true",
        default=False,
        help="Ablation: collapse frame-aligned text to global text in "
        "stream_generate_step (tests whether text layout is the root cause).",
    )
    parser.add_argument(
        "--step_future_traj_window",
        action="store_true",
        default=False,
        help="Ablation: extend stream_generate_step trajectory window to "
        "include future horizon (tests whether truncated traj window is the root cause).",
    )
    parser.add_argument(
        "--step_bypass_buf",
        action="store_true",
        default=False,
        help="Ablation: use stream_generate's traj encoding path "
        "(frame-level traj_features -> encode_traj_batch) inside "
        "stream_generate_step's denoising loop, bypassing TrajStreamBuffer entirely.",
    )
    parser.add_argument(
        "--step_fixed_model_window",
        action="store_true",
        default=False,
        help="Ablation: keep stream_generate_step's WanModel seq_len fixed to "
        "history_length while preserving the existing traj window.",
    )
    parser.add_argument(
        "--predroot_recovery_ablation",
        action="store_true",
        default=False,
        help="Ablation: compare web-demo pred-root trajectory rebuilding with "
        "projected-path recovery compensation.",
    )
    parser.add_argument(
        "--predroot_token_step_sweep",
        type=str,
        default=None,
        help="Comma-separated fixed token_step values for pred-root current-polyline "
        "diagnosis, e.g. '0.06,0.08,0.10,0.12,0.15,0.20'. "
        "Bypasses the full diagnostic matrix.",
    )
    parser.add_argument(
        "--predroot_timestamped_ablation",
        action="store_true",
        default=False,
        help="Ablation: build future trajectory by timestamp interpolation from "
        "dataset GT root, matching web-demo-like stream_generate_step without "
        "distance-based token_step estimation.",
    )
    parser.add_argument(
        "--timestamped_fixed_token_step",
        type=float,
        default=0.10,
        help="Fixed token_step baseline included in --predroot_timestamped_ablation.",
    )
    parser.add_argument(
        "--duration_waypoint_stride_seconds",
        type=float,
        default=1.0,
        help="Waypoint sampling stride used by --predroot_timestamped_ablation "
        "when building timestamped trajectories.",
    )
    parser.add_argument(
        "--render_video",
        action="store_true",
        default=False,
        help="Save rendered .mp4 videos alongside artifacts (slow).",
    )
    parser.add_argument(
        "--recovery_tokens",
        type=int,
        default=6,
        help="Number of future tokens used to blend closed-loop root drift back "
        "to the projected path in --predroot_recovery_ablation.",
    )
    parser.add_argument("--traj_horizon_tokens", type=int, default=20)
    parser.add_argument("--num_denoise_steps", type=int, default=10)
    parser.add_argument("--motion_fps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument(
        "--dataset",
        default="humanml3d",
        choices=["humanml3d", "babel"],
        help="Dataset to load the sample from (default: humanml3d)",
    )
    parser.add_argument(
        "--precomputed_text_emb_path",
        default=None,
        help="Path to pre-tokenized T5 embeddings .pt file. "
        "When set, enables use_precomputed_text_emb and skips live T5.",
    )
    parser.add_argument(
        "--sample_ids",
        default=None,
        help="Comma-separated BABEL sample IDs to merge into a long session "
        "(e.g. 9797_1,9797_2,9797_3). Requires --dataset babel.",
    )
    parser.add_argument(
        "--babel_mode",
        default=None,
        choices=["gt_suffix", "timestamped_gt_plan",
                 "duration_waypoints", "no_traj", "all"],
        help="Trajectory mode for BABEL long-session eval.",
    )
    parser.add_argument("--waypoint_dt", type=float, default=0.05,
                        help="Waypoint stride in seconds for duration_waypoints mode.")
    parser.add_argument("--token_dt", type=float, default=0.20,
                        help="Token time step in seconds.")
    parser.add_argument("--segment_boundary_smooth_frames", type=int, default=4,
                        help="Frames of Gaussian smoothing across segment boundaries.")
    parser.add_argument("--mid_session_update", action="store_true", default=False,
                        help="Ablation: split GT trajectory at --split_token, "
                        "reset timestamped plan mid-session, measure pre/post ADE.")
    parser.add_argument("--split_token", type=int, default=15,
                        help="Token index for --mid_session_update split.")
    parser.add_argument("--synthetic_trail", action="store_true", default=False,
                        help="Ablation: after sample ends, extend with a synthetic "
                        "trajectory segment rotated by --trail_angle degrees.")
    parser.add_argument("--trail_angle", type=float, default=50.0,
                        help="Degrees to rotate the synthetic trail extension.")
    parser.add_argument("--trail_tokens", type=int, default=50,
                        help="Number of extra tokens for synthetic trail extension.")
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(config_path=args.config)
    OmegaConf.update(cfg.config, "test_vae_ckpt", args.vae_ckpt)
    if args.precomputed_text_emb_path:
        OmegaConf.update(
            cfg.config, "model.params.use_precomputed_text_emb", True
        )
        OmegaConf.update(
            cfg.config,
            "model.params.precomputed_text_emb_path",
            args.precomputed_text_emb_path,
        )
    print(f"Loading VAE from {args.vae_ckpt} ...")
    vae = _load_vae(cfg, device)
    print(f"Loading model from {args.ckpt} ...")
    model = _load_model(cfg, args.ckpt, device)

    # ── BABEL multi-sample fast path ──────────────────────────────────
    if args.sample_ids:
        base = args.sample_ids.split(",")[0].strip().rsplit("_", 1)[0]
        out_root = os.path.join(args.out_dir, base)
        _run_babel_long_session(args, model, vae, device, out_root)
        return

    out_root = os.path.join(args.out_dir, args.sample_id)

    print(f"Loading sample {args.sample_id} (dataset={args.dataset}) ...")
    sample = _load_sample(args.raw_data_dir, args.sample_id, dataset=args.dataset)
    _text_preview = (
        sample["text"][0][:60]
        if isinstance(sample["text"], list)
        else sample["text"][:60]
    )
    print(
        f"  feature: {sample['feature'].shape}, token: {sample['token'].shape}, "
        f"text: {_text_preview}..."
    )

    gt_root = extract_root_trajectory_263(sample["feature"].numpy())
    target_traj = sample["traj"].numpy()  # (T, 3) xyz

    # --- history-length ablation (bypasses full matrix) ---
    if args.ablate_history:
        hl_values = [int(x.strip()) for x in args.ablate_history.split(",")]
        print(f"\nHistory-length ablation: {hl_values}")
        all_records = []
        for hl in hl_values:
            mode_name = f"step_full_xyz_gtroot_hl{hl}"
            print(f"\n--- {mode_name} ---")
            pred_motion, pred_root = run_stream_step(
                model, vae, sample, device,
                history_length=hl,
                num_denoise_steps=args.num_denoise_steps,
                horizon_tokens=None,
                use_pred_root=False,
            )
            metrics = _build_mode_metrics(
                pred_root, gt_root, target_traj, mode_name, None, "gt",
                traj_encoder_path="",
            )
            print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
            mode_dir = os.path.join(out_root, mode_name)
            _save_artifacts(
                mode_dir, pred_motion, pred_root,
                gt_root[:len(pred_root)], target_traj[:len(pred_root)], metrics,
            )
            all_records.append(metrics)
        summary = {
            "sample_id": args.sample_id, "dataset": args.dataset,
            "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt, "config": args.config,
            "ablation": "history_length",
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_hl_ablation.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nAblation summary saved to {summary_path}")
        print(f"\n{'Mode':<35} {'ADE':>8} {'FDE':>8}")
        print("-" * 53)
        for rec in all_records:
            print(f"{rec['mode']:<35} {rec['ADE']:>8.4f} {rec['FDE']:>8.4f}")
        return

    if args.diagnose_step_path:
        print("\n=== Step-path diagnosis ===")
        step_modes = [
            ("step_full_features_gtroot", {
                "history_length": args.history_length,
                "num_denoise_steps": args.num_denoise_steps,
                "horizon_tokens": None, "use_pred_root": False,
                "use_features_path": True, "no_traj": False,
            }),
            ("step_full_xyz_gtroot", {
                "history_length": args.history_length,
                "num_denoise_steps": args.num_denoise_steps,
                "horizon_tokens": None, "use_pred_root": False,
                "use_features_path": False, "no_traj": False,
            }),
            ("step_no_traj", {
                "history_length": args.history_length,
                "num_denoise_steps": args.num_denoise_steps,
                "horizon_tokens": None, "use_pred_root": False,
                "use_features_path": False, "no_traj": True,
            }),
        ]
        all_records = []
        for mode_name, kw in step_modes:
            print(f"\n--- {mode_name} ---")
            pred_motion, pred_root = run_stream_step(
                model, vae, sample, device, **kw,
            )
            metrics = _build_mode_metrics(
                pred_root, gt_root, target_traj, mode_name,
                kw.get("horizon_tokens"), "gt",
                traj_encoder_path=(
                    "traj_features (no anchor)" if kw.get("use_features_path")
                    else "xyz (anchor-subtract)" if not kw.get("no_traj")
                    else "none"
                ),
            )
            print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
            mode_dir = os.path.join(out_root, mode_name)
            _save_artifacts(
                mode_dir, pred_motion, pred_root,
                gt_root[:len(pred_root)], target_traj[:len(pred_root)], metrics,
            )
            all_records.append(metrics)

        # Also run offline decode comparison.
        print(f"\n--- step_offline_decode ---")
        latents, _ = run_stream_step(
            model, vae, sample, device,
            history_length=args.history_length,
            num_denoise_steps=args.num_denoise_steps,
            horizon_tokens=None, use_pred_root=False,
            use_features_path=False, no_traj=False, collect_latents=True,
        )
        decoded_offline = vae.decode(
            torch.from_numpy(latents).float().unsqueeze(0).to(device)
        )[0].float().cpu().numpy()
        pred_root_off = extract_root_trajectory_263(decoded_offline)
        metrics_off = _build_mode_metrics(
            pred_root_off, gt_root, target_traj, "step_offline_decode",
            None, "gt", traj_encoder_path="xyz (latents from step, decode offline)",
        )
        print(f"  ADE={metrics_off['ADE']:.4f}  FDE={metrics_off['FDE']:.4f}")
        mode_dir = os.path.join(out_root, "step_offline_decode")
        _save_artifacts(mode_dir, decoded_offline, pred_root_off,
                        gt_root[:len(pred_root_off)], target_traj[:len(pred_root_off)],
                        metrics_off)
        all_records.append(metrics_off)

        summary = {
            "sample_id": args.sample_id, "dataset": args.dataset,
            "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt, "config": args.config,
            "diagnosis": "step_path",
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_step_path.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")
        print(f"\n{'Mode':<35} {'ADE':>8} {'FDE':>8}")
        print("-" * 53)
        for rec in all_records:
            print(f"{rec['mode']:<35} {rec['ADE']:>8.4f} {rec['FDE']:>8.4f}")
        return

    if args.debug_traj_emb:
        print("\n=== Traj-emb debug ===")
        _bs = _wrap_flat_sample_for_suffix(sample)
        gen_seq_len = sample["token_length"] + model.chunk_size

        # 1. encode_traj_batch (used by generate / stream_generate)
        emb_gen = model._build_traj_emb(_bs, gen_seq_len, device)
        if emb_gen is not None:
            print(f"  [generate path] _build_traj_emb: shape={tuple(emb_gen.shape)} "
                  f"mean={emb_gen.mean().item():.6f} std={emb_gen.std().item():.6f} "
                  f"min={emb_gen.min().item():.6f} max={emb_gen.max().item():.6f}")
        else:
            print("  [generate path] _build_traj_emb: None")

        # 2. TrajStreamBuffer (used by stream_generate_step)
        model.init_generated(args.history_length, batch_size=1,
                            num_denoise_steps=args.num_denoise_steps)
        # Feed one GT-root step to populate the buffer.
        for ci in range(min(5, sample["token_length"])):
            ti = build_stream_suffix_conditioning(_bs, ci, prefer_xyz=True)
            sp = build_stream_step_model_input(
                sample["text"] if isinstance(sample["text"], str) else sample["text"][0],
                traj_input=ti,
            )
            model.stream_generate_step(sp, first_chunk=(ci == 0))

        end_idx = min(5 + model.chunk_size,
                      model.commit_index + model.chunk_size)
        emb_step = model._traj_buf.build_traj_emb(end_idx, args.history_length, device)
        if emb_step is not None:
            print(f"  [step   path] build_traj_emb:  shape={tuple(emb_step.shape)} "
                  f"mean={emb_step.mean().item():.6f} std={emb_step.std().item():.6f} "
                  f"min={emb_step.min().item():.6f} max={emb_step.max().item():.6f}")
            if emb_gen is not None:
                _match = emb_gen[:, :emb_step.shape[1], :] - emb_step
                print(f"  [compare]      L2-diff per token: "
                      f"mean={_match.norm(dim=-1).mean().item():.6f} "
                      f"max={_match.norm(dim=-1).max().item():.6f}")
        else:
            print("  [step   path] build_traj_emb:  None")

        print(f"  [traj-buf state] commit_index={model.commit_index} "
              f"feat_buf={'set' if model._traj_buf._feat_buf is not None else 'None'} "
              f"xyz_buf={'set' if model._traj_buf._xyz_buf is not None else 'None'} "
              f"mask_buf={'set' if model._traj_buf._mask_buf is not None else 'None'}")
        if model._traj_buf._xyz_buf is not None:
            _x = model._traj_buf._xyz_buf[0, :10, :]
            print(f"  [traj-buf xyz first 10 tokens] mean={_x.mean().item():.4f} "
                  f"nonzero={(_x.abs().sum(dim=-1) > 0).sum().item()}/10")

        return

    if args.debug_control_effect:
        print("\n=== ControlNet effect debug ===")
        _bs = _wrap_flat_sample_for_suffix(sample)

        # 1. Run one generate() denoise step with traj and without, compare
        #    ControlNet residuals directly.
        feat = _bs["token"].unsqueeze(0).to(device)  # (1, T, C)
        feat_len = torch.tensor([sample["token_length"]], device=device)
        seq_len = feat.shape[1]

        # Prepare a single-window state at time t ≈ 0.5
        time_steps = torch.tensor([0.5], device=device)
        noise_level = model._get_noise_levels(device, seq_len, time_steps)
        noisy_x, _ = model.add_noise(feat, noise_level)
        noisy_pre = model.preprocess(noisy_x)  # (B, C, T, 1, 1)
        noisy_in = [noisy_pre[b] for b in range(noisy_pre.shape[0])]

        text_list = [sample["text"]] if isinstance(sample["text"], str) else [sample["text"][0]]
        text_ctx = [u.to(model.param_dtype) for u in
                     model.encode_text_with_cache(text_list, device)]
        text_null = [u.to(model.param_dtype) for u in
                      model.encode_text_with_cache([""], device)]

        traj_emb = model._build_traj_emb(_bs, seq_len + model.chunk_size, device)
        traj_sl = model._get_traj_seq_lens(_bs, seq_len + model.chunk_size, device)
        t_scaled = noise_level * model.time_embedding_scale

        # All ControlNet / backbone forwards inside the same autocast
        # that the training loop uses (bf16-mixed).
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # --- ControlNet residual with traj ---
            cn_res_traj = model._controlnet_forward(
                noisy_in, t_scaled, text_ctx, seq_len, traj_emb, traj_sl,
            )
            cn_norm_traj = sum(r.norm().item() for r in cn_res_traj)

            # --- ControlNet residual without traj ---
            cn_res_null = model._controlnet_forward(
                noisy_in, t_scaled, text_ctx, seq_len, None, None,
            )
            cn_norm_null = sum(r.norm().item() for r in cn_res_null)

            cn_delta = sum((a - b).norm().item()
                           for a, b in zip(cn_res_traj, cn_res_null))

            # --- Backbone pred with traj vs null-traj ControlNet ---
            pred_traj = model.model(
                noisy_in, t_scaled, text_ctx, seq_len,
                y=None, traj_emb=None, traj_seq_lens=None,
                controlnet_residuals=cn_res_traj,
            )
            pred_null = model.model(
                noisy_in, t_scaled, text_ctx, seq_len,
                y=None, traj_emb=None, traj_seq_lens=None,
                controlnet_residuals=cn_res_null,
            )

        print(f"  [control-residual] with-traj norm={cn_norm_traj:.4f}  "
              f"without-traj norm={cn_norm_null:.4f}  delta={cn_delta:.4f}")

        delta_pred = sum((a - b).norm().item()
                         for a, b in zip(pred_traj, pred_null))
        pred_traj_norm = sum(a.norm().item() for a in pred_traj)

        # Per-token delta (mean across feature dims of the velocity).
        _dp = pred_traj[0] - pred_null[0]  # (C, T, 1, 1)
        _tok_norms = _dp.squeeze(-1).squeeze(-1).norm(dim=0)  # (T,)
        print(f"  [pred-delta] total norm={delta_pred:.4f}  "
              f"pred-traj norm={pred_traj_norm:.4f}")
        print(f"  [pred-delta per-token]  "
              f"min={_tok_norms.min().item():.6f}  "
              f"mean={_tok_norms.mean().item():.6f}  "
              f"max={_tok_norms.max().item():.6f}")
        print(f"  [diagnosis] cn_delta={'LARGE' if cn_delta > 0.1 else 'small'}  "
              f"pred_delta={'LARGE' if delta_pred > 0.1 else 'small'}")

        # --- Compare latent tokens: generate() vs stream_generate_step() ---
        model.init_generated(args.history_length, batch_size=1,
                            num_denoise_steps=args.num_denoise_steps)
        vae.clear_cache()
        latents_step = []
        _bs2 = _wrap_flat_sample_for_suffix(sample)
        _txt = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
        first_chunk = True
        for ci in range(min(10, sample["token_length"])):
            ti = build_stream_suffix_conditioning(_bs2, ci, prefer_xyz=True)
            sp = build_stream_step_model_input(_txt, traj_input=ti)
            out = model.stream_generate_step(sp, first_chunk=first_chunk)
            latents_step.append(out["generated"][0].float().cpu())
            first_chunk = False
        latents_step = torch.cat(latents_step, dim=0)  # (N, C)

        # Compare with generate() latents (first N tokens)
        model_batch = _make_model_batch(sample, device)
        gen_out = model.generate(model_batch, num_denoise_steps=args.num_denoise_steps)
        latents_gen = gen_out["generated"][0].float().cpu()  # (T, C)

        n_cmp = min(len(latents_step), len(latents_gen))
        l2_diff = (latents_step[:n_cmp] - latents_gen[:n_cmp]).norm(dim=-1)
        cos_sim = torch.nn.functional.cosine_similarity(
            latents_step[:n_cmp].float(), latents_gen[:n_cmp].float(), dim=-1
        )
        print(f"  [latent-compare] generate vs step (first {n_cmp} tokens):")
        print(f"    gen token stats: mean={latents_gen[:n_cmp].mean():.6f} "
              f"std={latents_gen[:n_cmp].std():.6f}")
        print(f"    step token stats: mean={latents_step[:n_cmp].mean():.6f} "
              f"std={latents_step[:n_cmp].std():.6f}")
        print(f"    L2-diff: min={l2_diff.min():.6f} mean={l2_diff.mean():.6f} "
              f"max={l2_diff.max():.6f}")
        print(f"    cosine-sim: min={cos_sim.min():.4f} mean={cos_sim.mean():.4f}")
        print(f"  [diagnosis] latent-diff={'SMALL (<1.0 means similar)' if l2_diff.mean() < 1.0 else 'LARGE (>1.0 means diverged)'}")
        return

    if args.step_global_text:
        print("\n=== Global-text ablation ===")
        _orig_dn = model._denoise_with_cfg

        def _global_text_dn(noisy_input, t_scaled, text_cond_ctx, text_null_ctx,
                            traj_emb, traj_seq_lens, seq_len, batch_size):
            # Collapse frame-aligned text to global: keep only first token's
            # context per sample.
            if len(text_cond_ctx) > batch_size and len(text_cond_ctx) % seq_len == 0:
                text_cond_ctx = text_cond_ctx[:batch_size]
            if len(text_null_ctx) > batch_size:
                text_null_ctx = text_null_ctx[:batch_size]
            return _orig_dn(noisy_input, t_scaled, text_cond_ctx, text_null_ctx,
                            traj_emb, traj_seq_lens, seq_len, batch_size)

        model._denoise_with_cfg = _global_text_dn

        pred_motion, pred_root = run_stream_step(
            model, vae, sample, device,
            history_length=args.history_length,
            num_denoise_steps=args.num_denoise_steps,
            horizon_tokens=None, use_pred_root=False,
            use_features_path=False, no_traj=False, collect_latents=False,
        )
        metrics = _build_mode_metrics(
            pred_root, gt_root, target_traj, "step_global_text_gtroot",
            None, "gt", traj_encoder_path="xyz (global text)",
        )
        print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
        model._denoise_with_cfg = _orig_dn

        # Also run with no-traj for comparison.
        model._denoise_with_cfg = _global_text_dn
        pred_motion2, pred_root2 = run_stream_step(
            model, vae, sample, device,
            history_length=args.history_length,
            num_denoise_steps=args.num_denoise_steps,
            horizon_tokens=None, use_pred_root=False,
            use_features_path=False, no_traj=True, collect_latents=False,
        )
        metrics2 = _build_mode_metrics(
            pred_root2, gt_root, target_traj, "step_global_text_no_traj",
            None, "none", traj_encoder_path="none (global text, no traj)",
        )
        print(f"  ADE={metrics2['ADE']:.4f}  FDE={metrics2['FDE']:.4f}")
        model._denoise_with_cfg = _orig_dn

        mode_dir = os.path.join(out_root, "step_global_text_gtroot")
        _save_artifacts(mode_dir, pred_motion, pred_root,
                        gt_root[:len(pred_root)], target_traj[:len(pred_root)], metrics)
        mode_dir2 = os.path.join(out_root, "step_global_text_no_traj")
        _save_artifacts(mode_dir2, pred_motion2, pred_root2,
                        gt_root[:len(pred_root2)], target_traj[:len(pred_root2)], metrics2)

        all_records = [metrics, metrics2]
        summary = {
            "sample_id": args.sample_id, "dataset": args.dataset,
            "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt,
            "ablation": "global_text",
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_global_text.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n{'Mode':<35} {'ADE':>8} {'FDE':>8}")
        print("-" * 53)
        for rec in all_records:
            print(f"{rec['mode']:<35} {rec['ADE']:>8.4f} {rec['FDE']:>8.4f}")
        return

    if args.step_fixed_model_window:
        print("\n=== Fixed-model-window ablation ===")
        _orig_step = model.stream_generate_step

        def _patched_step(x, first_chunk=True):
            device = next(model.parameters()).device
            if first_chunk:
                model.generated = model.generated.to(device)
            model._traj_buf.update(x, model.commit_index, device)

            if model.use_text_cond and "text" in x:
                text_list = x["text"]
                new_text_context = model.encode_text_with_cache(text_list, device)
                new_text_context = [u.to(model.param_dtype) for u in new_text_context]
            else:
                new_text_context = [""] * model.batch_size
                new_text_context = model.encode_text_with_cache(new_text_context, device)
                new_text_context = [u.to(model.param_dtype) for u in new_text_context]

            text_null_list = [""] * model.batch_size
            text_null_context = model.encode_text_with_cache(text_null_list, device)
            text_null_context = [u.to(model.param_dtype) for u in text_null_context]

            for i in range(model.batch_size):
                if first_chunk:
                    model.text_condition_list[i].extend(
                        [new_text_context[i]] * model.chunk_size
                    )
                else:
                    model.text_condition_list[i].extend([new_text_context[i]])

            end_step = (
                (model.commit_index + model.chunk_size)
                * model.num_denoise_steps
                / model.chunk_size
            )
            while model.current_step < end_step:
                current_time = model.current_step * model.dt
                start_index = max(0, int(model.chunk_size * (current_time - 1)) + 1)
                end_index = int(model.chunk_size * current_time) + 1
                time_steps = torch.full((model.batch_size,), current_time, device=device)

                noise_level_full = model._get_noise_levels(device, end_index, time_steps)
                noise_level_for_update = noise_level_full

                # ABLATION: keep WanModel's seq_len fixed even during early
                # stream_generate_step calls. The formal path passes
                # model_sl=min(end_index, seq_len), so the first steps run with
                # very short positional/time/context windows. stream_generate()
                # instead always calls the model with a fixed padded seq_len.
                model_sl = model.seq_len
                if end_index < model_sl:
                    noise_level = model._get_noise_levels(device, model_sl, time_steps)
                else:
                    noise_level = noise_level_full[:, -model_sl:]

                noisy_input = []
                for i in range(model.batch_size):
                    noisy_input.append(
                        model.generated[i, :, :end_index, ...][:, -model_sl:]
                    )

                text_condition = []
                for i in range(model.batch_size):
                    tc = model.text_condition_list[i][:end_index][-model_sl:]
                    if not tc:
                        tc = [new_text_context[i]]
                    if len(tc) < model_sl:
                        tc = tc + [tc[-1]] * (model_sl - len(tc))
                    else:
                        tc = tc[:model_sl]
                    text_condition.extend(tc)

                traj_emb = model._traj_buf.build_traj_emb(end_index, model_sl, device)
                if traj_emb is not None:
                    valid_lens = model._traj_buf.get_traj_valid_lens(
                        end_index, model_sl, device
                    )
                    traj_seq_lens = (
                        valid_lens
                        if valid_lens is not None
                        else torch.full(
                            (model.batch_size,),
                            min(end_index, model_sl),
                            device=device,
                            dtype=torch.long,
                        )
                    )
                else:
                    traj_seq_lens = None

                t_scaled = noise_level * model.time_embedding_scale
                predicted_result = model._denoise_with_cfg(
                    noisy_input, t_scaled,
                    text_condition, text_null_context,
                    traj_emb, traj_seq_lens, model_sl, model.batch_size,
                )

                for i in range(model.batch_size):
                    predicted_result_i = predicted_result[i]
                    if end_index > model_sl:
                        predicted_result_i = torch.cat(
                            [
                                torch.zeros(
                                    predicted_result_i.shape[0],
                                    end_index - model_sl,
                                    predicted_result_i.shape[2],
                                    predicted_result_i.shape[3],
                                    device=device,
                                ),
                                predicted_result_i,
                            ],
                            dim=1,
                        )
                    if model.prediction_type == "vel":
                        predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                        model.generated[i, :, start_index:end_index, ...] += (
                            predicted_vel * model.dt
                        )
                    elif model.prediction_type == "x0":
                        nl = (
                            noise_level_for_update[i, start_index:end_index]
                            .unsqueeze(0)
                            .unsqueeze(-1)
                            .unsqueeze(-1)
                            .clamp(min=1e-6)
                        )
                        predicted_vel = (
                            predicted_result_i[:, start_index:end_index, ...]
                            - model.generated[i, :, start_index:end_index, ...]
                        ) / nl
                        model.generated[i, :, start_index:end_index, ...] += (
                            predicted_vel * model.dt
                        )
                    elif model.prediction_type == "noise":
                        denom = (
                            1
                            + model.dt
                            - noise_level_for_update[i, start_index:end_index]
                            .unsqueeze(0)
                            .unsqueeze(-1)
                            .unsqueeze(-1)
                        ).clamp(min=1e-6)
                        predicted_vel = (
                            model.generated[i, :, start_index:end_index, ...]
                            - predicted_result_i[:, start_index:end_index, ...]
                        ) / denom
                        model.generated[i, :, start_index:end_index, ...] += (
                            predicted_vel * model.dt
                        )
                model.current_step += 1

            output = model.generated[:, :, model.commit_index : model.commit_index + 1, ...]
            output = model.postprocess(output)
            out = {"generated": output}
            model.commit_index += 1

            if model.commit_index == model.seq_len * 2:
                model.generated = torch.cat(
                    [
                        model.generated[:, :, model.seq_len :, ...],
                        torch.randn(
                            model.batch_size,
                            model.input_dim,
                            model.seq_len,
                            1,
                            1,
                            device=device,
                        ),
                    ],
                    dim=2,
                )
                model._traj_buf.roll(model.seq_len, device)
                model.current_step -= (
                    model.seq_len * model.num_denoise_steps / model.chunk_size
                )
                model.commit_index -= model.seq_len
                for i in range(model.batch_size):
                    model.text_condition_list[i] = model.text_condition_list[i][
                        model.seq_len:
                    ]
            return out

        model.stream_generate_step = _patched_step
        pred_motion, pred_root = run_stream_step(
            model, vae, sample, device,
            history_length=args.history_length,
            num_denoise_steps=args.num_denoise_steps,
            horizon_tokens=None, use_pred_root=False,
            use_features_path=False, no_traj=False, collect_latents=False,
        )
        metrics = _build_mode_metrics(
            pred_root, gt_root, target_traj, "step_fixed_model_window_gtroot",
            None, "gt", traj_encoder_path="xyz (fixed model seq_len)",
        )
        print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
        model.stream_generate_step = _orig_step

        mode_dir = os.path.join(out_root, "step_fixed_model_window_gtroot")
        _save_artifacts(
            mode_dir, pred_motion, pred_root,
            gt_root[:len(pred_root)], target_traj[:len(pred_root)], metrics,
        )

        all_records = [metrics]
        summary = {
            "sample_id": args.sample_id, "dataset": args.dataset,
            "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt,
            "ablation": "fixed_model_window",
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_fixed_model_window.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n{'Mode':<35} {'ADE':>8} {'FDE':>8}")
        print("-" * 53)
        for rec in all_records:
            print(f"{rec['mode']:<35} {rec['ADE']:>8.4f} {rec['FDE']:>8.4f}")
        return

    if args.step_future_traj_window:
        print("\n=== Future-traj-window ablation ===")
        _orig_step = model.stream_generate_step

        def _patched_step(x, first_chunk=True):
            device = next(model.parameters()).device
            if first_chunk:
                model.generated = model.generated.to(device)
            model._traj_buf.update(x, model.commit_index, device)

            if model.use_text_cond and "text" in x:
                text_list = x["text"]
                new_text_context = model.encode_text_with_cache(text_list, device)
                new_text_context = [u.to(model.param_dtype) for u in new_text_context]
            else:
                new_text_context = [""] * model.batch_size
                new_text_context = model.encode_text_with_cache(new_text_context, device)
                new_text_context = [u.to(model.param_dtype) for u in new_text_context]

            text_null_list = [""] * model.batch_size
            text_null_context = model.encode_text_with_cache(text_null_list, device)
            text_null_context = [u.to(model.param_dtype) for u in text_null_context]

            for i in range(model.batch_size):
                if first_chunk:
                    model.text_condition_list[i].extend(
                        [new_text_context[i]] * model.chunk_size
                    )
                else:
                    model.text_condition_list[i].extend([new_text_context[i]])

            end_step = (
                (model.commit_index + model.chunk_size)
                * model.num_denoise_steps
                / model.chunk_size
            )
            while model.current_step < end_step:
                current_time = model.current_step * model.dt
                start_index = max(0, int(model.chunk_size * (current_time - 1)) + 1)
                end_index = int(model.chunk_size * current_time) + 1
                time_steps = torch.full((model.batch_size,), current_time, device=device)

                noise_level_full = model._get_noise_levels(device, end_index, time_steps)
                noise_level_for_update = noise_level_full

                # ABLATION: keep the model window length unchanged, but shift
                # the rolling window so it contains a short history plus future
                # trajectory slots.  This avoids changing the model's positional
                # distribution while testing whether future traj keys matter:
                #
                #   [window_start, end_index)           = latent history
                #   [end_index, window_start + seq_len) = future traj only
                #
                # This mirrors full stream_generate(), where WanModel receives
                # a short latent prefix padded to a fixed seq_len plus future
                # traj_emb in the padded slots.
                model_window_len = model.seq_len
                horizon_len = min(max(0, int(args.traj_horizon_tokens)), model_window_len)
                history_context_len = max(1, model_window_len - horizon_len)
                window_start = max(0, end_index - history_context_len)
                window_end = min(window_start + model_window_len, model._traj_buf.buf_len)
                model_window_len = window_end - window_start
                if model_window_len <= 0:
                    continue

                noise_level_window_full = model._get_noise_levels(
                    device, window_end, time_steps
                )
                noise_level = noise_level_window_full[:, window_start:window_end]

                noisy_input = []
                for i in range(model.batch_size):
                    noisy_input.append(
                        model.generated[i, :, window_start:end_index, ...]
                    )

                text_condition = []
                for i in range(model.batch_size):
                    tc = model.text_condition_list[i][window_start:end_index]
                    if not tc:
                        tc = [new_text_context[i]]
                    # Pad future text with the most recent known condition.
                    if len(tc) < model_window_len:
                        tc = tc + [tc[-1]] * (model_window_len - len(tc))
                    else:
                        tc = tc[:model_window_len]
                    text_condition.extend(tc)

                traj_emb = model._traj_buf.build_traj_emb(
                    window_end, model_window_len, device
                )

                if traj_emb is not None:
                    valid_lens = model._traj_buf.get_traj_valid_lens(
                        window_end, model_window_len, device,
                    )
                    traj_seq_lens = (
                        valid_lens
                        if valid_lens is not None
                        else torch.full(
                            (model.batch_size,), model_window_len,
                            device=device, dtype=torch.long,
                        )
                    )
                else:
                    traj_seq_lens = None
                t_scaled = noise_level * model.time_embedding_scale
                predicted_result = model._denoise_with_cfg(
                    noisy_input, t_scaled,
                    text_condition, text_null_context,
                    traj_emb, traj_seq_lens, model_window_len, model.batch_size,
                )

                for i in range(model.batch_size):
                    predicted_result_i = predicted_result[i]
                    local_start = start_index - window_start
                    local_end = end_index - window_start
                    if model.prediction_type == "vel":
                        predicted_vel = predicted_result_i[:, local_start:local_end, ...]
                        model.generated[i, :, start_index:end_index, ...] += (
                            predicted_vel * model.dt
                        )
                    elif model.prediction_type == "x0":
                        nl = (
                            noise_level_for_update[i, start_index:end_index]
                            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                            .clamp(min=1e-6)
                        )
                        predicted_vel = (
                            predicted_result_i[:, local_start:local_end, ...]
                            - model.generated[i, :, start_index:end_index, ...]
                        ) / nl
                        model.generated[i, :, start_index:end_index, ...] += (
                            predicted_vel * model.dt
                        )
                    elif model.prediction_type == "noise":
                        denom = (
                            1 + model.dt
                            - noise_level_for_update[i, start_index:end_index]
                            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                        ).clamp(min=1e-6)
                        predicted_vel = (
                            model.generated[i, :, start_index:end_index, ...]
                            - predicted_result_i[:, local_start:local_end, ...]
                        ) / denom
                        model.generated[i, :, start_index:end_index, ...] += (
                            predicted_vel * model.dt
                        )
                model.current_step += 1

            output = model.generated[
                :, :, model.commit_index : model.commit_index + 1, ...
            ]
            output = model.postprocess(output)
            out = {"generated": output}
            model.commit_index += 1

            if model.commit_index == model.seq_len * 2:
                model.generated = torch.cat(
                    [
                        model.generated[:, :, model.seq_len:, ...],
                        torch.randn(
                            model.batch_size, model.input_dim,
                            model.seq_len, 1, 1, device=device,
                        ),
                    ],
                    dim=2,
                )
                model._traj_buf.roll(model.seq_len, device)
                model.current_step -= (
                    model.seq_len * model.num_denoise_steps / model.chunk_size
                )
                model.commit_index -= model.seq_len
                for i in range(model.batch_size):
                    model.text_condition_list[i] = model.text_condition_list[i][
                        model.seq_len:
                    ]
            return out

        model.stream_generate_step = _patched_step

        pred_motion, pred_root = run_stream_step(
            model, vae, sample, device,
            history_length=args.history_length,
            num_denoise_steps=args.num_denoise_steps,
            horizon_tokens=None, use_pred_root=False,
            use_features_path=False, no_traj=False, collect_latents=False,
        )
        metrics = _build_mode_metrics(
            pred_root, gt_root, target_traj, "step_future_traj_window_gtroot",
            None, "gt", traj_encoder_path="xyz (full traj window)",
        )
        print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
        model.stream_generate_step = _orig_step
        mode_dir = os.path.join(out_root, "step_future_traj_window_gtroot")
        _save_artifacts(mode_dir, pred_motion, pred_root,
                        gt_root[:len(pred_root)], target_traj[:len(pred_root)], metrics)

        all_records = [metrics]
        summary = {
            "sample_id": args.sample_id, "dataset": args.dataset,
            "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt,
            "ablation": "future_traj_window",
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_future_traj.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n{'Mode':<35} {'ADE':>8} {'FDE':>8}")
        print("-" * 53)
        for rec in all_records:
            print(f"{rec['mode']:<35} {rec['ADE']:>8.4f} {rec['FDE']:>8.4f}")
        return

    if args.step_bypass_buf:
        print("\n=== Bypass-buffer ablation ===")
        token_length = sample["token_length"]

        # 1. Populate TrajStreamBuffer with a few steps (same xyz path as
        #    stream_generate_step), then capture the buffer's traj_emb.
        model.init_generated(args.history_length, batch_size=1,
                            num_denoise_steps=args.num_denoise_steps)
        model.generated = model.generated.to(device)
        _bs = _wrap_flat_sample_for_suffix(sample)
        _txt = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
        first_chunk = True
        for ci in range(min(5, token_length)):
            ti = build_stream_suffix_conditioning(_bs, ci, prefer_xyz=True)
            sp = build_stream_step_model_input(_txt, traj_input=ti)
            model.stream_generate_step(sp, first_chunk=first_chunk)
            first_chunk = False
        _buf_emb = model._traj_buf.build_traj_emb(
            model.commit_index + model.chunk_size, model.seq_len, device
        )
        _buf_sl = model._traj_buf.get_traj_valid_lens(
            model.commit_index + model.chunk_size, model.seq_len, device,
        )

        # 2. Patch _build_traj_emb so generate() uses the buffer's emb,
        #    then call generate() which is known not to OOM.
        _orig_build = model._build_traj_emb
        def _patched_build(x, sl, dev):
            return _buf_emb
        model._build_traj_emb = _patched_build

        model_batch = _make_model_batch(sample, device)
        out = model.generate(model_batch, num_denoise_steps=args.num_denoise_steps)
        model._build_traj_emb = _orig_build
        generated_latent = out["generated"][0]
        decoded = vae.decode(
            generated_latent[None, :].to(device)
        )[0].float().cpu().numpy()
        pred_root = extract_root_trajectory_263(decoded)

        metrics = _build_mode_metrics(
            pred_root, gt_root, target_traj, "step_bypass_buf_gtroot",
            None, "gt",
            traj_encoder_path="traj from buffer xyz path, via generate loop",
        )
        print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
        mode_dir = os.path.join(out_root, "step_bypass_buf_gtroot")
        _save_artifacts(mode_dir, decoded, pred_root,
                        gt_root[:len(pred_root)], target_traj[:len(pred_root)], metrics)

        all_records = [metrics]
        summary = {
            "sample_id": args.sample_id, "dataset": args.dataset,
            "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt,
            "ablation": "bypass_buffer",
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_bypass_buf.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n{'Mode':<35} {'ADE':>8} {'FDE':>8}")
        print("-" * 53)
        for rec in all_records:
            print(f"{rec['mode']:<35} {rec['ADE']:>8.4f} {rec['FDE']:>8.4f}")
        return

    if args.predroot_recovery_ablation:
        print("\n=== Pred-root recovery ablation ===")
        oracle_step = _compute_oracle_token_step(sample)
        print(f"  oracle token_step={oracle_step:.4f}")
        mode_specs = [
            (
                f"predroot_current_h{args.traj_horizon_tokens}",
                "current_polyline",
                None,
                "web-demo current_root->projection polyline",
            ),
            (
                f"predroot_current_oracle_h{args.traj_horizon_tokens}",
                "current_polyline",
                oracle_step,
                "web-demo polyline with GT oracle token_step",
            ),
            (
                f"predroot_recovery_r{args.recovery_tokens}_h{args.traj_horizon_tokens}",
                "recovery",
                None,
                "projected suffix + lateral-error recovery blend",
            ),
            (
                f"predroot_recovery_oracle_r{args.recovery_tokens}_h{args.traj_horizon_tokens}",
                "recovery",
                oracle_step,
                "recovery blend with GT oracle token_step",
            ),
        ]
        all_records = []
        for mode_name, pred_mode, token_step_override, traj_path in mode_specs:
            print(f"\n--- {mode_name} ---")
            trace: list[dict] = []
            seed_everything(args.seed)
            pred_motion, pred_root = run_stream_step(
                model,
                vae,
                sample,
                device,
                history_length=args.history_length,
                num_denoise_steps=args.num_denoise_steps,
                horizon_tokens=args.traj_horizon_tokens,
                use_pred_root=True,
                predroot_mode=pred_mode,
                recovery_tokens=args.recovery_tokens,
                oracle_token_step=token_step_override,
                trace_sink=trace,
            )
            metrics = _build_mode_metrics(
                pred_root,
                gt_root,
                target_traj,
                mode_name,
                args.traj_horizon_tokens,
                "pred",
                traj_encoder_path=traj_path,
            )
            metrics["predroot_mode"] = pred_mode
            metrics["recovery_tokens"] = int(args.recovery_tokens)
            metrics["oracle_token_step"] = (
                None if token_step_override is None else float(token_step_override)
            )
            if trace:
                metrics["mean_cross_track_error"] = float(
                    np.mean([row["cross_track_error"] for row in trace])
                )
                metrics["max_cross_track_error"] = float(
                    np.max([row["cross_track_error"] for row in trace])
                )
            print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
            if trace:
                print(
                    "  cross_track="
                    f"{metrics['mean_cross_track_error']:.4f}/"
                    f"{metrics['max_cross_track_error']:.4f} mean/max"
                )
            mode_dir = os.path.join(out_root, mode_name)
            _save_artifacts(
                mode_dir,
                pred_motion,
                pred_root,
                gt_root[: len(pred_root)],
                target_traj[: len(pred_root)],
                metrics,
            )
            with open(os.path.join(mode_dir, "predroot_trace.json"), "w") as f:
                json.dump(trace, f, indent=2, default=str)
            all_records.append(metrics)

        summary = {
            "sample_id": args.sample_id,
            "dataset": args.dataset,
            "ckpt": args.ckpt,
            "vae_ckpt": args.vae_ckpt,
            "config": args.config,
            "ablation": "predroot_recovery",
            "history_length": args.history_length,
            "traj_horizon_tokens": args.traj_horizon_tokens,
            "recovery_tokens": args.recovery_tokens,
            "oracle_token_step": oracle_step,
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_predroot_recovery.json")
        os.makedirs(out_root, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")
        print(f"\n{'Mode':<48} {'ADE':>8} {'FDE':>8} {'XTE':>8}")
        print("-" * 76)
        for rec in all_records:
            print(
                f"{rec['mode']:<48} {rec['ADE']:>8.4f} "
                f"{rec['FDE']:>8.4f} "
                f"{rec.get('mean_cross_track_error', float('nan')):>8.4f}"
            )
        return

    if args.predroot_token_step_sweep:
        print("\n=== Pred-root token-step sweep ===")
        fixed_steps = [
            float(x.strip())
            for x in args.predroot_token_step_sweep.split(",")
            if x.strip()
        ]
        oracle_step = _compute_oracle_token_step(sample)
        print(f"  oracle token_step={oracle_step:.4f}")
        all_records = []

        sweep_specs = [(None, "online_estimate")]
        sweep_specs.extend((step, f"fixed_{step:.4f}") for step in fixed_steps)
        sweep_specs.append((oracle_step, "oracle"))

        for step_value, label in sweep_specs:
            mode_name = f"predroot_step_{label}_h{args.traj_horizon_tokens}"
            print(f"\n--- {mode_name} ---")
            trace: list[dict] = []
            seed_everything(args.seed)
            pred_motion, pred_root = run_stream_step(
                model,
                vae,
                sample,
                device,
                history_length=args.history_length,
                num_denoise_steps=args.num_denoise_steps,
                horizon_tokens=args.traj_horizon_tokens,
                use_pred_root=True,
                predroot_mode="current_polyline",
                oracle_token_step=step_value,
                trace_sink=trace,
            )
            metrics = _build_mode_metrics(
                pred_root,
                gt_root,
                target_traj,
                mode_name,
                args.traj_horizon_tokens,
                "pred",
                traj_encoder_path="web-demo current_root->projection polyline",
            )
            metrics["fixed_token_step"] = (
                None if step_value is None else float(step_value)
            )
            if trace:
                token_steps = [row["token_step"] for row in trace]
                metrics["mean_token_step"] = float(np.mean(token_steps))
                metrics["min_token_step"] = float(np.min(token_steps))
                metrics["max_token_step"] = float(np.max(token_steps))
                metrics["mean_cross_track_error"] = float(
                    np.mean([row["cross_track_error"] for row in trace])
                )
                metrics["max_cross_track_error"] = float(
                    np.max([row["cross_track_error"] for row in trace])
                )
            print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
            if trace:
                print(
                    "  token_step="
                    f"{metrics['mean_token_step']:.4f}/"
                    f"{metrics['min_token_step']:.4f}/"
                    f"{metrics['max_token_step']:.4f} mean/min/max"
                )
            mode_dir = os.path.join(out_root, mode_name)
            _save_artifacts(
                mode_dir,
                pred_motion,
                pred_root,
                gt_root[: len(pred_root)],
                target_traj[: len(pred_root)],
                metrics,
            )
            with open(os.path.join(mode_dir, "predroot_trace.json"), "w") as f:
                json.dump(trace, f, indent=2, default=str)
            all_records.append(metrics)

        summary = {
            "sample_id": args.sample_id,
            "dataset": args.dataset,
            "ckpt": args.ckpt,
            "vae_ckpt": args.vae_ckpt,
            "config": args.config,
            "ablation": "predroot_token_step_sweep",
            "history_length": args.history_length,
            "traj_horizon_tokens": args.traj_horizon_tokens,
            "fixed_steps": fixed_steps,
            "oracle_token_step": oracle_step,
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_predroot_token_step_sweep.json")
        os.makedirs(out_root, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")
        print(f"\n{'Mode':<48} {'ADE':>8} {'FDE':>8} {'step':>8}")
        print("-" * 76)
        for rec in all_records:
            print(
                f"{rec['mode']:<48} {rec['ADE']:>8.4f} "
                f"{rec['FDE']:>8.4f} "
                f"{rec.get('mean_token_step', float('nan')):>8.4f}"
            )
        return

    if args.predroot_timestamped_ablation:
        print("\n=== Pred-root timestamped trajectory ablation ===")
        oracle_step = _compute_oracle_token_step(sample)
        token_dt = 4.0 / float(args.motion_fps)
        timestamped_traj = _make_gt_timestamped_traj(sample, motion_fps=args.motion_fps)
        duration_times, duration_waypoints, _ = _make_duration_waypoint_traj(
            sample,
            motion_fps=args.motion_fps,
            waypoint_stride_seconds=args.duration_waypoint_stride_seconds,
        )
        duration_traj = (duration_times, duration_waypoints)
        print(f"  oracle token_step={oracle_step:.4f}")
        print(f"  token_dt={token_dt:.4f}s  motion_fps={args.motion_fps:.2f}")
        print(
            "  duration-waypoint: "
            f"{len(duration_waypoints)} points, "
            f"stride={args.duration_waypoint_stride_seconds:.2f}s"
        )

        all_records = []

        # First add a GT-root ceiling under the same H-token horizon.
        print(f"\n--- gtroot_h{args.traj_horizon_tokens} ---")
        seed_everything(args.seed)
        pred_motion, pred_root = run_stream_step(
            model,
            vae,
            sample,
            device,
            history_length=args.history_length,
            num_denoise_steps=args.num_denoise_steps,
            horizon_tokens=args.traj_horizon_tokens,
            use_pred_root=False,
        )
        metrics = _build_mode_metrics(
            pred_root,
            gt_root,
            target_traj,
            f"gtroot_h{args.traj_horizon_tokens}",
            args.traj_horizon_tokens,
            "gt",
            traj_encoder_path="GT suffix xyz, clipped horizon",
        )
        print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
        mode_dir = os.path.join(out_root, metrics["mode"])
        _save_artifacts(
            mode_dir,
            pred_motion,
            pred_root,
            gt_root[: len(pred_root)],
            target_traj[: len(pred_root)],
            metrics,
        )
        all_records.append(metrics)

        mode_specs = [
            (
                f"predroot_current_h{args.traj_horizon_tokens}",
                "current_polyline",
                None,
                None,
                "web-demo current root + online token_step",
            ),
            (
                f"predroot_fixed_{args.timestamped_fixed_token_step:.4f}_h{args.traj_horizon_tokens}",
                "current_polyline",
                float(args.timestamped_fixed_token_step),
                None,
                "web-demo current root + fixed token_step",
            ),
            (
                f"predroot_oracle_h{args.traj_horizon_tokens}",
                "current_polyline",
                oracle_step,
                None,
                "web-demo current root + GT oracle token_step",
            ),
            (
                f"timestamped_gt_h{args.traj_horizon_tokens}",
                "timestamped",
                None,
                timestamped_traj,
                "GT timestamped root sampled at token times",
            ),
            (
                f"duration_waypoints_s{args.duration_waypoint_stride_seconds:g}_h{args.traj_horizon_tokens}",
                "timestamped",
                None,
                duration_traj,
                "coarse spatial waypoints + total duration, arclength timing",
            ),
        ]

        for mode_name, pred_mode, token_step_override, ts_traj, traj_path in mode_specs:
            print(f"\n--- {mode_name} ---")
            trace: list[dict] = []
            seed_everything(args.seed)
            pred_motion, pred_root = run_stream_step(
                model,
                vae,
                sample,
                device,
                history_length=args.history_length,
                num_denoise_steps=args.num_denoise_steps,
                horizon_tokens=args.traj_horizon_tokens,
                use_pred_root=True,
                predroot_mode=pred_mode,
                oracle_token_step=token_step_override,
                timestamped_traj=ts_traj,
                token_dt=token_dt,
                trace_sink=trace,
            )
            metrics = _build_mode_metrics(
                pred_root,
                gt_root,
                target_traj,
                mode_name,
                args.traj_horizon_tokens,
                "pred" if pred_mode != "timestamped" else "timestamped_gt",
                traj_encoder_path=traj_path,
            )
            metrics["predroot_mode"] = pred_mode
            metrics["fixed_token_step"] = (
                None if token_step_override is None else float(token_step_override)
            )
            metrics["token_dt"] = float(token_dt)
            if mode_name.startswith("duration_waypoints"):
                metrics["duration_waypoint_count"] = int(len(duration_waypoints))
                metrics["duration_waypoint_stride_seconds"] = float(
                    args.duration_waypoint_stride_seconds
                )
            if trace:
                metrics["mean_future_token_step"] = float(
                    np.mean([row["future_token_step"] for row in trace])
                )
                metrics["mean_cross_track_error"] = float(
                    np.mean([row["cross_track_error"] for row in trace])
                )
                metrics["max_cross_track_error"] = float(
                    np.max([row["cross_track_error"] for row in trace])
                )
            print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
            if trace:
                print(
                    "  future_step="
                    f"{metrics['mean_future_token_step']:.4f}  "
                    "cross_track="
                    f"{metrics['mean_cross_track_error']:.4f}/"
                    f"{metrics['max_cross_track_error']:.4f} mean/max"
                )
            mode_dir = os.path.join(out_root, mode_name)
            _save_artifacts(
                mode_dir,
                pred_motion,
                pred_root,
                gt_root[: len(pred_root)],
                target_traj[: len(pred_root)],
                metrics,
            )
            if args.render_video and pred_motion.size > 0:
                _mp4 = os.path.join(mode_dir, "pred_motion.mp4")
                render_single_video(
                    motion=pred_motion, save_path=_mp4,
                    dim=263, render_setting={},
                )
                print(f"    video saved to {_mp4}")
                # Also render trajectory comparison (pred + target).
                _n = min(len(pred_root), len(gt_root))
                if _n > 1:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    from matplotlib.animation import FFMpegWriter
                    _traj_png = os.path.join(mode_dir, "traj_compare.png")
                    _fig, _ax = plt.subplots(figsize=(6, 6))
                    _ax.plot(gt_root[:_n, 0], gt_root[:_n, 2], "g-", linewidth=1.5, alpha=0.7, label="target")
                    _ax.plot(pred_root[:_n, 0], pred_root[:_n, 2], "r-", linewidth=1.5, alpha=0.7, label="pred")
                    _ax.plot(gt_root[0, 0], gt_root[0, 2], "go", markersize=6, label="start")
                    _ax.plot(gt_root[_n-1, 0], gt_root[_n-1, 2], "gx", markersize=10, label="end")
                    _ax.set_aspect("equal")
                    _ax.legend()
                    _ax.set_title(f"{mode_name}")
                    _fig.savefig(_traj_png, dpi=100, bbox_inches="tight")
                    plt.close(_fig)
                    print(f"    traj plot saved to {_traj_png}")

                    # Animated trajectory growth video.
                    _traj_mp4 = os.path.join(mode_dir, "traj_compare.mp4")
                    _fig2, _ax2 = plt.subplots(figsize=(6, 6))
                    _all_x = [gt_root[:_n, 0], pred_root[:_n, 0]]
                    _all_z = [gt_root[:_n, 2], pred_root[:_n, 2]]
                    _xlim = (min(a.min() for a in _all_x) - 0.5, max(a.max() for a in _all_x) + 0.5)
                    _zlim = (min(a.min() for a in _all_z) - 0.5, max(a.max() for a in _all_z) + 0.5)
                    _writer = FFMpegWriter(fps=20)
                    _step = max(1, _n // 120)  # ~6s video
                    with _writer.saving(_fig2, _traj_mp4, dpi=100):
                        for _f in range(1, _n + 1, _step):
                            _ax2.clear()
                            _ax2.plot(gt_root[:min(_f, _n), 0], gt_root[:min(_f, _n), 2], "g-", linewidth=1.5, alpha=0.7, label="target")
                            _ax2.plot(pred_root[:min(_f, _n), 0], pred_root[:min(_f, _n), 2], "r-", linewidth=1.5, alpha=0.7, label="pred")
                            _ax2.plot(gt_root[0, 0], gt_root[0, 2], "go", markersize=6)
                            _ax2.plot(gt_root[min(_f-1, _n-1), 0], gt_root[min(_f-1, _n-1), 2], "g.", markersize=8)
                            _ax2.plot(pred_root[min(_f-1, _n-1), 0], pred_root[min(_f-1, _n-1), 2], "r.", markersize=8)
                            _ax2.set_xlim(_xlim)
                            _ax2.set_ylim(_zlim)
                            _ax2.set_aspect("equal")
                            _ax2.legend(loc="upper right")
                            _ax2.set_title(f"{mode_name}  frame {min(_f,_n)}/{_n}")
                            _writer.grab_frame()
                    plt.close(_fig2)
                    print(f"    traj video saved to {_traj_mp4}")
            with open(os.path.join(mode_dir, "predroot_trace.json"), "w") as f:
                json.dump(trace, f, indent=2, default=str)
            all_records.append(metrics)

        summary = {
            "sample_id": args.sample_id,
            "dataset": args.dataset,
            "ckpt": args.ckpt,
            "vae_ckpt": args.vae_ckpt,
            "config": args.config,
            "ablation": "predroot_timestamped",
            "history_length": args.history_length,
            "traj_horizon_tokens": args.traj_horizon_tokens,
            "motion_fps": args.motion_fps,
            "token_dt": token_dt,
            "oracle_token_step": oracle_step,
            "timestamped_fixed_token_step": args.timestamped_fixed_token_step,
            "duration_waypoint_stride_seconds": args.duration_waypoint_stride_seconds,
            "duration_waypoint_count": int(len(duration_waypoints)),
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_predroot_timestamped.json")
        os.makedirs(out_root, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")
        print(f"\n{'Mode':<48} {'ADE':>8} {'FDE':>8} {'step':>8} {'XTE':>8}")
        print("-" * 86)
        for rec in all_records:
            print(
                f"{rec['mode']:<48} {rec['ADE']:>8.4f} "
                f"{rec['FDE']:>8.4f} "
                f"{rec.get('mean_future_token_step', float('nan')):>8.4f} "
                f"{rec.get('mean_cross_track_error', float('nan')):>8.4f}"
            )
        return

    if args.mid_session_update:
        print(f"\n=== Mid-session trajectory update (split at token "
              f"{args.split_token}) ===")
        tl = sample["token_length"]
        split_tok = max(5, min(args.split_token, tl - args.traj_horizon_tokens - 1))
        gt_root_arr = sample["traj"].numpy()
        total_f = len(gt_root_arr)

        # Time-based waypoints: original 20fps GT frames, t = frame_idx / 20.
        _time_pts = gt_root_arr
        _time_t = np.arange(total_f, dtype=np.float32) / 20.0
        # Arc-length waypoints: uniformly spaced along path, same count.
        segs = np.linalg.norm(np.diff(gt_root_arr[:, [0, 2]], axis=0), axis=1)
        cum_arc = np.concatenate([np.zeros(1), np.cumsum(segs)])
        total_arc = cum_arc[-1]
        num_arc = max(2, total_f)
        _arc_pts = np.column_stack([
            np.interp(np.linspace(0, total_arc, num_arc), cum_arc, gt_root_arr[:, d])
            for d in range(3)
        ]).astype(np.float32)
        _arc_t = np.arange(num_arc, dtype=np.float32) / 20.0

        all_records = []
        for _mode_label, _update_at, _wp, _times in [
            ("time_cont", None, _time_pts, _time_t),
            ("time_split", split_tok, _time_pts, _time_t),
            ("arc_cont", None, _arc_pts, _arc_t),
            ("arc_split", split_tok, _arc_pts, _arc_t),
        ]:
            total_frames = 1 + 4 * (tl - 1) if tl > 1 else 1
            vae.clear_cache()
            model.init_generated(args.history_length, batch_size=1,
                                num_denoise_steps=args.num_denoise_steps)
            model.generated = model.generated.to(device)
            stream_rec = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
            all_dec, all_pr = [], []
            first_chunk = True
            _plan_v = 0

            for ci in range(tl):
                if _update_at is not None and ci == _update_at:
                    _plan_v += 1
                _plan_start = 0 if _plan_v == 0 else split_tok
                _elapsed = ci - _plan_start
                qt = (float(_elapsed) * args.token_dt
                      + np.arange(args.traj_horizon_tokens, dtype=np.float32) * args.token_dt)
                ft = sample_timestamped_trajectory(_times, _wp, qt)
                # Pred root alignment: same as web_demo mid-session update.
                cur_root = np.zeros(3, dtype=np.float32)
                cur_root[[0, 2]] = stream_rec.r_pos_accum[[0, 2]].astype(np.float32)
                anchor = sample_timestamped_trajectory(
                    _times, _wp, np.asarray([qt[0]], dtype=np.float32))[0]
                ft = cur_root + (ft - anchor.astype(np.float32))

                ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0),
                      "token_mask": torch.ones(1, args.traj_horizon_tokens)}
                sp = build_stream_step_model_input(
                    sample["text"] if isinstance(sample["text"], str) else sample["text"][0],
                    traj_input=ti)
                out = model.stream_generate_step(sp, first_chunk=first_chunk)
                dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                         first_chunk=first_chunk)[0].float().cpu().numpy())
                first_chunk = False
                for frm in dec:
                    stream_rec.process_frame(frm)
                all_dec.append(dec)
                all_pr.append(extract_root_trajectory_263(dec))

            vae.clear_cache()
            pred_motion = (np.concatenate(all_dec, axis=0)[:total_frames]
                           if all_dec else np.zeros((0, 263)))
            pred_root = (np.concatenate(all_pr, axis=0)[:total_frames]
                         if all_pr else np.zeros((0, 3)))

            # Per-segment metrics.
            pre_root, gt_pre = pred_root[:split_tok], gt_root[:split_tok]
            post_root, gt_post = (pred_root[split_tok:split_tok * 2],
                                  gt_root[split_tok:split_tok * 2])
            metrics = _build_mode_metrics(
                pred_root, gt_root, target_traj,
                f"mid_update_{_mode_label}_split{args.split_token}", None, "gt",
                traj_encoder_path=f"timestamped, split={_update_at}")
            metrics["ADE_pre_split"] = _compute_ade(pre_root, gt_pre) if len(pre_root) > 1 else float("nan")
            metrics["ADE_post_split"] = _compute_ade(post_root, gt_post) if len(post_root) > 1 else float("nan")
            # Path-based metrics (arc-length reparam / chamfer).
            metrics["path_arc_ade"] = _path_arc_ade(pred_root, gt_root)
            metrics["path_chamfer"] = _path_chamfer(pred_root, gt_root)
            print(f"  ADE={metrics['ADE']:.4f}  pre={metrics['ADE_pre_split']:.4f}  "
                  f"post={metrics['ADE_post_split']:.4f}  "
                  f"arc_ADE={metrics['path_arc_ade']:.4f}  "
                  f"chamfer={metrics['path_chamfer']:.4f}")

            mode_dir = os.path.join(out_root, f"mid_update_{_mode_label}")
            _save_artifacts(mode_dir, pred_motion, pred_root,
                            gt_root[:len(pred_root)], target_traj[:len(pred_root)], metrics)
            if args.render_video and pred_motion.size > 0:
                _mp4 = os.path.join(mode_dir, "pred_motion.mp4")
                render_single_video(motion=pred_motion, save_path=_mp4,
                                    dim=263, render_setting={})
                print(f"    video saved to {_mp4}")
            all_records.append(metrics)

        summary = {
            "sample_id": args.sample_id, "split_token": split_tok,
            "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt,
            "ablation": "mid_session_update",
            "modes": {m["mode"]: {k: v for k, v in m.items()} for m in all_records},
        }
        summary_path = os.path.join(out_root, "summary_mid_update.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n{'Mode':<38} {'ADE':>8} {'pre':>8} {'post':>8} {'arc_ADE':>8} {'chamfer':>8}")
        print("-" * 82)
        for m in all_records:
            print(f"{m['mode']:<38} {m['ADE']:>8.4f} "
                  f"{m.get('ADE_pre_split', float('nan')):>8.4f} "
                  f"{m.get('ADE_post_split', float('nan')):>8.4f} "
                  f"{m.get('path_arc_ade', float('nan')):>8.4f} "
                  f"{m.get('path_chamfer', float('nan')):>8.4f}")
        return

    if args.synthetic_trail:
        print(f"\n=== Synthetic trail extension (angle={args.trail_angle}°, "
              f"tokens={args.trail_tokens}) ===")
        tl = sample["token_length"]
        total_frames = 1 + 4 * (tl - 1) if tl > 1 else 1
        extra_tokens = args.trail_tokens

        vae.clear_cache()
        model.init_generated(args.history_length, batch_size=1,
                            num_denoise_steps=args.num_denoise_steps)
        model.generated = model.generated.to(device)
        stream_rec = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
        all_dec, all_pr = [], []
        first_chunk = True

        # Get GT heading at last frame for trail direction.
        gt_root_arr = sample["traj"].numpy()
        last_gt_root = gt_root_arr[-1]
        # Heading from last two frames.
        d_gt = gt_root_arr[-1, [0, 2]] - gt_root_arr[max(0, len(gt_root_arr) - 5), [0, 2]]
        gt_heading = np.arctan2(float(d_gt[1]), float(d_gt[0]))
        trail_heading = gt_heading + np.deg2rad(args.trail_angle)

        # Generate through the full sample first (GT suffix).
        for ci in range(tl):
            _bs = _wrap_flat_sample_for_suffix(sample)
            ti = build_stream_suffix_conditioning(_bs, ci, prefer_xyz=True)
            sp = build_stream_step_model_input(
                sample["text"] if isinstance(sample["text"], str) else sample["text"][0],
                traj_input=ti)
            out = model.stream_generate_step(sp, first_chunk=first_chunk)
            dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                     first_chunk=first_chunk)[0].float().cpu().numpy())
            first_chunk = False
            for frm in dec:
                stream_rec.process_frame(frm)
            all_dec.append(dec)
            all_pr.append(extract_root_trajectory_263(dec))

        # Extend with synthetic trail.
        cur_root = np.zeros(3, dtype=np.float32)
        cur_root[[0, 2]] = stream_rec.r_pos_accum[[0, 2]].astype(np.float32)
        trail_step = 0.25  # meters per token
        trail_pts = np.array([
            [cur_root[0] + i * trail_step * np.cos(trail_heading),
             0.0,
             cur_root[2] + i * trail_step * np.sin(trail_heading)]
            for i in range(extra_tokens + 1)
        ], dtype=np.float32)
        trail_times = np.arange(len(trail_pts), dtype=np.float32) * args.token_dt

        for ci in range(extra_tokens):
            qt = (float(ci) * args.token_dt
                  + np.arange(args.traj_horizon_tokens, dtype=np.float32) * args.token_dt)
            ft = sample_timestamped_trajectory(trail_times, trail_pts, qt)
            # Align to predicted root.
            croot = np.zeros(3, dtype=np.float32)
            croot[[0, 2]] = stream_rec.r_pos_accum[[0, 2]].astype(np.float32)
            anchor = sample_timestamped_trajectory(
                trail_times, trail_pts, np.asarray([qt[0]], dtype=np.float32))[0]
            ft = croot + (ft - anchor.astype(np.float32))
            ti = {"traj": torch.from_numpy(ft).float().unsqueeze(0),
                  "token_mask": torch.ones(1, args.traj_horizon_tokens)}
            sp = build_stream_step_model_input(
                sample["text"] if isinstance(sample["text"], str) else sample["text"][0],
                traj_input=ti)
            out = model.stream_generate_step(sp, first_chunk=first_chunk)
            dec = (vae.stream_decode(out["generated"][0][None, :].to(device),
                                     first_chunk=first_chunk)[0].float().cpu().numpy())
            first_chunk = False
            for frm in dec:
                stream_rec.process_frame(frm)
            all_dec.append(dec)
            all_pr.append(extract_root_trajectory_263(dec))

        vae.clear_cache()
        pred_motion = (np.concatenate(all_dec, axis=0)[:total_frames + extra_tokens * 4]
                       if all_dec else np.zeros((0, 263)))
        pred_root = (np.concatenate(all_pr, axis=0)[:total_frames + extra_tokens * 4]
                     if all_pr else np.zeros((0, 3)))

        # GT for the sample portion only.
        gt_root_sample = extract_root_trajectory_263(sample["feature"].numpy())
        sample_ade = _compute_ade(pred_root[:len(gt_root_sample)], gt_root_sample)
        # Trail: measure how well pred follows the straight line.
        trail_pred = pred_root[len(gt_root_sample):]
        if len(trail_pred) > 1:
            trail_start = trail_pts[0, [0, 2]]
            trail_dir = np.array([np.cos(trail_heading), np.sin(trail_heading)])
            trail_lateral = [
                np.abs(np.cross(trail_dir, p[[0, 2]] - trail_start))
                for p in trail_pred
            ]
            trail_lateral_mean = float(np.mean(trail_lateral))
            trail_dist = [
                float(np.linalg.norm(p[[0, 2]] - trail_start))
                for p in trail_pred
            ]
        else:
            trail_lateral_mean = float("nan")

        print(f"  sample_ADE={sample_ade:.4f}  trail_lateral_mean={trail_lateral_mean:.4f}")

        mode_dir = os.path.join(out_root, f"synthetic_trail_{int(args.trail_angle)}deg")
        _save_artifacts(mode_dir, pred_motion, pred_root,
                        gt_root_sample, target_traj[:len(gt_root_sample)],
                        {"mode": f"synthetic_trail_{int(args.trail_angle)}deg",
                         "sample_ADE": sample_ade,
                         "trail_lateral_mean": trail_lateral_mean,
                         "trail_angle": args.trail_angle,
                         "trail_tokens": extra_tokens})
        # Path metrics on trail segment.
        trail_chamfer = _path_chamfer(trail_pred, trail_pts) if len(trail_pred) > 1 else float("nan")
        trail_arc_ade = _path_arc_ade(trail_pred, trail_pts) if len(trail_pred) > 1 else float("nan")
        print(f"  trail_chamfer={trail_chamfer:.4f}  trail_arc_ADE={trail_arc_ade:.4f}")

        # Build full target for trajectory comparison: GT sample + synthetic trail.
        _gt_sample_root = gt_root_sample
        _full_target = np.concatenate([_gt_sample_root, trail_pts], axis=0)

        if args.render_video and pred_motion.size > 0:
            _mp4 = os.path.join(mode_dir, "pred_motion.mp4")
            render_single_video(motion=pred_motion, save_path=_mp4,
                                dim=263, render_setting={})
            print(f"    video saved to {_mp4}")
            # Trajectory comparison video.
            _n = min(len(pred_root), len(_full_target))
            if _n > 1:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from matplotlib.animation import FFMpegWriter
                _t_mp4 = os.path.join(mode_dir, "traj_compare.mp4")
                _f2, _a2 = plt.subplots(figsize=(7, 7))
                _all_x = [_full_target[:_n, 0], pred_root[:_n, 0]]
                _all_z = [_full_target[:_n, 2], pred_root[:_n, 2]]
                _xl = (min(a.min() for a in _all_x) - 0.5, max(a.max() for a in _all_x) + 0.5)
                _zl = (min(a.min() for a in _all_z) - 0.5, max(a.max() for a in _all_z) + 0.5)
                _wr = FFMpegWriter(fps=20)
                _sf = max(1, _n // 150)
                with _wr.saving(_f2, _t_mp4, dpi=100):
                    for _f in range(1, _n + 1, _sf):
                        _a2.clear()
                        _a2.plot(_full_target[:min(_f, _n), 0], _full_target[:min(_f, _n), 2],
                                 "g-", lw=1.5, alpha=0.7, label="target")
                        _a2.plot(pred_root[:min(_f, _n), 0], pred_root[:min(_f, _n), 2],
                                 "r-", lw=1.5, alpha=0.7, label="pred")
                        _a2.plot(_full_target[0, 0], _full_target[0, 2], "go", ms=6)
                        # Mark sample/trail boundary.
                        _bd = len(_gt_sample_root)
                        if 0 < _bd < _n:
                            _a2.axvline(x=_full_target[min(_bd, _n - 1), 0],
                                        color="gray", ls="--", alpha=0.5, label="boundary")
                        _a2.set_xlim(_xl)
                        _a2.set_ylim(_zl)
                        _a2.set_aspect("equal")
                        _a2.legend(loc="upper right")
                        _a2.set_title(f"trail {int(args.trail_angle)}deg  f{min(_f,_n)}/{_n}")
                        _wr.grab_frame()
                plt.close(_f2)
                print(f"    traj video saved to {_t_mp4}")
        return

    # Encoding path labels per mode family.
    _TPATH_GENERATE = (
        "traj_features (frame-level) -> _build_traj_emb"
        " -> frames_to_tokens -> LocalTrajEncoder -> TrajEncoder"
        " (no anchor-subtract)"
    )
    _TPATH_STEP = (
        "traj (xyz) -> TrajStreamBuffer._build_from_xyz"
        " (anchor-subtract) -> token-level -> frames_to_tokens"
        " -> LocalTrajEncoder -> TrajEncoder"
    )

    # Diagnostic matrix: (mode_name, horizon, root_source, use_pred_root, traj_encoder_path)
    modes = [
        ("generate_full", None, "gt", False, _TPATH_GENERATE),
        ("stream_generate_full", None, "gt", False, _TPATH_GENERATE),
        ("step_full_xyz_gtroot", None, "gt", False, _TPATH_STEP),
        ("step_full_xyz_predroot", None, "pred", True, _TPATH_STEP),
        (f"step_h{args.traj_horizon_tokens}_xyz_gtroot",
         args.traj_horizon_tokens, "gt", False, _TPATH_STEP),
        (f"step_h{args.traj_horizon_tokens}_xyz_predroot",
         args.traj_horizon_tokens, "pred", True, _TPATH_STEP),
    ]

    all_records = []
    for mode_name, horizon, root_source, use_pred_root, traj_path in modes:
        print(f"\n--- {mode_name} ---")
        if mode_name == "generate_full":
            pred_motion, pred_root = run_generate_full(model, vae, sample, device)
        elif mode_name == "stream_generate_full":
            pred_motion, pred_root = run_stream_generate_full(model, vae, sample, device)
        else:
            pred_motion, pred_root = run_stream_step(
                model, vae, sample, device,
                history_length=args.history_length,
                num_denoise_steps=args.num_denoise_steps,
                horizon_tokens=horizon,
                use_pred_root=use_pred_root,
            )

        metrics = _build_mode_metrics(
            pred_root, gt_root, target_traj, mode_name, horizon, root_source,
            traj_encoder_path=traj_path,
        )
        print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")

        mode_dir = os.path.join(out_root, mode_name)
        _save_artifacts(
            mode_dir, pred_motion, pred_root,
            gt_root[: len(pred_root)], target_traj[: len(pred_root)], metrics,
        )
        all_records.append(metrics)

    # Summary
    summary = {
        "sample_id": args.sample_id,
        "dataset": args.dataset,
        "ckpt": args.ckpt,
        "vae_ckpt": args.vae_ckpt,
        "config": args.config,
        "history_length": args.history_length,
        "traj_horizon_tokens": args.traj_horizon_tokens,
        "num_denoise_steps": args.num_denoise_steps,
        "seed": args.seed,
        "modes": all_records,
    }
    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    print("\nDiagnostic Matrix:")
    print(f"{'Mode':<35} {'ADE':>8} {'FDE':>8}")
    print("-" * 53)
    for rec in all_records:
        print(f"{rec['mode']:<35} {rec['ADE']:>8.4f} {rec['FDE']:>8.4f}")


if __name__ == "__main__":
    main()
