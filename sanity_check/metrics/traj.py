"""
metrics/traj.py
===============
Reusable trajectory and control metric computation functions.

Shared by:
  - eval/eval_generation_metrics.py  (CLI offline evaluation)
  - eval/eval_runner.py        (inline training evaluation)
  - eval/eval_summary.py       (summary aggregation)
"""
import hashlib
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from utils.motion_process import extract_root_trajectory_263_torch, recover_joint_positions_263


# ─────────────────────────────────────────────────────────────────────────────
# Seed helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stable_eval_seed(base_seed: int, probe_tag: str, sample_name: str, run_idx: int) -> int:
    digest = hashlib.md5(f"{probe_tag}:{sample_name}:{run_idx}".encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return int(base_seed) + offset


def _seed_eval_locally(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    torch.random.set_rng_state(gen.get_state())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))


# ─────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_device(obj: Any, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _slice_single_sample_batch(batch: Dict, sample_idx: int) -> Dict:
    sample_batch: Dict = {}
    batch_size = len(batch["name"])
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] == batch_size:
                sample_batch[key] = value[sample_idx : sample_idx + 1]
            else:
                sample_batch[key] = value
        elif isinstance(value, list):
            if len(value) == batch_size:
                sample_batch[key] = [value[sample_idx]]
            else:
                sample_batch[key] = value
        else:
            sample_batch[key] = value
    return sample_batch


def _copy_traj_fields(batch: Dict, model_batch: Dict):
    for key in ("traj", "traj_length", "traj_mask", "token_mask", "traj_features"):
        if key in batch:
            model_batch[key] = batch[key]


def _build_model_batch(batch: Dict, device: torch.device) -> Dict:
    mb = batch.copy()
    mb["feature"] = batch["token"]
    mb["feature_length"] = batch["token_length"]
    if "token_text_end" in batch:
        mb["feature_text_end"] = batch["token_text_end"]
    _copy_traj_fields(batch, mb)
    return _to_device(mb, device)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_metric_statistics(
    values: np.ndarray, replication_times: int
) -> tuple:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(max(replication_times, 1))
    return mean, std, conf_interval


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory metrics
# ─────────────────────────────────────────────────────────────────────────────

def _moving_average_same(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def _resample_polyline_by_arclen(points: np.ndarray, num_samples: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected path points with shape (T,2), got {points.shape}")
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    num_samples = max(int(num_samples), 1)
    if len(points) == 1:
        return np.repeat(points[:1], num_samples, axis=0)

    seg_lens = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumlen = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(seg_lens, dtype=np.float32)], axis=0
    )
    total_len = float(cumlen[-1])
    if total_len < 1e-8:
        return np.repeat(points[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total_len, num_samples, dtype=np.float32)
    x = np.interp(targets, cumlen, points[:, 0])
    z = np.interp(targets, cumlen, points[:, 1])
    return np.stack([x, z], axis=-1).astype(np.float32)


def _compute_path_only_metrics(pred_path: np.ndarray, gt_path: np.ndarray) -> Dict:
    """
    Spatial-only path similarity (arc-length reparameterized ADE + Chamfer).
    Removes timing/speed differences; focuses on path shape.
    """
    pred_path = np.asarray(pred_path, dtype=np.float32)
    gt_path = np.asarray(gt_path, dtype=np.float32)
    if (
        pred_path.ndim != 2 or gt_path.ndim != 2
        or pred_path.shape[1] != 2 or gt_path.shape[1] != 2
    ):
        return {"path_arc_ade": float("nan"), "path_chamfer": float("nan")}
    if len(pred_path) == 0 or len(gt_path) == 0:
        return {"path_arc_ade": float("nan"), "path_chamfer": float("nan")}

    num_samples = int(min(max(len(pred_path), len(gt_path)), 256))
    pred_arc = _resample_polyline_by_arclen(pred_path, num_samples)
    gt_arc = _resample_polyline_by_arclen(gt_path, num_samples)

    diff = pred_arc - gt_arc
    path_arc_ade = float(np.linalg.norm(diff, axis=-1).mean())

    dmat = np.linalg.norm(pred_arc[:, None, :] - gt_arc[None, :, :], axis=-1)
    path_chamfer = float(0.5 * (dmat.min(axis=1).mean() + dmat.min(axis=0).mean()))

    return {"path_arc_ade": path_arc_ade, "path_chamfer": path_chamfer}


def _calculate_skating_ratio_from_joints(joints_xyz: np.ndarray) -> float:
    """HumanML3D-compatible skating ratio (OmniControl thresholds)."""
    if joints_xyz.ndim != 3 or joints_xyz.shape[0] < 2:
        return float("nan")
    fps = 20.0
    thresh_height = 0.05
    thresh_vel = 0.50
    avg_window = 5
    if joints_xyz.shape[1] == 22:
        foot_ids = [10, 11]
    elif joints_xyz.shape[1] == 21:
        foot_ids = [15, 20]
    else:
        return float("nan")

    feet = joints_xyz[:, foot_ids, :]  # (T, 2, 3)
    plane_vel = (
        np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1) * fps
    )  # (T-1, 2)
    vel_avg = np.stack(
        [_moving_average_same(plane_vel[:, j], avg_window) for j in range(plane_vel.shape[1])],
        axis=1,
    )
    feet_height = feet[:, :, 1]  # (T, 2)
    feet_contact = np.logical_and(
        feet_height[:-1] < thresh_height, feet_height[1:] < thresh_height
    )  # (T-1, 2)
    skating = np.logical_and(feet_contact, plane_vel > thresh_vel)
    skating = np.logical_and(skating, vel_avg > thresh_vel)
    skating = np.logical_or(skating[:, 0], skating[:, 1])  # (T-1,)
    return float(skating.mean()) if skating.size > 0 else float("nan")


def _compute_traj_metrics(
    decoded_generated: torch.Tensor,  # (T_pred, 263)
    batch: Dict,
    sample_idx: int,
    seg_size: int,
) -> Dict:
    """Return dict with time-aligned traj metrics plus path-only spatial metrics."""
    with torch.no_grad():
        pred_traj_xyz = extract_root_trajectory_263_torch(
            decoded_generated[None, :]
        )[0]  # (T_pred, 3)
        pred_xz = pred_traj_xyz[:, [0, 2]].cpu()  # (T_pred, 2)

    traj_len = (
        int(batch["traj_length"][sample_idx].item())
        if "traj_length" in batch
        else batch["traj"][sample_idx].shape[0]
    )
    gt_xz = batch["traj"][sample_idx][:traj_len, [0, 2]].float().cpu()  # (T_gt, 2)
    mask = batch["traj_mask"][sample_idx][:traj_len].float().cpu()  # (T_gt,)

    T = min(pred_xz.shape[0], gt_xz.shape[0])
    pred_xz, gt_xz, mask = pred_xz[:T], gt_xz[:T], mask[:T]
    n_masked = mask.sum().item()

    result: Dict = {"T": T, "masked_ratio": n_masked / max(T, 1)}

    if n_masked > 0:
        diff = pred_xz - gt_xz  # (T, 2)
        l2_t = diff.norm(dim=-1)  # (T,)
        sq_t = (diff ** 2).sum(dim=-1)  # (T,)
        result["ade"] = float((mask * l2_t).sum().item() / n_masked)
        result["mse"] = float((mask * sq_t).sum().item() / n_masked)
        last_idx = mask.nonzero(as_tuple=False)[-1].item()
        result["fde"] = float(l2_t[last_idx].item())
    else:
        result["ade"] = result["fde"] = result["mse"] = float("nan")

    mask_bool = mask > 0
    if int(mask_bool.sum().item()) > 0:
        pred_path = pred_xz[mask_bool].numpy().astype(np.float32)
        gt_path = gt_xz[mask_bool].numpy().astype(np.float32)
        result.update(_compute_path_only_metrics(pred_path, gt_path))
    else:
        result["path_arc_ade"] = float("nan")
        result["path_chamfer"] = float("nan")

    # Segment MSE: non-overlapping windows of seg_size frames
    seg_mse: List[Optional[float]] = []
    n_segs = (T + seg_size - 1) // seg_size
    for s in range(n_segs):
        sf, ef = s * seg_size, min((s + 1) * seg_size, T)
        m_s = mask[sf:ef]
        n_s = m_s.sum().item()
        if n_s > 0:
            sq = ((pred_xz[sf:ef] - gt_xz[sf:ef]) ** 2).sum(dim=-1)
            seg_mse.append(float((m_s * sq).sum().item() / n_s))
        else:
            seg_mse.append(None)
    result["seg_mse"] = seg_mse

    # Trajectory jitter: mean squared acceleration
    if pred_xz.shape[0] >= 3:
        accel = pred_xz[2:] - 2 * pred_xz[1:-1] + pred_xz[:-2]  # (T-2, 2)
        result["traj_jitter"] = float(accel.pow(2).sum(dim=-1).mean().item())
    else:
        result["traj_jitter"] = float("nan")

    # Prefix MSE: cumulative [0:ef] at ef = seg_size, 2*seg_size, ...
    prefix_mse: List[Optional[float]] = []
    for ef in range(seg_size, T + 1, seg_size):
        m_p = mask[:ef]
        n_p = m_p.sum().item()
        if n_p > 0:
            sq = ((pred_xz[:ef] - gt_xz[:ef]) ** 2).sum(dim=-1)
            prefix_mse.append(float((m_p * sq).sum().item() / n_p))
        else:
            prefix_mse.append(None)
    result["prefix_mse"] = prefix_mse

    return result


def _average_traj_metrics(run_metrics: List[Dict]) -> Dict:
    """Average _compute_traj_metrics dicts across multiple runs."""
    if len(run_metrics) == 1:
        return run_metrics[0].copy()
    result: Dict = {}
    if "T" in run_metrics[0]:
        result["T"] = run_metrics[0]["T"]
    if "masked_ratio" in run_metrics[0]:
        result["masked_ratio"] = run_metrics[0]["masked_ratio"]
    for key in ("ade", "fde", "mse", "traj_jitter", "path_arc_ade", "path_chamfer"):
        vals = [r[key] for r in run_metrics if key in r and r[key] == r[key]]
        if vals:
            result[key] = float(np.mean(vals))
            result[f"{key}_std"] = float(np.std(vals))
    for list_key in ("seg_mse", "prefix_mse"):
        if list_key not in run_metrics[0]:
            continue
        n = max(len(r.get(list_key, [])) for r in run_metrics)
        avg = []
        for s in range(n):
            vals = [
                r[list_key][s]
                for r in run_metrics
                if s < len(r.get(list_key, [])) and r[list_key][s] is not None
            ]
            avg.append(float(np.mean(vals)) if vals else None)
        result[list_key] = avg
    return result


def _compute_omni_control_metrics(
    decoded_generated: torch.Tensor,  # (T_pred, 263)
    batch: Dict,
    sample_idx: int,
) -> Dict:
    """OmniControl / MotionLCM-compatible pelvis-control metrics."""
    with torch.no_grad():
        pred_traj_xyz = extract_root_trajectory_263_torch(
            decoded_generated[None, :]
        )[0].cpu()  # (T_pred, 3)

    traj_len = (
        int(batch["traj_length"][sample_idx].item())
        if "traj_length" in batch
        else batch["traj"][sample_idx].shape[0]
    )
    gt_traj_xyz = batch["traj"][sample_idx][:traj_len].float().cpu()
    mask = batch["traj_mask"][sample_idx][:traj_len].float().cpu() > 0

    T = min(pred_traj_xyz.shape[0], gt_traj_xyz.shape[0])
    pred_traj_xyz = pred_traj_xyz[:T]
    gt_traj_xyz = gt_traj_xyz[:T]
    mask = mask[:T]
    n_masked = int(mask.sum().item())

    result: Dict = {
        "control_T": T,
        "control_masked_ratio": float(n_masked / max(T, 1)),
    }
    if n_masked > 0:
        dist_error = (pred_traj_xyz - gt_traj_xyz).norm(dim=-1)[mask].numpy()
        mean_error = float(dist_error.mean())
        result["control_l2_dist"] = mean_error
        result["traj_fail_20cm"] = float(1.0 - float((dist_error <= 0.2).all()))
        result["traj_fail_50cm"] = float(1.0 - float((dist_error <= 0.5).all()))
        result["kps_fail_20cm"] = float((dist_error > 0.2).mean())
        result["kps_fail_50cm"] = float((dist_error > 0.5).mean())
        result["kps_mean_err_m"] = mean_error
    else:
        for key in (
            "control_l2_dist",
            "traj_fail_20cm",
            "traj_fail_50cm",
            "kps_fail_20cm",
            "kps_fail_50cm",
            "kps_mean_err_m",
        ):
            result[key] = float("nan")

    try:
        joints_np = recover_joint_positions_263(
            decoded_generated.detach().cpu().numpy(), joints_num=22
        )
        result["skating_ratio"] = _calculate_skating_ratio_from_joints(joints_np)
    except Exception:
        result["skating_ratio"] = float("nan")

    return result


def _average_control_metrics(run_metrics: List[Dict]) -> Dict:
    """Average OmniControl-compatible scalar metrics across multiple runs."""
    if len(run_metrics) == 1:
        return run_metrics[0].copy()
    result: Dict = {}
    for key in (
        "control_l2_dist",
        "skating_ratio",
        "traj_fail_20cm",
        "traj_fail_50cm",
        "kps_fail_20cm",
        "kps_fail_50cm",
        "kps_mean_err_m",
    ):
        vals = [r[key] for r in run_metrics if key in r and r[key] == r[key]]
        if vals:
            result[key] = float(np.mean(vals))
            result[f"{key}_std"] = float(np.std(vals))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Forward control loss (training-equivalent active-window XZ loss)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fwd_ctrl_loss_per_sample(
    pred_list: List[torch.Tensor],
    traj: torch.Tensor,
    traj_mask: torch.Tensor,
    traj_length: torch.Tensor,
    vae,
    device: torch.device,
    train_mode: int = 3,
    chunk_size_tokens: Optional[int] = None,
    token_to_frame: int = 4,
) -> List[Dict]:
    """XZ control loss per sample, matching train_ldf::_compute_control_loss_xz semantics."""
    out = []
    use_active_window = train_mode in (1, 2, 5, 6)
    detach_past = train_mode in (2, 4)
    relative_disp = train_mode in (5, 6)
    relative_disp_gt_anchor = train_mode == 6
    for i in range(len(pred_list)):
        pred_latent_full = pred_list[i].to(device)
        t_tok = pred_latent_full.size(0)

        if chunk_size_tokens is not None and t_tok > chunk_size_tokens:
            start_tok = t_tok - chunk_size_tokens
            start_f = 0 if start_tok == 0 else 4 * start_tok - 3
            end_f = t_tok * token_to_frame
        else:
            start_tok = 0
            start_f = 0
            end_f = None

        if detach_past and start_tok > 0:
            latent_for_decode = torch.cat(
                [pred_latent_full[:start_tok].detach(), pred_latent_full[start_tok:]], dim=0
            )
        else:
            latent_for_decode = pred_latent_full

        decoded = vae.decode(latent_for_decode.unsqueeze(0))[0].float()
        l_motion = decoded.size(0)
        l_gt_total = min(int(traj_length[i].item()), traj.shape[1])

        if use_active_window and end_f is not None:
            pred_sl = slice(min(start_f, l_motion), min(end_f, l_motion))
            gt_sl = slice(min(start_f, l_gt_total), min(end_f, l_gt_total))
        else:
            pred_sl = slice(0, l_motion)
            gt_sl = slice(0, l_gt_total)

        l = min(pred_sl.stop - pred_sl.start, gt_sl.stop - gt_sl.start)
        if l <= 0:
            out.append({"loss": float("nan"), "n_valid": 0, "window_len": 0})
            continue

        pred_traj_full = extract_root_trajectory_263_torch(decoded.unsqueeze(0))
        pred_traj = pred_traj_full[:, pred_sl, :][:, :l, :]
        gt_traj = (
            traj[i, gt_sl, :][:l]
            .unsqueeze(0)
            .to(pred_traj.device, dtype=pred_traj.dtype)
        )
        mask = (
            traj_mask[i, gt_sl][:l]
            .unsqueeze(0)
            .to(pred_traj.device, dtype=pred_traj.dtype)
        )

        pred_xz = pred_traj[..., [0, 2]]
        gt_xz = gt_traj[..., [0, 2]]

        if relative_disp:
            if relative_disp_gt_anchor:
                gt_anchor = gt_xz[:, 0:1, :].detach()
                pred_xz = pred_xz - pred_xz[:, 0:1, :].detach()
                gt_xz = gt_xz - gt_anchor
            else:
                anchor = pred_xz[:, 0:1, :].detach()
                pred_xz = pred_xz - anchor
                gt_xz = gt_xz - gt_xz[:, 0:1, :]

        sq_err = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
        n_valid = float(mask.sum().item())
        loss_val = (
            float((mask * sq_err).sum().item() / n_valid) if n_valid > 0 else float("nan")
        )
        out.append({"loss": loss_val, "n_valid": n_valid, "window_len": l})
    return out


def _iter_deterministic_time_steps(
    valid_len: int, chunk_size: int, mode: str = "mean_chunk_windows"
) -> List[float]:
    valid_len = int(valid_len)
    if valid_len <= 0:
        return []
    if mode == "last_window":
        return [float((valid_len - 1) / chunk_size)]

    end_indices = list(range(1, valid_len + 1, chunk_size))
    if end_indices[-1] != valid_len:
        end_indices.append(valid_len)
    return [float((end_idx - 1) / chunk_size) for end_idx in end_indices]


def _compute_deterministic_fwd_ctrl_loss_sample(
    model,
    sample_batch: Dict,
    vae,
    device: torch.device,
    train_mode: int,
    chunk_size_tokens: Optional[int] = None,
    window_mode: str = "mean_chunk_windows",
) -> Dict:
    valid_len = int(sample_batch["token_length"][0].item())
    time_steps = _iter_deterministic_time_steps(valid_len, model.chunk_size, mode=window_mode)
    if not time_steps:
        return {
            "loss": float("nan"),
            "loss_std": float("nan"),
            "n_valid": 0.0,
            "window_len": 0.0,
            "num_windows": 0,
        }

    losses = []
    n_valids = []
    win_lens = []
    for t in time_steps:
        model_batch = _build_model_batch(sample_batch, device)
        model_batch["_time_steps_override"] = torch.tensor(
            [t], device=device, dtype=torch.float32
        )
        with torch.no_grad():
            fwd_out = model(model_batch)
        if "control_aux" not in fwd_out:
            continue
        pred_list = fwd_out["control_aux"]["pred_x0_latent_list"]
        stats = _compute_fwd_ctrl_loss_per_sample(
            pred_list=pred_list,
            traj=model_batch["traj"],
            traj_mask=model_batch["traj_mask"],
            traj_length=model_batch["traj_length"],
            vae=vae,
            device=device,
            train_mode=train_mode,
            chunk_size_tokens=chunk_size_tokens,
        )
        if not stats:
            continue
        stat = stats[0]
        if stat["loss"] == stat["loss"]:
            losses.append(stat["loss"])
        n_valids.append(stat["n_valid"])
        win_lens.append(stat["window_len"])

    if not losses:
        return {
            "loss": float("nan"),
            "loss_std": float("nan"),
            "n_valid": float(np.mean(n_valids)) if n_valids else 0.0,
            "window_len": float(np.mean(win_lens)) if win_lens else 0.0,
            "num_windows": len(time_steps),
        }

    return {
        "loss": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "n_valid": float(np.mean(n_valids)) if n_valids else 0.0,
        "window_len": float(np.mean(win_lens)) if win_lens else 0.0,
        "num_windows": len(time_steps),
    }
