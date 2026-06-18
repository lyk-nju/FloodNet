"""Artifact helpers for LDF stream evaluation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from eval.common.visualization import (
    plot_xz_trajectories,
    plot_yaw_series,
    render_motion_video,
    yaw_from_7d,
    yaw_from_root_path,
)
from metrics.stream import (
    compute_root_path_yaw_error,
    summarize_stream_records,
)
from utils.motion_process import extract_root_trajectory_263_torch


def _format_text(sample_batch: Dict) -> str:
    lines = []
    segment_names = sample_batch.get("segment_names", None)
    if isinstance(segment_names, list) and len(segment_names) == 1 and isinstance(segment_names[0], list):
        segment_names = segment_names[0]
    if segment_names:
        lines.append("segments: " + ", ".join(str(name) for name in segment_names))
    text_value = sample_batch.get("text", [""])[0]
    if isinstance(text_value, list):
        end_list = sample_batch.get("feature_text_end", [[]])[0]
        for idx, segment in enumerate(text_value):
            end_frame = end_list[idx] if idx < len(end_list) else None
            lines.append(f"[{idx}] end={end_frame}: {segment}")
        return "\n".join(lines)
    lines.append(str(text_value))
    return "\n".join(lines)


_TRAJECTORY_BATCH_KEYS = {
    "traj_cond_7d",
    "traj_cond",
    "traj",
    "traj_features",
    "traj_length",
    "traj_cond_mask",
    "traj_mask",
    "traj_loss_mask",
    "token_mask",
}


def _remove_trajectory_conditioning(sample_batch: Dict) -> Dict:
    return {
        key: value
        for key, value in sample_batch.items()
        if key not in _TRAJECTORY_BATCH_KEYS
    }


def _root_numpy(feature: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if feature is None:
        return None
    with torch.no_grad():
        return extract_root_trajectory_263_torch(feature[None, :])[0].cpu().numpy()


def _root_path_yaw_error(
    pred_feature: torch.Tensor,
    sample_batch: Dict,
) -> float:
    gt_feature = sample_batch.get("feature")
    if gt_feature is None:
        return float("nan")
    gt_single = gt_feature[0].float().cpu() if torch.is_tensor(gt_feature) and gt_feature.ndim == 3 else gt_feature
    pred_root = _root_numpy(pred_feature)
    gt_root = _root_numpy(gt_single)
    if pred_root is None or gt_root is None:
        return float("nan")
    return compute_root_path_yaw_error(pred_root, gt_root)


def _condition_root_numpy(sample_batch: Dict) -> Optional[np.ndarray]:
    traj7 = sample_batch.get("traj_cond_7d")
    if traj7 is not None:
        value = traj7[0] if torch.is_tensor(traj7) and traj7.ndim == 3 else traj7
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if arr.ndim == 2 and arr.shape[-1] >= 3:
            return arr[:, :3].astype(np.float32)
    traj = sample_batch.get("traj")
    if traj is not None:
        value = traj[0] if torch.is_tensor(traj) and traj.ndim == 3 else traj
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if arr.ndim == 2 and arr.shape[-1] >= 3:
            return arr[:, :3].astype(np.float32)
    return None


def _condition_yaw_numpy(sample_batch: Dict) -> Optional[np.ndarray]:
    traj7 = sample_batch.get("traj_cond_7d")
    if traj7 is None:
        return None
    value = traj7[0] if torch.is_tensor(traj7) and traj7.ndim == 3 else traj7
    yaw = yaw_from_7d(value)
    return yaw if yaw.shape[0] > 0 else None


def _save_sample_outputs(
    sample_dir: Path,
    sample_batch: Dict,
    sample_record: Dict,
    stream_feature: torch.Tensor,
    gt_feature: Optional[torch.Tensor],
    offline_feature: Optional[torch.Tensor],
    stream_no_traj_feature: Optional[torch.Tensor] = None,
    stream_latent: Optional[torch.Tensor] = None,
    offline_latent: Optional[torch.Tensor] = None,
    save_feature_npy: bool = True,
    save_latent_npy: bool = False,
    save_plots: bool = True,
    render_video: bool = False,
    render_offline_video: bool = False,
    render_no_traj_video: bool = False,
):
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "text.txt", "w") as f:
        f.write(_format_text(sample_batch))
    with open(sample_dir / "metrics.json", "w") as f:
        json.dump(sample_record, f, indent=2)

    gt_root = _root_numpy(gt_feature)
    stream_root = _root_numpy(stream_feature)
    offline_root = _root_numpy(offline_feature)
    no_traj_root = _root_numpy(stream_no_traj_feature)
    condition_root = _condition_root_numpy(sample_batch)

    if gt_root is not None:
        np.save(sample_dir / "gt_root.npy", gt_root.astype(np.float32))
    if condition_root is not None:
        np.save(sample_dir / "condition_root.npy", condition_root.astype(np.float32))
    if stream_root is not None:
        np.save(sample_dir / "stream_gt_root.npy", stream_root.astype(np.float32))
    if offline_root is not None:
        np.save(sample_dir / "offline_gt_root.npy", offline_root.astype(np.float32))
    if no_traj_root is not None:
        np.save(sample_dir / "stream_no_traj_root.npy", no_traj_root.astype(np.float32))

    if save_plots:
        plot_xz_trajectories(
            sample_dir / "plot_xz.png",
            {
                "gt_root": gt_root,
                "condition_root": condition_root,
                "stream_gt": stream_root,
                "offline_gt": offline_root,
                "stream_no_traj": no_traj_root,
            },
            title=str(sample_batch.get("name", ["sample"])[0]),
        )
        plot_yaw_series(
            sample_dir / "plot_yaw.png",
            {
                "gt_yaw": yaw_from_root_path(gt_root),
                "condition_yaw": _condition_yaw_numpy(sample_batch),
                "stream_gt_yaw": yaw_from_root_path(stream_root),
                "offline_gt_yaw": yaw_from_root_path(offline_root),
                "stream_no_traj_yaw": yaw_from_root_path(no_traj_root),
            },
            title=str(sample_batch.get("name", ["sample"])[0]),
        )

    if save_feature_npy:
        np.save(sample_dir / "stream_feature.npy", stream_feature.cpu().numpy())
        if gt_feature is not None:
            np.save(sample_dir / "gt_feature.npy", gt_feature.cpu().numpy())
        if offline_feature is not None:
            np.save(sample_dir / "offline_feature.npy", offline_feature.cpu().numpy())
        if stream_no_traj_feature is not None:
            np.save(
                sample_dir / "stream_no_traj_feature.npy",
                stream_no_traj_feature.cpu().numpy(),
            )

    if save_latent_npy and stream_latent is not None:
        np.save(sample_dir / "stream_latent.npy", stream_latent.cpu().numpy())
        if offline_latent is not None:
            np.save(sample_dir / "offline_latent.npy", offline_latent.cpu().numpy())

    if render_video:
        render_motion_video(
            stream_feature,
            sample_dir / "video_stream_gt.mp4",
            dim=263,
            traj_xz=condition_root[:, [0, 2]] if condition_root is not None else None,
        )
    if render_offline_video and offline_feature is not None:
        render_motion_video(
            offline_feature,
            sample_dir / "video_offline_gt.mp4",
            dim=263,
            traj_xz=condition_root[:, [0, 2]] if condition_root is not None else None,
        )
    if render_no_traj_video and stream_no_traj_feature is not None:
        render_motion_video(
            stream_no_traj_feature,
            sample_dir / "video_stream_no_traj.mp4",
            dim=263,
            traj_xz=condition_root[:, [0, 2]] if condition_root is not None else None,
        )


def _stream_eval_artifact_dirs(out_root: Path, dataset_id: str, probe_tag: str, step_tag: str) -> Dict[str, Path]:
    base_dir = Path(out_root) / dataset_id
    return {
        "text": base_dir / "text" / probe_tag / step_tag,
        "token": base_dir / "token" / probe_tag / step_tag,
        "feature": base_dir / "feature" / probe_tag / step_tag,
        "traj_xz": base_dir / "traj_xz" / probe_tag / step_tag,
        "traj_mask": base_dir / "traj_mask" / probe_tag / step_tag,
        "frames": base_dir / "frames" / probe_tag / step_tag,
        "metrics": base_dir / "metrics" / probe_tag / step_tag,
        "video": base_dir / "video" / probe_tag / step_tag,
        "composite": base_dir / "composite" / probe_tag / step_tag,
        "condition_compare": base_dir / "condition_compare" / probe_tag / step_tag,
    }


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _frames_numpy(sample_batch: Dict) -> Optional[np.ndarray]:
    frames = sample_batch.get("feature_text_end")
    if frames is None:
        return None
    value = frames[0] if isinstance(frames, list) and len(frames) > 0 else frames
    arr = _to_numpy(value).reshape(-1)
    return arr.astype(np.int64) if arr.size > 0 else None


def _traj_mask_numpy(sample_batch: Dict, feat_len: int) -> Optional[np.ndarray]:
    mask = sample_batch.get("traj_mask")
    if mask is None:
        return None
    value = mask[0] if torch.is_tensor(mask) and mask.ndim >= 2 else mask
    arr = _to_numpy(value).reshape(-1)
    if feat_len > 0:
        arr = arr[:feat_len]
    return arr.astype(np.float32)


def _save_eval_style_sample_outputs(
    *,
    out_root: Path,
    dataset_id: str,
    probe_tag: str,
    step_tag: str,
    sample_name: str,
    sample_batch: Dict,
    sample_record: Dict,
    stream_feature: Optional[torch.Tensor],
    stream_latent: Optional[torch.Tensor] = None,
    gt_feature: Optional[torch.Tensor] = None,
    offline_feature: Optional[torch.Tensor] = None,
    stream_no_traj_feature: Optional[torch.Tensor] = None,
) -> None:
    dirs = _stream_eval_artifact_dirs(Path(out_root), dataset_id, probe_tag, step_tag)
    for out_dir in dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(dirs["text"] / f"{sample_name}.txt", "w") as f:
        f.write(_format_text(sample_batch))

    with open(dirs["metrics"] / f"{sample_name}.json", "w") as f:
        json.dump(sample_record, f, indent=2)

    feat_len = 0
    if stream_feature is not None:
        stream_np = _to_numpy(stream_feature).astype(np.float32)
        feat_len = int(stream_np.shape[0]) if stream_np.ndim > 0 else 0
        np.save(dirs["feature"] / f"{sample_name}.npy", stream_np)

    if stream_latent is not None:
        np.save(
            dirs["token"] / f"{sample_name}.npy",
            _to_numpy(stream_latent).astype(np.float32),
        )

    condition_root = _condition_root_numpy(sample_batch)
    if condition_root is not None:
        cond_xz = condition_root[:, [0, 2]].astype(np.float32)
        if feat_len > 0:
            cond_xz = cond_xz[:feat_len]
        np.save(dirs["traj_xz"] / f"{sample_name}.npy", cond_xz)

    traj_mask = _traj_mask_numpy(sample_batch, feat_len)
    if traj_mask is not None:
        np.save(dirs["traj_mask"] / f"{sample_name}.npy", traj_mask)

    frames = _frames_numpy(sample_batch)
    if frames is not None:
        np.save(dirs["frames"] / f"{sample_name}.npy", frames)

    plot_xz_trajectories(
        dirs["condition_compare"] / f"{sample_name}.png",
        {
            "gt_root": _root_numpy(gt_feature),
            "condition_root": condition_root,
            "stream_gt": _root_numpy(stream_feature),
            "offline_gt": _root_numpy(offline_feature),
            "stream_no_traj": _root_numpy(stream_no_traj_feature),
        },
        title=str(sample_name),
    )


def _build_run_name(ckpt_path: str, probe_tag: str, stream_mode: str) -> str:
    ckpt_tag = Path(ckpt_path).stem.replace("=", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{probe_tag}_{stream_mode}_{ckpt_tag}"


def _resolve_run_name(
    *,
    ckpt_path: str,
    probe_tag: str,
    stream_mode: str,
    requested_run_name: str | None,
) -> str:
    if requested_run_name:
        return requested_run_name
    return _build_run_name(ckpt_path, probe_tag, stream_mode)


def _average_scalar_metric(run_metrics: List[Dict], key: str) -> float:
    vals = [metric[key] for metric in run_metrics if key in metric and metric[key] == metric[key]]
    return float(np.mean(vals)) if vals else float("nan")



def _rank_payload_path(run_dir: Path, rank: int) -> Path:
    return run_dir / "_rank_records" / f"rank_{rank}.json"


def _write_summary_payload(run_dir: Path, payload: Dict) -> None:
    with open(run_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_eval_style_summaries(
    *,
    out_root: Path,
    payload: Dict,
    probe_tag: str,
    step_tag: str,
) -> None:
    records_by_dataset: Dict[str, List[Dict]] = {}
    for record in payload.get("samples", []):
        dataset_id = record.get("dataset") or record.get("dataset_id")
        if dataset_id is None:
            continue
        records_by_dataset.setdefault(str(dataset_id), []).append(record)

    for dataset_id, records in records_by_dataset.items():
        dirs = _stream_eval_artifact_dirs(Path(out_root), dataset_id, probe_tag, step_tag)
        dirs["metrics"].mkdir(parents=True, exist_ok=True)
        summary = summarize_stream_records(records)
        with open(dirs["metrics"] / "summary.json", "w") as f:
            json.dump({"summary": summary, "samples": records}, f, indent=2)


def _render_eval_style_outputs(
    *,
    cfg,
    out_root: Path,
    sample_records: List[Dict],
    probe_tag: str,
    step_tag: str,
    enabled: bool,
) -> None:
    if not enabled:
        return
    try:
        from utils.visualization.video import (
            make_composite_compare_videos,
            render_video as render_eval_video,
        )
    except Exception as exc:
        print(f"[stream-eval render] imports failed: {exc}")
        return

    render_setting = cfg.get("test_setting", {}) or {}
    try:
        render_setting = OmegaConf.to_container(render_setting, resolve=True)
    except Exception:
        render_setting = dict(render_setting)
    render_setting.setdefault("recover_dim", 263)

    dataset_ids = sorted(
        {
            str(record.get("dataset") or record.get("dataset_id"))
            for record in sample_records
            if record.get("dataset") or record.get("dataset_id")
        }
    )
    for dataset_id in dataset_ids:
        dirs = _stream_eval_artifact_dirs(Path(out_root), dataset_id, probe_tag, step_tag)
        if not dirs["feature"].exists():
            continue
        try:
            render_eval_video(
                motion_dir=str(dirs["feature"]),
                save_dir=str(dirs["video"]),
                render_setting=render_setting,
                frames_dir=str(dirs["frames"]),
                traj_mask_dir=str(dirs["traj_mask"]),
                cond_traj_dir=str(dirs["traj_xz"]),
            )
            make_composite_compare_videos(
                result_folder=str(dirs["video"]),
                compare_folders=render_setting.get(dataset_id, {}).get(
                    "compare_folders", None
                ),
                compare_names=render_setting.get(dataset_id, {}).get(
                    "compare_names", None
                ),
                text_folder=str(dirs["text"]),
                save_dir=str(dirs["composite"]),
            )
        except Exception as exc:
            print(f"[stream-eval render] dataset={dataset_id} failed: {exc}")


def _aggregate_rank_payloads(
    *,
    run_dir: Path,
    world_size: int,
    probe_tag: str,
    ckpt_path: str,
    vae_ckpt_path: str,
    stream_mode: str,
    num_runs: int,
    out_root: Optional[Path] = None,
    step_tag: Optional[str] = None,
) -> Dict:
    sample_records = []
    for rank in range(world_size):
        rank_path = _rank_payload_path(run_dir, rank)
        if not rank_path.is_file():
            raise FileNotFoundError(f"Missing stream eval rank payload: {rank_path}")
        with open(rank_path) as f:
            rank_payload = json.load(f)
        sample_records.extend(rank_payload.get("samples", []))

    sample_records.sort(key=lambda record: int(record.get("_sample_index", 0)))
    summary = summarize_stream_records(sample_records)
    payload = {
        "probe_tag": probe_tag,
        "ckpt": ckpt_path,
        "vae_ckpt": vae_ckpt_path,
        "stream_mode": stream_mode,
        "num_samples": len(sample_records),
        "num_runs": num_runs,
        "summary": summary,
        "samples": sample_records,
    }
    _write_summary_payload(run_dir, payload)
    if out_root is not None and step_tag is not None:
        _write_eval_style_summaries(
            out_root=Path(out_root),
            payload=payload,
            probe_tag=probe_tag,
            step_tag=step_tag,
        )
    return payload



__all__ = [
    "_aggregate_rank_payloads",
    "_average_scalar_metric",
    "_condition_root_numpy",
    "_condition_yaw_numpy",
    "_remove_trajectory_conditioning",
    "_resolve_run_name",
    "_root_numpy",
    "_root_path_yaw_error",
    "_save_eval_style_sample_outputs",
    "_save_sample_outputs",
    "_stream_eval_artifact_dirs",
    "_write_summary_payload",
]
