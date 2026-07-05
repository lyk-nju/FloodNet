"""Sweep stream_generate_step CFG scales without reloading the model per combo."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from statistics import mean, pstdev


for _key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf

from eval.common.visualization import render_motion_video
from eval.ldf.stream_generation import run_stream_generate_step_sample
from eval.ldf.stream_setup import (
    _set_seed,
    build_eval_dataloader,
    enable_cpu_text_encoding,
    load_eval_model_and_vae,
)
from metrics.traj import (
    _compute_traj_metrics,
    _seed_eval_locally,
    _slice_single_sample_batch,
    _stable_eval_seed,
)
from utils.initialize import load_config
from utils.motion_process import replace_root_channels_263_from_7d


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search cfg_scale_text/cfg_scale_traj for stream_generate_step."
    )
    parser.add_argument("--config", default="configs/ldf.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vae_ckpt", default=None)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--min_cfg", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--num_values", type=int, default=16)
    parser.add_argument(
        "--cfg_text_values",
        default=None,
        help="Optional comma-separated cfg_scale_text values. Overrides grid text values.",
    )
    parser.add_argument(
        "--cfg_traj_values",
        default=None,
        help="Optional comma-separated cfg_scale_traj values. Overrides grid traj values.",
    )
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--history_length", type=int, default=30)
    parser.add_argument("--horizon_tokens", type=int, default=20)
    parser.add_argument(
        "--extra_frames",
        type=int,
        default=0,
        help="Generate this many frames past the GT length and report tail FDE.",
    )
    parser.add_argument(
        "--root_replace_condition",
        action="store_true",
        help="Also evaluate a copy whose 263D root channels are replaced from traj_cond_7d.",
    )
    parser.add_argument(
        "--root_replace_feedback",
        action="store_true",
        help="Run a second stream pass that re-encodes GT-root-corrected chunks into latent history.",
    )
    parser.add_argument(
        "--root_feedback_xz_blend_alpha",
        type=float,
        default=1.0,
        help="Blend generated root XZ with condition root XZ during feedback; 1.0 is hard replacement.",
    )
    parser.add_argument(
        "--render_video",
        action="store_true",
        help="Render run0 original stream and root-replaced videos when enabled.",
    )
    parser.add_argument("--num_denoise_steps", type=int, default=None)
    parser.add_argument("--frames_per_token", type=int, default=4)
    parser.add_argument("--token_dt", type=float, default=0.20)
    parser.add_argument(
        "--best_of_k",
        type=int,
        default=1,
        help="Number of stream_generate_step candidates to sample and select by XZ score.",
    )
    parser.add_argument("--best_of_k_score", default="xz")
    parser.add_argument("--best_of_k_xz_weight", type=float, default=1.0)
    parser.add_argument("--best_of_k_fde_weight", type=float, default=1.0)
    parser.add_argument("--best_of_k_cont_weight", type=float, default=0.0)
    parser.add_argument("--best_of_k_vel_weight", type=float, default=0.5)
    parser.add_argument("--best_of_k_rel_margin", type=float, default=0.10)
    parser.add_argument("--best_of_k_abs_margin", type=float, default=0.03)
    parser.add_argument("--best_of_k_cont_tol", type=float, default=0.03)
    parser.add_argument("--best_of_k_force_candidate0", action="store_true")
    parser.add_argument("--best_of_k_switch_cooldown_steps", type=int, default=0)
    parser.add_argument("--best_of_k_debug", action="store_true")
    parser.add_argument("--sample_name", default="000021")
    parser.add_argument("--probe_tag", default="cfg_sweep_000021")
    parser.add_argument(
        "--caption_index",
        type=int,
        default=None,
        help="Optional 0-based caption index from sample_batch['text_all'] to force.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def _cfg_values(min_cfg: float, step: float, num_values: int) -> list[float]:
    return [round(float(min_cfg) + i * float(step), 6) for i in range(int(num_values))]


def _parse_cfg_values(raw: str) -> list[float]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(round(float(item), 6))
    if not values:
        raise ValueError("cfg value list must contain at least one value")
    return values


def _load_sample(args: argparse.Namespace, cfg):
    _, dataloader = build_eval_dataloader(
        cfg,
        meta_paths=[args.meta_path],
        batch_size=1,
        num_workers=0,
        group_present_segments=False,
    )
    batch = next(iter(dataloader))
    sample_batch = _slice_single_sample_batch(batch, 0)
    name = str(sample_batch["name"][0])
    if args.sample_name and name != str(args.sample_name):
        raise ValueError(
            f"Expected sample {args.sample_name!r} from {args.meta_path}, got {name!r}"
        )
    if args.caption_index is not None:
        _apply_caption_index(sample_batch, int(args.caption_index))
    return sample_batch


def _apply_caption_index(sample_batch: dict, caption_index: int) -> dict:
    text_all = sample_batch.get("text_all")
    if not text_all:
        raise ValueError("caption_index requires sample_batch['text_all']")
    captions = text_all[0] if isinstance(text_all, list) and text_all else text_all
    if captions and isinstance(captions[0], list):
        captions = captions[0]
    index = int(caption_index)
    if index < 0 or index >= len(captions):
        raise ValueError(
            f"caption_index={index} out of range for {len(captions)} captions"
        )
    caption = str(captions[index])
    sample_batch["text"] = [caption]
    sample_batch["_caption_index"] = index
    sample_batch["_caption_text"] = caption
    return sample_batch


def _stats(values: list[float]) -> tuple[float, float]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return float("nan"), float("nan")
    return float(mean(values)), float(pstdev(values) if len(values) > 1 else 0.0)


def _combo_path(out_dir: Path, cfg_text: float, cfg_traj: float) -> Path:
    return out_dir / "results" / f"text_{cfg_text:.2f}_traj_{cfg_traj:.2f}.json"


def _condition_traj_xz(sample_batch: dict):
    traj7 = sample_batch.get("traj_cond_7d")
    if traj7 is None:
        return None
    value = traj7[0] if torch.is_tensor(traj7) and traj7.ndim == 3 else traj7
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim != 2 or value.shape[-1] < 3:
        return None
    return value[:, [0, 2]].detach().cpu()


def _root_replace_from_condition(decoded_feature: torch.Tensor, sample_batch: dict) -> torch.Tensor:
    traj7 = sample_batch.get("traj_cond_7d")
    if traj7 is None:
        raise ValueError("root replacement requires sample_batch['traj_cond_7d']")
    value = traj7[0] if torch.is_tensor(traj7) and traj7.ndim == 3 else traj7
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    return replace_root_channels_263_from_7d(decoded_feature, value)


def _render_run0_videos(
    out_dir: Path,
    *,
    cfg_text: float,
    cfg_traj: float,
    decoded_feature: torch.Tensor,
    root_replaced_feature: torch.Tensor | None,
    feedback_feature: torch.Tensor | None,
    sample_batch: dict,
) -> dict[str, str]:
    combo_dir = out_dir / "videos" / f"text_{cfg_text:.2f}_traj_{cfg_traj:.2f}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    traj_xz = _condition_traj_xz(sample_batch)
    original_path = combo_dir / "stream_original_run0.mp4"
    render_motion_video(
        decoded_feature,
        original_path,
        dim=263,
        traj_xz=traj_xz,
    )
    paths = {"video_stream_original": str(original_path)}
    if root_replaced_feature is not None:
        replaced_path = combo_dir / "stream_root_replaced_run0.mp4"
        render_motion_video(
            root_replaced_feature,
            replaced_path,
            dim=263,
            traj_xz=traj_xz,
        )
        paths["video_stream_root_replaced"] = str(replaced_path)
    if feedback_feature is not None:
        feedback_path = combo_dir / "stream_root_feedback_run0.mp4"
        render_motion_video(
            feedback_feature,
            feedback_path,
            dim=263,
            traj_xz=traj_xz,
        )
        paths["video_stream_root_feedback"] = str(feedback_path)
    return paths


def _run_worker(worker_id: int, gpu_id: int, combos: list[tuple[float, float]], args_dict):
    args = argparse.Namespace(**args_dict)
    out_dir = Path(args.out_dir)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = out_dir / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(int(gpu_id))
    device = torch.device(f"cuda:{int(gpu_id)}")
    _set_seed(int(args.seed) + int(worker_id))

    cfg = load_config(config_path=args.config)
    vae_ckpt = args.vae_ckpt or cfg.get("test_vae_ckpt", None)
    if vae_ckpt is None:
        raise ValueError("Missing VAE checkpoint; pass --vae_ckpt or set cfg.test_vae_ckpt")

    model, vae = load_eval_model_and_vae(
        cfg,
        ckpt_path=args.ckpt,
        vae_ckpt_path=vae_ckpt,
        device=device,
        use_ema=True,
    )
    if str(cfg.get("eval.text_device", "cpu")).lower() == "cpu":
        enable_cpu_text_encoding(model)
    sample_batch = _load_sample(args, cfg)

    worker_rows = []
    for combo_index, (cfg_text, cfg_traj) in enumerate(combos):
        result_path = _combo_path(out_dir, cfg_text, cfg_traj)
        if args.skip_existing and result_path.exists():
            with result_path.open("r", encoding="utf-8") as handle:
                worker_rows.append(json.load(handle))
            continue

        model.cfg_scale_text = float(cfg_text)
        model.cfg_scale_traj = float(cfg_traj)
        run_metrics = []
        for run_idx in range(int(args.num_runs)):
            sample_seed = _stable_eval_seed(
                int(args.seed),
                str(args.probe_tag),
                str(sample_batch["name"][0]),
                int(run_idx),
            )
            _seed_eval_locally(sample_seed)
            start_time = time.perf_counter()
            with torch.no_grad():
                stream_out = run_stream_generate_step_sample(
                    model=model,
                    vae=vae,
                    sample_batch=sample_batch,
                    device=device,
                    history_length=int(args.history_length),
                    num_denoise_steps=args.num_denoise_steps,
                    traj_horizon_tokens=int(args.horizon_tokens),
                    token_dt=float(args.token_dt),
                    frames_per_token=int(args.frames_per_token),
                    extra_frames=int(args.extra_frames),
                    best_of_k=int(args.best_of_k),
                    best_of_k_score=str(args.best_of_k_score),
                    best_of_k_xz_weight=float(args.best_of_k_xz_weight),
                    best_of_k_fde_weight=float(args.best_of_k_fde_weight),
                    best_of_k_cont_weight=float(args.best_of_k_cont_weight),
                    best_of_k_vel_weight=float(args.best_of_k_vel_weight),
                    best_of_k_rel_margin=float(args.best_of_k_rel_margin),
                    best_of_k_abs_margin=float(args.best_of_k_abs_margin),
                    best_of_k_cont_tol=float(args.best_of_k_cont_tol),
                    best_of_k_force_candidate0=bool(args.best_of_k_force_candidate0),
                    best_of_k_switch_cooldown_steps=int(
                        args.best_of_k_switch_cooldown_steps
                    ),
                    best_of_k_debug=bool(args.best_of_k_debug),
                )
            original_time = time.perf_counter() - start_time
            metrics = _compute_traj_metrics(
                stream_out["decoded_feature"],
                sample_batch,
                0,
                seg_size=20,
                tail_fde_frames=int(args.extra_frames),
            )
            root_replaced_feature = None
            root_replaced_metrics = {}
            posthoc_start_time = time.perf_counter()
            if bool(args.root_replace_condition):
                root_replaced_feature = _root_replace_from_condition(
                    stream_out["decoded_feature"],
                    sample_batch,
                )
                root_replaced_metrics = _compute_traj_metrics(
                    root_replaced_feature,
                    sample_batch,
                    0,
                    seg_size=20,
                    tail_fde_frames=int(args.extra_frames),
                )
            posthoc_time = time.perf_counter() - posthoc_start_time
            feedback_feature = None
            feedback_metrics = {}
            feedback_time = float("nan")
            if bool(args.root_replace_feedback):
                _seed_eval_locally(sample_seed)
                feedback_start_time = time.perf_counter()
                with torch.no_grad():
                    feedback_out = run_stream_generate_step_sample(
                        model=model,
                        vae=vae,
                        sample_batch=sample_batch,
                        device=device,
                        history_length=int(args.history_length),
                        num_denoise_steps=args.num_denoise_steps,
                        traj_horizon_tokens=int(args.horizon_tokens),
                        token_dt=float(args.token_dt),
                        frames_per_token=int(args.frames_per_token),
                        extra_frames=int(args.extra_frames),
                        root_replace_feedback=True,
                        root_feedback_xz_blend_alpha=float(
                            args.root_feedback_xz_blend_alpha
                        ),
                    )
                feedback_time = time.perf_counter() - feedback_start_time
                feedback_feature = feedback_out["decoded_feature"]
                feedback_metrics = _compute_traj_metrics(
                    feedback_feature,
                    sample_batch,
                    0,
                    seg_size=20,
                    tail_fde_frames=int(args.extra_frames),
                )
            video_paths = {}
            if bool(args.render_video) and run_idx == 0:
                video_paths = _render_run0_videos(
                    out_dir,
                    cfg_text=float(cfg_text),
                    cfg_traj=float(cfg_traj),
                    decoded_feature=stream_out["decoded_feature"],
                    root_replaced_feature=root_replaced_feature,
                    feedback_feature=feedback_feature,
                    sample_batch=sample_batch,
                )
            run_metrics.append(
                {
                    "run_idx": int(run_idx),
                    "seed": int(sample_seed),
                    "ade": float(metrics.get("ade", float("nan"))),
                    "fde": float(metrics.get("fde", float("nan"))),
                    "fde_plus_extra": float(
                        metrics.get("fde_plus_extra", float("nan"))
                    ),
                    "fde_tail_min_extra": float(
                        metrics.get("fde_tail_min_extra", float("nan"))
                    ),
                    "target_total_frames": int(stream_out.get("target_total_frames", 0)),
                    "original_total_frames": int(stream_out.get("original_total_frames", 0)),
                    "path_arc_ade": float(metrics.get("path_arc_ade", float("nan"))),
                    "path_chamfer": float(metrics.get("path_chamfer", float("nan"))),
                    "mse": float(metrics.get("mse", float("nan"))),
                    "traj_jitter": float(metrics.get("traj_jitter", float("nan"))),
                    "root_replaced_ade": float(
                        root_replaced_metrics.get("ade", float("nan"))
                    ),
                    "root_replaced_fde": float(
                        root_replaced_metrics.get("fde", float("nan"))
                    ),
                    "root_replaced_mse": float(
                        root_replaced_metrics.get("mse", float("nan"))
                    ),
                    "root_feedback_ade": float(
                        feedback_metrics.get("ade", float("nan"))
                    ),
                    "root_feedback_fde": float(
                        feedback_metrics.get("fde", float("nan"))
                    ),
                    "root_feedback_mse": float(
                        feedback_metrics.get("mse", float("nan"))
                    ),
                    "time_stream_original_sec": float(original_time),
                    "time_root_replaced_posthoc_sec": float(posthoc_time),
                    "time_root_replaced_feedback_sec": float(feedback_time),
                    **video_paths,
                }
            )

        ade_mean, ade_std = _stats([m["ade"] for m in run_metrics])
        fde_mean, fde_std = _stats([m["fde"] for m in run_metrics])
        fde_plus_mean, fde_plus_std = _stats(
            [m["fde_plus_extra"] for m in run_metrics]
        )
        fde_tail_min_mean, fde_tail_min_std = _stats(
            [m["fde_tail_min_extra"] for m in run_metrics]
        )
        root_replaced_ade_mean, root_replaced_ade_std = _stats(
            [m["root_replaced_ade"] for m in run_metrics]
        )
        root_replaced_fde_mean, root_replaced_fde_std = _stats(
            [m["root_replaced_fde"] for m in run_metrics]
        )
        root_feedback_ade_mean, root_feedback_ade_std = _stats(
            [m["root_feedback_ade"] for m in run_metrics]
        )
        root_feedback_fde_mean, root_feedback_fde_std = _stats(
            [m["root_feedback_fde"] for m in run_metrics]
        )
        time_original_mean, time_original_std = _stats(
            [m["time_stream_original_sec"] for m in run_metrics]
        )
        time_posthoc_mean, time_posthoc_std = _stats(
            [m["time_root_replaced_posthoc_sec"] for m in run_metrics]
        )
        time_feedback_mean, time_feedback_std = _stats(
            [m["time_root_replaced_feedback_sec"] for m in run_metrics]
        )
        arc_mean, arc_std = _stats([m["path_arc_ade"] for m in run_metrics])
        record = {
            "cfg_text": float(cfg_text),
            "cfg_traj": float(cfg_traj),
            "ade_mean": ade_mean,
            "ade_std": ade_std,
            "fde_mean": fde_mean,
            "fde_std": fde_std,
            "fde_plus_extra_mean": fde_plus_mean,
            "fde_plus_extra_std": fde_plus_std,
            "fde_tail_min_extra_mean": fde_tail_min_mean,
            "fde_tail_min_extra_std": fde_tail_min_std,
            "root_replaced_ade_mean": root_replaced_ade_mean,
            "root_replaced_ade_std": root_replaced_ade_std,
            "root_replaced_fde_mean": root_replaced_fde_mean,
            "root_replaced_fde_std": root_replaced_fde_std,
            "root_feedback_ade_mean": root_feedback_ade_mean,
            "root_feedback_ade_std": root_feedback_ade_std,
            "root_feedback_fde_mean": root_feedback_fde_mean,
            "root_feedback_fde_std": root_feedback_fde_std,
            "time_stream_original_sec_mean": time_original_mean,
            "time_stream_original_sec_std": time_original_std,
            "stream_original_fps_mean": (
                float(run_metrics[0]["target_total_frames"]) / time_original_mean
                if math.isfinite(time_original_mean) and time_original_mean > 0
                else float("nan")
            ),
            "time_root_replaced_posthoc_sec_mean": time_posthoc_mean,
            "time_root_replaced_posthoc_sec_std": time_posthoc_std,
            "time_root_replaced_feedback_sec_mean": time_feedback_mean,
            "time_root_replaced_feedback_sec_std": time_feedback_std,
            "root_feedback_fps_mean": (
                float(run_metrics[0]["target_total_frames"]) / time_feedback_mean
                if math.isfinite(time_feedback_mean) and time_feedback_mean > 0
                else float("nan")
            ),
            "path_arc_ade_mean": arc_mean,
            "path_arc_ade_std": arc_std,
            "num_runs": int(args.num_runs),
            "horizon_tokens": int(args.horizon_tokens),
            "extra_frames": int(args.extra_frames),
            "root_replace_condition": bool(args.root_replace_condition),
            "root_replace_feedback": bool(args.root_replace_feedback),
            "root_feedback_xz_blend_alpha": float(args.root_feedback_xz_blend_alpha),
            "best_of_k": int(args.best_of_k),
            "best_of_k_score": str(args.best_of_k_score),
            "stream_mode": "stream_generate_step",
            "sample_name": str(sample_batch["name"][0]),
            "caption_index": sample_batch.get("_caption_index"),
            "caption_text": sample_batch.get("_caption_text"),
            "gpu_id": int(gpu_id),
            "worker_id": int(worker_id),
            "run_metrics": run_metrics,
        }
        tmp_path = result_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        tmp_path.replace(result_path)
        worker_rows.append(record)
        print(
            f"[worker {worker_id} gpu {gpu_id}] {combo_index + 1}/{len(combos)} "
            f"text={cfg_text:.2f} traj={cfg_traj:.2f} ADE={ade_mean:.4f}±{ade_std:.4f}",
            flush=True,
        )

    worker_csv = worker_dir / f"worker_{worker_id}_gpu_{gpu_id}.csv"
    _write_csv(worker_csv, worker_rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "cfg_text",
        "cfg_traj",
        "ade_mean",
        "ade_std",
        "fde_mean",
        "fde_std",
        "fde_plus_extra_mean",
        "fde_plus_extra_std",
        "fde_tail_min_extra_mean",
        "fde_tail_min_extra_std",
        "root_replaced_ade_mean",
        "root_replaced_ade_std",
        "root_replaced_fde_mean",
        "root_replaced_fde_std",
        "root_feedback_ade_mean",
        "root_feedback_ade_std",
        "root_feedback_fde_mean",
        "root_feedback_fde_std",
        "time_stream_original_sec_mean",
        "time_stream_original_sec_std",
        "stream_original_fps_mean",
        "time_root_replaced_posthoc_sec_mean",
        "time_root_replaced_posthoc_sec_std",
        "time_root_replaced_feedback_sec_mean",
        "time_root_replaced_feedback_sec_std",
        "root_feedback_fps_mean",
        "path_arc_ade_mean",
        "path_arc_ade_std",
        "num_runs",
        "horizon_tokens",
        "extra_frames",
        "root_replace_condition",
        "root_replace_feedback",
        "root_feedback_xz_blend_alpha",
        "best_of_k",
        "best_of_k_score",
        "sample_name",
        "gpu_id",
        "worker_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["ade_mean"], item["cfg_text"], item["cfg_traj"])):
            writer.writerow({field: row.get(field) for field in fields})


def _aggregate(out_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((out_dir / "results").glob("text_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    rows.sort(key=lambda item: (item["ade_mean"], item["cfg_text"], item["cfg_traj"]))
    _write_csv(out_dir / "all_results.csv", rows)
    summary = {
        "num_combos": len(rows),
        "best_by_ade": rows[0] if rows else None,
        "top10_by_ade": rows[:10],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return rows


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = _cfg_values(args.min_cfg, args.step, args.num_values)
    text_values = (
        _parse_cfg_values(args.cfg_text_values)
        if args.cfg_text_values is not None
        else values
    )
    traj_values = (
        _parse_cfg_values(args.cfg_traj_values)
        if args.cfg_traj_values is not None
        else values
    )
    combos = [(cfg_text, cfg_traj) for cfg_text in text_values for cfg_traj in traj_values]
    metadata = {
        "config": args.config,
        "ckpt": args.ckpt,
        "vae_ckpt": args.vae_ckpt,
        "meta_path": args.meta_path,
        "gpus": args.gpus,
        "values": values,
        "text_values": text_values,
        "traj_values": traj_values,
        "num_combos": len(combos),
        "num_runs": args.num_runs,
        "horizon_tokens": args.horizon_tokens,
        "extra_frames": args.extra_frames,
        "root_replace_condition": args.root_replace_condition,
        "root_replace_feedback": args.root_replace_feedback,
        "root_feedback_xz_blend_alpha": args.root_feedback_xz_blend_alpha,
        "best_of_k": args.best_of_k,
        "best_of_k_score": args.best_of_k_score,
        "render_video": args.render_video,
    }
    with (out_dir / "sweep_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    OmegaConf.save(load_config(args.config).config, out_dir / "resolved_config.yaml")

    shards = [[] for _ in args.gpus]
    for index, combo in enumerate(combos):
        shards[index % len(shards)].append(combo)

    ctx = get_context("spawn")
    processes = []
    args_dict = vars(args)
    for worker_id, (gpu_id, shard) in enumerate(zip(args.gpus, shards)):
        process = ctx.Process(
            target=_run_worker,
            args=(worker_id, int(gpu_id), shard, args_dict),
        )
        process.start()
        processes.append(process)
    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(process.exitcode)
    rows = _aggregate(out_dir)
    if failed:
        raise SystemExit(f"Worker failures: {failed}")
    if rows:
        best = rows[0]
        print(
            "best_by_ade: "
            f"text={best['cfg_text']:.2f} traj={best['cfg_traj']:.2f} "
            f"ADE={best['ade_mean']:.4f}±{best['ade_std']:.4f} "
            f"FDE={best['fde_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
