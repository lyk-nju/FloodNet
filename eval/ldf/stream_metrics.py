import argparse
import json
import os
import random
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Multi-GPU stream eval spawns one Python process per device. Cap BLAS/OpenMP
# thread pools before importing numpy/torch so workers do not exhaust RLIMIT_NPROC.
for _thread_env_key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_key, "1")

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from FloodNet.metrics.stream import (
        compute_stream_boundary_metrics,
        compute_root_path_yaw_error,
        compute_stream_vs_offline_metrics,
        decode_stream_chunks,
        summarize_stream_records,
    )
    from FloodNet.metrics.traj import (
        _average_control_metrics,
        _average_traj_metrics,
        _compute_omni_control_metrics,
        _compute_traj_metrics,
        _seed_eval_locally,
        _slice_single_sample_batch,
        _stable_eval_seed,
    )
    from FloodNet.eval.ldf.conditioning import LdfEvalStreamConditioner
    from FloodNet.utils.training.ldf.validation_conditioning import (
        prepare_ldf_eval_model_batch,
    )
    from FloodNet.eval.common.visualization import (
        plot_xz_trajectories,
        plot_yaw_series,
        render_motion_video,
        yaw_from_7d,
        yaw_from_root_path,
    )
    from FloodNet.utils.initialize import get_function, instantiate, load_config
    from FloodNet.utils.motion_process import (
        StreamJointRecovery263,
        extract_root_trajectory_263_torch,
    )
    from FloodNet.utils.inference.stream_generator import StreamGenerator
    from FloodNet.utils.training.ldf.model_factory import instantiate_ldf_model
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from metrics.stream import (
        compute_stream_boundary_metrics,
        compute_root_path_yaw_error,
        compute_stream_vs_offline_metrics,
        decode_stream_chunks,
        summarize_stream_records,
    )
    from metrics.traj import (
        _average_control_metrics,
        _average_traj_metrics,
        _compute_omni_control_metrics,
        _compute_traj_metrics,
        _seed_eval_locally,
        _slice_single_sample_batch,
        _stable_eval_seed,
    )
    from eval.ldf.conditioning import LdfEvalStreamConditioner
    from utils.training.ldf.validation_conditioning import prepare_ldf_eval_model_batch
    from eval.common.visualization import (
        plot_xz_trajectories,
        plot_yaw_series,
        render_motion_video,
        yaw_from_7d,
        yaw_from_root_path,
    )
    from utils.initialize import get_function, instantiate, load_config
    from utils.motion_process import (
        StreamJointRecovery263,
        extract_root_trajectory_263_torch,
    )
    from utils.inference.stream_generator import StreamGenerator
    from utils.training.ldf.model_factory import instantiate_ldf_model
from eval.ldf.stream_artifacts import (
    _aggregate_rank_payloads,
    _average_scalar_metric,
    _condition_root_numpy,
    _condition_yaw_numpy,
    _rank_payload_path,
    _remove_trajectory_conditioning,
    _render_eval_style_outputs,
    _resolve_run_name,
    _root_numpy,
    _root_path_yaw_error,
    _write_eval_style_summaries,
    _save_eval_style_sample_outputs,
    _save_sample_outputs,
    _stream_eval_artifact_dirs,
    _write_summary_payload,
)
from eval.ldf.stream_generation import (
    StreamTextRolloutController,
    build_stream_input,
    run_offline_generate_sample,
    run_stream_generate_sample,
    run_stream_generate_step_sample,
)
from eval.ldf.stream_setup import (
    InMemorySampleDataset,
    _infer_meta_tag,
    _parse_devices_arg,
    _parse_overrides,
    _resolve_accelerator,
    _resolve_ema_params,
    _resolve_meta_paths_and_probe_tag,
    _select_eval_device,
    _set_seed,
    _should_process_batch_on_rank,
    build_eval_dataloader,
    enable_cpu_text_encoding,
    load_eval_model_and_vae,
)






def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate BABEL long-horizon streaming generation via stream_generate()."
    )
    parser.add_argument("--config", type=str, default="configs/eval_babel_stream.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--stream_mode",
        type=str,
        choices=["stream_generate", "stream_generate_step"],
        default=None,
    )
    parser.add_argument("--num_denoise_steps", type=int, default=None)
    parser.add_argument("--compute_offline_baseline", action="store_true")
    parser.add_argument("--no_compute_offline_baseline", action="store_true")
    parser.add_argument("--compute_no_traj_baseline", action="store_true")
    parser.add_argument("--no_compute_no_traj_baseline", action="store_true")
    parser.add_argument("--save_feature_npy", action="store_true")
    parser.add_argument("--save_latent_npy", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--no_save_plots", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--render_offline_video", action="store_true")
    parser.add_argument("--render_no_traj_video", action="store_true")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Optional deterministic run directory name under --out_dir. "
            "Defaults to a timestamped ckpt/probe tag."
        ),
    )
    parser.add_argument("--probe_tag", type=str, default=None)
    parser.add_argument("--meta_paths", nargs="+", default=None)
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help=(
            "Number of GPU worker processes to launch, or a comma-separated "
            "visible device list such as 0,1,2,3. Stream eval shards samples "
            "across workers; each worker still runs batch_size=1."
        ),
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["gpu", "cpu"],
        default=None,
        help="Execution device type. Defaults to gpu when CUDA is available.",
    )
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="OmegaConf dot-path overrides, e.g. --set model.params.cfg_scale_traj=3.0",
    )
    return parser.parse_args(argv)


def parse_args_from_list(argv):
    return parse_args(argv)


def resolve_eval_path_flags(args, cfg) -> tuple[bool, bool]:
    """Resolve LDF stream eval diagnostic paths.

    The design-default LDF suite runs stream GT, offline GT, and stream no-traj.
    CLI disable flags are explicit opt-outs.
    """
    compute_offline_baseline = bool(
        args.compute_offline_baseline
        or cfg.get("eval.compute_offline_baseline", True)
    )
    compute_no_traj_baseline = bool(
        args.compute_no_traj_baseline
        or cfg.get("eval.compute_no_traj_baseline", True)
    )
    if getattr(args, "no_compute_offline_baseline", False):
        compute_offline_baseline = False
    if args.no_compute_no_traj_baseline:
        compute_no_traj_baseline = False
    return compute_offline_baseline, compute_no_traj_baseline










def _resolve_launch_context(args) -> Dict:
    overrides = _parse_overrides(args.set)
    cfg = load_config(config_path=args.config, override_args=overrides)
    ckpt_path = args.ckpt or cfg.get("test_ckpt", None) or cfg.get("resume_ckpt", None)
    vae_ckpt_path = args.vae_ckpt or cfg.get("test_vae_ckpt", None)
    if ckpt_path is None:
        raise ValueError("No checkpoint provided via --ckpt / cfg.test_ckpt / cfg.resume_ckpt")
    if vae_ckpt_path is None:
        raise ValueError("No VAE checkpoint provided via --vae_ckpt / cfg.test_vae_ckpt")

    meta_paths, probe_tag = _resolve_meta_paths_and_probe_tag(args, cfg)
    stream_mode = args.stream_mode or cfg.get("eval.stream_mode", "stream_generate")
    num_runs = args.num_runs or int(cfg.get("eval.num_runs", 1))
    out_root = Path(args.out_dir or cfg.get("eval.out_dir", "./outputs_stream_eval"))
    run_name = _resolve_run_name(
        ckpt_path=ckpt_path,
        probe_tag=probe_tag,
        stream_mode=stream_mode,
        requested_run_name=args.run_name,
    )
    step_tag = Path(str(ckpt_path)).stem.replace("=", "_")
    return {
        "cfg": cfg,
        "ckpt_path": ckpt_path,
        "vae_ckpt_path": vae_ckpt_path,
        "meta_paths": meta_paths,
        "probe_tag": probe_tag,
        "stream_mode": stream_mode,
        "num_runs": num_runs,
        "out_root": out_root,
        "run_name": run_name,
        "step_tag": step_tag,
        "run_dir": out_root / run_name,
    }


def _run_stream_eval(
    args,
    *,
    rank: int = 0,
    world_size: int = 1,
    accelerator: str | None = None,
    device_index: int | None = None,
    write_summary: bool = True,
) -> Dict:
    context = _resolve_launch_context(args)
    cfg = context["cfg"]
    ckpt_path = context["ckpt_path"]
    vae_ckpt_path = context["vae_ckpt_path"]
    meta_paths = context["meta_paths"]
    probe_tag = context["probe_tag"]
    stream_mode = context["stream_mode"]
    num_runs = context["num_runs"]
    out_root = context["out_root"]
    step_tag = context["step_tag"]
    run_dir = context["run_dir"]

    _set_seed(args.seed + rank)
    if accelerator is None:
        accelerator = _resolve_accelerator(
            args,
            [device_index] if device_index is not None else [0],
        )
    device = _select_eval_device(accelerator, device_index)

    batch_size = args.batch_size or int(cfg.data.test_bs)
    if batch_size != 1:
        raise NotImplementedError(
            "Streaming evaluator currently requires batch_size=1 because rollout and VAE streaming decode are single-sample."
        )

    num_workers = args.num_workers if args.num_workers is not None else int(cfg.data.num_workers)
    seg_size = int(cfg.get("eval.seg_size", 20))
    num_denoise_steps = (
        args.num_denoise_steps
        if args.num_denoise_steps is not None
        else cfg.get("eval.num_denoise_steps", None)
    )
    compute_offline_baseline, compute_no_traj_baseline = resolve_eval_path_flags(
        args, cfg
    )
    save_feature_npy = bool(args.save_feature_npy or cfg.get("eval.save_feature_npy", True))
    save_latent_npy = bool(args.save_latent_npy or cfg.get("eval.save_latent_npy", False))
    save_plots = bool(cfg.get("eval.save_plots", True) or args.save_plots)
    if args.no_save_plots:
        save_plots = False
    render_video = bool(args.render_video or cfg.get("eval.render_video", False))
    render_offline_video = bool(
        args.render_offline_video or cfg.get("eval.render_offline_video", False)
    )
    render_no_traj_video = bool(
        args.render_no_traj_video or cfg.get("eval.render_no_traj_video", False)
    )
    max_batches = args.max_batches or int(cfg.get("eval.max_batches", 0))
    max_samples = args.max_samples or int(cfg.get("eval.max_samples", 0))
    text_device = str(cfg.get("eval.text_device", "cpu")).lower()
    history_length = int(cfg.get("eval.history_length", 30))
    group_present_segments = bool(cfg.get("eval.group_present_segments", False))
    _traj_horizon_raw = cfg.get("eval.traj_horizon_tokens", None)
    traj_horizon_tokens = int(_traj_horizon_raw) if _traj_horizon_raw is not None else None
    token_dt = float(cfg.get("eval.token_dt", cfg.get("stream.token_dt", 0.20)))
    frames_per_token = int(cfg.get("eval.frames_per_token", cfg.get("data.frames_per_token", 4)))
    best_of_k = int(cfg.get("eval.stream_best_of_k", cfg.get("eval_stream_best_of_k", 1)))
    best_of_k_score = str(
        cfg.get("eval.stream_best_of_k_score", cfg.get("eval_stream_best_of_k_score", "xz"))
    )
    best_of_k_xz_weight = float(
        cfg.get("eval.stream_best_of_k_xz_weight", cfg.get("eval_stream_best_of_k_xz_weight", 1.0))
    )
    best_of_k_fde_weight = float(
        cfg.get("eval.stream_best_of_k_fde_weight", cfg.get("eval_stream_best_of_k_fde_weight", 1.0))
    )
    best_of_k_cont_weight = float(
        cfg.get("eval.stream_best_of_k_cont_weight", cfg.get("eval_stream_best_of_k_cont_weight", 0.0))
    )
    best_of_k_vel_weight = float(
        cfg.get("eval.stream_best_of_k_vel_weight", cfg.get("eval_stream_best_of_k_vel_weight", 0.5))
    )
    best_of_k_rel_margin = float(
        cfg.get("eval.stream_best_of_k_rel_margin", cfg.get("eval_stream_best_of_k_rel_margin", 0.10))
    )
    best_of_k_abs_margin = float(
        cfg.get("eval.stream_best_of_k_abs_margin", cfg.get("eval_stream_best_of_k_abs_margin", 0.03))
    )
    best_of_k_cont_tol = float(
        cfg.get("eval.stream_best_of_k_cont_tol", cfg.get("eval_stream_best_of_k_cont_tol", 0.03))
    )
    best_of_k_force_candidate0 = bool(
        cfg.get("eval.stream_best_of_k_force_candidate0", cfg.get("eval_stream_best_of_k_force_candidate0", False))
    )
    best_of_k_switch_cooldown_steps = int(
        cfg.get("eval.stream_best_of_k_switch_cooldown_steps", cfg.get("eval_stream_best_of_k_switch_cooldown_steps", 0))
    )
    best_of_k_debug = bool(
        cfg.get("eval.stream_best_of_k_debug", cfg.get("eval_stream_best_of_k_debug", False))
    )

    sample_root = run_dir / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        OmegaConf.save(cfg.config, run_dir / "config.yaml")

    model, vae = load_eval_model_and_vae(
        cfg,
        ckpt_path=ckpt_path,
        vae_ckpt_path=vae_ckpt_path,
        device=device,
        use_ema=not args.no_ema,
    )
    if text_device == "cpu":
        enable_cpu_text_encoding(model)
    _, dataloader = build_eval_dataloader(
        cfg,
        meta_paths=meta_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        group_present_segments=group_present_segments,
    )

    print(
        f"[stream-eval][rank {rank}/{world_size}] ckpt={ckpt_path} "
        f"probe={probe_tag} stream_mode={stream_mode} device={device} "
        f"num_runs={num_runs} batch_size={batch_size} text_device={text_device} "
        f"group_present_segments={int(group_present_segments)} history_length={history_length} "
        f"traj_horizon_tokens={traj_horizon_tokens} best_of_k={best_of_k} "
        f"out_dir={run_dir}"
    )

    sample_records = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        if max_samples > 0 and batch_idx >= max_samples:
            break
        if not _should_process_batch_on_rank(
            batch_idx,
            rank=rank,
            world_size=world_size,
            max_batches=max_batches,
            max_samples=max_samples,
        ):
            continue
        sample_batch = _slice_single_sample_batch(batch, 0)
        sample_name = sample_batch["name"][0]
        sample_dataset = sample_batch["dataset"][0]

        traj_runs = []
        control_runs = []
        stream_runs = []
        stream_feature_run0 = None
        stream_latent_run0 = None
        offline_feature_run0 = None
        offline_latent_run0 = None
        no_traj_runs = []
        stream_no_traj_feature_run0 = None

        for run_idx in range(num_runs):
            sample_seed = _stable_eval_seed(args.seed, probe_tag, sample_name, run_idx)
            _seed_eval_locally(sample_seed)

            stream_started = time.perf_counter()
            with torch.no_grad():
                if stream_mode == "stream_generate":
                    stream_out = run_stream_generate_sample(
                        model=model,
                        vae=vae,
                        sample_batch=sample_batch,
                        device=device,
                        num_denoise_steps=num_denoise_steps,
                    )
                elif stream_mode == "stream_generate_step":
                    stream_out = run_stream_generate_step_sample(
                        model=model,
                        vae=vae,
                        sample_batch=sample_batch,
                        device=device,
                        history_length=history_length,
                        num_denoise_steps=num_denoise_steps,
                        traj_horizon_tokens=traj_horizon_tokens,
                        token_dt=token_dt,
                        frames_per_token=frames_per_token,
                        best_of_k=best_of_k,
                        best_of_k_score=best_of_k_score,
                        best_of_k_xz_weight=best_of_k_xz_weight,
                        best_of_k_fde_weight=best_of_k_fde_weight,
                        best_of_k_cont_weight=best_of_k_cont_weight,
                        best_of_k_vel_weight=best_of_k_vel_weight,
                        best_of_k_rel_margin=best_of_k_rel_margin,
                        best_of_k_abs_margin=best_of_k_abs_margin,
                        best_of_k_cont_tol=best_of_k_cont_tol,
                        best_of_k_force_candidate0=best_of_k_force_candidate0,
                        best_of_k_switch_cooldown_steps=best_of_k_switch_cooldown_steps,
                        best_of_k_debug=best_of_k_debug,
                    )
                else:
                    raise ValueError(f"Unsupported stream_mode: {stream_mode}")
            stream_elapsed_sec = float(time.perf_counter() - stream_started)

            decoded_stream = stream_out["decoded_feature"]
            traj_runs.append(_compute_traj_metrics(decoded_stream, sample_batch, 0, seg_size=seg_size))
            control_runs.append(_compute_omni_control_metrics(decoded_stream, sample_batch, 0))

            boundary_metrics = compute_stream_boundary_metrics(
                decoded_stream,
                stream_out["chunk_frame_ends"],
            )
            stream_metric = {
                "stream_root_jump_mean": boundary_metrics["root_jump_mean"],
                "stream_root_jump_max": boundary_metrics["root_jump_max"],
                "stream_joint_jump_mean": boundary_metrics["joint_jump_mean"],
                "stream_num_boundaries": boundary_metrics["n_boundaries"],
                "stream_yaw_error": _root_path_yaw_error(
                    decoded_stream,
                    sample_batch,
                ),
                "stream_rollout_time_sec": stream_elapsed_sec,
            }
            best_of_k_info = stream_out.get("stream_best_of_k", {})
            if best_of_k_info.get("enabled", False):
                stream_metric["stream_best_of_k"] = float(best_of_k_info.get("k", 1))
                stream_metric["stream_best_of_k_total_elapsed_sec"] = float(
                    best_of_k_info.get("total_elapsed_sec", 0.0)
                )
                stream_metric["stream_best_of_k_switch_count"] = float(
                    best_of_k_info.get("switch_count", 0)
                )
                stream_metric["stream_best_of_k_step_count"] = float(
                    best_of_k_info.get("step_count", 0)
                )
                if best_of_k_debug:
                    stream_metric["stream_best_of_k_records"] = best_of_k_info.get(
                        "records",
                        [],
                    )

            if compute_offline_baseline:
                _seed_eval_locally(sample_seed)
                with torch.no_grad():
                    offline_out = run_offline_generate_sample(
                        model=model,
                        vae=vae,
                        sample_batch=sample_batch,
                        device=device,
                        num_denoise_steps=num_denoise_steps,
                    )
                offline_cmp = compute_stream_vs_offline_metrics(
                    decoded_stream,
                    offline_out["decoded_feature"],
                )
                stream_metric["stream_offline_feature_l2_mean"] = offline_cmp["feature_l2_mean"]
                stream_metric["stream_offline_feature_l2_max"] = offline_cmp["feature_l2_max"]
                stream_metric["stream_offline_root_ade"] = offline_cmp["root_ade"]
                stream_metric["stream_offline_length_delta"] = offline_cmp["length_delta"]
                if run_idx == 0:
                    offline_feature_run0 = offline_out["decoded_feature"]
                    offline_latent_run0 = offline_out["latent"]

            if compute_no_traj_baseline:
                no_traj_batch = _remove_trajectory_conditioning(sample_batch)
                _seed_eval_locally(sample_seed)
                with torch.no_grad():
                    if stream_mode == "stream_generate":
                        no_traj_out = run_stream_generate_sample(
                            model=model,
                            vae=vae,
                            sample_batch=no_traj_batch,
                            device=device,
                            num_denoise_steps=num_denoise_steps,
                        )
                    elif stream_mode == "stream_generate_step":
                        no_traj_out = run_stream_generate_step_sample(
                            model=model,
                            vae=vae,
                            sample_batch=no_traj_batch,
                            device=device,
                            history_length=history_length,
                            num_denoise_steps=num_denoise_steps,
                            traj_horizon_tokens=traj_horizon_tokens,
                            token_dt=token_dt,
                            frames_per_token=frames_per_token,
                            best_of_k=1,
                        )
                    else:
                        raise ValueError(f"Unsupported stream_mode: {stream_mode}")
                no_traj_decoded = no_traj_out["decoded_feature"]
                no_traj_metrics = _compute_traj_metrics(
                    no_traj_decoded,
                    sample_batch,
                    0,
                    seg_size=seg_size,
                )
                no_traj_runs.append(no_traj_metrics)
                if run_idx == 0:
                    stream_no_traj_feature_run0 = no_traj_decoded

            if run_idx == 0:
                stream_feature_run0 = decoded_stream
                stream_latent_run0 = stream_out["latent_stream"]
            stream_runs.append(stream_metric)

        sample_record = {
            "_sample_index": batch_idx,
            "name": sample_name,
            "dataset": sample_dataset,
            "stream_mode": stream_mode,
            "num_runs": num_runs,
        }
        if "segment_names" in sample_batch:
            segment_names = sample_batch["segment_names"][0]
            sample_record["segment_names"] = list(segment_names)
        sample_record.update(_average_traj_metrics(traj_runs))
        sample_record.update(_average_control_metrics(control_runs))
        sample_record["stream_root_jump_mean"] = _average_scalar_metric(stream_runs, "stream_root_jump_mean")
        sample_record["stream_root_jump_max"] = _average_scalar_metric(stream_runs, "stream_root_jump_max")
        sample_record["stream_joint_jump_mean"] = _average_scalar_metric(stream_runs, "stream_joint_jump_mean")
        sample_record["stream_num_boundaries"] = _average_scalar_metric(stream_runs, "stream_num_boundaries")
        sample_record["stream_yaw_error"] = _average_scalar_metric(stream_runs, "stream_yaw_error")
        sample_record["stream_rollout_time_sec"] = _average_scalar_metric(stream_runs, "stream_rollout_time_sec")
        if best_of_k > 1:
            sample_record["stream_best_of_k"] = float(best_of_k)
            sample_record["stream_best_of_k_total_elapsed_sec"] = _average_scalar_metric(
                stream_runs,
                "stream_best_of_k_total_elapsed_sec",
            )
            sample_record["stream_best_of_k_switch_count"] = _average_scalar_metric(
                stream_runs,
                "stream_best_of_k_switch_count",
            )
            sample_record["stream_best_of_k_step_count"] = _average_scalar_metric(
                stream_runs,
                "stream_best_of_k_step_count",
            )
        if compute_offline_baseline:
            sample_record["stream_offline_feature_l2_mean"] = _average_scalar_metric(stream_runs, "stream_offline_feature_l2_mean")
            sample_record["stream_offline_feature_l2_max"] = _average_scalar_metric(stream_runs, "stream_offline_feature_l2_max")
            sample_record["stream_offline_root_ade"] = _average_scalar_metric(stream_runs, "stream_offline_root_ade")
            sample_record["stream_offline_length_delta"] = _average_scalar_metric(stream_runs, "stream_offline_length_delta")
        if compute_no_traj_baseline and no_traj_runs:
            no_traj_avg = _average_traj_metrics(no_traj_runs)
            for key, value in no_traj_avg.items():
                sample_record[f"stream_no_traj/{key}"] = value
        sample_record["_traj_runs"] = traj_runs
        sample_record["_control_runs"] = control_runs
        sample_record["_stream_runs"] = stream_runs
        if compute_no_traj_baseline:
            sample_record["_stream_no_traj_runs"] = no_traj_runs
        sample_records.append(sample_record)

        gt_feature = sample_batch["feature"][0].float().cpu() if "feature" in sample_batch else None
        _save_sample_outputs(
            sample_dir=sample_root / sample_name,
            sample_batch=sample_batch,
            sample_record=sample_record,
            stream_feature=stream_feature_run0,
            gt_feature=gt_feature,
            offline_feature=offline_feature_run0,
            stream_no_traj_feature=stream_no_traj_feature_run0,
            stream_latent=stream_latent_run0,
            offline_latent=offline_latent_run0,
            save_feature_npy=save_feature_npy,
            save_latent_npy=save_latent_npy,
            save_plots=save_plots,
            render_video=render_video,
            render_offline_video=render_offline_video,
            render_no_traj_video=render_no_traj_video,
        )
        _save_eval_style_sample_outputs(
            out_root=out_root,
            dataset_id=sample_dataset,
            probe_tag=probe_tag,
            step_tag=step_tag,
            sample_name=sample_name,
            sample_batch=sample_batch,
            sample_record=sample_record,
            stream_feature=stream_feature_run0,
            stream_latent=stream_latent_run0,
            gt_feature=gt_feature,
            offline_feature=offline_feature_run0,
            stream_no_traj_feature=stream_no_traj_feature_run0,
        )

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
    if world_size > 1:
        rank_path = _rank_payload_path(run_dir, rank)
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rank_path, "w") as f:
            json.dump(payload, f, indent=2)
    elif write_summary:
        _write_summary_payload(run_dir, payload)
        _write_eval_style_summaries(
            out_root=out_root,
            payload=payload,
            probe_tag=probe_tag,
            step_tag=step_tag,
        )
        _render_eval_style_outputs(
            cfg=cfg,
            out_root=out_root,
            sample_records=sample_records,
            probe_tag=probe_tag,
            step_tag=step_tag,
            enabled=render_video,
        )

    print(
        f"[stream-eval][rank {rank}/{world_size}] finished {len(sample_records)} samples | "
        f"ADE={summary.get('traj/ADE_mean', float('nan')):.4f} "
        f"FDE={summary.get('traj/FDE_mean', float('nan')):.4f} "
        f"RootJump={summary.get('stream_boundary/root_jump_mean', float('nan')):.4f}"
    )
    return payload


def _distributed_worker(rank: int, args, device_ids: list[int]) -> None:
    _run_stream_eval(
        args,
        rank=rank,
        world_size=len(device_ids),
        accelerator="gpu",
        device_index=device_ids[rank],
        write_summary=False,
    )


def main():
    args = parse_args()
    device_ids = _parse_devices_arg(args.devices)
    accelerator = _resolve_accelerator(args, device_ids)
    if accelerator == "gpu" and len(device_ids) > 1:
        context = _resolve_launch_context(args)
        args.run_name = context["run_name"]
        context["run_dir"].mkdir(parents=True, exist_ok=True)
        torch.multiprocessing.spawn(
            _distributed_worker,
            args=(args, device_ids),
            nprocs=len(device_ids),
            join=True,
        )
        payload = _aggregate_rank_payloads(
            run_dir=context["run_dir"],
            world_size=len(device_ids),
            probe_tag=context["probe_tag"],
            ckpt_path=context["ckpt_path"],
            vae_ckpt_path=context["vae_ckpt_path"],
            stream_mode=context["stream_mode"],
            num_runs=context["num_runs"],
            out_root=context["out_root"],
            step_tag=context["step_tag"],
        )
        render_video = bool(
            args.render_video
            or context["cfg"].get("eval.render_video", False)
        )
        _render_eval_style_outputs(
            cfg=context["cfg"],
            out_root=context["out_root"],
            sample_records=payload["samples"],
            probe_tag=context["probe_tag"],
            step_tag=context["step_tag"],
            enabled=render_video,
        )
        summary = payload["summary"]
        print(
            f"[stream-eval] merged {payload['num_samples']} samples from "
            f"{len(device_ids)} ranks | "
            f"ADE={summary.get('traj/ADE_mean', float('nan')):.4f} "
            f"FDE={summary.get('traj/FDE_mean', float('nan')):.4f} "
            f"RootJump={summary.get('stream_boundary/root_jump_mean', float('nan')):.4f}"
        )
        return

    device_index = device_ids[0] if accelerator == "gpu" and device_ids else None
    _run_stream_eval(
        args,
        rank=0,
        world_size=1,
        accelerator=accelerator,
        device_index=device_index,
    )


if __name__ == "__main__":
    main()
