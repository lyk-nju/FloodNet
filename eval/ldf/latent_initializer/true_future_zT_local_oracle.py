"""Local true future z_T oracle for stream_generate_step.

This eval-only line-B prototype differs from active noisy-state correction:
it only optimizes future tokens whose current diffusion schedule is still
pure initial Gaussian noise (beta ~= 1).  Tokens already in the active
denoising state x_beta and committed z0 history are not optimization variables.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.common.visualization import render_motion_video  # noqa: E402
from eval.ldf.conditioning import LdfEvalStreamConditioner  # noqa: E402
from eval.ldf.latent_initializer.optimize_noise import (  # noqa: E402
    _cap_sample_to_frames,
    _ckpt_tag,
    _load_single_sample,
    _masked_metrics,
    _noise_stats,
    _plot_paths,
    _root_xz_from_feature,
    _set_seed,
    _target_xz_and_mask,
)
from eval.ldf.latent_initializer.optimize_stream_chunk_noise import (  # noqa: E402
    _anchored_absolute_xz_loss,
    _baseline_optimized_xz_diff,
    _extract_window_noise,
    _extract_window_tokens,
    _inject_window_noise,
    _snapshot_model_state,
    _restore_model_state,
    _build_step,
)
from eval.ldf.latent_initializer.optimize_stream_noise import (  # noqa: E402
    _attach_root_path_metrics,
    _decode_latent_stream,
    _heading_error_deg_from_xz,
    _noise_l2,
    _path_length,
    _run_stream_latents,
)
from eval.ldf.stream_generation import (  # noqa: E402
    StreamTextRolloutController,
    _restore_vae_decode_cache,
    _snapshot_vae_decode_cache,
)
from eval.ldf.stream_setup import (  # noqa: E402
    enable_cpu_text_encoding,
    load_eval_model_and_vae,
)
from models.diffusion_forcing_wan import DiffForcingWanModel  # noqa: E402
from utils.inference.stream_generator import StreamGenerator  # noqa: E402
from utils.initialize import load_config  # noqa: E402
from utils.motion_process import StreamJointRecovery263  # noqa: E402
from utils.token_frame import num_tokens_for_frame_len  # noqa: E402


def _local_true_zT_method_metadata() -> dict[str, str]:
    return {
        "method": "local_true_future_zT_oracle",
        "optimized_variable": "future model.generated tokens with beta ~= 1",
        "not_optimized_variable": "active x_beta or committed z0",
        "oracle_line": "B_local_true_zT_initializer",
        "commit_strategy": "receding_horizon_commit_one",
    }


def _token_beta(*, token_idx: int, current_time: float, chunk_size: int) -> float:
    return float(
        max(
            0.0,
            min(1.0, 1.0 + float(token_idx) / float(max(1, chunk_size)) - float(current_time)),
        )
    )


def _select_future_zT_window(
    *,
    commit_index: int,
    current_step: int,
    dt: float,
    chunk_size: int,
    generated_tokens: int,
    target_tokens: int,
    zT_horizon_tokens: int,
    beta_threshold: float,
) -> dict:
    commit_index = int(commit_index)
    generated_tokens = int(generated_tokens)
    target_tokens = int(target_tokens)
    current_time = float(current_step) * float(dt)
    start = max(0, commit_index)
    end_limit = max(start + 1, min(generated_tokens, target_tokens + int(zT_horizon_tokens)))
    while start < end_limit:
        beta = _token_beta(
            token_idx=start,
            current_time=current_time,
            chunk_size=int(chunk_size),
        )
        if beta >= float(beta_threshold):
            break
        start += 1
    end = min(end_limit, start + max(1, int(zT_horizon_tokens)))
    token_indices = list(range(start, end))
    token_beta = [
        _token_beta(token_idx=idx, current_time=current_time, chunk_size=int(chunk_size))
        for idx in token_indices
    ]
    return {
        "start": int(start),
        "end": int(end),
        "token_indices": token_indices,
        "token_beta": token_beta,
        "current_time": current_time,
        "beta_threshold": float(beta_threshold),
    }


def _should_optimize_commit(commit_index: int, *, optimize_every_tokens: int) -> bool:
    optimize_every_tokens = int(optimize_every_tokens)
    if optimize_every_tokens <= 1:
        return True
    return int(commit_index) % optimize_every_tokens == 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ldf_test.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vae_ckpt", default=None)
    parser.add_argument(
        "--meta_path",
        default="/data1/yuankai/text2Motion/FloodDiffusion/raw_data/HumanML3D/test_000021_codex.txt",
    )
    parser.add_argument("--sample_name", default="000021")
    parser.add_argument("--caption_index", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cfg_text", type=float, default=1.25)
    parser.add_argument("--cfg_traj", type=float, default=3.0)
    parser.add_argument("--history_length", type=int, default=30)
    parser.add_argument("--traj_horizon_tokens", type=int, default=20)
    parser.add_argument("--token_dt", type=float, default=0.20)
    parser.add_argument("--frames_per_token", type=int, default=4)
    parser.add_argument("--num_denoise_steps", type=int, default=None)
    parser.add_argument("--optim_steps", type=int, default=20)
    parser.add_argument("--optimize_every_tokens", type=int, default=1)
    parser.add_argument("--zT_horizon_tokens", type=int, default=5)
    parser.add_argument("--loss_horizon_tokens", type=int, default=20)
    parser.add_argument("--beta_threshold", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--lambda_noise", type=float, default=1e-4)
    parser.add_argument("--lambda_vel", type=float, default=0.05)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--progress_every_commit", type=int, default=1)
    parser.add_argument("--partial_every_commit", type=int, default=0)
    parser.add_argument("--no_render_video", action="store_true")
    parser.add_argument(
        "--out_dir",
        default="eval/out_eval/tset_local_true_future_zT_oracle",
    )
    return parser.parse_args()


def _clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _rollout_shadow_latents(
    *,
    model,
    vae,
    stream: StreamGenerator,
    text_rollout: StreamTextRolloutController,
    conditioner: LdfEvalStreamConditioner,
    recovery: StreamJointRecovery263,
    start_commit: int,
    tokens: int,
    first_chunk: bool,
    device: torch.device,
) -> torch.Tensor:
    local_conditioner = copy.deepcopy(conditioner)
    local_recovery = copy.deepcopy(recovery)
    local_first = bool(first_chunk)
    latents = []
    with torch.enable_grad():
        for offset in range(int(tokens)):
            commit = int(start_commit) + int(offset)
            step_payload, condition_provider = _build_step(
                model=model,
                stream=stream,
                text_rollout=text_rollout,
                conditioner=local_conditioner,
                commit_index=commit,
                first_chunk=local_first,
                device=device,
            )
            output = DiffForcingWanModel.stream_generate_step.__wrapped__(
                model,
                step_payload,
                first_chunk=local_first,
                condition=condition_provider,
            )
            latent_token = output["generated"][0]
            latents.append(latent_token)
            with torch.no_grad():
                decoded_chunk = vae.stream_decode(
                    latent_token.detach()[None, :],
                    first_chunk=local_first,
                )[0].float().detach().cpu()
                local_conditioner.append_decoded(
                    decoded_chunk,
                    commit_idx=commit + 1,
                    recovery=local_recovery,
                )
            local_first = False
    return torch.cat(latents, dim=0)


def _decode_local_latents(vae, local_latents: torch.Tensor) -> torch.Tensor:
    vae.clear_cache()
    return vae.decode(local_latents.unsqueeze(0))[0].float()


def _optimize_future_zT_window(
    *,
    model,
    vae,
    stream: StreamGenerator,
    text_rollout: StreamTextRolloutController,
    conditioner: LdfEvalStreamConditioner,
    recovery: StreamJointRecovery263,
    target_xz: torch.Tensor,
    mask: torch.Tensor,
    commit_index: int,
    target_tokens: int,
    first_chunk: bool,
    device: torch.device,
    frames_per_token: int,
    history_length: int,
    zT_horizon_tokens: int,
    loss_horizon_tokens: int,
    beta_threshold: float,
    optim_steps: int,
    lr: float,
    lambda_noise: float,
    lambda_vel: float,
) -> tuple[list[dict], dict]:
    state = _snapshot_model_state(model)
    window = _select_future_zT_window(
        commit_index=int(getattr(model, "commit_index", commit_index)),
        current_step=int(getattr(model, "current_step", 0)),
        dt=float(getattr(model, "dt", 1.0 / max(1, int(getattr(model, "num_denoise_steps", 1))))),
        chunk_size=int(getattr(model, "chunk_size", 1)),
        generated_tokens=int(state["generated"].shape[2]),
        target_tokens=int(target_tokens),
        zT_horizon_tokens=int(zT_horizon_tokens),
        beta_threshold=float(beta_threshold),
    )
    start = int(window["start"])
    end = int(window["end"])
    if start >= end:
        return [], {**window, "skipped": True}
    base_window = _extract_window_noise(state["generated"], start=start, end=end)
    opt_window = base_window.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_window], lr=float(lr))
    loss_curve = []
    best = {"loss": math.inf, "window": None}
    shadow_tokens = max(1, min(int(loss_horizon_tokens), int(target_tokens) - int(commit_index)))
    for step in range(int(optim_steps)):
        optimizer.zero_grad(set_to_none=True)
        vae_cache = _snapshot_vae_decode_cache(vae)
        generated = _inject_window_noise(state["generated"].to(device), opt_window, start=start)
        _restore_model_state(model, state, generated=generated)
        shadow_latents = _rollout_shadow_latents(
            model=model,
            vae=vae,
            stream=stream,
            text_rollout=text_rollout,
            conditioner=conditioner,
            recovery=recovery,
            start_commit=int(commit_index),
            tokens=shadow_tokens,
            first_chunk=bool(first_chunk),
            device=device,
        )
        history_start = max(0, int(commit_index) - int(history_length))
        history_tokens = _extract_window_tokens(
            state["generated"].to(device),
            start=history_start,
            end=int(commit_index),
            detach=True,
        )[0]
        active_latents = torch.cat([history_tokens, shadow_latents], dim=0)
        active_feature = _decode_local_latents(vae, active_latents)
        pred_xz = _root_xz_from_feature(active_feature)
        frame_start = int(history_start) * int(frames_per_token)
        frame_end = frame_start + int(pred_xz.shape[0])
        local_target = target_xz[frame_start:frame_end]
        local_mask = mask[frame_start:frame_end]
        history_frames = (int(commit_index) - int(history_start)) * int(frames_per_token)
        local_loss, parts = _anchored_absolute_xz_loss(
            pred_xz,
            local_target,
            local_mask,
            history_frames=int(history_frames),
            lambda_vel=float(lambda_vel),
            anchor_mode="target_anchor_abs",
        )
        noise_reg = ((opt_window - base_window.to(opt_window.device)) ** 2).mean()
        loss = local_loss + float(lambda_noise) * noise_reg
        loss.backward()
        grad_norm = (
            float(torch.linalg.vector_norm(opt_window.grad.detach().float()).cpu().item())
            if opt_window.grad is not None
            else 0.0
        )
        optimizer.step()
        with torch.no_grad():
            opt_window.clamp_(-4.0, 4.0)
        _restore_vae_decode_cache(vae, vae_cache)
        _restore_model_state(model, state)
        row = {
            "step": int(step),
            "commit_index": int(commit_index),
            "loss": float(loss.detach().cpu().item()),
            **parts,
            "noise_reg": float(noise_reg.detach().cpu().item()),
            "opt_window_grad_norm": grad_norm,
            "zT_start": start,
            "zT_end": end,
        }
        loss_curve.append(row)
        if row["loss"] < best["loss"]:
            best["loss"] = row["loss"]
            best["window"] = opt_window.detach().clone()
    chosen = best["window"] if best["window"] is not None else opt_window.detach()
    optimized_generated = _inject_window_noise(state["generated"].to(device), chosen, start=start)
    _restore_model_state(model, state, generated=optimized_generated.detach())
    diagnostics = {
        **window,
        "skipped": False,
        "base_zT_mean": float(base_window.detach().float().mean().cpu().item()),
        "base_zT_std": float(base_window.detach().float().std(unbiased=False).cpu().item()),
        "optimized_zT_mean": float(chosen.detach().float().mean().cpu().item()),
        "optimized_zT_std": float(chosen.detach().float().std(unbiased=False).cpu().item()),
        "optimized_minus_base_l2": _noise_l2(chosen, base_window),
        "shadow_tokens": int(shadow_tokens),
        "best_loss": float(best["loss"]),
    }
    return loss_curve, diagnostics


def _run_local_zT_stream(
    *,
    model,
    vae,
    sample_batch: dict,
    device: torch.device,
    history_length: int,
    traj_horizon_tokens: int,
    token_dt: float,
    frames_per_token: int,
    num_denoise_steps: int | None,
    initial_generated: torch.Tensor,
    target_xz: torch.Tensor,
    mask: torch.Tensor,
    zT_horizon_tokens: int,
    loss_horizon_tokens: int,
    beta_threshold: float,
    optim_steps: int,
    optimize_every_tokens: int,
    lr: float,
    lambda_noise: float,
    lambda_vel: float,
    progress_every_commit: int,
    partial_every_commit: int,
    partial_path: Path | None,
) -> tuple[torch.Tensor, list[dict], list[dict]]:
    total_frames = int(sample_batch["feature_length"][0].item())
    target_tokens = num_tokens_for_frame_len(total_frames, int(frames_per_token))
    model.init_generated(
        int(history_length),
        batch_size=1,
        num_denoise_steps=num_denoise_steps,
        traj_buffer=None,
        initial_generated=initial_generated,
    )
    stream = StreamGenerator(
        ldf_model=model,
        device=device,
        history_length=int(history_length),
        traj_horizon_tokens=int(traj_horizon_tokens),
        token_dt=float(token_dt),
    )
    text_rollout = StreamTextRolloutController.from_sample_batch(sample_batch)
    conditioner = LdfEvalStreamConditioner(
        sample_batch,
        history_length=int(history_length),
        traj_horizon_tokens=int(traj_horizon_tokens),
        token_dt=float(token_dt),
        frames_per_token=int(frames_per_token),
        device=device,
    )
    recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    vae.clear_cache()
    first_chunk = True
    latent_tokens = []
    loss_curve = []
    windows = []
    for commit_index in range(int(target_tokens)):
        commit_t0 = time.time()
        if _should_optimize_commit(
            commit_index, optimize_every_tokens=int(optimize_every_tokens)
        ):
            rows, diagnostics = _optimize_future_zT_window(
                model=model,
                vae=vae,
                stream=stream,
                text_rollout=text_rollout,
                conditioner=conditioner,
                recovery=recovery,
                target_xz=target_xz,
                mask=mask,
                commit_index=int(commit_index),
                target_tokens=int(target_tokens),
                first_chunk=first_chunk,
                device=device,
                frames_per_token=int(frames_per_token),
                history_length=int(history_length),
                zT_horizon_tokens=int(zT_horizon_tokens),
                loss_horizon_tokens=int(loss_horizon_tokens),
                beta_threshold=float(beta_threshold),
                optim_steps=int(optim_steps),
                lr=float(lr),
                lambda_noise=float(lambda_noise),
                lambda_vel=float(lambda_vel),
            )
        else:
            rows = []
            diagnostics = {
                "commit_index": int(commit_index),
                "skipped": True,
                "skip_reason": "optimize_every_tokens",
                "optimize_every_tokens": int(optimize_every_tokens),
                "best_loss": float("nan"),
            }
        loss_curve.extend(rows)
        diagnostics["elapsed_sec"] = float(time.time() - commit_t0)
        windows.append(diagnostics)
        if int(progress_every_commit) > 0 and (
            commit_index == 0
            or (commit_index + 1) % int(progress_every_commit) == 0
            or commit_index + 1 == int(target_tokens)
        ):
            print(
                "local-zT "
                f"commit={commit_index + 1}/{int(target_tokens)} "
                f"zT=[{diagnostics.get('start')},{diagnostics.get('end')}) "
                f"best_loss={diagnostics.get('best_loss', float('nan')):.6g} "
                f"elapsed={diagnostics['elapsed_sec']:.2f}s",
                flush=True,
            )
        if (
            partial_path is not None
            and int(partial_every_commit) > 0
            and (
                (commit_index + 1) % int(partial_every_commit) == 0
                or commit_index + 1 == int(target_tokens)
            )
        ):
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps(
                    {
                        "completed_commits": int(commit_index + 1),
                        "target_tokens": int(target_tokens),
                        "window_summary": _summarize_windows(windows),
                        "last_window": diagnostics,
                    },
                    indent=2,
                )
            )
        step_payload, condition_provider = _build_step(
            model=model,
            stream=stream,
            text_rollout=text_rollout,
            conditioner=conditioner,
            commit_index=int(commit_index),
            first_chunk=first_chunk,
            device=device,
        )
        output = model.stream_generate_step(
            step_payload,
            first_chunk=first_chunk,
            condition=condition_provider,
        )
        latent_token = output["generated"][0].detach()
        latent_tokens.append(latent_token.detach().cpu())
        decoded_chunk = vae.stream_decode(
            latent_token[None, :],
            first_chunk=first_chunk,
        )[0].float().detach().cpu()
        conditioner.append_decoded(
            decoded_chunk,
            commit_idx=int(commit_index) + 1,
            recovery=recovery,
        )
        first_chunk = False
        model.generated = model.generated.detach()
    vae.clear_cache()
    return torch.cat(latent_tokens, dim=0), loss_curve, windows


def _summarize_windows(windows: list[dict]) -> dict:
    active = [row for row in windows if not row.get("skipped", False)]
    if not active:
        return {"num_windows": len(windows), "num_optimized_windows": 0}
    return {
        "num_windows": len(windows),
        "num_optimized_windows": len(active),
        "mean_optimized_minus_base_l2": float(
            sum(float(row["optimized_minus_base_l2"]) for row in active) / len(active)
        ),
        "mean_best_loss": float(sum(float(row["best_loss"]) for row in active) / len(active)),
    }


def main() -> int:
    args = _parse_args()
    _set_seed(int(args.seed))
    torch.cuda.set_device(int(args.gpu))
    device = torch.device(f"cuda:{int(args.gpu)}")

    cfg = load_config(config_path=args.config)
    vae_ckpt = args.vae_ckpt or cfg.get("test_vae_ckpt", None)
    model, vae = load_eval_model_and_vae(
        cfg,
        ckpt_path=args.ckpt,
        vae_ckpt_path=vae_ckpt,
        device=device,
        use_ema=True,
    )
    if str(cfg.get("eval.text_device", "cpu")).lower() == "cpu":
        enable_cpu_text_encoding(model)
    model.cfg_scale_text = float(args.cfg_text)
    model.cfg_scale_traj = float(args.cfg_traj)
    model.eval()
    vae.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for param in vae.parameters():
        param.requires_grad_(False)

    sample_batch = _load_single_sample(args, cfg)
    sample_batch = _cap_sample_to_frames(
        sample_batch,
        int(args.max_frames),
        frames_per_token=int(args.frames_per_token),
    )
    target_xz, mask = _target_xz_and_mask(sample_batch, device)
    total_frames = int(sample_batch["feature_length"][0].item())
    target_num_tokens = num_tokens_for_frame_len(
        total_frames, int(args.frames_per_token)
    )
    initial_shape = (
        1,
        int(args.history_length) * 2 + int(model.chunk_size),
        int(model.input_dim),
    )
    base_noise = torch.randn(initial_shape, device=device)

    with torch.no_grad():
        baseline_latent = _run_stream_latents(
            model=model,
            vae=vae,
            sample_batch=sample_batch,
            device=device,
            history_length=int(args.history_length),
            traj_horizon_tokens=int(args.traj_horizon_tokens),
            token_dt=float(args.token_dt),
            frames_per_token=int(args.frames_per_token),
            num_denoise_steps=args.num_denoise_steps,
            initial_generated=base_noise,
            differentiable=False,
        )
        baseline_feature = _decode_latent_stream(vae, baseline_latent).detach()
        baseline_xz = _root_xz_from_feature(baseline_feature)
        baseline_metrics = _attach_root_path_metrics(
            _masked_metrics(baseline_xz, target_xz, mask),
            xz=baseline_xz,
            target_xz=target_xz,
            mask=mask,
        )

    out_dir = (
        Path(args.out_dir)
        / _ckpt_tag(args.ckpt)
        / f"sample_{args.sample_name}"
        / f"cap{int(args.caption_index)}"
        / f"h{int(args.history_length):03d}_hor{int(args.traj_horizon_tokens):03d}"
        / (
            f"localzT{int(args.zT_horizon_tokens):02d}_"
            f"loss{int(args.loss_horizon_tokens):02d}_"
            f"every{int(args.optimize_every_tokens):02d}_"
            f"steps{int(args.optim_steps)}_lr{str(args.lr).replace('.', 'p')}"
        )
        / str(args.run_tag or f"seed{int(args.seed)}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    optimized_latent, loss_curve, windows = _run_local_zT_stream(
        model=model,
        vae=vae,
        sample_batch=sample_batch,
        device=device,
        history_length=int(args.history_length),
        traj_horizon_tokens=int(args.traj_horizon_tokens),
        token_dt=float(args.token_dt),
        frames_per_token=int(args.frames_per_token),
        num_denoise_steps=args.num_denoise_steps,
        initial_generated=base_noise.detach(),
        target_xz=target_xz,
        mask=mask,
        zT_horizon_tokens=int(args.zT_horizon_tokens),
        loss_horizon_tokens=int(args.loss_horizon_tokens),
        beta_threshold=float(args.beta_threshold),
        optim_steps=int(args.optim_steps),
        optimize_every_tokens=int(args.optimize_every_tokens),
        lr=float(args.lr),
        lambda_noise=float(args.lambda_noise),
        lambda_vel=float(args.lambda_vel),
        progress_every_commit=int(args.progress_every_commit),
        partial_every_commit=int(args.partial_every_commit),
        partial_path=out_dir / "partial_progress.json",
    )
    with torch.no_grad():
        optimized_feature = _decode_latent_stream(vae, optimized_latent.to(device)).detach()
        optimized_xz = _root_xz_from_feature(optimized_feature)
        optimized_metrics = _attach_root_path_metrics(
            _masked_metrics(optimized_xz, target_xz, mask),
            xz=optimized_xz,
            target_xz=target_xz,
            mask=mask,
        )

    summary = {
        "mode": "stream_generate_step_local_true_future_zT_oracle",
        **_local_true_zT_method_metadata(),
        "sample_name": str(sample_batch["name"][0]),
        "caption_index": sample_batch.get("_caption_index", int(args.caption_index)),
        "caption_text": sample_batch.get("_caption_text", sample_batch.get("text", [""])[0]),
        "ckpt": str(args.ckpt),
        "vae_ckpt": str(vae_ckpt),
        "cfg_text": float(args.cfg_text),
        "cfg_traj": float(args.cfg_traj),
        "history_length": int(args.history_length),
        "traj_horizon_tokens": int(args.traj_horizon_tokens),
        "zT_horizon_tokens": int(args.zT_horizon_tokens),
        "loss_horizon_tokens": int(args.loss_horizon_tokens),
        "beta_threshold": float(args.beta_threshold),
        "num_denoise_steps": args.num_denoise_steps,
        "optim_steps": int(args.optim_steps),
        "optimize_every_tokens": int(args.optimize_every_tokens),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "lambda_noise": float(args.lambda_noise),
        "lambda_vel": float(args.lambda_vel),
        "num_frames": int(total_frames),
        "target_frames": int(target_xz.shape[0]),
        "target_num_tokens": int(target_num_tokens),
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "delta": {
            "ade": optimized_metrics["ade"] - baseline_metrics["ade"],
            "fde": optimized_metrics["fde"] - baseline_metrics["fde"],
            "mse": optimized_metrics["mse"] - baseline_metrics["mse"],
            "heading_error_deg": optimized_metrics["heading_error_deg"]
            - baseline_metrics["heading_error_deg"],
            "path_ratio": optimized_metrics["path_ratio"] - baseline_metrics["path_ratio"],
        },
        "noise": {"base": _noise_stats(base_noise)},
        "baseline_optimized_xz_diff": _baseline_optimized_xz_diff(
            baseline_xz,
            optimized_xz,
        ),
        "opt_steps_zero_sanity": (
            _baseline_optimized_xz_diff(baseline_xz, optimized_xz)
            if int(args.optim_steps) == 0
            else None
        ),
        "gradient_check": {
            "optimized_variable_is_independent_window": True,
            "model_parameters_require_grad": False,
            "nonzero_window_grad_steps": int(
                sum(
                    1
                    for row in loss_curve
                    if float(row.get("opt_window_grad_norm", 0.0)) > 0.0
                )
            ),
            "total_optimization_steps": int(len(loss_curve)),
        },
        "window_summary": _summarize_windows(windows),
        "windows": windows,
        "loss_curve": loss_curve,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2))
    np.savez_compressed(
        out_dir / "motions.npz",
        target_xz=target_xz.detach().cpu().numpy(),
        mask=mask.detach().cpu().numpy(),
        baseline_feature=baseline_feature.detach().cpu().numpy(),
        optimized_feature=optimized_feature.detach().cpu().numpy(),
        baseline_xz=baseline_xz.detach().cpu().numpy(),
        optimized_xz=optimized_xz.detach().cpu().numpy(),
        baseline_latent=baseline_latent.detach().cpu().numpy(),
        optimized_latent=optimized_latent.detach().cpu().numpy(),
    )
    _plot_paths(
        out_dir / "root_xz_compare.png",
        target_xz.detach().cpu().numpy(),
        baseline_xz.detach().cpu().numpy(),
        optimized_xz.detach().cpu().numpy(),
    )
    if not bool(args.no_render_video):
        render_setting = {
            "cond_traj_show_full": True,
            "traj_mask_point_radius": 3,
            "cond_traj_point_radius": 4,
        }
        traj_mask = torch.ones(
            int(target_xz.shape[0]), dtype=torch.float32, device=target_xz.device
        )
        render_motion_video(
            baseline_feature,
            out_dir / "baseline.mp4",
            dim=263,
            traj_xz=target_xz,
            traj_mask=traj_mask,
            cond_traj_mask=mask,
            render_setting=render_setting,
        )
        render_motion_video(
            optimized_feature,
            out_dir / "optimized_local_true_zT.mp4",
            dim=263,
            traj_xz=target_xz,
            traj_mask=traj_mask,
            cond_traj_mask=mask,
            render_setting=render_setting,
        )
    print(json.dumps(summary, indent=2))
    print(f"wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
