"""Oracle optimize stream noise locally for active chunk windows.

This eval-only prototype keeps the final rollout full-length, but bounds each
autograd graph to a short active window.  The intended use is testing whether
local z_T search can improve ``stream_generate_step`` trajectory tracking
without backpropagating through the whole motion.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
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
from eval.ldf.latent_initializer.optimize_stream_noise import (  # noqa: E402
    _decode_latent_stream,
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
    parser.add_argument("--optim_steps", type=int, default=100)
    parser.add_argument("--local_horizon_tokens", type=int, default=5)
    parser.add_argument(
        "--loss_anchor_mode",
        choices=("history_anchor_abs", "relative_local"),
        default="history_anchor_abs",
    )
    parser.add_argument(
        "--anchor_mode",
        choices=("target_anchor_abs", "generated_anchor_abs"),
        default="target_anchor_abs",
        help=(
            "Anchor used by history_anchor_abs. target_anchor_abs preserves the "
            "original local-oracle behavior; generated_anchor_abs keeps the "
            "generated history world anchor and exposes accumulated drift."
        ),
    )
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--start_seed", type=int, default=1234)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--lambda_noise", type=float, default=1e-4)
    parser.add_argument("--lambda_vel", type=float, default=0.05)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--no_render_video", action="store_true")
    parser.add_argument(
        "--out_dir",
        default="eval/out_eval/tset_z_optimize_chunk_horizon_full",
    )
    return parser.parse_args()


def _active_token_slice(
    commit_index: int,
    *,
    horizon_tokens: int,
    target_tokens: int,
) -> tuple[int, int]:
    start = max(0, int(commit_index))
    target_tokens = max(start + 1, int(target_tokens))
    horizon_tokens = max(1, int(horizon_tokens))
    end = min(target_tokens, start + horizon_tokens)
    return start, max(start + 1, end)


def _active_state_method_metadata() -> dict[str, str]:
    return {
        "method": "active_noisy_state_opt",
        "optimized_variable": "model.generated[start:end] current x_beta",
        "not_optimized_variable": "initial Gaussian z_T",
        "oracle_line": "A_active_noisy_state",
    }


def _beta_for_tokens(
    *,
    token_indices: list[int],
    current_time: float,
    chunk_size: int,
) -> list[float]:
    chunk_size = max(1, int(chunk_size))
    return [
        float(max(0.0, min(1.0, 1.0 + float(token_idx) / float(chunk_size) - current_time)))
        for token_idx in token_indices
    ]


def _window_stats(prefix: str, value: torch.Tensor) -> dict[str, float]:
    tensor = value.detach().float()
    return {
        f"{prefix}_mean": float(tensor.mean().cpu().item()),
        f"{prefix}_std": float(tensor.std(unbiased=False).cpu().item()),
    }


def _build_window_diagnostics(
    *,
    state_generated: torch.Tensor,
    optimized_generated: torch.Tensor,
    base_window: torch.Tensor,
    optimized_window: torch.Tensor,
    start: int,
    end: int,
    current_step: int,
    commit_index: int,
    dt: float,
    chunk_size: int,
) -> dict:
    start = int(start)
    end = int(end)
    current_step = int(current_step)
    commit_index = int(commit_index)
    current_time = float(current_step) * float(dt)
    token_indices = list(range(start, end))
    history_before = state_generated[:, :, :start, ...].detach()
    history_after = optimized_generated[:, :, :start, ...].detach()
    if history_before.numel() == 0:
        history_diff = 0.0
    else:
        history_diff = float((history_after - history_before).abs().max().cpu().item())
    delta = (optimized_window.detach().float() - base_window.detach().float()).reshape(-1)
    diagnostics = {
        **_window_stats("base_window", base_window),
        **_window_stats("optimized_window", optimized_window),
        "optimized_minus_base_l2": float(torch.linalg.vector_norm(delta).cpu().item()),
        "optimized_minus_base_mean_abs": float(delta.abs().mean().cpu().item()),
        "current_step": current_step,
        "commit_index": commit_index,
        "current_time": current_time,
        "token_indices": token_indices,
        "token_beta": _beta_for_tokens(
            token_indices=token_indices,
            current_time=current_time,
            chunk_size=int(chunk_size),
        ),
        "token_denoise_stage": [
            "initial_noise_zT" if beta >= 0.999 else "clean_z0" if beta <= 0.001 else "active_x_beta"
            for beta in _beta_for_tokens(
                token_indices=token_indices,
                current_time=current_time,
                chunk_size=int(chunk_size),
            )
        ],
        "history_before_after_diff": history_diff,
    }
    return diagnostics


def _baseline_optimized_xz_diff(
    baseline_xz: torch.Tensor,
    optimized_xz: torch.Tensor,
) -> dict[str, float]:
    n = min(int(baseline_xz.shape[0]), int(optimized_xz.shape[0]))
    if n <= 0:
        return {"mean_l2": 0.0, "max_l2": 0.0}
    diff = torch.linalg.vector_norm(
        optimized_xz[:n].detach().float() - baseline_xz[:n].detach().float(),
        dim=-1,
    )
    return {
        "mean_l2": float(diff.mean().cpu().item()),
        "max_l2": float(diff.max().cpu().item()),
    }


def _extract_window_noise(
    generated_preprocessed: torch.Tensor,
    *,
    start: int,
    end: int,
) -> torch.Tensor:
    if generated_preprocessed.dim() != 5:
        raise ValueError(
            "generated_preprocessed must have shape [B,C,T,1,1]; "
            f"got {tuple(generated_preprocessed.shape)}"
        )
    start = int(start)
    end = int(end)
    if not (0 <= start < end <= int(generated_preprocessed.shape[2])):
        raise ValueError(
            f"invalid token window [{start}, {end}) for generated length "
            f"{generated_preprocessed.shape[2]}"
        )
    return (
        generated_preprocessed[:, :, start:end, :, :]
        .squeeze(-1)
        .squeeze(-1)
        .permute(0, 2, 1)
        .detach()
        .clone()
    )


def _extract_window_tokens(
    generated_preprocessed: torch.Tensor,
    *,
    start: int,
    end: int,
    detach: bool,
) -> torch.Tensor:
    if generated_preprocessed.dim() != 5:
        raise ValueError(
            "generated_preprocessed must have shape [B,C,T,1,1]; "
            f"got {tuple(generated_preprocessed.shape)}"
        )
    start = int(start)
    end = int(end)
    if not (0 <= start <= end <= int(generated_preprocessed.shape[2])):
        raise ValueError(
            f"invalid token window [{start}, {end}) for generated length "
            f"{generated_preprocessed.shape[2]}"
        )
    tokens = (
        generated_preprocessed[:, :, start:end, :, :]
        .squeeze(-1)
        .squeeze(-1)
        .permute(0, 2, 1)
    )
    if detach:
        return tokens.detach().clone()
    return tokens


def _inject_window_noise(
    generated_preprocessed: torch.Tensor,
    window_noise: torch.Tensor,
    *,
    start: int,
) -> torch.Tensor:
    if window_noise.dim() != 3:
        raise ValueError(f"window_noise must have shape [B,T,C], got {tuple(window_noise.shape)}")
    start = int(start)
    length = int(window_noise.shape[1])
    end = start + length
    if not (0 <= start < end <= int(generated_preprocessed.shape[2])):
        raise ValueError(
            f"invalid token window [{start}, {end}) for generated length "
            f"{generated_preprocessed.shape[2]}"
        )
    replacement = window_noise.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    out = generated_preprocessed.clone()
    out[:, :, start:end, :, :] = replacement.to(device=out.device, dtype=out.dtype)
    return out


def _relative_xz_loss(
    pred_xz: torch.Tensor,
    target_xz: torch.Tensor,
    mask: torch.Tensor,
    *,
    lambda_vel: float,
) -> tuple[torch.Tensor, dict]:
    n = min(int(pred_xz.shape[0]), int(target_xz.shape[0]), int(mask.shape[0]))
    if n <= 0:
        zero = pred_xz.sum() * 0.0
        return zero, {"traj_loss": 0.0, "vel_loss": 0.0}
    pred = pred_xz[:n] - pred_xz[:1]
    target = target_xz[:n] - target_xz[:1]
    m = mask[:n].float()
    denom = m.sum().clamp(min=1.0)
    traj_loss = (((pred - target) ** 2).sum(dim=-1) * m).sum() / denom
    if n >= 2:
        pred_v = pred[1:] - pred[:-1]
        target_v = target[1:] - target[:-1]
        vm = (m[1:] * m[:-1]).float()
        vel_loss = (((pred_v - target_v) ** 2).sum(dim=-1) * vm).sum() / vm.sum().clamp(min=1.0)
    else:
        vel_loss = traj_loss.new_zeros(())
    loss = traj_loss + float(lambda_vel) * vel_loss
    return loss, {
        "traj_loss": float(traj_loss.detach().cpu().item()),
        "vel_loss": float(vel_loss.detach().cpu().item()),
    }


def _anchored_absolute_xz_loss(
    pred_xz: torch.Tensor,
    target_xz: torch.Tensor,
    mask: torch.Tensor,
    *,
    history_frames: int,
    lambda_vel: float,
    anchor_mode: str,
    generated_anchor_xz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    n = min(int(pred_xz.shape[0]), int(target_xz.shape[0]), int(mask.shape[0]))
    if n <= 0:
        zero = pred_xz.sum() * 0.0
        return zero, {
            "traj_loss": 0.0,
            "vel_loss": 0.0,
            "history_frames": 0,
            "optimized_frames": 0,
        }
    history_frames = max(0, min(int(history_frames), n))
    if history_frames >= n:
        zero = pred_xz[:n].sum() * 0.0
        return zero, {
            "traj_loss": 0.0,
            "vel_loss": 0.0,
            "history_frames": int(history_frames),
            "optimized_frames": 0,
        }
    pred = pred_xz[:n]
    target = target_xz[:n]
    m = mask[:n].float()
    pred_anchor = pred[:1].detach()
    anchor_mode = str(anchor_mode)
    if anchor_mode == "target_anchor_abs":
        world_anchor = target[:1].detach()
    elif anchor_mode == "generated_anchor_abs":
        if generated_anchor_xz is None:
            world_anchor = pred_anchor
        else:
            world_anchor = generated_anchor_xz.to(device=pred.device, dtype=pred.dtype).view(1, 2).detach()
    else:
        raise ValueError(f"unknown anchor_mode: {anchor_mode!r}")
    pred_world = pred - pred_anchor + world_anchor

    opt = slice(history_frames, n)
    opt_mask = m[opt]
    denom = opt_mask.sum().clamp(min=1.0)
    traj_loss = (((pred_world[opt] - target[opt]) ** 2).sum(dim=-1) * opt_mask).sum() / denom
    if n - history_frames >= 2:
        pred_v = pred_world[history_frames + 1 : n] - pred_world[history_frames : n - 1]
        target_v = target[history_frames + 1 : n] - target[history_frames : n - 1]
        vm = (m[history_frames + 1 : n] * m[history_frames : n - 1]).float()
        vel_loss = (((pred_v - target_v) ** 2).sum(dim=-1) * vm).sum() / vm.sum().clamp(min=1.0)
    else:
        vel_loss = traj_loss.new_zeros(())
    loss = traj_loss + float(lambda_vel) * vel_loss
    return loss, {
        "traj_loss": float(traj_loss.detach().cpu().item()),
        "vel_loss": float(vel_loss.detach().cpu().item()),
        "history_frames": int(history_frames),
        "optimized_frames": int(n - history_frames),
        "anchor_mode": anchor_mode,
    }


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


def _snapshot_model_state(model) -> dict:
    return {
        "generated": model.generated.detach().clone(),
        "current_step": int(getattr(model, "current_step", 0)),
        "commit_index": int(getattr(model, "commit_index", 0)),
        "text_condition_list": _clone_value(getattr(model, "text_condition_list", [])),
    }


def _restore_model_state(model, state: dict, *, generated: torch.Tensor | None = None) -> None:
    model.generated = state["generated"].detach().clone() if generated is None else generated
    model.current_step = int(state["current_step"])
    model.commit_index = int(state["commit_index"])
    model.text_condition_list = _clone_value(state["text_condition_list"])


def _build_step(
    *,
    model,
    stream: StreamGenerator,
    text_rollout: StreamTextRolloutController,
    conditioner: LdfEvalStreamConditioner,
    commit_index: int,
    first_chunk: bool,
    device: torch.device,
) -> tuple[dict, object]:
    current_text = text_rollout.get_text_for_commit_index(int(commit_index))
    local_commit_index = int(getattr(model, "commit_index", commit_index))
    traj_input = conditioner.build_step_payload(
        local_commit_index=local_commit_index,
        absolute_commit_index=int(commit_index),
        chunk_size=int(getattr(model, "chunk_size", 1)),
    )
    step_payload = stream.build_step_input(current_text, traj_input=traj_input)
    condition_provider = stream.build_ldf_condition_provider(
        step_payload,
        first_chunk=bool(first_chunk),
        device=device,
    )
    return step_payload, condition_provider


def _rollout_local_latents(
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


def _optimize_active_window(
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
    local_horizon_tokens: int,
    optim_steps: int,
    lr: float,
    lambda_noise: float,
    lambda_vel: float,
    history_length: int,
    loss_anchor_mode: str,
    anchor_mode: str,
    progress_every: int = 0,
) -> tuple[torch.Tensor, list[dict], dict]:
    state = _snapshot_model_state(model)
    start, end = _active_token_slice(
        int(getattr(model, "commit_index", commit_index)),
        horizon_tokens=int(local_horizon_tokens),
        target_tokens=min(int(target_tokens), int(model.generated.shape[2])),
    )
    base_window = _extract_window_noise(state["generated"], start=start, end=end)
    opt_window = base_window.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_window], lr=float(lr))
    loss_curve = []
    best = {"loss": math.inf, "window": None}
    for step in range(int(optim_steps)):
        optimizer.zero_grad(set_to_none=True)
        vae_cache = _snapshot_vae_decode_cache(vae)
        generated = _inject_window_noise(state["generated"].to(device), opt_window, start=start)
        _restore_model_state(model, state, generated=generated)
        local_latents = _rollout_local_latents(
            model=model,
            vae=vae,
            stream=stream,
            text_rollout=text_rollout,
            conditioner=conditioner,
            recovery=recovery,
            start_commit=int(commit_index),
            tokens=end - start,
            first_chunk=bool(first_chunk),
            device=device,
        )
        if str(loss_anchor_mode) == "relative_local":
            local_feature = _decode_local_latents(vae, local_latents)
            pred_xz = _root_xz_from_feature(local_feature)
            frame_start = int(commit_index) * int(frames_per_token)
            frame_end = frame_start + int(pred_xz.shape[0])
            local_target = target_xz[frame_start:frame_end]
            local_mask = mask[frame_start:frame_end]
            local_loss, parts = _relative_xz_loss(
                pred_xz,
                local_target,
                local_mask,
                lambda_vel=float(lambda_vel),
            )
            parts["loss_anchor_mode"] = "relative_local"
        elif str(loss_anchor_mode) == "history_anchor_abs":
            history_start = max(0, int(start) - int(history_length))
            history_tokens = _extract_window_tokens(
                state["generated"].to(device),
                start=history_start,
                end=int(start),
                detach=True,
            )[0]
            active_latents = torch.cat([history_tokens, local_latents], dim=0)
            active_feature = _decode_local_latents(vae, active_latents)
            pred_xz = _root_xz_from_feature(active_feature)
            frame_start = int(history_start) * int(frames_per_token)
            frame_end = frame_start + int(pred_xz.shape[0])
            local_target = target_xz[frame_start:frame_end]
            local_mask = mask[frame_start:frame_end]
            history_frames = (int(start) - int(history_start)) * int(frames_per_token)
            local_loss, parts = _anchored_absolute_xz_loss(
                pred_xz,
                local_target,
                local_mask,
                history_frames=int(history_frames),
                lambda_vel=float(lambda_vel),
                anchor_mode=str(anchor_mode),
                generated_anchor_xz=pred_xz[:1].detach(),
            )
            parts["loss_anchor_mode"] = "history_anchor_abs"
            parts["history_start_token"] = int(history_start)
        else:
            raise ValueError(f"unknown loss_anchor_mode: {loss_anchor_mode!r}")
        noise_reg = ((opt_window - base_window.to(opt_window.device)) ** 2).mean()
        loss = local_loss + float(lambda_noise) * noise_reg
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            opt_window.clamp_(-4.0, 4.0)
        _restore_vae_decode_cache(vae, vae_cache)
        _restore_model_state(model, state)
        row = {
            "commit_index": int(commit_index),
            "step": int(step),
            "loss": float(loss.detach().cpu().item()),
            **parts,
            "noise_reg": float(noise_reg.detach().cpu().item()),
        }
        loss_curve.append(row)
        if row["loss"] < best["loss"]:
            best["loss"] = row["loss"]
            best["window"] = opt_window.detach().clone()
        if int(progress_every) > 0 and (
            step == 0 or (step + 1) % int(progress_every) == 0 or step + 1 == int(optim_steps)
        ):
            print(
                "chunk-opt "
                f"commit={int(commit_index)} window={end - start} "
                f"step={step + 1}/{int(optim_steps)} "
                f"loss={row['loss']:.6g}",
                flush=True,
            )
    chosen = best["window"] if best["window"] is not None else opt_window.detach()
    optimized_generated = _inject_window_noise(state["generated"].to(device), chosen, start=start)
    _restore_model_state(model, state, generated=optimized_generated.detach())
    diagnostics = _build_window_diagnostics(
        state_generated=state["generated"].to(device),
        optimized_generated=optimized_generated.detach(),
        base_window=base_window,
        optimized_window=chosen.detach(),
        start=start,
        end=end,
        current_step=int(state["current_step"]),
        commit_index=int(state["commit_index"]),
        dt=float(getattr(model, "dt", 1.0 / max(1, int(getattr(model, "num_denoise_steps", 1))))),
        chunk_size=int(getattr(model, "chunk_size", 1)),
    )
    return chosen.detach(), loss_curve, diagnostics


def _run_stream_with_chunk_optimization(
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
    local_horizon_tokens: int,
    optim_steps: int,
    lr: float,
    lambda_noise: float,
    lambda_vel: float,
    loss_anchor_mode: str,
    anchor_mode: str,
    progress_every: int = 0,
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
    window_rows = []
    commit_index = 0
    while commit_index < int(target_tokens):
        window_tokens = min(int(local_horizon_tokens), int(target_tokens) - int(commit_index))
        _, rows, diagnostics = _optimize_active_window(
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
            local_horizon_tokens=int(window_tokens),
            optim_steps=int(optim_steps),
            lr=float(lr),
            lambda_noise=float(lambda_noise),
            lambda_vel=float(lambda_vel),
            history_length=int(history_length),
            loss_anchor_mode=str(loss_anchor_mode),
            anchor_mode=str(anchor_mode),
            progress_every=int(progress_every),
        )
        loss_curve.extend(rows)
        window_rows.append(
            {
                "commit_index": int(commit_index),
                "tokens": int(window_tokens),
                "loss_anchor_mode": str(loss_anchor_mode),
                "anchor_mode": str(anchor_mode),
                "best_loss": min((row["loss"] for row in rows), default=float("nan")),
                "diagnostics": diagnostics,
            }
        )
        for _ in range(int(window_tokens)):
            if commit_index >= int(target_tokens):
                break
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
            commit_index += 1
        model.generated = model.generated.detach()
    vae.clear_cache()
    return torch.cat(latent_tokens, dim=0), loss_curve, window_rows


def _run_one(
    *,
    args: argparse.Namespace,
    cfg,
    model,
    vae,
    vae_ckpt,
    device: torch.device,
    sample_batch: dict,
    seed: int,
    out_dir: Path,
) -> dict:
    _set_seed(int(seed))
    sample_batch = _cap_sample_to_frames(
        sample_batch,
        int(args.max_frames),
        frames_per_token=int(args.frames_per_token),
    )
    target_xz, mask = _target_xz_and_mask(sample_batch, device)
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
        baseline_metrics = _masked_metrics(baseline_xz, target_xz, mask)

    optimized_latent, loss_curve, window_rows = _run_stream_with_chunk_optimization(
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
        local_horizon_tokens=int(args.local_horizon_tokens),
        optim_steps=int(args.optim_steps),
        lr=float(args.lr),
        lambda_noise=float(args.lambda_noise),
        lambda_vel=float(args.lambda_vel),
        loss_anchor_mode=str(args.loss_anchor_mode),
        anchor_mode=str(args.anchor_mode),
        progress_every=int(args.progress_every),
    )
    with torch.no_grad():
        optimized_feature = _decode_latent_stream(vae, optimized_latent.to(device)).detach()
        optimized_xz = _root_xz_from_feature(optimized_feature)
        optimized_metrics = _masked_metrics(optimized_xz, target_xz, mask)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "stream_generate_step_active_noisy_state_opt",
        **_active_state_method_metadata(),
        "loss_anchor_mode": str(args.loss_anchor_mode),
        "anchor_mode": str(args.anchor_mode),
        "sample_name": str(sample_batch["name"][0]),
        "caption_index": sample_batch.get("_caption_index", int(args.caption_index)),
        "caption_text": sample_batch.get("_caption_text", sample_batch.get("text", [""])[0]),
        "ckpt": str(args.ckpt),
        "vae_ckpt": str(vae_ckpt),
        "cfg_text": float(args.cfg_text),
        "cfg_traj": float(args.cfg_traj),
        "history_length": int(args.history_length),
        "traj_horizon_tokens": int(args.traj_horizon_tokens),
        "local_horizon_tokens": int(args.local_horizon_tokens),
        "num_denoise_steps": args.num_denoise_steps,
        "optim_steps": int(args.optim_steps),
        "progress_every": int(args.progress_every),
        "lr": float(args.lr),
        "seed": int(seed),
        "lambda_noise": float(args.lambda_noise),
        "lambda_vel": float(args.lambda_vel),
        "initial_buffer_tokens": int(initial_shape[1]),
        "target_frames": int(target_xz.shape[0]),
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "delta": {
            "ade": optimized_metrics["ade"] - baseline_metrics["ade"],
            "fde": optimized_metrics["fde"] - baseline_metrics["fde"],
            "mse": optimized_metrics["mse"] - baseline_metrics["mse"],
        },
        "noise": {"base": _noise_stats(base_noise)},
        "baseline_optimized_xz_diff": _baseline_optimized_xz_diff(
            baseline_xz,
            optimized_xz,
        ),
        "optim_steps_zero_baseline_diff": (
            _baseline_optimized_xz_diff(baseline_xz, optimized_xz)
            if int(args.optim_steps) == 0
            else None
        ),
        "windows": window_rows,
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
            out_dir / "optimized_active_noisy_state.mp4",
            dim=263,
            traj_xz=target_xz,
            traj_mask=traj_mask,
            cond_traj_mask=mask,
            render_setting=render_setting,
        )
    print(json.dumps(summary, indent=2))
    print(f"wrote: {out_dir}")
    return summary


def _metric_mean_std(rows: list[dict], section: str, metric: str) -> dict:
    values = [float(row[section][metric]) for row in rows if section in row and metric in row[section]]
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    mean = float(sum(values) / len(values))
    if len(values) <= 1:
        std = 0.0
    else:
        std = float((sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5)
    return {"mean": mean, "std": std}


def _aggregate_summaries(summaries: list[dict]) -> dict:
    aggregate = {
        **_active_state_method_metadata(),
        "num_runs": int(len(summaries)),
        "seeds": [int(row["seed"]) for row in summaries],
        "loss_anchor_mode": summaries[0].get("loss_anchor_mode") if summaries else None,
        "anchor_mode": summaries[0].get("anchor_mode") if summaries else None,
        "baseline": {},
        "optimized": {},
        "delta": {},
        "runs": summaries,
    }
    for section in ("baseline", "optimized", "delta"):
        for metric in ("ade", "fde", "mse"):
            aggregate[section][metric] = _metric_mean_std(summaries, section, metric)
    return aggregate


def _base_output_dir(args: argparse.Namespace) -> Path:
    return (
        Path(args.out_dir)
        / _ckpt_tag(args.ckpt)
        / f"sample_{args.sample_name}"
        / f"cap{int(args.caption_index)}"
        / f"h{int(args.history_length):03d}_hor{int(args.traj_horizon_tokens):03d}"
        / f"local{int(args.local_horizon_tokens):02d}_steps{int(args.optim_steps)}_lr{str(args.lr).replace('.', 'p')}"
    )


def main() -> int:
    args = _parse_args()
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
    num_runs = max(1, int(args.num_runs))
    start_seed = int(args.start_seed if num_runs > 1 else args.seed)
    base_dir = _base_output_dir(args)
    if num_runs > 1:
        run_parent = base_dir / f"runs{num_runs}"
    else:
        run_parent = base_dir
    summaries = []
    for run_idx in range(num_runs):
        seed = start_seed + run_idx
        if num_runs > 1:
            out_dir = run_parent / f"seed{seed}"
        else:
            out_dir = run_parent / str(args.run_tag or f"seed{seed}")
        print(
            f"starting run {run_idx + 1}/{num_runs}: "
            f"seed={seed}, loss_anchor_mode={args.loss_anchor_mode}",
            flush=True,
        )
        summary = _run_one(
            args=args,
            cfg=cfg,
            model=model,
            vae=vae,
            vae_ckpt=vae_ckpt,
            device=device,
            sample_batch=dict(sample_batch),
            seed=seed,
            out_dir=out_dir,
        )
        summaries.append(summary)
    if num_runs > 1:
        aggregate = _aggregate_summaries(summaries)
        run_parent.mkdir(parents=True, exist_ok=True)
        (run_parent / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2))
        print(json.dumps(aggregate, indent=2))
        print(f"wrote aggregate: {run_parent / 'aggregate_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
