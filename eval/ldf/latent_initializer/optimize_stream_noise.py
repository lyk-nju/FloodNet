"""Oracle optimize the initial rolling noise buffer for stream_generate_step."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.ldf.conditioning import LdfEvalStreamConditioner  # noqa: E402
from eval.common.visualization import render_motion_video  # noqa: E402
from eval.ldf.latent_initializer.optimize_noise import (  # noqa: E402
    _cap_sample_to_frames,
    _ckpt_tag,
    _load_single_sample,
    _loss_from_feature,
    _masked_metrics,
    _noise_stats,
    _plot_paths,
    _root_xz_from_feature,
    _set_seed,
    _target_xz_and_mask,
)
from eval.ldf.stream_generation import StreamTextRolloutController  # noqa: E402
from eval.ldf.stream_setup import (  # noqa: E402
    enable_cpu_text_encoding,
    load_eval_model_and_vae,
)
from models.diffusion_forcing_wan import DiffForcingWanModel  # noqa: E402
from utils.inference.stream_generator import StreamGenerator  # noqa: E402
from utils.initialize import load_config  # noqa: E402
from utils.motion_process import StreamJointRecovery263  # noqa: E402
from utils.token_frame import num_tokens_for_frame_len  # noqa: E402


def _true_zT_method_metadata() -> dict[str, str]:
    return {
        "method": "true_future_zT_oracle",
        "optimized_variable": "independent initial_generated Gaussian z_T buffer",
        "not_optimized_variable": "model.generated[start:end] current x_beta",
        "oracle_line": "B_true_zT_initializer",
        "commit_strategy": "full_stream_shadow_rollout",
        "decode_surrogate": "offline_vae_decode",
        "decode_surrogate_note": (
            "Optimization loss is computed with vae.decode(latent_stream) as a "
            "differentiable offline surrogate; final runtime behavior may differ "
            "when using VAE stream_decode/cache."
        ),
    }


def _path_length(xz: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    n = int(xz.shape[0])
    if n < 2:
        return 0.0
    points = xz.detach().float()
    if mask is not None:
        valid = mask.detach().float().view(-1)[:n] > 0
        if not bool(valid.any()):
            return 0.0
        last = int(valid.nonzero(as_tuple=False)[-1].item()) + 1
        points = points[:last]
    if int(points.shape[0]) < 2:
        return 0.0
    return float(torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).sum().cpu().item())


def _heading_error_deg_from_xz(
    pred_xz: torch.Tensor,
    target_xz: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    n = min(int(pred_xz.shape[0]), int(target_xz.shape[0]), int(mask.shape[0]))
    if n < 2:
        return float("nan")
    pred = pred_xz[:n].detach().float()
    target = target_xz[:n].detach().float()
    m = mask[:n].detach().float()
    pred_v = pred[1:] - pred[:-1]
    target_v = target[1:] - target[:-1]
    vm = (m[1:] * m[:-1]) > 0
    speed_ok = (torch.linalg.vector_norm(pred_v, dim=-1) > 1e-6) & (
        torch.linalg.vector_norm(target_v, dim=-1) > 1e-6
    )
    valid = vm & speed_ok
    if not bool(valid.any()):
        return float("nan")
    pred_yaw = torch.atan2(pred_v[:, 0], pred_v[:, 1])
    target_yaw = torch.atan2(target_v[:, 0], target_v[:, 1])
    diff = pred_yaw - target_yaw
    wrapped = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()
    return float(torch.rad2deg(wrapped[valid]).mean().cpu().item())


def _attach_root_path_metrics(
    metrics: dict,
    *,
    xz: torch.Tensor,
    target_xz: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    out = dict(metrics)
    if "ade" in out:
        out["ADE"] = out["ade"]
    if "fde" in out:
        out["FDE"] = out["fde"]
    pred_len = _path_length(xz, mask)
    target_len = _path_length(target_xz, mask)
    out["heading_error_deg"] = _heading_error_deg_from_xz(xz, target_xz, mask)
    out["path_length"] = pred_len
    out["target_path_length"] = target_len
    out["path_ratio"] = float(pred_len / target_len) if target_len > 1e-8 else float("nan")
    return out


def _xz_diff_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return {"mean_l2": 0.0, "max_l2": 0.0}
    diff = torch.linalg.vector_norm(a[:n].detach().float() - b[:n].detach().float(), dim=-1)
    return {
        "mean_l2": float(diff.mean().cpu().item()),
        "max_l2": float(diff.max().cpu().item()),
    }


def _noise_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((a.detach().float() - b.detach().float()).reshape(-1)).cpu().item())


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
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--lambda_noise", type=float, default=1e-4)
    parser.add_argument("--lambda_vel", type=float, default=0.05)
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--skip_zero_step_sanity", action="store_true")
    parser.add_argument("--no_render_video", action="store_true")
    parser.add_argument(
        "--out_dir",
        default="eval/output_eval/ldf/latent_initializer/stream_oracle_noise",
    )
    return parser.parse_args()


def _decode_latent_stream(vae, latent_stream: torch.Tensor) -> torch.Tensor:
    vae.clear_cache()
    return vae.decode(latent_stream.unsqueeze(0))[0].float()


def _run_stream_latents(
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
    differentiable: bool,
) -> torch.Tensor:
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
    for commit_index in range(int(target_tokens)):
        current_text = text_rollout.get_text_for_commit_index(commit_index)
        local_commit_index = int(getattr(model, "commit_index", commit_index))
        traj_input = conditioner.build_step_payload(
            local_commit_index=local_commit_index,
            absolute_commit_index=commit_index,
            chunk_size=int(getattr(model, "chunk_size", 1)),
        )
        step_payload = stream.build_step_input(current_text, traj_input=traj_input)
        condition_provider = stream.build_ldf_condition_provider(
            step_payload,
            first_chunk=first_chunk,
            device=device,
        )
        if differentiable:
            output = DiffForcingWanModel.stream_generate_step.__wrapped__(
                model,
                step_payload,
                first_chunk=first_chunk,
                condition=condition_provider,
            )
        else:
            output = model.stream_generate_step(
                step_payload,
                first_chunk=first_chunk,
                condition=condition_provider,
            )
        latent_token = output["generated"][0]
        latent_tokens.append(latent_token)
        with torch.no_grad():
            decoded_chunk = vae.stream_decode(
                latent_token.detach()[None, :],
                first_chunk=first_chunk,
            )[0].float().detach().cpu()
            conditioner.append_decoded(
                decoded_chunk,
                commit_idx=commit_index + 1,
                recovery=recovery,
            )
        first_chunk = False
    vae.clear_cache()
    return torch.cat(latent_tokens, dim=0)


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

        zero_step_sanity = None
        if not bool(args.skip_zero_step_sanity):
            zero_step_latent = _run_stream_latents(
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
            zero_step_feature = _decode_latent_stream(vae, zero_step_latent).detach()
            zero_step_xz = _root_xz_from_feature(zero_step_feature)
            latent_max_abs = float(
                (zero_step_latent.detach().float() - baseline_latent.detach().float())
                .abs()
                .max()
                .cpu()
                .item()
            )
            xz_diff = _xz_diff_metrics(baseline_xz, zero_step_xz)
            zero_step_sanity = {
                "enabled": True,
                "baseline_vs_zero_step_xz": xz_diff,
                "baseline_vs_zero_step_latent_max_abs": latent_max_abs,
                "passes": bool(latent_max_abs <= 1e-6 and xz_diff["max_l2"] <= 1e-6),
            }
        else:
            zero_step_sanity = {"enabled": False}

    opt_noise = base_noise.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_noise], lr=float(args.lr))
    loss_curve = []
    gradient_check = {
        "model_trainable_param_count": int(sum(1 for p in model.parameters() if p.requires_grad)),
        "vae_trainable_param_count": int(sum(1 for p in vae.parameters() if p.requires_grad)),
        "opt_noise_grad_nonzero_steps": 0,
        "first_opt_noise_grad_norm": None,
        "last_opt_noise_grad_norm": None,
        "only_independent_initial_generated_buffer": False,
    }
    best = {
        "loss": math.inf,
        "noise": None,
        "latent": None,
        "feature": None,
        "metrics": None,
    }
    for step in range(int(args.optim_steps)):
        optimizer.zero_grad(set_to_none=True)
        latent = _run_stream_latents(
            model=model,
            vae=vae,
            sample_batch=sample_batch,
            device=device,
            history_length=int(args.history_length),
            traj_horizon_tokens=int(args.traj_horizon_tokens),
            token_dt=float(args.token_dt),
            frames_per_token=int(args.frames_per_token),
            num_denoise_steps=args.num_denoise_steps,
            initial_generated=opt_noise,
            differentiable=True,
        )
        feature = _decode_latent_stream(vae, latent)
        loss, parts = _loss_from_feature(
            feature,
            target_xz=target_xz,
            mask=mask,
            base_noise=base_noise,
            opt_noise=opt_noise,
            lambda_noise=float(args.lambda_noise),
            lambda_vel=float(args.lambda_vel),
        )
        loss.backward()
        grad_norm = None
        if opt_noise.grad is not None:
            grad_norm = float(torch.linalg.vector_norm(opt_noise.grad.detach().float()).cpu().item())
            if grad_norm > 0.0:
                gradient_check["opt_noise_grad_nonzero_steps"] += 1
            if gradient_check["first_opt_noise_grad_norm"] is None:
                gradient_check["first_opt_noise_grad_norm"] = grad_norm
            gradient_check["last_opt_noise_grad_norm"] = grad_norm
        optimizer.step()
        with torch.no_grad():
            opt_noise.clamp_(-4.0, 4.0)
        loss_curve.append({"step": int(step), **parts, "opt_noise_grad_norm": grad_norm})
        if parts["loss"] < best["loss"]:
            with torch.no_grad():
                best["loss"] = parts["loss"]
                best["noise"] = opt_noise.detach().clone()
                best["latent"] = latent.detach().clone()
                best["feature"] = feature.detach().clone()
                best["metrics"] = _masked_metrics(
                    _root_xz_from_feature(feature.detach()), target_xz, mask
                )
    gradient_check["only_independent_initial_generated_buffer"] = bool(
        gradient_check["model_trainable_param_count"] == 0
        and gradient_check["vae_trainable_param_count"] == 0
        and gradient_check["opt_noise_grad_nonzero_steps"] > 0
    )

    optimized_noise = best["noise"] if best["noise"] is not None else opt_noise.detach()
    with torch.no_grad():
        optimized_latent = _run_stream_latents(
            model=model,
            vae=vae,
            sample_batch=sample_batch,
            device=device,
            history_length=int(args.history_length),
            traj_horizon_tokens=int(args.traj_horizon_tokens),
            token_dt=float(args.token_dt),
            frames_per_token=int(args.frames_per_token),
            num_denoise_steps=args.num_denoise_steps,
            initial_generated=optimized_noise,
            differentiable=False,
        )
        optimized_feature = _decode_latent_stream(vae, optimized_latent).detach()
        optimized_xz = _root_xz_from_feature(optimized_feature)
        optimized_metrics = _attach_root_path_metrics(
            _masked_metrics(optimized_xz, target_xz, mask),
            xz=optimized_xz,
            target_xz=target_xz,
            mask=mask,
        )

    out_dir = (
        Path(args.out_dir)
        / _ckpt_tag(args.ckpt)
        / f"sample_{args.sample_name}"
        / f"cap{int(args.caption_index)}"
        / f"h{int(args.history_length):03d}_hor{int(args.traj_horizon_tokens):03d}"
        / f"steps{int(args.optim_steps)}_lr{str(args.lr).replace('.', 'p')}"
        / str(args.run_tag or f"seed{int(args.seed)}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "stream_generate_step_true_initial_zT_oracle",
        **_true_zT_method_metadata(),
        "sample_name": str(sample_batch["name"][0]),
        "caption_index": sample_batch.get("_caption_index", int(args.caption_index)),
        "caption_text": sample_batch.get("_caption_text", sample_batch.get("text", [""])[0]),
        "ckpt": str(args.ckpt),
        "vae_ckpt": str(vae_ckpt),
        "cfg_text": float(args.cfg_text),
        "cfg_traj": float(args.cfg_traj),
        "history_length": int(args.history_length),
        "traj_horizon_tokens": int(args.traj_horizon_tokens),
        "num_denoise_steps": args.num_denoise_steps,
        "optim_steps": int(args.optim_steps),
        "lr": float(args.lr),
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
            "heading_error_deg": (
                optimized_metrics["heading_error_deg"]
                - baseline_metrics["heading_error_deg"]
            ),
            "path_ratio": optimized_metrics["path_ratio"] - baseline_metrics["path_ratio"],
        },
        "noise": {
            "base": _noise_stats(base_noise),
            "optimized": _noise_stats(optimized_noise),
            "delta": _noise_stats(optimized_noise - base_noise),
        },
        "optimized_zT": {
            **_noise_stats(optimized_noise),
            "optimized_minus_base_l2": _noise_l2(optimized_noise, base_noise),
        },
        "gradient_check": gradient_check,
        "opt_steps_zero_sanity": zero_step_sanity,
        "loss_curve": loss_curve,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2))
    torch.save(optimized_noise.detach().cpu(), out_dir / "optimized_stream_zT.pt")
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
            out_dir / "optimized_zT.mp4",
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
