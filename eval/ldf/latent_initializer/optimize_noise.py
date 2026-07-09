"""Oracle optimize initial diffusion noise for one LDF sample.

This is an eval-only prototype for testing whether a better z_T exists under a
frozen LDF denoiser and frozen VAE. It uses full-sequence ``model.generate`` so
the first version can keep a differentiable path from initial noise to decoded
root trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

from eval.ldf.stream_setup import (  # noqa: E402
    build_eval_dataloader,
    enable_cpu_text_encoding,
    load_eval_model_and_vae,
)
from metrics.traj import _slice_single_sample_batch  # noqa: E402
from utils.initialize import load_config  # noqa: E402
from utils.motion_process import extract_root_trajectory_263_torch  # noqa: E402
from utils.token_frame import num_tokens_for_frame_len  # noqa: E402
from utils.training.ldf.validation_conditioning import (  # noqa: E402
    prepare_ldf_eval_model_batch,
)


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
    parser.add_argument("--num_denoise_steps", type=int, default=None)
    parser.add_argument("--optim_steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--lambda_noise", type=float, default=1e-4)
    parser.add_argument("--lambda_vel", type=float, default=0.05)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        default="eval/output_eval/ldf/latent_initializer/oracle_noise",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _select_caption(sample_batch: dict, caption_index: int) -> None:
    text_all = sample_batch.get("text_all")
    if not text_all:
        return
    captions = text_all[0] if isinstance(text_all, list) else text_all
    if captions and isinstance(captions[0], list):
        captions = captions[0]
    index = int(caption_index)
    if index < 0 or index >= len(captions):
        raise ValueError(f"caption_index={index} out of range for {len(captions)} captions")
    sample_batch["text"] = [str(captions[index])]
    sample_batch["_caption_index"] = index
    sample_batch["_caption_text"] = str(captions[index])


def _load_single_sample(args: argparse.Namespace, cfg) -> dict:
    _, loader = build_eval_dataloader(
        cfg,
        meta_paths=[args.meta_path],
        batch_size=1,
        num_workers=0,
        group_present_segments=False,
    )
    last_name = None
    for batch in loader:
        sample = _slice_single_sample_batch(batch, 0)
        name = str(sample["name"][0])
        last_name = name
        if not args.sample_name or name == str(args.sample_name):
            _select_caption(sample, int(args.caption_index))
            return sample
    raise ValueError(
        f"sample {args.sample_name!r} not found in {args.meta_path!r}; "
        f"last sample seen was {last_name!r}"
    )


def _cap_sample_to_frames(sample_batch: dict, max_frames: int, *, frames_per_token: int = 4) -> dict:
    max_frames = int(max_frames)
    if max_frames <= 0:
        return sample_batch
    current_frames = int(sample_batch["feature_length"][0].item())
    capped_frames = max(1, min(max_frames, current_frames))
    capped_tokens = num_tokens_for_frame_len(capped_frames, int(frames_per_token))
    out = dict(sample_batch)
    for key in ("feature", "traj", "traj_cond", "traj_cond_7d", "traj_cond_mask", "traj_mask", "traj_loss_mask"):
        value = out.get(key)
        if value is None:
            continue
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == 1:
            out[key] = value[:, :capped_frames].clone()
    token = out.get("token")
    if torch.is_tensor(token) and token.ndim >= 2 and token.shape[0] == 1:
        out["token"] = token[:, :capped_tokens].clone()
    out["feature_length"] = torch.as_tensor([capped_frames], dtype=sample_batch["feature_length"].dtype)
    out["traj_length"] = torch.as_tensor([capped_frames], dtype=sample_batch.get("traj_length", sample_batch["feature_length"]).dtype)
    out["token_length"] = torch.as_tensor([capped_tokens], dtype=sample_batch["token_length"].dtype)
    if "token_text_end" in out:
        token_end = out["token_text_end"]
        if torch.is_tensor(token_end):
            out["token_text_end"] = torch.clamp(token_end.clone(), max=capped_tokens)
        elif isinstance(token_end, list):
            out["token_text_end"] = [[min(int(v), capped_tokens) for v in token_end[0]]]
    return out


def _target_xz_and_mask(sample_batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    traj = sample_batch.get("traj_cond_7d")
    if traj is None:
        raise KeyError("sample_batch must contain traj_cond_7d")
    traj = traj[0] if torch.is_tensor(traj) and traj.ndim == 3 else torch.as_tensor(traj)
    traj = traj.to(device=device, dtype=torch.float32)
    xz = traj[:, [0, 2]]
    mask = sample_batch.get("traj_cond_mask", sample_batch.get("traj_mask"))
    if mask is None:
        mask_t = torch.ones(xz.shape[0], device=device, dtype=torch.float32)
    else:
        mask_t = mask[0] if torch.is_tensor(mask) and mask.ndim == 2 else torch.as_tensor(mask)
        mask_t = mask_t.to(device=device, dtype=torch.float32).view(-1)
    valid_frames = int(sample_batch["feature_length"][0].item())
    valid_frames = min(valid_frames, int(xz.shape[0]), int(mask_t.shape[0]))
    return xz[:valid_frames], mask_t[:valid_frames]


def _decode_latent(vae, latent: torch.Tensor) -> torch.Tensor:
    return vae.decode(latent.unsqueeze(0))[0].float()


def _root_xz_from_feature(feature: torch.Tensor) -> torch.Tensor:
    return extract_root_trajectory_263_torch(feature.unsqueeze(0))[0][:, [0, 2]]


def _masked_metrics(pred_xz: torch.Tensor, target_xz: torch.Tensor, mask: torch.Tensor) -> dict:
    n = min(int(pred_xz.shape[0]), int(target_xz.shape[0]), int(mask.shape[0]))
    pred = pred_xz[:n].detach()
    target = target_xz[:n].detach()
    m = mask[:n].detach().float()
    valid = m > 0
    if not bool(valid.any()):
        return {"ade": float("nan"), "fde": float("nan"), "mse": float("nan"), "valid_frames": 0}
    diff = pred - target
    l2 = torch.linalg.norm(diff, dim=-1)
    sq = (diff * diff).sum(dim=-1)
    last_idx = int(valid.nonzero(as_tuple=False)[-1].item())
    return {
        "ade": float((l2 * m).sum().item() / m.sum().item()),
        "fde": float(l2[last_idx].item()),
        "mse": float((sq * m).sum().item() / m.sum().item()),
        "valid_frames": int(valid.sum().item()),
    }


def _loss_from_feature(
    feature: torch.Tensor,
    *,
    target_xz: torch.Tensor,
    mask: torch.Tensor,
    base_noise: torch.Tensor,
    opt_noise: torch.Tensor,
    lambda_noise: float,
    lambda_vel: float,
) -> tuple[torch.Tensor, dict]:
    pred_xz = _root_xz_from_feature(feature)
    n = min(int(pred_xz.shape[0]), int(target_xz.shape[0]), int(mask.shape[0]))
    pred = pred_xz[:n]
    target = target_xz[:n]
    m = mask[:n].float()
    denom = m.sum().clamp(min=1.0)
    diff = pred - target
    traj_loss = ((diff * diff).sum(dim=-1) * m).sum() / denom
    if n >= 2:
        pred_v = pred[1:] - pred[:-1]
        target_v = target[1:] - target[:-1]
        vm = (m[1:] * m[:-1]).float()
        vel_loss = (((pred_v - target_v) ** 2).sum(dim=-1) * vm).sum() / vm.sum().clamp(min=1.0)
    else:
        vel_loss = traj_loss.new_zeros(())
    noise_reg = ((opt_noise - base_noise) ** 2).mean()
    loss = traj_loss + float(lambda_vel) * vel_loss + float(lambda_noise) * noise_reg
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "traj_loss": float(traj_loss.detach().cpu().item()),
        "vel_loss": float(vel_loss.detach().cpu().item()),
        "noise_reg": float(noise_reg.detach().cpu().item()),
    }


def _plot_paths(path: Path, target_xz, baseline_xz, optimized_xz) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.plot(target_xz[:, 0], target_xz[:, 1], label="target", linewidth=2)
    plt.plot(baseline_xz[:, 0], baseline_xz[:, 1], label="gaussian", linewidth=1.5)
    plt.plot(optimized_xz[:, 0], optimized_xz[:, 1], label="optimized_zT", linewidth=1.5)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _noise_stats(noise: torch.Tensor) -> dict:
    value = noise.detach().float().cpu()
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "rms": float(torch.sqrt((value * value).mean()).item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def _ckpt_tag(path: str) -> str:
    stem = Path(path).stem
    if stem.startswith("step_"):
        return "ckpt_" + stem[len("step_") :]
    return stem


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
        frames_per_token=int(cfg.validation.get("eval_stream_frames_per_token", 4)),
    )
    model_batch = prepare_ldf_eval_model_batch(sample_batch, device, model=model)
    condition = model_batch["ldf_condition"]
    target_xz, mask = _target_xz_and_mask(sample_batch, device)
    latent_len = int(model_batch["feature_length"][0].item())
    noise_shape = (1, latent_len + int(model.chunk_size), int(model.input_dim))
    base_noise = torch.randn(noise_shape, device=device)

    with torch.no_grad():
        baseline_latent = model.generate(
            model_batch,
            condition=condition,
            num_denoise_steps=args.num_denoise_steps,
            initial_noise=base_noise,
        )["generated"][0]
        baseline_feature = _decode_latent(vae, baseline_latent).detach()
        baseline_xz = _root_xz_from_feature(baseline_feature)
        baseline_metrics = _masked_metrics(baseline_xz, target_xz, mask)

    opt_noise = base_noise.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_noise], lr=float(args.lr))
    loss_curve = []
    best = {
        "loss": math.inf,
        "noise": None,
        "latent": None,
        "feature": None,
        "metrics": None,
    }
    for step in range(int(args.optim_steps)):
        optimizer.zero_grad(set_to_none=True)
        latent = model.generate(
            model_batch,
            condition=condition,
            num_denoise_steps=args.num_denoise_steps,
            initial_noise=opt_noise,
        )["generated"][0]
        feature = _decode_latent(vae, latent)
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
        optimizer.step()
        with torch.no_grad():
            opt_noise.clamp_(-4.0, 4.0)
        loss_curve.append({"step": int(step), **parts})
        if parts["loss"] < best["loss"]:
            with torch.no_grad():
                best["loss"] = parts["loss"]
                best["noise"] = opt_noise.detach().clone()
                best["latent"] = latent.detach().clone()
                best["feature"] = feature.detach().clone()
                best["metrics"] = _masked_metrics(
                    _root_xz_from_feature(feature.detach()), target_xz, mask
                )

    optimized_noise = best["noise"] if best["noise"] is not None else opt_noise.detach()
    with torch.no_grad():
        optimized_latent = model.generate(
            model_batch,
            condition=condition,
            num_denoise_steps=args.num_denoise_steps,
            initial_noise=optimized_noise,
        )["generated"][0]
        optimized_feature = _decode_latent(vae, optimized_latent).detach()
        optimized_xz = _root_xz_from_feature(optimized_feature)
        optimized_metrics = _masked_metrics(optimized_xz, target_xz, mask)

    out_dir = (
        Path(args.out_dir)
        / _ckpt_tag(args.ckpt)
        / f"sample_{args.sample_name}"
        / f"cap{int(args.caption_index)}"
        / f"steps{int(args.optim_steps)}_lr{str(args.lr).replace('.', 'p')}"
        / str(args.run_tag or f"seed{int(args.seed)}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "sample_name": str(sample_batch["name"][0]),
        "caption_index": sample_batch.get("_caption_index", int(args.caption_index)),
        "caption_text": sample_batch.get("_caption_text", sample_batch.get("text", [""])[0]),
        "ckpt": str(args.ckpt),
        "vae_ckpt": str(vae_ckpt),
        "cfg_text": float(args.cfg_text),
        "cfg_traj": float(args.cfg_traj),
        "num_denoise_steps": args.num_denoise_steps,
        "optim_steps": int(args.optim_steps),
        "lr": float(args.lr),
        "lambda_noise": float(args.lambda_noise),
        "lambda_vel": float(args.lambda_vel),
        "latent_len": int(latent_len),
        "target_frames": int(target_xz.shape[0]),
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "delta": {
            "ade": optimized_metrics["ade"] - baseline_metrics["ade"],
            "fde": optimized_metrics["fde"] - baseline_metrics["fde"],
            "mse": optimized_metrics["mse"] - baseline_metrics["mse"],
        },
        "noise": {
            "base": _noise_stats(base_noise),
            "optimized": _noise_stats(optimized_noise),
            "delta": _noise_stats(optimized_noise - base_noise),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2))
    torch.save(optimized_noise.detach().cpu(), out_dir / "optimized_zT.pt")
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
    print(json.dumps(summary, indent=2))
    print(f"wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
