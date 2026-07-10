"""Evaluate a trained frontier NoiseInitializer on one streaming sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.ldf.conditioning import LdfEvalStreamConditioner  # noqa: E402
from eval.ldf.latent_initializer.optimize_noise import (  # noqa: E402
    _cap_sample_to_frames,
    _load_single_sample,
    _masked_metrics,
    _plot_paths,
    _root_xz_from_feature,
    _target_xz_and_mask,
)
from eval.ldf.latent_initializer.optimize_stream_chunk_noise import (  # noqa: E402
    _baseline_optimized_xz_diff,
    _build_step,
)
from eval.ldf.latent_initializer.optimize_stream_noise import (  # noqa: E402
    _attach_root_path_metrics,
    _decode_latent_stream,
)
from eval.ldf.stream_generation import StreamTextRolloutController  # noqa: E402
from eval.ldf.stream_setup import enable_cpu_text_encoding, load_eval_model_and_vae  # noqa: E402
from models.noise_initializer import NoiseInitializer  # noqa: E402
from utils.inference.stream_generator import StreamGenerator  # noqa: E402
from utils.initialize import load_config  # noqa: E402
from utils.motion_process import StreamJointRecovery263  # noqa: E402
from utils.token_frame import (  # noqa: E402
    num_tokens_for_frame_len,
    token_range_to_frame_slice,
)
from utils.training.noise_initializer.overfit_runner import (  # noqa: E402
    _build_initializer_context_for_commit,
    _build_runtime_traj_payload,
    advance_model_token_update_count,
    apply_initializer_to_stream_state,
    should_apply_initializer,
)
from utils.training.noise_initializer.text_encoder import (  # noqa: E402
    resolve_noise_initializer_text_encoder,
)


def _parse_overrides(items: list[str] | None) -> dict[str, str]:
    overrides = {}
    for item in items or []:
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/noise_initializer_overfit.yaml")
    parser.add_argument("--initializer_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--alphas", default="0,1")
    parser.add_argument("--include_gaussian", action="store_true")
    return parser.parse_args()


def _load_initializer(
    ckpt_path: Path,
    cfg: dict,
    device: torch.device,
) -> tuple[NoiseInitializer, dict]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("cfg") or {}
    model_params = dict(((ckpt_cfg.get("model") or {}).get("params") or {}))
    model_params.update(dict(((cfg.get("model") or {}).get("params") or {})))
    initializer = NoiseInitializer(**model_params).to(device)
    initializer.load_state_dict(checkpoint["state_dict"], strict=True)
    initializer.eval()
    for parameter in initializer.parameters():
        parameter.requires_grad_(False)
    return initializer, dict(ckpt_cfg)


def _run_stream_variant(
    *,
    model,
    vae,
    initializer: NoiseInitializer | None,
    initializer_text_encoder,
    sample_batch: dict,
    cfg: dict,
    initializer_training_cfg: dict,
    device: torch.device,
    initial_generated: torch.Tensor,
    alpha: float | None,
) -> tuple[torch.Tensor, list[dict]]:
    history_tokens = int(cfg.get("history_tokens", cfg.get("history_length", 30)))
    traj_horizon_tokens = int(cfg.get("traj_horizon_tokens", 20))
    frames_per_token = int(cfg.get("frames_per_token", 4))
    target_tokens = num_tokens_for_frame_len(
        int(sample_batch["feature_length"][0].item()),
        frames_per_token,
    )
    max_commits = min(int(cfg.get("max_commits", target_tokens)), int(target_tokens))
    require_zero_update_count = bool(cfg.get("require_zero_update_count", False))

    model.init_generated(
        history_tokens,
        batch_size=1,
        num_denoise_steps=cfg.get("num_denoise_steps", None),
        initial_generated=initial_generated,
        traj_buffer=None,
    )
    model.token_update_count = torch.zeros(
        int(model.generated.shape[2]),
        device=model.generated.device,
        dtype=torch.long,
    )
    stream = StreamGenerator(
        ldf_model=model,
        device=device,
        history_length=history_tokens,
        traj_horizon_tokens=traj_horizon_tokens,
        token_dt=float(cfg.get("token_dt", 0.20)),
    )
    text_rollout = StreamTextRolloutController.from_sample_batch(sample_batch)
    conditioner = LdfEvalStreamConditioner(
        sample_batch,
        history_length=history_tokens,
        traj_horizon_tokens=traj_horizon_tokens,
        token_dt=float(cfg.get("token_dt", 0.20)),
        frames_per_token=frames_per_token,
        device=device,
    )
    recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    vae.clear_cache()
    first_chunk = True
    latents = []
    apply_events = []

    for commit_index in range(max_commits):
        runtime_traj_payload = _build_runtime_traj_payload(
            model=model,
            conditioner=conditioner,
            commit_index=commit_index,
        )
        if (
            initializer is not None
            and alpha is not None
            and should_apply_initializer(initializer_training_cfg, commit_index)
        ):
            context = _build_initializer_context_for_commit(
                model=model,
                sample_batch=sample_batch,
                text_rollout=text_rollout,
                commit_index=commit_index,
                device=device,
                text_encoder=initializer_text_encoder,
                history_tokens=history_tokens,
                frontier_tokens=int(cfg.get("frontier_tokens", 5)),
                traj_horizon_tokens=traj_horizon_tokens,
                frames_per_token=frames_per_token,
                beta_threshold=float(cfg.get("beta_threshold", 0.999)),
                traj_payload=runtime_traj_payload,
                token_update_count=getattr(model, "token_update_count", None),
                require_zero_update_count=require_zero_update_count,
            )
            event = apply_initializer_to_stream_state(
                model=model,
                initializer=initializer,
                context=context,
                alpha=float(alpha),
                max_delta_norm_ratio=cfg.get("max_delta_norm_ratio", None),
            )
            event["commit_index"] = int(commit_index)
            apply_events.append(event)

        step_payload, condition_provider = _build_step(
            model=model,
            stream=stream,
            text_rollout=text_rollout,
            conditioner=conditioner,
            commit_index=commit_index,
            first_chunk=first_chunk,
            device=device,
        )
        with torch.no_grad():
            update_start_step = int(model.current_step)
            update_start_commit = int(model.commit_index)
            output = model.stream_generate_step(
                step_payload,
                first_chunk=first_chunk,
                condition=condition_provider,
            )
            advance_model_token_update_count(
                model,
                start_step=update_start_step,
                start_commit=update_start_commit,
            )
            latent_token = output["generated"][0].detach()
            latents.append(latent_token.cpu())
            decoded_chunk = vae.stream_decode(
                latent_token[None, :],
                first_chunk=first_chunk,
            )[0].float().detach().cpu()
            conditioner.append_decoded(
                decoded_chunk,
                commit_idx=commit_index + 1,
                recovery=recovery,
            )
        first_chunk = False
        model.generated = model.generated.detach()

    vae.clear_cache()
    return torch.cat(latents, dim=0), apply_events


def _metrics_for_latents(vae, latents: torch.Tensor, target_xz: torch.Tensor, mask: torch.Tensor, device: torch.device):
    with torch.no_grad():
        feature = _decode_latent_stream(vae, latents.to(device)).detach()
        xz = _root_xz_from_feature(feature)
        metrics = _attach_root_path_metrics(
            _masked_metrics(xz, target_xz, mask),
            xz=xz,
            target_xz=target_xz,
            mask=mask,
        )
    return feature, xz, metrics


def _metrics_for_token_window(
    pred_xz: torch.Tensor,
    target_xz: torch.Tensor,
    mask: torch.Tensor,
    *,
    start_token: int,
    num_tokens: int,
    frames_per_token: int,
) -> dict:
    frame_slice = token_range_to_frame_slice(
        int(start_token),
        int(num_tokens),
        int(frames_per_token),
    )
    return _masked_metrics(
        pred_xz[frame_slice],
        target_xz[frame_slice],
        mask[frame_slice],
    )


def main() -> int:
    args = parse_args()
    cfg_obj = load_config(
        config_path=args.config,
        override_args=_parse_overrides(args.override),
    )
    cfg = OmegaConf.to_container(cfg_obj.config, resolve=True)

    device = torch.device(str(cfg.get("device", "cuda:0")))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = int(cfg.get("seed", 1234))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    ldf_config = load_config(config_path=str(cfg.get("ldf_config", "configs/ldf_test.yaml")))
    ldf_cfg = ldf_config.config
    vae_ckpt = cfg.get("vae_ckpt") or ldf_config.get("test_vae_ckpt", None)
    model, vae = load_eval_model_and_vae(
        ldf_cfg,
        ckpt_path=str(cfg["ckpt"]),
        vae_ckpt_path=str(vae_ckpt),
        device=device,
        use_ema=bool(cfg.get("use_ema", True)),
    )
    if str(ldf_config.get("eval.text_device", "cpu")).lower() == "cpu":
        enable_cpu_text_encoding(model)
    model.cfg_scale_text = float(cfg.get("cfg_text", 1.25))
    model.cfg_scale_traj = float(cfg.get("cfg_traj", 3.0))
    for module in (model, vae):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    namespace = argparse.Namespace(
        meta_path=str(cfg["meta_path"]),
        sample_name=str(cfg.get("sample_name", "")),
        caption_index=int(cfg.get("caption_index", 0)),
    )
    sample_batch = _load_single_sample(namespace, ldf_config)
    sample_batch = _cap_sample_to_frames(
        sample_batch,
        int(cfg.get("max_frames", 0)),
        frames_per_token=int(cfg.get("frames_per_token", 4)),
    )
    target_xz, mask = _target_xz_and_mask(sample_batch, device)
    initializer, initializer_training_cfg = _load_initializer(
        Path(args.initializer_ckpt),
        cfg,
        device,
    )
    initializer_text_encoder = resolve_noise_initializer_text_encoder(
        cfg,
        text_emb_dim=int(((cfg.get("model") or {}).get("params") or {}).get("text_dim", 4096)),
    )

    history_tokens = int(cfg.get("history_tokens", cfg.get("history_length", 30)))
    initial_shape = (
        1,
        history_tokens * 2 + int(model.chunk_size),
        int(model.input_dim),
    )
    base_noise = torch.randn(initial_shape, device=device)

    variants: dict[str, dict] = {}
    if bool(args.include_gaussian):
        variants["gaussian"] = {"initializer": None, "alpha": None}
    for raw in str(args.alphas).split(","):
        if raw.strip():
            alpha = float(raw.strip())
            variants[f"alpha_{str(alpha).replace('.', 'p')}"] = {
                "initializer": initializer,
                "alpha": alpha,
            }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "eval_noise_initializer",
        "initializer_ckpt": str(args.initializer_ckpt),
        "sample_name": str(sample_batch["name"][0]),
        "caption_index": int(cfg.get("caption_index", 0)),
        "ckpt": str(cfg["ckpt"]),
        "vae_ckpt": str(vae_ckpt),
        "cfg_text": float(cfg.get("cfg_text", 1.25)),
        "cfg_traj": float(cfg.get("cfg_traj", 3.0)),
        "max_commits": int(cfg.get("max_commits", 0)),
        "optimize_every_tokens": int(cfg.get("optimize_every_tokens", 5)),
        "initializer_training_policy": {
            "training_mode": initializer_training_cfg.get("training_mode"),
            "fixed_commit_index": initializer_training_cfg.get("fixed_commit_index"),
            "optimize_every_tokens": initializer_training_cfg.get("optimize_every_tokens"),
        },
        "variants": {},
    }
    arrays = {
        "target_xz": target_xz.detach().cpu().numpy(),
        "mask": mask.detach().cpu().numpy(),
        "base_noise": base_noise.detach().cpu().numpy(),
    }
    xz_by_name = {}
    feature_by_name = {}
    latent_by_name = {}
    for name, spec in variants.items():
        latents, apply_events = _run_stream_variant(
            model=model,
            vae=vae,
            initializer=spec["initializer"],
            initializer_text_encoder=initializer_text_encoder,
            sample_batch=sample_batch,
            cfg=cfg,
            initializer_training_cfg=initializer_training_cfg,
            device=device,
            initial_generated=base_noise.detach(),
            alpha=spec["alpha"],
        )
        feature, xz, metrics = _metrics_for_latents(vae, latents, target_xz, mask, device)
        affected_windows = []
        for event in apply_events:
            offsets = list(event.get("frontier_offsets", []))
            if not offsets:
                continue
            affected_start = int(event["commit_index"]) + int(offsets[0])
            affected_stop = min(
                int(cfg.get("max_commits", 0)),
                int(event["commit_index"]) + int(cfg.get("loss_horizon_tokens", 10)),
            )
            if affected_stop <= affected_start:
                continue
            affected_windows.append(
                {
                    "commit_index": int(event["commit_index"]),
                    "affected_start_token": affected_start,
                    "affected_num_tokens": affected_stop - affected_start,
                    "metrics": _metrics_for_token_window(
                        xz,
                        target_xz,
                        mask,
                        start_token=affected_start,
                        num_tokens=affected_stop - affected_start,
                        frames_per_token=int(cfg.get("frames_per_token", 4)),
                    ),
                }
            )
        suffix_metrics = None
        if affected_windows:
            suffix_start = min(row["affected_start_token"] for row in affected_windows)
            suffix_tokens = int(cfg.get("max_commits", 0)) - suffix_start
            if suffix_tokens > 0:
                suffix_metrics = {
                    "start_token": suffix_start,
                    "metrics": _metrics_for_token_window(
                        xz,
                        target_xz,
                        mask,
                        start_token=suffix_start,
                        num_tokens=suffix_tokens,
                        frames_per_token=int(cfg.get("frames_per_token", 4)),
                    ),
                }
        result["variants"][name] = {
            "alpha": spec["alpha"],
            "metrics": metrics,
            "apply_events": apply_events,
            "affected_windows": affected_windows,
            "suffix_metrics": suffix_metrics,
        }
        arrays[f"{name}_latent"] = latents.detach().cpu().numpy()
        arrays[f"{name}_feature"] = feature.detach().cpu().numpy()
        arrays[f"{name}_xz"] = xz.detach().cpu().numpy()
        xz_by_name[name] = xz
        feature_by_name[name] = feature
        latent_by_name[name] = latents

    if "gaussian" in xz_by_name:
        for name, xz in xz_by_name.items():
            if name == "gaussian":
                continue
            result["variants"][name]["diff_from_gaussian"] = _baseline_optimized_xz_diff(
                xz_by_name["gaussian"],
                xz,
            )
            metrics = result["variants"][name]["metrics"]
            base_metrics = result["variants"]["gaussian"]["metrics"]
            result["variants"][name]["delta_vs_gaussian"] = {
                key: float(metrics[key] - base_metrics[key])
                for key in ("ade", "fde", "mse", "heading_error_deg", "path_ratio")
                if key in metrics and key in base_metrics
            }
            affected_windows = result["variants"][name].get("affected_windows", [])
            for window in affected_windows:
                base_window_metrics = _metrics_for_token_window(
                    xz_by_name["gaussian"],
                    target_xz,
                    mask,
                    start_token=int(window["affected_start_token"]),
                    num_tokens=int(window["affected_num_tokens"]),
                    frames_per_token=int(cfg.get("frames_per_token", 4)),
                )
                window["delta_vs_gaussian"] = {
                    key: float(window["metrics"][key] - base_window_metrics[key])
                    for key in ("ade", "fde", "mse")
                }
            suffix = result["variants"][name].get("suffix_metrics")
            if suffix is not None:
                suffix_start = int(suffix["start_token"])
                base_suffix = _metrics_for_token_window(
                    xz_by_name["gaussian"],
                    target_xz,
                    mask,
                    start_token=suffix_start,
                    num_tokens=int(cfg.get("max_commits", 0)) - suffix_start,
                    frames_per_token=int(cfg.get("frames_per_token", 4)),
                )
                suffix["delta_vs_gaussian"] = {
                    key: float(suffix["metrics"][key] - base_suffix[key])
                    for key in ("ade", "fde", "mse")
                }
                if suffix_start > 0:
                    pre_effect_slice = token_range_to_frame_slice(
                        0,
                        suffix_start,
                        int(cfg.get("frames_per_token", 4)),
                    )
                    result["variants"][name]["pre_effect_diff_from_gaussian"] = (
                        _baseline_optimized_xz_diff(
                            xz_by_name["gaussian"][pre_effect_slice],
                            xz[pre_effect_slice],
                        )
                    )

    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out_dir / "motions.npz", **arrays)
    if "gaussian" in xz_by_name:
        for name, xz in xz_by_name.items():
            if name != "gaussian":
                _plot_paths(
                    out_dir / f"root_xz_compare_{name}.png",
                    target_xz.detach().cpu().numpy(),
                    xz_by_name["gaussian"].detach().cpu().numpy(),
                    xz.detach().cpu().numpy(),
                )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
