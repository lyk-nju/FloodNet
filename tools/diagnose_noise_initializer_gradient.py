"""One-step frontier z_T gradient diagnostics for NoiseInitializer."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.ldf.conditioning import LdfEvalStreamConditioner  # noqa: E402
from eval.ldf.latent_initializer.optimize_noise import (  # noqa: E402
    _cap_sample_to_frames,
    _load_single_sample,
    _target_xz_and_mask,
)
from eval.ldf.latent_initializer.optimize_stream_chunk_noise import _build_step  # noqa: E402
from eval.ldf.stream_generation import StreamTextRolloutController  # noqa: E402
from eval.ldf.stream_setup import enable_cpu_text_encoding, load_eval_model_and_vae  # noqa: E402
from models.noise_initializer import NoiseInitializer  # noqa: E402
from utils.inference.stream_generator import StreamGenerator  # noqa: E402
from utils.initialize import load_config  # noqa: E402
from utils.motion_process import StreamJointRecovery263  # noqa: E402
from utils.token_frame import token_range_to_frame_slice  # noqa: E402
from utils.training.noise_initializer.free_delta import optimize_free_delta  # noqa: E402
from utils.training.noise_initializer.losses import anchored_root_xz_loss  # noqa: E402
from utils.training.noise_initializer.overfit_runner import (  # noqa: E402
    _build_initializer_context_for_commit,
    _build_runtime_traj_payload,
    _decode_latents_to_root_xz,
    advance_model_token_update_count,
    affected_history_frames,
    sync_vae_decode_cache,
)
from utils.training.noise_initializer.shadow_rollout import (  # noqa: E402
    inject_context_frontier_zT,
    restore_stream_state,
    restore_vae_cache,
    snapshot_stream_state,
    snapshot_vae_cache,
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
    parser.add_argument("--initializer_ckpt", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--commits", default="0,5,10,15")
    parser.add_argument("--etas", default="0.001,0.005,0.01,0.02")
    parser.add_argument("--free_delta_steps", type=int, default=0)
    parser.add_argument("--free_delta_lr", type=float, default=0.02)
    parser.add_argument("--free_delta_max_norm_ratio", type=float, default=None)
    parser.add_argument("--free_delta_lambda", type=float, default=0.0)
    parser.add_argument("--free_delta_log_every", type=int, default=50)
    parser.add_argument("--override", nargs="*", default=None)
    return parser.parse_args()


def _load_initializer(path: str | None, cfg: dict, device: torch.device):
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("cfg") or {}
    params = dict(((ckpt_cfg.get("model") or {}).get("params") or {}))
    params.update(dict(((cfg.get("model") or {}).get("params") or {})))
    initializer = NoiseInitializer(**params).to(device)
    initializer.load_state_dict(checkpoint["state_dict"], strict=True)
    initializer.eval()
    for parameter in initializer.parameters():
        parameter.requires_grad_(False)
    return initializer


def _initializer_accepts_frontier_base(initializer) -> bool:
    signature = inspect.signature(initializer.forward)
    return "frontier_base_zT" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _rollin_to_commit(
    *,
    model,
    vae,
    sample_batch: dict,
    cfg: dict,
    device: torch.device,
    initial_generated: torch.Tensor,
    target_commit: int,
):
    history_tokens = int(cfg.get("history_tokens", cfg.get("history_length", 30)))
    traj_horizon_tokens = int(cfg.get("traj_horizon_tokens", 20))
    frames_per_token = int(cfg.get("frames_per_token", 4))
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
    committed_latent_tokens = []
    vae.clear_cache()
    first_chunk = True
    for commit_index in range(int(target_commit)):
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
            committed_latent_tokens.append(output["generated"].detach().clone())
            latent_token = output["generated"][0].detach()
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
    return (
        stream,
        text_rollout,
        conditioner,
        recovery,
        first_chunk,
        committed_latent_tokens,
    )


def _make_rollout_fn(*, vae, stream, text_rollout, conditioner, recovery, start_commit, device):
    def rollout_fn(model, *, rollout_tokens: int, first_chunk: bool):
        local_conditioner = copy.deepcopy(conditioner)
        local_recovery = copy.deepcopy(recovery)
        local_first = bool(first_chunk)
        latents = []
        with torch.enable_grad():
            for offset in range(int(rollout_tokens)):
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
                output = model.stream_generate_step.__wrapped__(
                    model,
                    step_payload,
                    first_chunk=local_first,
                    condition=condition_provider,
                )
                latent_token = output["generated"]
                latents.append(latent_token)
                with torch.no_grad():
                    decoded_chunk = vae.stream_decode(
                        latent_token.detach(),
                        first_chunk=local_first,
                    )[0].float().detach().cpu()
                    local_conditioner.append_decoded(
                        decoded_chunk,
                        commit_idx=commit + 1,
                        recovery=local_recovery,
                    )
                local_first = False
        return torch.cat(latents, dim=1)

    return rollout_fn


def _loss_for_frontier(
    *,
    model,
    vae,
    context,
    frontier_zT: torch.Tensor,
    rollout_fn,
    rollout_tokens: int,
    first_chunk: bool,
    target_xz: torch.Tensor,
    target_mask: torch.Tensor,
    target_frame_slice: slice,
    history_frames: int,
    generated_anchor_xz: torch.Tensor,
    loss_cfg: dict,
    committed_prefix_latents: torch.Tensor,
    shadow_start_token: int,
    frames_per_token: int,
) -> torch.Tensor:
    stream_state = snapshot_stream_state(model)
    vae_state = snapshot_vae_cache(vae)
    try:
        model.generated = inject_context_frontier_zT(
            model.generated,
            context,
            frontier_zT,
        )
        shadow_latents = rollout_fn(
            model,
            rollout_tokens=int(rollout_tokens),
            first_chunk=bool(first_chunk),
        )
        pred_xz = _decode_latents_to_root_xz(
            vae,
            shadow_latents,
            committed_prefix_latents=committed_prefix_latents,
            shadow_start_token=int(shadow_start_token),
            frames_per_token=int(frames_per_token),
        )
        loss, _ = anchored_root_xz_loss(
            pred_xz[0] if pred_xz.dim() == 3 else pred_xz,
            target_xz[target_frame_slice],
            target_mask[target_frame_slice],
            history_frames=int(history_frames),
            lambda_vel=float(loss_cfg.get("lambda_vel", 0.0)),
            anchor_mode=str(loss_cfg.get("anchor_mode", "generated_anchor_abs")),
            generated_anchor_xz=generated_anchor_xz,
        )
        return loss
    finally:
        restore_stream_state(model, stream_state)
        restore_vae_cache(vae, vae_state)


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
    training_vae = copy.deepcopy(vae).to(device).eval()
    for parameter in training_vae.parameters():
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
    target_xz, target_mask = _target_xz_and_mask(sample_batch, device)
    initializer = _load_initializer(args.initializer_ckpt, cfg, device)
    text_encoder = resolve_noise_initializer_text_encoder(
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
    frames_per_token = int(cfg.get("frames_per_token", 4))
    rollout_tokens = int(cfg.get("loss_horizon_tokens", 10))
    loss_cfg = dict(cfg.get("loss") or {})
    commits = [int(v) for v in str(args.commits).split(",") if v.strip()]
    etas = [float(v) for v in str(args.etas).split(",") if v.strip()]
    results = []

    for commit in commits:
        (
            stream,
            text_rollout,
            conditioner,
            recovery,
            first_chunk,
            committed_latent_tokens,
        ) = _rollin_to_commit(
            model=model,
            vae=vae,
            sample_batch=sample_batch,
            cfg=cfg,
            device=device,
            initial_generated=base_noise.detach(),
            target_commit=commit,
        )
        runtime_traj_payload = _build_runtime_traj_payload(
            model=model,
            conditioner=conditioner,
            commit_index=commit,
        )
        context = _build_initializer_context_for_commit(
            model=model,
            sample_batch=sample_batch,
            text_rollout=text_rollout,
            commit_index=commit,
            device=device,
            text_encoder=text_encoder,
            history_tokens=history_tokens,
            frontier_tokens=int(cfg.get("frontier_tokens", 5)),
            traj_horizon_tokens=int(cfg.get("traj_horizon_tokens", 20)),
            frames_per_token=frames_per_token,
            beta_threshold=float(cfg.get("beta_threshold", 0.999)),
            traj_payload=runtime_traj_payload,
            token_update_count=getattr(model, "token_update_count", None),
            require_zero_update_count=bool(cfg.get("require_zero_update_count", False)),
        )
        rollout_fn = _make_rollout_fn(
            vae=training_vae,
            stream=stream,
            text_rollout=text_rollout,
            conditioner=conditioner,
            recovery=recovery,
            start_commit=commit,
            device=device,
        )
        target_frame_slice = token_range_to_frame_slice(
            commit,
            rollout_tokens,
            frames_per_token,
        )
        history_frames = affected_history_frames(
            context,
            frames_per_token=frames_per_token,
        )
        generated_anchor_xz = conditioner.timeline.head.world_xz.detach().clone()
        sync_vae_decode_cache(vae, training_vae)
        z_base = context.frontier_base_zT.detach()
        committed_prefix_latents = (
            torch.cat(committed_latent_tokens, dim=1)
            if committed_latent_tokens
            else z_base[:, :0]
        )
        z_var = z_base.clone().requires_grad_(True)
        base_loss = _loss_for_frontier(
            model=model,
            vae=training_vae,
            context=context,
            frontier_zT=z_var,
            rollout_fn=rollout_fn,
            rollout_tokens=rollout_tokens,
            first_chunk=first_chunk,
            target_xz=target_xz,
            target_mask=target_mask,
            target_frame_slice=target_frame_slice,
            history_frames=history_frames,
            generated_anchor_xz=generated_anchor_xz,
            loss_cfg=loss_cfg,
            committed_prefix_latents=committed_prefix_latents,
            shadow_start_token=commit,
            frames_per_token=frames_per_token,
        )
        base_loss.backward()
        grad = z_var.grad.detach()
        grad_norm = float(grad.float().norm().cpu().item())
        row = {
            "commit_index": int(commit),
            "frontier_ids": [int(v) for v in context.frontier_ids.detach().cpu().tolist()],
            "history_frames": int(history_frames),
            "base_loss": float(base_loss.detach().cpu().item()),
            "base_zT_norm": float(z_base.float().norm().cpu().item()),
            "grad_norm": grad_norm,
            "eta_results": [],
        }
        if initializer is not None:
            with torch.no_grad():
                kwargs = context.as_model_kwargs()
                if _initializer_accepts_frontier_base(initializer):
                    kwargs["frontier_base_zT"] = context.frontier_base_zT
                learned_delta = initializer(**kwargs).detach()
                neg_grad = -grad
                denom = learned_delta.float().norm() * neg_grad.float().norm()
                cosine = (
                    float((learned_delta.float() * neg_grad.float()).sum().cpu().item())
                    / float(denom.cpu().item())
                    if float(denom.cpu().item()) > 1e-12
                    else float("nan")
                )
                row["learned_delta_norm"] = float(learned_delta.float().norm().cpu().item())
                row["learned_delta_neg_grad_cosine"] = cosine
                learned_loss = _loss_for_frontier(
                    model=model,
                    vae=training_vae,
                    context=context,
                    frontier_zT=z_base + learned_delta,
                    rollout_fn=rollout_fn,
                    rollout_tokens=rollout_tokens,
                    first_chunk=first_chunk,
                    target_xz=target_xz,
                    target_mask=target_mask,
                    target_frame_slice=target_frame_slice,
                    history_frames=history_frames,
                    generated_anchor_xz=generated_anchor_xz,
                    loss_cfg=loss_cfg,
                    committed_prefix_latents=committed_prefix_latents,
                    shadow_start_token=commit,
                    frames_per_token=frames_per_token,
                )
                row["learned_alpha1_loss"] = float(learned_loss.detach().cpu().item())
        for eta in etas:
            candidate = z_base - float(eta) * grad
            loss = _loss_for_frontier(
                model=model,
                vae=training_vae,
                context=context,
                frontier_zT=candidate,
                rollout_fn=rollout_fn,
                rollout_tokens=rollout_tokens,
                first_chunk=first_chunk,
                target_xz=target_xz,
                target_mask=target_mask,
                target_frame_slice=target_frame_slice,
                history_frames=history_frames,
                generated_anchor_xz=generated_anchor_xz,
                loss_cfg=loss_cfg,
                committed_prefix_latents=committed_prefix_latents,
                shadow_start_token=commit,
                frames_per_token=frames_per_token,
            )
            row["eta_results"].append(
                {
                    "eta": float(eta),
                    "loss": float(loss.detach().cpu().item()),
                    "delta_vs_base": float(loss.detach().cpu().item() - row["base_loss"]),
                }
            )
        if int(args.free_delta_steps) > 0:
            def free_delta_loss(frontier_zT: torch.Tensor) -> torch.Tensor:
                return _loss_for_frontier(
                    model=model,
                    vae=training_vae,
                    context=context,
                    frontier_zT=frontier_zT,
                    rollout_fn=rollout_fn,
                    rollout_tokens=rollout_tokens,
                    first_chunk=first_chunk,
                    target_xz=target_xz,
                    target_mask=target_mask,
                    target_frame_slice=target_frame_slice,
                    history_frames=history_frames,
                    generated_anchor_xz=generated_anchor_xz,
                    loss_cfg=loss_cfg,
                    committed_prefix_latents=committed_prefix_latents,
                    shadow_start_token=commit,
                    frames_per_token=frames_per_token,
                )

            def report_free_delta_progress(progress_row: dict) -> None:
                print(
                    json.dumps(
                        {
                            "event": "free_delta_progress",
                            "commit_index": int(commit),
                            **progress_row,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            free_result = optimize_free_delta(
                base_zT=z_base,
                loss_fn=free_delta_loss,
                steps=int(args.free_delta_steps),
                lr=float(args.free_delta_lr),
                lambda_delta=float(args.free_delta_lambda),
                max_delta_norm_ratio=args.free_delta_max_norm_ratio,
                log_every=int(args.free_delta_log_every),
                progress_fn=report_free_delta_progress,
            )
            row["free_delta"] = {
                "steps": int(args.free_delta_steps),
                "lr": float(args.free_delta_lr),
                "lambda_delta": float(args.free_delta_lambda),
                "max_delta_norm_ratio": args.free_delta_max_norm_ratio,
                "initial_task_loss": free_result.initial_task_loss,
                "final_task_loss": free_result.final_task_loss,
                "final_total_loss": free_result.final_total_loss,
                "relative_task_improvement": (
                    (free_result.initial_task_loss - free_result.final_task_loss)
                    / max(abs(free_result.initial_task_loss), 1e-12)
                ),
                "raw_delta_norm": float(free_result.raw_delta_zT.float().norm().cpu().item()),
                "clipped_delta_norm": float(free_result.delta_zT.float().norm().cpu().item()),
                "base_zT_norm": float(z_base.float().norm().cpu().item()),
                "clip_saturation_ratio": free_result.clip_saturation_ratio,
                "loss_curve": free_result.loss_curve,
            }
            print(
                json.dumps(
                    {
                        "event": "free_delta_commit_complete",
                        "commit_index": int(commit),
                        **{key: value for key, value in row["free_delta"].items() if key != "loss_curve"},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        results.append(row)

    out = {
        "mode": (
            "frontier_zT_gradient_and_free_delta_diagnostic"
            if int(args.free_delta_steps) > 0
            else "one_step_frontier_zT_gradient_diagnostic"
        ),
        "initializer_ckpt": args.initializer_ckpt,
        "commits": commits,
        "etas": etas,
        "loss_horizon_tokens": int(rollout_tokens),
        "free_delta_steps": int(args.free_delta_steps),
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
