"""Single-sample online residual overfit runner for NoiseInitializer."""

from __future__ import annotations

import copy
import inspect
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from models.diffusion_forcing_wan import DiffForcingWanModel
from models.noise_initializer import NoiseInitializer
from utils.inference.latent_state_view import StreamLatentStateView
from utils.inference.stream_generator import StreamGenerator
from utils.motion_process import StreamJointRecovery263
from utils.token_frame import token_range_to_frame_slice, token_start_frame
from utils.training.noise_initializer.context_builder import (
    NoiseInitializerContext,
    build_noise_initializer_context,
)
from utils.training.noise_initializer.config_validate import (
    validate_noise_initializer_overfit_config,
)
from utils.training.noise_initializer.lightning_module import (
    NoiseInitializerLightningModule,
)
from utils.training.noise_initializer.shadow_rollout import (
    clip_delta_to_base_norm,
    inject_context_frontier_zT,
    restore_vae_cache,
    snapshot_vae_cache,
)
from utils.training.noise_initializer.text_encoder import (
    resolve_noise_initializer_text_encoder,
)


def should_train_commit(commit_index: int, *, optimize_every_tokens: int) -> bool:
    optimize_every_tokens = int(optimize_every_tokens)
    if optimize_every_tokens <= 1:
        return True
    return int(commit_index) % optimize_every_tokens == 0


def should_apply_initializer(training_cfg: dict, commit_index: int) -> bool:
    """Match evaluation apply decisions to the checkpoint's training policy."""

    mode = str(training_cfg.get("training_mode", "online_multi_commit"))
    if mode == "fixed_snapshot_overfit":
        return int(commit_index) == int(training_cfg.get("fixed_commit_index", 0))
    if mode != "online_multi_commit":
        raise ValueError(f"unknown noise initializer training_mode: {mode!r}")
    interval = int(
        training_cfg.get(
            "optimize_every_tokens",
            training_cfg.get("apply_initializer_every_tokens", 5),
        )
    )
    return should_train_commit(commit_index, optimize_every_tokens=interval)


def resolve_training_commit_indices(cfg: dict, *, target_tokens: int) -> list[int]:
    """Resolve decision points for fixed-snapshot or online training."""

    mode = str(cfg.get("training_mode", "online_multi_commit"))
    target_tokens = max(0, int(target_tokens))
    if mode == "fixed_snapshot_overfit":
        commit = int(cfg.get("fixed_commit_index", 0))
        if commit < 0 or commit >= target_tokens:
            raise ValueError(
                "fixed_commit_index must select a token in the sample: "
                f"got {commit}, target_tokens={target_tokens}"
            )
        return [commit]
    if mode != "online_multi_commit":
        raise ValueError(f"unknown noise initializer training_mode: {mode!r}")
    max_commits = min(int(cfg.get("max_commits", target_tokens)), target_tokens)
    optimize_every = int(cfg.get("optimize_every_tokens", 5))
    return [
        commit
        for commit in range(max_commits)
        if should_train_commit(commit, optimize_every_tokens=optimize_every)
    ]


def advance_token_update_count(
    counts: torch.Tensor,
    *,
    start_step: int,
    end_step: int,
    dt: float,
    chunk_size: int,
) -> None:
    """Count the exact token ranges touched by triangular denoise updates."""

    token_count = int(counts.numel())
    for step in range(int(start_step), int(end_step)):
        current_time = float(step) * float(dt)
        start_index = max(0, math.floor(int(chunk_size) * (current_time - 1.0)) + 1)
        end_index = min(token_count, int(int(chunk_size) * current_time) + 1)
        if end_index > start_index:
            counts[start_index:end_index] += 1


def advance_model_token_update_count(
    model,
    *,
    start_step: int,
    start_commit: int,
) -> None:
    """Advance and roll the strict-frontier tracker with the model buffer."""

    end_step = int(
        (int(start_commit) + int(model.chunk_size))
        * int(model.num_denoise_steps)
        / int(model.chunk_size)
    )
    advance_token_update_count(
        model.token_update_count,
        start_step=int(start_step),
        end_step=end_step,
        dt=float(model.dt),
        chunk_size=int(model.chunk_size),
    )
    if int(start_commit) + 1 == int(model.seq_len) * 2:
        seq_len = int(model.seq_len)
        model.token_update_count = torch.cat(
            [
                model.token_update_count[seq_len:],
                torch.zeros(
                    seq_len,
                    device=model.token_update_count.device,
                    dtype=model.token_update_count.dtype,
                ),
            ],
            dim=0,
        )


def should_log_train_progress(train_step: int, *, log_every_train_steps: int) -> bool:
    log_every = int(log_every_train_steps)
    step = int(train_step)
    return log_every > 0 and step > 0 and step % log_every == 0


def resolve_train_progress_log_step(
    *,
    completed_steps: int,
    inner_step: int,
    log_every_train_steps: int,
) -> int | None:
    """Return the displayed step, including the pre-update snapshot baseline."""

    if int(log_every_train_steps) <= 0:
        return None
    if int(inner_step) == 0:
        return max(0, int(completed_steps) - 1)
    if should_log_train_progress(
        int(completed_steps),
        log_every_train_steps=int(log_every_train_steps),
    ):
        return int(completed_steps)
    return None


def format_train_progress_log(*, train_step: int, row: dict) -> dict:
    payload = {
        "event": "noise_initializer_train_progress",
        "train_step": int(train_step),
        "commit_index": int(row["commit_index"]),
        "inner_step": int(row["inner_step"]),
        "loss": float(row["loss"]),
        "grad_norm_sum": float(row["grad_norm_sum"]),
        "history_frames": int(row["history_frames"]),
    }
    for key in (
        "traj_loss",
        "vel_loss",
        "delta_reg",
        "applied_delta_norm",
        "applied_delta_ratio",
        "delta_ratio_reg",
        "raw_delta_norm",
        "clipped_delta_norm",
        "base_zT_norm",
        "clipped_to_base_ratio",
        "delta_scale_mean",
        "clip_saturation_ratio",
    ):
        payload[key] = float(row[key])
    payload["optimized_frames"] = int(row["optimized_frames"])
    return payload


def encode_initializer_text_embedding(
    model,
    text: str,
    device: torch.device,
    *,
    text_encoder=None,
) -> torch.Tensor:
    """Encode text for the initializer as one pooled vector per sample."""

    if text_encoder is not None:
        encoded = text_encoder.encode([str(text)], device=device)
        if encoded.dim() != 2:
            raise ValueError(
                "initializer text_encoder.encode must return [B,D], "
                f"got {tuple(encoded.shape)}"
            )
        return encoded.to(device=device, dtype=torch.float32)
    if not hasattr(model, "encode_text_with_cache"):
        raise AttributeError("ldf model must expose encode_text_with_cache")
    encoded = model.encode_text_with_cache([str(text)], device)[0]
    encoded = encoded.to(device=device, dtype=torch.float32)
    if encoded.dim() == 1:
        return encoded.unsqueeze(0)
    if encoded.dim() == 2:
        return encoded.mean(dim=0, keepdim=True)
    if encoded.dim() == 3 and int(encoded.shape[0]) == 1:
        return encoded.mean(dim=1)
    raise ValueError(f"unsupported text embedding shape: {tuple(encoded.shape)}")


def slice_future_traj_frames(
    traj_cond_7d: torch.Tensor,
    *,
    commit_index: int,
    traj_horizon_tokens: int,
    frames_per_token: int,
) -> torch.Tensor:
    """Slice a future frame-level trajectory window and zero-pad if needed."""

    if traj_cond_7d.dim() != 3:
        raise ValueError(
            f"traj_cond_7d must have shape [B,F,C], got {tuple(traj_cond_7d.shape)}"
        )
    frame_slice = token_range_to_frame_slice(
        int(commit_index),
        int(traj_horizon_tokens),
        int(frames_per_token),
    )
    start_frame = int(frame_slice.start)
    stop_frame = int(frame_slice.stop)
    frame_count = stop_frame - start_frame
    window = traj_cond_7d[:, start_frame:min(stop_frame, int(traj_cond_7d.shape[1]))]
    if int(window.shape[1]) < frame_count:
        pad = traj_cond_7d.new_zeros(
            int(traj_cond_7d.shape[0]),
            frame_count - int(window.shape[1]),
            int(traj_cond_7d.shape[2]),
        )
        window = torch.cat([window, pad], dim=1)
    return window


def slice_initializer_traj_payload(
    traj_payload: dict,
    *,
    absolute_commit_index: int,
    traj_tokens: int,
    frames_per_token: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Crop a runtime payload to a commit-aligned initializer trajectory window."""

    frames = traj_payload["traj_cond_7d_frame"]
    if frames.dim() == 2:
        frames = frames.unsqueeze(0)
    payload_abs_start = int(
        traj_payload.get(
            "traj_abs_start_token",
            traj_payload.get("traj_start_token", 0),
        )
    )
    desired = token_range_to_frame_slice(
        int(absolute_commit_index),
        int(traj_tokens),
        int(frames_per_token),
    )
    payload_origin_frame = token_start_frame(payload_abs_start, int(frames_per_token))
    payload_stop_frame = payload_origin_frame + int(frames.shape[1])
    copy_start = max(int(desired.start), payload_origin_frame)
    copy_stop = min(int(desired.stop), payload_stop_frame)
    frame_count = int(desired.stop) - int(desired.start)
    window = frames.new_zeros(int(frames.shape[0]), frame_count, int(frames.shape[2]))
    if copy_stop > copy_start:
        src_start = copy_start - payload_origin_frame
        src_stop = copy_stop - payload_origin_frame
        dst_start = copy_start - int(desired.start)
        dst_stop = copy_stop - int(desired.start)
        window[:, dst_start:dst_stop] = frames[:, src_start:src_stop]

    payload_mask = traj_payload.get(
        "traj_cond_frame_mask",
        traj_payload.get("traj_cond_mask"),
    )
    window_mask = None
    if payload_mask is not None:
        if payload_mask.dim() == 1:
            payload_mask = payload_mask.unsqueeze(0)
        window_mask = payload_mask.new_zeros(int(payload_mask.shape[0]), frame_count)
        if copy_stop > copy_start:
            window_mask[:, dst_start:dst_stop] = payload_mask[:, src_start:src_stop]

    offsets = torch.arange(
        int(traj_tokens),
        device=frames.device,
        dtype=torch.long,
    )
    return window, window_mask, offsets


def affected_history_frames(
    context: NoiseInitializerContext,
    *,
    frames_per_token: int,
) -> int:
    """Frames before the first frontier token can affect decoded motion."""

    if int(context.frontier_ids.numel()) == 0:
        return 0
    no_effect_tokens = max(0, int(context.frontier_offsets[0].item()))
    frame_slice = token_range_to_frame_slice(
        int(context.local_commit_index),
        int(no_effect_tokens),
        int(frames_per_token),
    )
    return int(frame_slice.stop) - int(frame_slice.start)


def append_terminal_hold(
    target_xz: torch.Tensor,
    mask: torch.Tensor,
    *,
    hold_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append a repeated endpoint segment so velocity loss can teach stopping."""

    hold_frames = int(hold_frames)
    if hold_frames <= 0 or int(target_xz.shape[0]) == 0:
        return target_xz, mask
    tail = target_xz[-1:].expand(hold_frames, -1)
    tail_mask = mask.new_ones(hold_frames)
    return torch.cat([target_xz, tail], dim=0), torch.cat([mask, tail_mask], dim=0)


def sync_vae_decode_cache(source_vae, target_vae) -> None:
    """Clone stream decode cache from the real VAE into an isolated training VAE."""

    restore_vae_cache(target_vae, snapshot_vae_cache(source_vae))


def build_noise_initializer_from_ldf(cfg: dict, ldf_model) -> NoiseInitializer:
    """Build an initializer, optionally copying the trained LDF trajectory encoder."""

    model_cfg = dict(cfg.get("model", {}) or {})
    params = dict(model_cfg.get("params", {}) or {})
    if bool(model_cfg.get("reuse_ldf_traj_encoder", False)):
        source = getattr(ldf_model, "traj_encoder", None)
        if source is None:
            raise ValueError(
                "model.reuse_ldf_traj_encoder=true requires ldf_model.traj_encoder"
            )
        encoder = copy.deepcopy(source)
        source_out_dim = int(getattr(encoder, "out_dim", params.get("traj_emb_dim", 128)))
        configured_out_dim = int(params.get("traj_emb_dim", source_out_dim))
        if configured_out_dim != source_out_dim:
            raise ValueError(
                "initializer traj_emb_dim must match LDF traj_encoder.out_dim: "
                f"got {configured_out_dim} and {source_out_dim}"
            )
        params["traj_emb_dim"] = source_out_dim
        params["traj_encoder"] = encoder
        params.setdefault("freeze_traj_encoder", True)
    return NoiseInitializer(**params)


def _initializer_accepts_frontier_base(initializer) -> bool:
    signature = inspect.signature(initializer.forward)
    return "frontier_base_zT" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def apply_initializer_to_stream_state(
    *,
    model,
    initializer,
    context: NoiseInitializerContext,
    alpha: float,
    max_delta_norm_ratio: float | None = None,
) -> dict:
    """Apply learned residual to real stream state before committing a token."""

    was_training = bool(getattr(initializer, "training", False))
    initializer.eval()
    with torch.no_grad():
        kwargs = context.as_model_kwargs()
        if _initializer_accepts_frontier_base(initializer):
            kwargs["frontier_base_zT"] = context.frontier_base_zT
        raw_delta_zT = initializer(**kwargs)
        delta_zT, delta_scale = clip_delta_to_base_norm(
            raw_delta_zT,
            context.frontier_base_zT,
            max_delta_norm_ratio=max_delta_norm_ratio,
        )
        frontier_zT = context.frontier_base_zT.to(
            device=delta_zT.device,
            dtype=delta_zT.dtype,
        ) + float(alpha) * delta_zT
        generated = inject_context_frontier_zT(
            model.generated.to(device=frontier_zT.device),
            context,
            frontier_zT,
        )
        model.generated = generated.detach()
    if was_training:
        initializer.train()
    return {
        "applied": True,
        "frontier_ids": [int(v) for v in context.frontier_ids.detach().cpu().tolist()],
        "frontier_offsets": [
            int(v) for v in context.frontier_offsets.detach().cpu().tolist()
        ],
        "raw_delta_norm": float(raw_delta_zT.detach().float().norm().cpu().item()),
        "delta_norm": float(delta_zT.detach().float().norm().cpu().item()),
        "base_zT_norm": float(context.frontier_base_zT.detach().float().norm().cpu().item()),
        "delta_scale_mean": float(delta_scale.detach().float().mean().cpu().item()),
        "max_delta_norm_ratio": (
            None if max_delta_norm_ratio is None else float(max_delta_norm_ratio)
        ),
        "alpha": float(alpha),
    }


def _decode_latents_to_root_xz(
    vae,
    latents: torch.Tensor,
    *,
    committed_prefix_latents: torch.Tensor | None = None,
    shadow_start_token: int = 0,
    frames_per_token: int = 4,
) -> torch.Tensor:
    from eval.ldf.latent_initializer.optimize_noise import _root_xz_from_feature

    if committed_prefix_latents is None:
        committed_prefix_latents = latents[:, :0].detach()
    prefix = committed_prefix_latents.to(device=latents.device, dtype=latents.dtype).detach()
    full_latents = torch.cat([prefix, latents], dim=1)
    decoded = vae.decode(full_latents)[0].float()
    full_root_xz = _root_xz_from_feature(decoded)
    frame_slice = token_range_to_frame_slice(
        int(shadow_start_token),
        int(latents.shape[1]),
        int(frames_per_token),
    )
    return full_root_xz[frame_slice].unsqueeze(0)


def make_ldf_shadow_rollout_fn(
    *,
    vae,
    stream: StreamGenerator,
    text_rollout,
    conditioner,
    recovery: StreamJointRecovery263,
    start_commit: int,
    device: torch.device,
):
    """Create the differentiable LDF short-rollout callback used by Lightning."""

    def rollout_fn(model, *, rollout_tokens: int, first_chunk: bool):
        from eval.ldf.latent_initializer.optimize_stream_chunk_noise import _build_step

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
                output = DiffForcingWanModel.stream_generate_step.__wrapped__(
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


def _build_initializer_context_for_commit(
    *,
    model,
    sample_batch: dict,
    text_rollout,
    commit_index: int,
    device: torch.device,
    text_encoder,
    history_tokens: int,
    frontier_tokens: int,
    traj_horizon_tokens: int,
    frames_per_token: int,
    beta_threshold: float,
    traj_payload: dict | None = None,
    token_update_count: torch.Tensor | None = None,
    require_zero_update_count: bool = False,
):
    view = StreamLatentStateView.from_model(
        model,
        beta_threshold=float(beta_threshold),
        frontier_tokens=int(frontier_tokens),
        token_update_count=token_update_count,
        require_zero_update_count=bool(require_zero_update_count),
    )
    current_text = text_rollout.get_text_for_commit_index(int(commit_index))
    text_embedding = encode_initializer_text_embedding(
        model,
        current_text,
        device,
        text_encoder=text_encoder,
    )
    traj_frame_mask = None
    traj_offsets = None
    if traj_payload is not None and traj_payload.get("traj_cond_7d_frame") is not None:
        payload = dict(traj_payload)
        payload["traj_cond_7d_frame"] = payload["traj_cond_7d_frame"].to(
            device=device,
            dtype=torch.float32,
        )
        for mask_key in ("traj_cond_frame_mask", "traj_cond_mask"):
            if payload.get(mask_key) is not None:
                payload[mask_key] = payload[mask_key].to(
                    device=device,
                    dtype=torch.float32,
                )
        traj_frames, traj_frame_mask, traj_offsets = slice_initializer_traj_payload(
            payload,
            absolute_commit_index=int(commit_index),
            traj_tokens=int(traj_horizon_tokens),
            frames_per_token=int(frames_per_token),
        )
    else:
        traj_cond_7d = sample_batch["traj_cond_7d"].to(device=device, dtype=torch.float32)
        traj_frames = slice_future_traj_frames(
            traj_cond_7d,
            commit_index=int(commit_index),
            traj_horizon_tokens=int(traj_horizon_tokens),
            frames_per_token=int(frames_per_token),
        )
        traj_mask = sample_batch.get("traj_cond_mask", sample_batch.get("traj_mask"))
        if traj_mask is not None:
            traj_frame_mask = slice_future_traj_frames(
                traj_mask.to(device=device, dtype=torch.float32).unsqueeze(-1),
                commit_index=int(commit_index),
                traj_horizon_tokens=int(traj_horizon_tokens),
                frames_per_token=int(frames_per_token),
            ).squeeze(-1)
        traj_offsets = torch.arange(
            int(traj_horizon_tokens),
            device=device,
            dtype=torch.long,
        )
    return build_noise_initializer_context(
        view,
        text_embedding=text_embedding,
        traj_local_frames=traj_frames,
        traj_frame_mask=traj_frame_mask,
        traj_offsets=traj_offsets,
        history_tokens=int(history_tokens),
        frontier_tokens=int(frontier_tokens),
        traj_tokens=int(traj_horizon_tokens),
        traj_start_token=int(commit_index),
        frames_per_token=int(frames_per_token),
    )


def _build_runtime_traj_payload(
    *,
    model,
    conditioner,
    commit_index: int,
) -> dict | None:
    return conditioner.build_step_payload(
        local_commit_index=int(getattr(model, "commit_index", commit_index)),
        absolute_commit_index=int(commit_index),
        chunk_size=int(getattr(model, "chunk_size", 1)),
    )


def run_single_sample_overfit(cfg: dict) -> dict:
    """Run Gaussian-rollin online residual overfit for one configured sample."""

    validate_noise_initializer_overfit_config(cfg)

    from eval.ldf.conditioning import LdfEvalStreamConditioner
    from eval.ldf.latent_initializer.optimize_noise import (
        _cap_sample_to_frames,
        _load_single_sample,
        _target_xz_and_mask,
    )
    from eval.ldf.latent_initializer.optimize_stream_chunk_noise import _build_step
    from eval.ldf.stream_generation import StreamTextRolloutController
    from eval.ldf.stream_setup import enable_cpu_text_encoding, load_eval_model_and_vae
    from utils.initialize import load_config
    from utils.token_frame import num_tokens_for_frame_len

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
    if not vae_ckpt:
        raise ValueError("noise initializer overfit requires vae_ckpt or test_vae_ckpt")
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

    args = SimpleNamespace(
        meta_path=str(cfg["meta_path"]),
        sample_name=str(cfg.get("sample_name", "")),
        caption_index=int(cfg.get("caption_index", 0)),
    )
    sample_batch = _load_single_sample(args, ldf_config)
    sample_batch = _cap_sample_to_frames(
        sample_batch,
        int(cfg.get("max_frames", 0)),
        frames_per_token=int(cfg.get("frames_per_token", 4)),
    )
    target_xz, target_mask = _target_xz_and_mask(sample_batch, device)
    target_xz, target_mask = append_terminal_hold(
        target_xz,
        target_mask,
        hold_frames=int(cfg.get("terminal_hold_frames", 0)),
    )
    target_tokens = num_tokens_for_frame_len(
        int(sample_batch["feature_length"][0].item()),
        int(cfg.get("frames_per_token", 4)),
    )

    initializer = build_noise_initializer_from_ldf(cfg, model).to(device)
    initializer_text_encoder = resolve_noise_initializer_text_encoder(
        cfg,
        text_emb_dim=int(cfg["model"]["params"]["text_dim"]),
    )
    module_cfg = OmegaConf.to_container(OmegaConf.create(dict(cfg)), resolve=True)
    module_cfg.setdefault("rollout", {})
    module_cfg["rollout"].setdefault("alpha", float(cfg.get("alpha", 1.0)))
    module_cfg["rollout"].setdefault(
        "loss_horizon_tokens",
        int(cfg.get("loss_horizon_tokens", 20)),
    )
    module_cfg["rollout"].setdefault(
        "max_delta_norm_ratio",
        cfg.get("max_delta_norm_ratio", None),
    )
    module_cfg.setdefault("loss", {})
    module_cfg["loss"].setdefault("lambda_vel", float(cfg.get("lambda_vel", 0.05)))
    module_cfg["loss"].setdefault("lambda_delta", float(cfg.get("lambda_delta", 1e-4)))
    lightning = NoiseInitializerLightningModule(
        cfg=module_cfg,
        initializer=initializer,
        ldf_model=model,
        vae=training_vae,
        rollout_fn=None,
        decode_latents_fn=_decode_latents_to_root_xz,
    )
    optimizer = lightning.configure_optimizers()

    if str(cfg.get("training_mode", "online_multi_commit")) == "two_stage_snapshot_replay":
        from utils.training.noise_initializer.replay_runner import (
            run_two_stage_snapshot_replay,
        )

        return run_two_stage_snapshot_replay(
            cfg=cfg,
            model=model,
            vae=vae,
            training_vae=training_vae,
            initializer=initializer,
            initializer_text_encoder=initializer_text_encoder,
            lightning=lightning,
            optimizer=optimizer,
            sample_batch=sample_batch,
            target_xz=target_xz,
            target_mask=target_mask,
            target_tokens=int(target_tokens),
            device=device,
        )

    history_tokens = int(cfg.get("history_tokens", cfg.get("history_length", 30)))
    traj_horizon_tokens = int(cfg.get("traj_horizon_tokens", 20))
    frames_per_token = int(cfg.get("frames_per_token", 4))
    model.init_generated(
        history_tokens,
        batch_size=1,
        num_denoise_steps=cfg.get("num_denoise_steps", None),
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
    training_vae.clear_cache()
    first_chunk = True
    rows = []
    training_mode = str(cfg.get("training_mode", "online_multi_commit"))
    training_commits = resolve_training_commit_indices(cfg, target_tokens=int(target_tokens))
    training_commit_set = set(training_commits)
    if training_mode == "fixed_snapshot_overfit":
        max_commits = training_commits[0] + 1
    else:
        max_commits = min(int(cfg.get("max_commits", target_tokens)), int(target_tokens))
    train_steps_per_commit = int(cfg.get("train_steps_per_commit", 1))
    optimize_every_tokens = int(cfg.get("optimize_every_tokens", 5))
    log_every_train_steps = int(cfg.get("log_every_train_steps", 0) or 0)
    apply_initializer_rollin = bool(cfg.get("apply_initializer_rollin", True))
    apply_every_tokens = int(cfg.get("apply_initializer_every_tokens", optimize_every_tokens))
    require_zero_update_count = bool(cfg.get("require_zero_update_count", False))
    apply_events = []
    committed_latent_tokens: list[torch.Tensor] = []

    for commit_index in range(max_commits):
        runtime_traj_payload = _build_runtime_traj_payload(
            model=model,
            conditioner=conditioner,
            commit_index=commit_index,
        )
        if commit_index in training_commit_set:
            for inner_step in range(train_steps_per_commit):
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
                lightning.rollout_fn = make_ldf_shadow_rollout_fn(
                    vae=training_vae,
                    stream=stream,
                    text_rollout=text_rollout,
                    conditioner=conditioner,
                    recovery=recovery,
                    start_commit=commit_index,
                    device=device,
                )
                loss_horizon_tokens = int(cfg.get("loss_horizon_tokens", 20))
                target_frame_slice = token_range_to_frame_slice(
                    commit_index,
                    loss_horizon_tokens,
                    frames_per_token,
                )
                if committed_latent_tokens:
                    committed_prefix_latents = torch.cat(committed_latent_tokens, dim=1)
                else:
                    committed_prefix_latents = context.frontier_base_zT[:, :0]
                batch = {
                    "context": context,
                    "target_xz": target_xz[target_frame_slice],
                    "target_mask": target_mask[target_frame_slice],
                    "history_frames": affected_history_frames(
                        context,
                        frames_per_token=frames_per_token,
                    ),
                    "first_chunk": first_chunk,
                    "generated_anchor_xz": conditioner.timeline.head.world_xz.detach().clone(),
                    "committed_prefix_latents": committed_prefix_latents,
                    "commit_index": int(commit_index),
                    "frames_per_token": int(frames_per_token),
                }
                optimizer.zero_grad(set_to_none=True)
                sync_vae_decode_cache(vae, training_vae)
                loss = lightning.training_step(batch, inner_step)
                loss.backward()
                delta_norm = 0.0
                for parameter in initializer.parameters():
                    if parameter.grad is not None:
                        delta_norm += float(parameter.grad.detach().float().norm().cpu().item())
                optimizer.step()
                row = {
                        "commit_index": int(commit_index),
                        "inner_step": int(inner_step),
                        "loss": float(loss.detach().cpu().item()),
                        "grad_norm_sum": float(delta_norm),
                        "history_frames": int(batch["history_frames"]),
                }
                row.update(lightning.last_step_diagnostics)
                rows.append(row)
                progress_step = resolve_train_progress_log_step(
                    completed_steps=len(rows),
                    inner_step=inner_step,
                    log_every_train_steps=log_every_train_steps,
                )
                if progress_step is not None:
                    print(
                        json.dumps(
                            format_train_progress_log(
                                train_step=progress_step,
                                row=rows[-1],
                            ),
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        if training_mode == "fixed_snapshot_overfit" and commit_index in training_commit_set:
            break

        if apply_initializer_rollin and should_train_commit(
            commit_index,
            optimize_every_tokens=apply_every_tokens,
        ):
            apply_context = _build_initializer_context_for_commit(
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
            apply_event = apply_initializer_to_stream_state(
                model=model,
                initializer=initializer,
                context=apply_context,
                alpha=float(cfg.get("alpha", 1.0)),
                max_delta_norm_ratio=cfg.get("max_delta_norm_ratio", None),
            )
            apply_event["commit_index"] = int(commit_index)
            apply_events.append(apply_event)

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

    out_dir = Path(str(cfg.get("out_dir", "eval/out_eval/noise_initializer_overfit")))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = out_dir / f"noise_initializer_{tag}.pt"
    debug_path = out_dir / f"debug_{tag}.json"
    torch.save({"state_dict": initializer.state_dict(), "cfg": dict(cfg)}, ckpt_path)
    summary = {
        "checkpoint": str(ckpt_path),
        "debug_json": str(debug_path),
        "num_train_steps": len(rows),
        "target_tokens": int(target_tokens),
        "completed_commits": int(len(committed_latent_tokens)),
        "training_mode": training_mode,
        "training_commits": training_commits,
        "loss_curve": rows,
        "apply_events": apply_events,
        "terminal_hold_frames": int(cfg.get("terminal_hold_frames", 0)),
        "require_zero_update_count": bool(require_zero_update_count),
    }
    debug_path.write_text(json.dumps(summary, indent=2))
    return summary


__all__ = [
    "apply_initializer_to_stream_state",
    "affected_history_frames",
    "advance_token_update_count",
    "advance_model_token_update_count",
    "append_terminal_hold",
    "build_noise_initializer_from_ldf",
    "encode_initializer_text_embedding",
    "format_train_progress_log",
    "resolve_train_progress_log_step",
    "resolve_training_commit_indices",
    "make_ldf_shadow_rollout_fn",
    "run_single_sample_overfit",
    "should_log_train_progress",
    "should_apply_initializer",
    "should_train_commit",
    "slice_future_traj_frames",
    "slice_initializer_traj_payload",
    "sync_vae_decode_cache",
]
