"""Config consistency validation for the flag-gated 4D→7D traj migration (T_B_10).

The 7D path is controlled by two flags that MUST agree:
  - data.traj_feat_dim                 (dataset emits 4D or 7D traj_cond)
  - model.params.traj_encoder_in_dim   (encoder consumes 4D or 7D)
A mismatch silently feeds 7D data into a 4D encoder (or vice-versa). This module
fails fast at startup instead.
"""

from __future__ import annotations

from omegaconf import OmegaConf


def validate_traj_dim_consistency(cfg) -> int:
    """Check the two traj-dim flags agree and equal 7. Returns the dim (7).

    The 4D legacy encoder was removed (LocalTrajEncoder is 7D-only), so a 4D
    config now crashes at model construction — fail fast here with a clear
    message. Defaults are 7 (matching the model's `traj_in_dim=7` default).

    Raises ValueError on mismatch or any non-7 value.
    """
    data_dim = int(OmegaConf.select(cfg, "data.traj_feat_dim", default=7))
    model_dim = int(OmegaConf.select(cfg, "model.params.traj_encoder_in_dim", default=7))
    if data_dim != model_dim:
        raise ValueError(
            f"traj dim mismatch: data.traj_feat_dim={data_dim} != "
            f"model.params.traj_encoder_in_dim={model_dim}. They must agree."
        )
    if data_dim != 7:
        raise ValueError(
            f"traj dim must be 7 (the 4D legacy encoder was removed), got {data_dim}."
        )
    return data_dim


def validate_7d_requires_self_forcing(cfg) -> None:
    """7D path requires self-forcing — fail fast otherwise. No-op for 4D.

    The two correctness guarantees of the 7D traj path live ONLY inside the
    self-forcing trainer:
      - body_aux_loss, which is the SOLE supervision of the new 7D heading
        channels (computed in SelfForcingTrainer; the regular _step path never
        computes it), and
      - the body-window world->local canonicalize (apply_body_window_canonicalize,
        called only from the SF path) that matches the streaming-inference
        distribution.
    So a 7D config with self_forcing_enabled=false would silently train the new
    heading channels unsupervised on uncanonicalized world-frame traj cond. The
    in-SF "traj_encoder_in_dim=7 requires body_aux_loss" guard never runs when SF
    is off, so enforce 7D => self_forcing here (at module construction).
    """
    dim = int(OmegaConf.select(cfg, "model.params.traj_encoder_in_dim", default=7))
    if dim != 7:
        return
    sf = bool(OmegaConf.select(cfg, "model.params.self_forcing_enabled", default=False))
    if not sf:
        raise ValueError(
            "traj_encoder_in_dim=7 requires model.params.self_forcing_enabled=true: "
            "7D heading supervision (body_aux_loss) and the body-window "
            "canonicalize are self-forcing-only, so a non-SF 7D run trains the new "
            "heading channels unsupervised on world-frame traj cond. Enable "
            "self_forcing or use the 4D path. See T_B_10."
        )


def validate_stream_training_config(cfg) -> None:
    """Validate optional window-local limited-history training settings."""
    enabled = bool(OmegaConf.select(cfg, "stream_training.enabled", default=False))
    if not enabled:
        return
    chunk_size = int(OmegaConf.select(cfg, "model.params.chunk_size", default=5))
    context_tokens = int(OmegaConf.select(cfg, "stream_training.context_tokens", default=0))
    window_sampling_enabled = bool(
        OmegaConf.select(cfg, "stream_training.window_sampling.enabled", default=False)
    )
    min_history_tokens = int(
        OmegaConf.select(cfg, "stream_training.min_history_tokens", default=chunk_size)
    )
    horizon_tokens = int(OmegaConf.select(cfg, "stream_training.horizon_tokens", default=0))
    sample_policy = str(
        OmegaConf.select(cfg, "stream_training.sample_policy", default="variable_history")
    )
    latent_source = str(
        OmegaConf.select(cfg, "stream_training.latent_source", default="precomputed_slice")
    )
    motion_aux_loss = str(
        OmegaConf.select(cfg, "stream_training.motion_aux_loss", default="latent_only")
    )
    anchor_move = bool(
        OmegaConf.select(cfg, "stream_training.anchor_move_in_rollout", default=False)
    )
    if context_tokens <= 0:
        raise ValueError(
            "stream_training.context_tokens must be > 0 when stream_training is enabled"
        )
    if window_sampling_enabled:
        ws_prefix = "stream_training.window_sampling"
        history_min = int(OmegaConf.select(cfg, f"{ws_prefix}.history_tokens_min", default=0))
        history_max = OmegaConf.select(cfg, f"{ws_prefix}.history_tokens_max", default="auto")
        horizon_min = int(OmegaConf.select(cfg, f"{ws_prefix}.horizon_tokens_min", default=0))
        horizon_max = int(OmegaConf.select(cfg, f"{ws_prefix}.horizon_tokens_max", default=0))
        stride = int(
            OmegaConf.select(cfg, "model.params.self_forcing_stride_tokens", default=1)
        )
        schedule = OmegaConf.select(
            cfg, "model.params.self_forcing_k_schedule", default=[[0.0, 1]]
        )
        max_k = 1
        for row in schedule:
            max_k = max(max_k, int(row[1]))
        rollout_span = max(0, (max_k - 1) * stride)
        auto_history_max = context_tokens - chunk_size - rollout_span
        if history_min < 0:
            raise ValueError(
                f"{ws_prefix}.history_tokens_min must be >= 0, got {history_min}"
            )
        if auto_history_max < history_min:
            raise ValueError(
                f"{ws_prefix}.history_tokens_max=auto leaves no valid history "
                f"range for context_tokens={context_tokens}, chunk_size={chunk_size}, "
                f"rollout_span={rollout_span}, history_tokens_min={history_min}"
            )
        if history_max is not None and str(history_max).lower() != "auto":
            if int(history_max) < history_min:
                raise ValueError(
                    f"{ws_prefix}.history_tokens_max must be >= history_tokens_min "
                    f"or 'auto'; got {history_max}"
                )
        if horizon_min < 0 or horizon_max < max(horizon_min, 1):
            raise ValueError(
                f"{ws_prefix}.horizon_tokens_min/max must define a non-negative "
                f"range with at least one complete future token; got "
                f"min={horizon_min}, max={horizon_max}"
            )
        hs_enabled = bool(OmegaConf.select(cfg, "horizon_sim.enabled", default=False))
        if hs_enabled:
            raise ValueError(
                "horizon_sim.enabled=true must not be mixed with "
                "stream_training.window_sampling.enabled=true; variable horizon "
                "is sampled by window_sampling in stream-training v2."
            )
        if latent_source != "precomputed_slice":
            raise ValueError(
                "stream_training.latent_source must be 'precomputed_slice' for v2; "
                f"got {latent_source!r}. VAE window re-encoding is intentionally "
                "not part of the window-local training contract."
            )
        if motion_aux_loss not in {"latent_only", "full_prefix", "disabled"}:
            raise ValueError(
                "stream_training.motion_aux_loss must be 'latent_only', "
                f"'full_prefix', or 'disabled'; got {motion_aux_loss!r}."
            )
        if anchor_move:
            raise ValueError(
                "stream_training.anchor_move_in_rollout=true is not implemented yet. "
                "Keep it false until trajectory/text/loss windows are rebuilt per "
                "rollout step."
            )
        return
    if min_history_tokens < chunk_size:
        raise ValueError(
            "stream_training.min_history_tokens must be >= model.params.chunk_size; "
            f"got min_history_tokens={min_history_tokens}, chunk_size={chunk_size}"
        )
    if context_tokens < min_history_tokens:
        raise ValueError(
            "stream_training.context_tokens must be >= min_history_tokens; "
            f"got context_tokens={context_tokens}, min_history_tokens={min_history_tokens}"
        )
    if horizon_tokens < 0:
        raise ValueError(
            f"stream_training.horizon_tokens must be >= 0, got {horizon_tokens}"
        )
    if sample_policy not in {"variable_history", "fixed_window"}:
        raise ValueError(
            "stream_training.sample_policy must be 'variable_history' or "
            f"'fixed_window'; got {sample_policy!r}."
        )
    if latent_source != "precomputed_slice":
        raise ValueError(
            "stream_training.latent_source must be 'precomputed_slice' for v1; "
            f"got {latent_source!r}. VAE window re-encoding is intentionally "
            "not part of the window-local training contract."
        )
    if motion_aux_loss not in {"latent_only", "full_prefix", "disabled"}:
        raise ValueError(
            "stream_training.motion_aux_loss must be 'latent_only', "
            f"'full_prefix', or 'disabled'; got {motion_aux_loss!r}."
        )
    if anchor_move:
        raise ValueError(
            "stream_training.anchor_move_in_rollout=true is not implemented yet. "
            "Keep it false until trajectory/text/loss windows are rebuilt per "
            "rollout step."
        )


def validate_stream_eval_config(cfg) -> None:
    """Validate optional async stream-eval checkpoint gate settings."""
    enabled = bool(OmegaConf.select(cfg, "validation.stream_eval.enabled", default=False))
    if not enabled:
        return
    stream_mode = str(
        OmegaConf.select(
            cfg,
            "validation.stream_eval.stream_mode",
            default="stream_generate_step",
        )
    )
    num_runs = int(OmegaConf.select(cfg, "validation.stream_eval.num_runs", default=1))
    max_samples = int(
        OmegaConf.select(cfg, "validation.stream_eval.max_samples", default=5)
    )
    max_batches = int(
        OmegaConf.select(cfg, "validation.stream_eval.max_batches", default=0)
    )
    if stream_mode not in {"stream_generate", "stream_generate_step"}:
        raise ValueError(
            "validation.stream_eval.stream_mode must be 'stream_generate' or "
            f"'stream_generate_step'; got {stream_mode!r}."
        )
    if num_runs <= 0:
        raise ValueError(
            f"validation.stream_eval.num_runs must be > 0, got {num_runs}"
        )
    if max_samples < 0:
        raise ValueError(
            f"validation.stream_eval.max_samples must be >= 0, got {max_samples}"
        )
    if max_batches < 0:
        raise ValueError(
            f"validation.stream_eval.max_batches must be >= 0, got {max_batches}"
        )


__all__ = [
    "validate_traj_dim_consistency",
    "validate_7d_requires_self_forcing",
    "validate_stream_eval_config",
    "validate_stream_training_config",
]
