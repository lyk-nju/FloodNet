"""LDF training config consistency checks."""

from __future__ import annotations

from omegaconf import OmegaConf

from utils.training.ldf.self_forcing_config import (
    self_forcing_enabled,
    self_forcing_k_schedule,
    self_forcing_stride_tokens,
)
from utils.training.ldf.t2m_generation_modes import resolve_t2m_generation_modes


_WINDOW_POLICIES = {"prefix", "rolling"}
_SAMPLE_POLICIES = {"variable_history", "fixed_window"}


def validate_traj_dim_consistency(cfg) -> int:
    """Check dataset and model trajectory feature dimensions."""
    data_dim = int(OmegaConf.select(cfg, "data.traj_feat_dim", default=7))
    model_dim = int(
        OmegaConf.select(cfg, "model.params.traj_encoder_in_dim", default=7)
    )
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
    """Require self-forcing for the 7D trajectory path."""
    traj_dim = int(OmegaConf.select(cfg, "model.params.traj_encoder_in_dim", default=7))
    if traj_dim != 7:
        return
    if not self_forcing_enabled(cfg):
        raise ValueError(
            "traj_encoder_in_dim=7 requires self_forcing.enabled=true: "
            "7D heading supervision (body_aux_loss) and the body-window "
            "canonicalize are self-forcing-only, so a non-SF 7D run trains the new "
            "heading channels unsupervised on world-frame traj cond. Enable "
            "self_forcing or use the 4D path. See T_B_10."
        )


def validate_ldf_training_config(cfg) -> None:
    """Validate the windowed LDF training contract.

    The new mainline has no separate full-sequence/full-latent training task.
    ``ldf_training.window_policy`` selects prefix or rolling window sampling,
    while runtime files may still use "stream" in their names.
    """
    resolve_t2m_generation_modes(cfg)
    if "stream_training" in cfg:
        raise ValueError(
            "stream_training training config was removed; use "
            "ldf_training.window_policy and ldf_training.window_sampling instead."
        )
    ldf_cfg = OmegaConf.select(cfg, "ldf_training", default={}) or {}
    if "motion_aux_loss" in ldf_cfg:
        raise ValueError(
            "ldf_training.motion_aux_loss is no longer configurable; "
            "windowed body auxiliary loss is controlled by body_aux_loss."
        )
    if "latent_source" in ldf_cfg:
        raise ValueError(
            "ldf_training.latent_source was removed; rolling window training "
            "always uses online VAE encode."
        )
    formulation = str(
        OmegaConf.select(cfg, "ldf_training.formulation", default="windowed")
    )
    if formulation != "windowed":
        raise ValueError(
            "ldf_training.formulation must be 'windowed'; "
            f"got {formulation!r}."
        )
    policy = str(OmegaConf.select(cfg, "ldf_training.window_policy", default="prefix"))
    if policy not in _WINDOW_POLICIES:
        raise ValueError(
            "ldf_training.window_policy must be 'prefix' or 'rolling'; "
            f"got {policy!r}."
        )
    chunk_size = int(OmegaConf.select(cfg, "model.params.chunk_size", default=5))
    window_sampling_enabled = bool(
        OmegaConf.select(cfg, "ldf_training.window_sampling.enabled", default=False)
    )
    if policy == "prefix":
        if "context_tokens" in ldf_cfg:
            raise ValueError(
                "ldf_training.context_tokens is not used for prefix training; "
                "prefix active right is sampled from [1, token_length]."
            )
        if "horizon_tokens" in ldf_cfg:
            raise ValueError(
                "ldf_training.horizon_tokens is not used for prefix training; "
                "prefix trajectory condition always extends to token_length."
            )
        if "window_sampling" in ldf_cfg:
            raise ValueError(
                "ldf_training.window_sampling is only used for rolling training; "
                "prefix active right is sampled from [1, token_length]."
            )
        if "min_history_tokens" in ldf_cfg:
            raise ValueError(
                "ldf_training.min_history_tokens is not used for prefix training; "
                "prefix active right is sampled from [1, token_length]."
            )
    else:
        context_tokens = int(
            OmegaConf.select(cfg, "ldf_training.context_tokens", default=1)
        )
        if context_tokens <= 0:
            raise ValueError(
                "ldf_training.context_tokens must be > 0 for rolling training; "
                f"got {context_tokens}."
            )
    min_history_tokens = int(
        OmegaConf.select(
            cfg,
            "ldf_training.min_history_tokens",
            default=chunk_size if policy == "rolling" else 1,
        )
    )
    horizon_tokens = int(
        OmegaConf.select(cfg, "ldf_training.horizon_tokens", default=0)
    )
    sample_policy = str(
        OmegaConf.select(cfg, "ldf_training.sample_policy", default="variable_history")
    )
    anchor_move = bool(
        OmegaConf.select(cfg, "ldf_training.anchor_move_in_rollout", default=False)
    )
    if sample_policy not in _SAMPLE_POLICIES:
        raise ValueError(
            "ldf_training.sample_policy must be 'variable_history' or "
            f"'fixed_window'; got {sample_policy!r}."
        )
    if policy == "rolling" and not window_sampling_enabled:
        raise ValueError(
            "ldf_training.window_policy='rolling' requires "
            "ldf_training.window_sampling.enabled=true."
        )
    if window_sampling_enabled:
        if policy != "rolling":
            raise ValueError(
                "ldf_training.window_sampling.enabled=true requires "
                "ldf_training.window_policy='rolling'."
            )
        ws_prefix = "ldf_training.window_sampling"
        history_min = int(
            OmegaConf.select(cfg, f"{ws_prefix}.history_tokens_min", default=0)
        )
        history_max = OmegaConf.select(
            cfg,
            f"{ws_prefix}.history_tokens_max",
            default="auto",
        )
        horizon_min = int(
            OmegaConf.select(cfg, f"{ws_prefix}.horizon_tokens_min", default=0)
        )
        horizon_max = int(
            OmegaConf.select(cfg, f"{ws_prefix}.horizon_tokens_max", default=0)
        )
        stride = self_forcing_stride_tokens(cfg)
        schedule = self_forcing_k_schedule(cfg)
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
                "ldf_training.window_sampling.enabled=true; variable horizon "
                "is sampled by windowed rolling training."
            )
        if anchor_move:
            raise ValueError(
                "ldf_training.anchor_move_in_rollout=true is not implemented yet. "
                "Keep it false until trajectory/text/loss windows are rebuilt per "
                "rollout step."
            )
        return
    if min_history_tokens <= 0:
        raise ValueError(
            "ldf_training.min_history_tokens must be > 0; "
            f"got min_history_tokens={min_history_tokens}"
        )
    if policy == "rolling" and min_history_tokens < chunk_size:
        raise ValueError(
            "ldf_training.min_history_tokens must be >= model.params.chunk_size "
            "for rolling training; "
            f"got min_history_tokens={min_history_tokens}, chunk_size={chunk_size}"
        )
    if policy == "rolling" and context_tokens < min_history_tokens:
        raise ValueError(
            "ldf_training.context_tokens must be >= min_history_tokens; "
            f"got context_tokens={context_tokens}, "
            f"min_history_tokens={min_history_tokens}"
        )
    if policy == "rolling" and horizon_tokens < 0:
        raise ValueError(
            f"ldf_training.horizon_tokens must be >= 0, got {horizon_tokens}"
        )
    if anchor_move:
        raise ValueError(
            "ldf_training.anchor_move_in_rollout=true is not implemented yet. "
            "Keep it false until trajectory/text/loss windows are rebuilt per "
            "rollout step."
        )


__all__ = [
    "validate_traj_dim_consistency",
    "validate_7d_requires_self_forcing",
    "validate_ldf_training_config",
    "resolve_t2m_generation_modes",
]
