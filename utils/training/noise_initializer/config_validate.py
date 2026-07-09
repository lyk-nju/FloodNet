"""Config validation for NoiseInitializer overfit/debug training."""

from __future__ import annotations


def _require(cfg: dict, key: str) -> None:
    if key not in cfg or cfg[key] in (None, ""):
        raise ValueError(f"noise initializer config requires {key!r}")


def _positive_int(cfg: dict, key: str, default: int | None = None) -> None:
    value = cfg.get(key, default)
    if value is None:
        return
    if int(value) <= 0:
        raise ValueError(f"{key} must be positive, got {value}")


def _positive_float(cfg: dict, key: str, default: float | None = None) -> None:
    value = cfg.get(key, default)
    if value is None:
        return
    if float(value) <= 0.0:
        raise ValueError(f"{key} must be positive, got {value}")


def validate_noise_initializer_overfit_config(cfg: dict) -> None:
    _require(cfg, "ckpt")
    _require(cfg, "meta_path")
    model_params = ((cfg.get("model") or {}).get("params") or {})
    for key in ("latent_dim", "text_dim", "frontier_tokens"):
        _positive_int(model_params, key)
    for key in (
        "history_tokens",
        "traj_horizon_tokens",
        "frames_per_token",
        "loss_horizon_tokens",
        "train_steps_per_commit",
        "max_commits",
    ):
        _positive_int(cfg, key, default=1)
    _positive_float(cfg, "alpha", default=1.0)
    optimizer = cfg.get("optimizer") or {}
    _positive_float(optimizer, "lr", default=1e-4)
    text_encoder = cfg.get("text_encoder") or {}
    if text_encoder.get("type") == "precomputed_t5_pool":
        if not text_encoder.get("precomputed_text_emb_path"):
            raise ValueError(
                "text_encoder.type=precomputed_t5_pool requires "
                "text_encoder.precomputed_text_emb_path"
            )


__all__ = ["validate_noise_initializer_overfit_config"]
