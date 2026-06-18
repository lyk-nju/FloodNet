"""Self-forcing config accessors and runtime guards."""

from __future__ import annotations

from omegaconf import OmegaConf


DEFAULT_SELF_FORCING_K_SCHEDULE = ((0.0, 2), (0.4, 3), (0.7, 5))


def self_forcing_enabled(cfg) -> bool:
    return bool(_select(cfg, "self_forcing.enabled", False))


def self_forcing_stride_tokens(cfg) -> int:
    return int(_select(cfg, "self_forcing.stride_tokens", 1))


def self_forcing_detach_between_steps(cfg) -> bool:
    return bool(_select(cfg, "self_forcing.detach_between_steps", True))


def self_forcing_k_schedule(cfg) -> list[tuple[float, int]]:
    schedule = _select(
        cfg,
        "self_forcing.k_schedule",
        DEFAULT_SELF_FORCING_K_SCHEDULE,
    )
    rows = [(float(progress), int(k_value)) for progress, k_value in schedule]
    if not rows:
        raise ValueError("self_forcing.k_schedule must not be empty")
    rows.sort(key=lambda item: item[0])
    return rows


def validate_self_forcing_runtime_config(cfg, *, prediction_type: str) -> None:
    if self_forcing_stride_tokens(cfg) != 1:
        raise ValueError("v1 self-forcing only supports stride_tokens == 1")
    if self_forcing_enabled(cfg) and not self_forcing_detach_between_steps(cfg):
        raise NotImplementedError(
            "v1 self-forcing only supports detach_between_steps=True"
        )
    if self_forcing_enabled(cfg) and prediction_type not in ("vel", "x0"):
        raise ValueError(
            "self-forcing only supports prediction_type in {'vel', 'x0'}"
        )
    self_forcing_k_schedule(cfg)


def _select(cfg, key: str, default=None):
    if cfg is None:
        return default
    try:
        return OmegaConf.select(cfg, key, default=default)
    except (AttributeError, TypeError, ValueError):
        pass

    current = cfg
    for part in key.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, "get"):
            current = current.get(part, default)
            if current is default:
                return default
        else:
            return default
    return current


__all__ = [
    "DEFAULT_SELF_FORCING_K_SCHEDULE",
    "self_forcing_detach_between_steps",
    "self_forcing_enabled",
    "self_forcing_k_schedule",
    "self_forcing_stride_tokens",
    "validate_self_forcing_runtime_config",
]
