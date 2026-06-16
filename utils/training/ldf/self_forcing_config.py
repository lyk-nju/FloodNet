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
    out = [(float(p), int(k)) for p, k in schedule]
    if not out:
        raise ValueError("self_forcing.k_schedule must not be empty")
    out.sort(key=lambda x: x[0])
    return out


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

    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        elif hasattr(cur, "get"):
            cur = cur.get(part, default)
            if cur is default:
                return default
        else:
            return default
    return cur
