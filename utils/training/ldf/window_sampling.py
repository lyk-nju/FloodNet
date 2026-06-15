"""Stream-training v2 active-window sampling helpers."""

from __future__ import annotations

import torch


def _as_long_1d(value, *, batch_size: int, device, name: str) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.long).view(-1)
    else:
        out = torch.as_tensor(value, device=device, dtype=torch.long).view(-1)
    if out.numel() == 1 and batch_size > 1:
        out = out.expand(batch_size)
    if out.numel() != batch_size:
        raise ValueError(
            f"{name} must be scalar or length {batch_size}; got shape {tuple(out.shape)}"
        )
    return out


def resolve_history_tokens_max(
    value,
    *,
    context_tokens: int,
    chunk_size: int,
    rollout_span: int,
) -> int:
    context_tokens = int(context_tokens)
    chunk_size = int(chunk_size)
    rollout_span = int(rollout_span)
    if context_tokens <= 0:
        raise ValueError(f"context_tokens must be > 0, got {context_tokens}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if rollout_span < 0:
        raise ValueError(f"rollout_span must be >= 0, got {rollout_span}")
    auto_max = context_tokens - chunk_size - rollout_span
    if auto_max < 0:
        raise ValueError(
            "context_tokens is too small for chunk_size + rollout_span; "
            f"context_tokens={context_tokens}, chunk_size={chunk_size}, "
            f"rollout_span={rollout_span}"
        )
    if value is None or str(value).lower() == "auto":
        return auto_max
    configured = int(value)
    if configured < 0:
        raise ValueError(f"history_tokens_max must be >= 0 or 'auto', got {value!r}")
    return min(configured, auto_max)


def sample_stream_window_indices(
    token_length,
    *,
    context_tokens: int,
    chunk_size: int,
    rollout_span: int,
    history_tokens_min: int,
    history_tokens_max,
    horizon_tokens_min: int,
    horizon_tokens_max: int,
    active_left_tokens=None,
    history_tokens=None,
    horizon_tokens=None,
) -> dict[str, torch.Tensor | int]:
    """Sample stream-training v2 windows centered on active-left token A.

    The hard contract is:
      B = A - G
      latent window = [B, A + chunk_size + rollout_span)
      traj window = [B, A + chunk_size + rollout_span + H)
      G + chunk_size + rollout_span <= context_tokens
      A + chunk_size + rollout_span + H <= token_length

    ``horizon_tokens_min`` is a preferred minimum. Short clips may fall back to
    a smaller complete horizon, down to the internal absolute floor of 1 token.
    """
    if torch.is_tensor(token_length):
        device = token_length.device
        lengths = token_length.to(device=device, dtype=torch.long).view(-1)
    else:
        lengths = torch.as_tensor(token_length, dtype=torch.long).view(-1)
        device = lengths.device
    batch_size = int(lengths.numel())
    if batch_size <= 0:
        raise ValueError("token_length must contain at least one sample")
    if bool((lengths <= 0).any()):
        raise ValueError(f"token_length must be > 0, got {lengths.tolist()}")

    chunk_size = int(chunk_size)
    rollout_span = int(rollout_span)
    history_tokens_min = int(history_tokens_min)
    horizon_tokens_min = int(horizon_tokens_min)
    horizon_tokens_max = int(horizon_tokens_max)
    if history_tokens_min < 0:
        raise ValueError(
            f"history_tokens_min must be >= 0, got {history_tokens_min}"
        )
    if horizon_tokens_min < 0:
        raise ValueError(
            f"horizon_tokens_min must be >= 0, got {horizon_tokens_min}"
        )
    if horizon_tokens_max < horizon_tokens_min:
        raise ValueError(
            "horizon_tokens_max must be >= horizon_tokens_min; "
            f"got min={horizon_tokens_min}, max={horizon_tokens_max}"
        )
    horizon_abs_min = 1
    horizon_pref_min = max(horizon_tokens_min, horizon_abs_min)
    if horizon_tokens_max < horizon_abs_min:
        raise ValueError(
            f"horizon_tokens_max must be >= {horizon_abs_min}, got {horizon_tokens_max}"
        )

    history_max_eff = resolve_history_tokens_max(
        history_tokens_max,
        context_tokens=context_tokens,
        chunk_size=chunk_size,
        rollout_span=rollout_span,
    )
    if history_max_eff < history_tokens_min:
        raise ValueError(
            "history_tokens_max effective value is below history_tokens_min; "
            f"history_tokens_max_effective={history_max_eff}, "
            f"history_tokens_min={history_tokens_min}"
        )

    active_override = (
        None
        if active_left_tokens is None
        else _as_long_1d(
            active_left_tokens,
            batch_size=batch_size,
            device=device,
            name="active_left_tokens",
        )
    )
    history_override = (
        None
        if history_tokens is None
        else _as_long_1d(
            history_tokens,
            batch_size=batch_size,
            device=device,
            name="history_tokens",
        )
    )
    horizon_override = (
        None
        if horizon_tokens is None
        else _as_long_1d(
            horizon_tokens,
            batch_size=batch_size,
            device=device,
            name="horizon_tokens",
        )
    )

    active_values: list[torch.Tensor] = []
    history_values: list[torch.Tensor] = []
    horizon_values: list[torch.Tensor] = []
    horizon_cap_values: list[torch.Tensor] = []
    horizon_short_fallback_values: list[torch.Tensor] = []
    for b in range(batch_size):
        t_len = int(lengths[b].item())
        horizon_cap_clip = t_len - chunk_size - rollout_span
        if horizon_cap_clip < horizon_abs_min:
            raise ValueError(
                "stream window sampling requires a full horizon inside the clip; "
                f"sample={b}, token_length={t_len}, "
                f"horizon_cap_clip={horizon_cap_clip}, "
                f"horizon_abs_min={horizon_abs_min}"
            )
        short_fallback = horizon_cap_clip < horizon_pref_min
        h_low = horizon_pref_min if not short_fallback else horizon_abs_min
        h_hi = min(horizon_tokens_max, horizon_cap_clip)
        if h_hi < h_low:
            raise ValueError(
                "stream window sampling found no valid horizon range; "
                f"sample={b}, horizon_low={h_low}, horizon_high={h_hi}, "
                f"horizon_cap_clip={horizon_cap_clip}"
            )
        if horizon_override is None:
            h = torch.randint(h_low, h_hi + 1, (1,), device=device)[0]
        else:
            h = horizon_override[b]
            h_int = int(h.item())
            if h_int < h_low or h_int > h_hi:
                raise ValueError(
                    "horizon_tokens must be within the complete per-clip range; "
                    f"sample={b}, horizon_tokens={h_int}, valid_range="
                    f"[{h_low}, {h_hi}]"
                )

        low_a = history_tokens_min
        high_a = t_len - chunk_size - rollout_span - int(h.item())
        if high_a < low_a:
            raise ValueError(
                "stream window sampling requires room for active chunk, rollout, "
                "chosen horizon, and minimum history; "
                f"sample={b}, token_length={t_len}, low_active_left={low_a}, "
                f"high_active_left={high_a}, chosen_horizon={int(h.item())}"
            )

        if active_override is None:
            a = torch.randint(low_a, high_a + 1, (1,), device=device)[0]
        else:
            a = active_override[b]
            if int(a.item()) < low_a or int(a.item()) > high_a:
                raise ValueError(
                    "active_left_tokens must allow full horizon and minimum history; "
                    f"sample={b}, active_left={int(a.item())}, "
                    f"valid_range=[{low_a}, {high_a}]"
                )

        g_hi = min(history_max_eff, int(a.item()))
        if g_hi < history_tokens_min:
            raise ValueError(
                "stream window sampling found no valid history length; "
                f"sample={b}, active_left={int(a.item())}, "
                f"history_tokens_min={history_tokens_min}, "
                f"history_tokens_max_effective={history_max_eff}"
            )
        if history_override is None:
            g = torch.randint(history_tokens_min, g_hi + 1, (1,), device=device)[0]
        else:
            g = history_override[b]
            g_int = int(g.item())
            if g_int < history_tokens_min or g_int > g_hi:
                raise ValueError(
                    "history_tokens must fit before active_left and inside context; "
                    f"sample={b}, history_tokens={g_int}, valid_range="
                    f"[{history_tokens_min}, {g_hi}]"
                )

        active_values.append(a.to(dtype=torch.long))
        history_values.append(g.to(dtype=torch.long))
        horizon_values.append(h.to(dtype=torch.long))
        horizon_cap_values.append(
            torch.as_tensor(horizon_cap_clip, device=device, dtype=torch.long)
        )
        horizon_short_fallback_values.append(
            torch.as_tensor(short_fallback, device=device, dtype=torch.bool)
        )

    active = torch.stack(active_values).to(device=device, dtype=torch.long)
    history = torch.stack(history_values).to(device=device, dtype=torch.long)
    horizon = torch.stack(horizon_values).to(device=device, dtype=torch.long)
    horizon_cap = torch.stack(horizon_cap_values).to(device=device, dtype=torch.long)
    horizon_short_fallback = torch.stack(horizon_short_fallback_values).to(
        device=device, dtype=torch.bool
    )
    window_left = active - history
    latent_num_tokens = history + int(chunk_size) + int(rollout_span)
    traj_num_tokens = latent_num_tokens + horizon

    return {
        "window_left_tokens": window_left,
        "active_left_tokens": active,
        "history_tokens": history,
        "horizon_tokens": horizon,
        "latent_num_tokens": latent_num_tokens,
        "traj_num_tokens": traj_num_tokens,
        "horizon_cap_clip": horizon_cap,
        "horizon_short_fallback": horizon_short_fallback,
        "history_tokens_max_effective": int(history_max_eff),
        "rollout_span": int(rollout_span),
    }


__all__ = ["resolve_history_tokens_max", "sample_stream_window_indices"]
