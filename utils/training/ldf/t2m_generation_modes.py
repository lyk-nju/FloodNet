"""T2M metric generation-mode config helpers."""

from __future__ import annotations

from collections.abc import Iterable

from omegaconf import OmegaConf


T2M_GENERATE = "generate"
T2M_STREAM_GENERATE = "stream_generate"
T2M_GENERATION_MODES = (T2M_GENERATE, T2M_STREAM_GENERATE)


def _normalize_mode(value) -> str:
    mode = str(value).strip().lower()
    if mode == "both":
        return mode
    if mode not in T2M_GENERATION_MODES:
        raise ValueError(
            "validation.t2m_generation_modes must contain only "
            f"{T2M_GENERATION_MODES}; got {value!r}."
        )
    return mode


def _raw_mode_list(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def resolve_t2m_generation_modes(cfg) -> tuple[str, ...]:
    """Return configured T2M generation modes.

    Defaults to ``("generate",)`` to preserve the old full-sequence T2M
    validation path. Use ``validation.t2m_generation_modes`` to opt into
    ``stream_generate`` or both modes. ``validation.t2m_generation_mode=both`` is
    accepted as a convenience alias.
    """
    modes_value = OmegaConf.select(cfg, "validation.t2m_generation_modes", default=None)
    mode_value = OmegaConf.select(cfg, "validation.t2m_generation_mode", default=None)
    if modes_value is not None and mode_value is not None:
        raise ValueError(
            "Set only one of validation.t2m_generation_modes or "
            "validation.t2m_generation_mode."
        )
    if modes_value is None and mode_value is None:
        return (T2M_GENERATE,)

    raw_modes = _raw_mode_list(mode_value if mode_value is not None else modes_value)
    resolved: list[str] = []
    for raw in raw_modes:
        mode = _normalize_mode(raw)
        if mode == "both":
            expanded = list(T2M_GENERATION_MODES)
        else:
            expanded = [mode]
        for item in expanded:
            if item in resolved:
                raise ValueError(
                    f"validation.t2m_generation_modes contains duplicate mode {item!r}."
                )
            resolved.append(item)
    if not resolved:
        raise ValueError("validation.t2m_generation_modes must not be empty.")
    return tuple(resolved)


__all__ = [
    "T2M_GENERATE",
    "T2M_STREAM_GENERATE",
    "T2M_GENERATION_MODES",
    "resolve_t2m_generation_modes",
]
