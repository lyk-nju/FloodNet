"""Compatibility wrapper for LDF T2M generation helpers."""

from __future__ import annotations

from utils.training.ldf import t2m_generation as _compat_impl
from utils.training.ldf.t2m_generation import *  # noqa: F401,F403
from utils.training.ldf.t2m_generation import run_t2m_generation_mode

for _compat_name in dir(_compat_impl):
    if _compat_name.startswith("_") and not _compat_name.startswith("__"):
        globals()[_compat_name] = getattr(_compat_impl, _compat_name)

del _compat_impl, _compat_name
