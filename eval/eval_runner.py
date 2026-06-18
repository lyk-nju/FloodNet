"""Compatibility wrapper for LDF validation generation helpers."""

from __future__ import annotations

from utils.training.ldf import validation_generation as _compat_impl
from utils.training.ldf.validation_generation import *  # noqa: F401,F403
from utils.training.ldf.validation_generation import run_validation_generation_eval

for _compat_name in dir(_compat_impl):
    if _compat_name.startswith("_") and not _compat_name.startswith("__"):
        globals()[_compat_name] = getattr(_compat_impl, _compat_name)

del _compat_impl, _compat_name
