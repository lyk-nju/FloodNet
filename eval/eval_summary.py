"""Compatibility wrapper for LDF validation summary helpers."""

from __future__ import annotations

from utils.training.ldf import validation_summary as _compat_impl
from utils.training.ldf.validation_summary import *  # noqa: F401,F403
from utils.training.ldf.validation_summary import (
    build_summary,
    flatten_validation_eval_summary,
    process_validation_generation_results,
    save_eval_payloads,
)

for _compat_name in dir(_compat_impl):
    if _compat_name.startswith("_") and not _compat_name.startswith("__"):
        globals()[_compat_name] = getattr(_compat_impl, _compat_name)

del _compat_impl, _compat_name
