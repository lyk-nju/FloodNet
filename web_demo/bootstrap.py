"""Bootstrap helpers for constructing the web demo runtime."""

from __future__ import annotations


def build_runtime(*args, **kwargs):
    from .model_manager import get_model_manager

    return get_model_manager(*args, **kwargs)


__all__ = ["build_runtime"]

