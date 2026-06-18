"""Response helpers for the staged web demo API refactor."""

from __future__ import annotations


def success(**payload):
    return {"status": "success", **payload}


def error(message: str, **payload):
    return {"status": "error", "message": message, **payload}


__all__ = ["error", "success"]

