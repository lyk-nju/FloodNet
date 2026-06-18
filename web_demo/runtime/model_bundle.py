"""Loaded model bundle contract for the web runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    vae: Any
    ldf_model: Any
    cfg: Any
    device: str
    stream_generator: Any
    root_refiner: Any | None = None
    root_text_encoder: Any | None = None


__all__ = ["ModelBundle"]
