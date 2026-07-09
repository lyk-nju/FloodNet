"""Text embedding resolver for NoiseInitializer training."""

from __future__ import annotations

import torch.nn as nn

from utils.training.root_refiner.text_encoder import (
    PrecomputedT5PooledTextEncoder,
)


def resolve_noise_initializer_text_encoder(
    cfg: dict,
    *,
    text_encoder: nn.Module | None = None,
    text_emb_dim: int | None = None,
) -> nn.Module | None:
    """Resolve optional text encoder for initializer training.

    ``None`` means callers should fall back to the frozen LDF model's own text
    cache/encoder.  Real dataset training should pass a configured precomputed
    encoder so it does not depend on online T5.
    """

    if text_encoder is not None:
        return text_encoder
    text_cfg = (cfg.get("text_encoder", {}) if hasattr(cfg, "get") else {}) or {}
    text_type = text_cfg.get("type")
    if text_type in (None, "", "ldf_model"):
        return None
    if text_type == "precomputed_t5_pool":
        path = text_cfg.get("precomputed_text_emb_path")
        if not path:
            raise ValueError(
                "text_encoder.type=precomputed_t5_pool requires "
                "text_encoder.precomputed_text_emb_path"
            )
        return PrecomputedT5PooledTextEncoder(
            path,
            pooling=text_cfg.get("pooling", "mean"),
            expected_dim=text_emb_dim,
        )
    raise ValueError(f"unknown noise initializer text_encoder.type={text_type!r}")


__all__ = ["resolve_noise_initializer_text_encoder"]
