"""RootRefiner text encoder resolver."""

from __future__ import annotations

import hashlib
import logging

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class FrozenStubTextEncoder(nn.Module):
    """Deterministic debug-only text embeddings from stable caption hashes."""

    def __init__(self, emb_dim: int, vocab: int = 4096):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab = vocab
        self.table = nn.Embedding(vocab, emb_dim)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _stable_id(text: str, vocab: int) -> int:
        """Process-stable hash of `text` into `[0, vocab)`."""
        digest = hashlib.md5(text.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big") % vocab

    @torch.no_grad()
    def encode(self, texts: list[str], device=None) -> Tensor:
        text_ids = torch.tensor(
            [self._stable_id(text, self.vocab) for text in texts],
            dtype=torch.long,
        )
        if device is not None:
            text_ids = text_ids.to(device)
        return self.table(text_ids)


class PrecomputedT5PooledTextEncoder(nn.Module):
    """Pool precomputed T5 caption embeddings to one vector per caption."""

    def __init__(self, path, pooling: str = "mean", expected_dim: int | None = None):
        super().__init__()
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if "embeddings" not in cache:
            raise ValueError(f"{path}: precomputed T5 cache missing 'embeddings' key")
        self._embeddings = cache["embeddings"]
        if pooling not in ("mean", "first"):
            raise ValueError(f"unknown pooling '{pooling}' (use 'mean' or 'first')")
        self.pooling = pooling
        sample_embedding = next(iter(self._embeddings.values()))
        self.text_dim = int(cache.get("text_dim", sample_embedding.shape[-1]))
        if expected_dim is not None and int(expected_dim) != self.text_dim:
            raise ValueError(
                f"model.text_emb_dim={expected_dim} != precomputed T5 text_dim="
                f"{self.text_dim}; set model.text_emb_dim to {self.text_dim}."
            )

    def _lookup(self, text: str) -> Tensor:
        if text in self._embeddings:
            return self._embeddings[text]
        stripped = text.strip()
        if stripped in self._embeddings:
            return self._embeddings[stripped]
        raise KeyError(
            f"Caption not in precomputed T5 table: {text!r}. "
            "Re-run tools/pretokenize_t5_text.py with the same config."
        )

    def _pool(self, seq: Tensor) -> Tensor:
        return seq.mean(dim=0) if self.pooling == "mean" else seq[0]

    @torch.no_grad()
    def encode(self, texts: list[str], device=None) -> Tensor:
        out = torch.stack(
            [self._pool(self._lookup(text).float()) for text in texts]
        )
        if device is not None:
            out = out.to(device)
        return out


def resolve_text_encoder(cfg, text_encoder=None, text_emb_dim: int | None = None):
    if text_encoder is not None:
        return text_encoder
    text_cfg = (cfg.get("text_encoder", {}) if hasattr(cfg, "get") else {}) or {}
    text_type = text_cfg.get("type")
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
    if text_cfg.get("debug_stub", False):
        log.warning(
            "Using FrozenStubTextEncoder (DEBUG ONLY: hashed caption-id "
            "embeddings, NOT semantic T5). Fine for smoke/overfit tests but will "
            "NOT generalize; real training must use text_encoder.type="
            "precomputed_t5_pool or pass a real encoder."
        )
        return FrozenStubTextEncoder(text_emb_dim)
    raise NotImplementedError(
        "No text encoder available: set text_encoder.type=precomputed_t5_pool "
        "(real training), text_encoder.debug_stub=true (smoke/tests), or pass "
        "text_encoder=."
    )


__all__ = [
    "FrozenStubTextEncoder",
    "PrecomputedT5PooledTextEncoder",
    "resolve_text_encoder",
]
