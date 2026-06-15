"""Shared text-encoder resolver for RootRefiner train + benchmark (P0-1 / P0-3).

train_refiner.py and eval/root_refiner_benchmark.py must build the SAME text
encoder from a config, so a checkpoint trained with real (precomputed-T5) text
conditioning can be evaluated with identical lookup / pooling. Resolution order:

  1. explicit `text_encoder=` object (e.g. an externally-wired encoder);
  2. cfg.text_encoder.type == "precomputed_t5_pool" → PrecomputedT5PooledTextEncoder
     (reuses the LDF offline T5 cache; pools [L,4096]→[4096]);
  3. cfg.text_encoder.debug_stub == true → FrozenStubTextEncoder (DEBUG ONLY:
     hashed caption-id embeddings, no semantic generalization — smoke/tests);
  4. otherwise → NotImplementedError.
"""

from __future__ import annotations

import hashlib
import logging

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class FrozenStubTextEncoder(nn.Module):
    """Deterministic per-text embedding via hashing + a frozen embedding table.

    DEBUG ONLY. Real training must use a semantic encoder (precomputed_t5_pool or
    an explicitly-wired one) — the stub only memorizes caption ids.
    """

    def __init__(self, emb_dim: int, vocab: int = 4096):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab = vocab
        self.table = nn.Embedding(vocab, emb_dim)
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _stable_id(text: str, vocab: int) -> int:
        """Process-stable hash of `text` → [0, vocab).

        ⚠ Must NOT use builtin hash(): str hashing is salted per process
        (PYTHONHASHSEED), so the same caption would map to different rows across
        runs / between train and benchmark, silently defeating conditioning.
        """
        digest = hashlib.md5(text.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big") % vocab

    @torch.no_grad()
    def encode(self, texts: list[str], device=None) -> Tensor:
        ids = torch.tensor(
            [self._stable_id(t, self.vocab) for t in texts], dtype=torch.long,
        )
        if device is not None:
            ids = ids.to(device)
        return self.table(ids)


class PrecomputedT5PooledTextEncoder(nn.Module):
    """Pool the LDF offline T5 cache (`{"embeddings": {cap: [L, 4096]}, ...}`) to
    a single per-caption vector. Has NO learnable params (the table is a plain
    dict, not a buffer) so it is not bloated into the refiner checkpoint.
    """

    def __init__(self, path, pooling: str = "mean", expected_dim: int | None = None):
        super().__init__()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if "embeddings" not in blob:
            raise ValueError(f"{path}: precomputed T5 cache missing 'embeddings' key")
        self._emb = blob["embeddings"]            # caption -> [L, 4096]
        if pooling not in ("mean", "first"):
            raise ValueError(f"unknown pooling '{pooling}' (use 'mean' or 'first')")
        self.pooling = pooling
        any_vec = next(iter(self._emb.values()))
        self.text_dim = int(blob.get("text_dim", any_vec.shape[-1]))
        if expected_dim is not None and int(expected_dim) != self.text_dim:
            raise ValueError(
                f"model.text_emb_dim={expected_dim} != precomputed T5 text_dim="
                f"{self.text_dim}; set model.text_emb_dim to {self.text_dim}."
            )

    def _lookup(self, text: str) -> Tensor:
        if text in self._emb:
            return self._emb[text]
        stripped = text.strip()
        if stripped in self._emb:
            return self._emb[stripped]
        raise KeyError(
            f"Caption not in precomputed T5 table: {text!r}. "
            "Re-run tools/pretokenize_t5_text.py with the same config."
        )

    def _pool(self, seq: Tensor) -> Tensor:
        return seq.mean(dim=0) if self.pooling == "mean" else seq[0]

    @torch.no_grad()
    def encode(self, texts: list[str], device=None) -> Tensor:
        out = torch.stack([self._pool(self._lookup(t).float()) for t in texts])  # [B, D]
        if device is not None:
            out = out.to(device)
        return out


def resolve_text_encoder(cfg, text_encoder=None, text_emb_dim: int | None = None):
    """Build the text encoder per the resolution order in the module docstring."""
    if text_encoder is not None:
        return text_encoder
    te_cfg = (cfg.get("text_encoder", {}) if hasattr(cfg, "get") else {}) or {}
    te_type = te_cfg.get("type")
    if te_type == "precomputed_t5_pool":
        path = te_cfg.get("precomputed_text_emb_path")
        if not path:
            raise ValueError(
                "text_encoder.type=precomputed_t5_pool requires "
                "text_encoder.precomputed_text_emb_path"
            )
        return PrecomputedT5PooledTextEncoder(
            path, pooling=te_cfg.get("pooling", "mean"), expected_dim=text_emb_dim,
        )
    if te_cfg.get("debug_stub", False):
        log.warning(
            "Using FrozenStubTextEncoder (DEBUG ONLY: hashed caption-id "
            "embeddings, NOT semantic T5). Fine for smoke/overfit tests but will "
            "NOT generalize — real training must use text_encoder.type="
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
