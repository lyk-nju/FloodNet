"""Tests for the shared text-encoder resolver (P0-1 / P0-3).

PrecomputedT5PooledTextEncoder + resolve_text_encoder are verified with a tiny
in-memory T5 cache (text_dim 16) — no real data needed. Also confirms a
RefinerLightningModule built from a precomputed_t5_pool config (the path the
benchmark uses to reload a real ckpt) no longer raises NotImplementedError.
"""

from __future__ import annotations

import pytest
import torch

from utils.text_encoder_resolver import (
    FrozenStubTextEncoder,
    PrecomputedT5PooledTextEncoder,
    resolve_text_encoder,
)

_DIM = 16


@pytest.fixture
def t5_cache(tmp_path):
    emb = {
        "walk": torch.ones(3, _DIM),                       # mean → ones
        "run forward": torch.arange(2 * _DIM, dtype=torch.float32).reshape(2, _DIM),
        "": torch.zeros(2, _DIM),
    }
    path = tmp_path / "t5.pt"
    torch.save({"embeddings": emb, "text_dim": _DIM}, path)
    return str(path)


# ---------------------------------------------------------------------------
# PrecomputedT5PooledTextEncoder
# ---------------------------------------------------------------------------


def test_encode_shape_and_mean_pool(t5_cache):
    enc = PrecomputedT5PooledTextEncoder(t5_cache, pooling="mean")
    out = enc.encode(["walk"])
    assert out.shape == (1, _DIM)
    assert torch.allclose(out[0], torch.ones(_DIM))        # mean of all-ones rows


def test_first_pooling(t5_cache):
    enc = PrecomputedT5PooledTextEncoder(t5_cache, pooling="first")
    out = enc.encode(["run forward"])
    assert torch.allclose(out[0], torch.arange(_DIM, dtype=torch.float32))


def test_strip_fallback(t5_cache):
    enc = PrecomputedT5PooledTextEncoder(t5_cache)
    out = enc.encode(["  walk  "])                          # trailing/leading ws
    assert torch.allclose(out[0], torch.ones(_DIM))


def test_missing_caption_raises_keyerror(t5_cache):
    enc = PrecomputedT5PooledTextEncoder(t5_cache)
    with pytest.raises(KeyError):
        enc.encode(["jump"])


def test_expected_dim_mismatch_raises(t5_cache):
    with pytest.raises(ValueError):
        PrecomputedT5PooledTextEncoder(t5_cache, expected_dim=999)


def test_bad_pooling_raises(t5_cache):
    with pytest.raises(ValueError):
        PrecomputedT5PooledTextEncoder(t5_cache, pooling="attention")


def test_no_learnable_params(t5_cache):
    enc = PrecomputedT5PooledTextEncoder(t5_cache)
    # embeddings are a plain dict, not buffers → not bloating the ckpt
    assert len(list(enc.parameters())) == 0
    assert len(enc.state_dict()) == 0


# ---------------------------------------------------------------------------
# resolve_text_encoder dispatch
# ---------------------------------------------------------------------------


def test_resolve_precomputed(t5_cache):
    cfg = {"text_encoder": {"type": "precomputed_t5_pool",
                            "precomputed_text_emb_path": t5_cache, "pooling": "mean"}}
    enc = resolve_text_encoder(cfg, text_emb_dim=_DIM)
    assert isinstance(enc, PrecomputedT5PooledTextEncoder)


def test_resolve_precomputed_missing_path_raises():
    cfg = {"text_encoder": {"type": "precomputed_t5_pool"}}
    with pytest.raises(ValueError):
        resolve_text_encoder(cfg, text_emb_dim=_DIM)


def test_resolve_debug_stub():
    enc = resolve_text_encoder({"text_encoder": {"debug_stub": True}}, text_emb_dim=32)
    assert isinstance(enc, FrozenStubTextEncoder)


def test_resolve_explicit_passthrough():
    sentinel = torch.nn.Identity()
    assert resolve_text_encoder({}, text_encoder=sentinel) is sentinel


def test_resolve_none_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        resolve_text_encoder({"text_encoder": {}}, text_emb_dim=16)


# ---------------------------------------------------------------------------
# P0-3: RefinerLightningModule(precomputed_t5_pool) builds (benchmark reload path)
# ---------------------------------------------------------------------------


def test_refiner_module_with_precomputed_builds(t5_cache):
    from train_refiner import RefinerLightningModule

    cfg = {
        "model": {
            "d_model": 32, "n_layers": 2, "n_heads": 4, "ff_dim": 64,
            "max_tokens": 8, "min_tokens": 2, "frames_per_token": 4,
            "n_path": 16, "n_hist": 8, "text_emb_dim": _DIM, "dropout": 0.0,
        },
        "training": {"lr": 1e-3, "weight_decay": 0.01},
        "loss": {"heading_form": "cosine"},
        "loss_weights": {"num_token": 1.0, "xyz": 5.0, "heading": 1.0,
                         "fwd_delta": 0.5, "yaw_delta": 0.5, "smoothness": 0.0},
        "text_encoder": {"type": "precomputed_t5_pool",
                         "precomputed_text_emb_path": t5_cache, "pooling": "mean"},
    }
    module = RefinerLightningModule(cfg)          # must NOT raise NotImplementedError
    assert isinstance(module.text_encoder, PrecomputedT5PooledTextEncoder)
