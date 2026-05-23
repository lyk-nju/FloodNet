"""Tests for T_B_02 Wan model fields + backward-compatible ckpt loading.

- WanModel.mask_emb / z_mean / z_std exist, are [in_dim], and are in state_dict.
- WanModel.load_z_stats overwrites the buffers from disk.
- backfill_compat_state_dict back-fills the new optional keys for an OLD ckpt
  (verified against the real step_460000.ckpt key set when available).
- A WanModel-level strict reload of an old-style state_dict (missing the new
  keys) succeeds after back-fill.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from models.diffusion_forcing_wan import (
    _BACKWARD_COMPAT_OPTIONAL_NAMES,
    _is_optional_compat_key,
    backfill_compat_state_dict,
)
from models.tools.wan_model import WanModel

_CKPT = Path(__file__).resolve().parent.parent / "step_460000.ckpt"


def _tiny_wan():
    return WanModel(
        in_dim=4, dim=16, ffn_dim=32, freq_dim=16, text_dim=32, out_dim=4,
        num_heads=2, num_layers=1, patch_size=(1, 1, 1), text_len=8,
    )


# ---------------------------------------------------------------------------
# Model fields
# ---------------------------------------------------------------------------


def test_wan_has_mask_emb_and_z_buffers_with_in_dim_shape():
    m = _tiny_wan()
    assert isinstance(m.mask_emb, torch.nn.Parameter)
    assert tuple(m.mask_emb.shape) == (m.in_dim,)
    assert tuple(m.z_mean.shape) == (m.in_dim,)
    assert tuple(m.z_std.shape) == (m.in_dim,)
    # default buffer values
    assert torch.allclose(m.z_mean, torch.zeros(m.in_dim))
    assert torch.allclose(m.z_std, torch.ones(m.in_dim))


def test_new_fields_are_persistent_in_state_dict():
    m = _tiny_wan()
    sd = m.state_dict()
    assert "mask_emb" in sd
    assert "z_mean" in sd
    assert "z_std" in sd


def test_load_z_stats_overwrites_buffers(tmp_path):
    m = _tiny_wan()
    np.save(tmp_path / "z_mean.npy", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    np.save(tmp_path / "z_std.npy", np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32))
    m.load_z_stats(str(tmp_path))
    assert torch.allclose(m.z_mean, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(m.z_std, torch.tensor([0.5, 0.6, 0.7, 0.8]))


# ---------------------------------------------------------------------------
# Back-fill helper
# ---------------------------------------------------------------------------


def test_backfill_adds_missing_optional_keys():
    own = {
        "model.patch_embedding.weight": torch.zeros(2),
        "model.mask_emb": torch.randn(4),
        "model.z_mean": torch.zeros(4),
        "model.z_std": torch.ones(4),
    }
    old = {"model.patch_embedding.weight": torch.ones(2)}   # 4D ckpt: no new fields
    filled, n = backfill_compat_state_dict(old, own)
    assert n == 3
    assert "model.mask_emb" in filled and "model.z_mean" in filled and "model.z_std" in filled
    # original key preserved (not overwritten by own)
    assert torch.allclose(filled["model.patch_embedding.weight"], torch.ones(2))
    # back-filled values come from own
    assert torch.allclose(filled["model.mask_emb"], own["model.mask_emb"])


def test_backfill_noop_when_keys_present():
    own = {"x.z_std": torch.ones(4)}
    incoming = {"x.z_std": torch.full((4,), 9.0)}
    filled, n = backfill_compat_state_dict(incoming, own)
    assert n == 0
    assert torch.allclose(filled["x.z_std"], torch.full((4,), 9.0))   # untouched


def test_backfill_names_constant_and_leaf_matcher():
    assert _BACKWARD_COMPAT_OPTIONAL_NAMES == ("mask_emb", "z_mean", "z_std")
    # leaf-name match works for both bare and prefixed keys
    assert _is_optional_compat_key("mask_emb")
    assert _is_optional_compat_key("model.mask_emb")
    assert _is_optional_compat_key("a.b.z_std")
    assert not _is_optional_compat_key("model.patch_embedding.weight")


def test_wan_strict_reload_after_backfill_for_old_style_state_dict():
    """Simulate an old ckpt (state_dict without the 3 new keys) and confirm a
    strict load succeeds after back-fill at the WanModel level (bare keys)."""
    m = _tiny_wan()
    full = m.state_dict()
    old = {k: v for k, v in full.items() if not _is_optional_compat_key(k)}
    assert "mask_emb" not in old
    filled, n = backfill_compat_state_dict(old, m.state_dict())
    assert n == 3
    # strict load must not raise
    missing, unexpected = m.load_state_dict(filled, strict=True)
    assert not missing and not unexpected


# ---------------------------------------------------------------------------
# Real ckpt (skipped if the test fixture ckpt isn't present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CKPT.is_file(), reason="step_460000.ckpt not present")
def test_real_ckpt_missing_new_keys_then_backfilled():
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    # The old ckpt must NOT contain the new fields.
    assert not any(_is_optional_compat_key(k) for k in sd)
    n_before = len(sd)
    # Simulate the new model's own_state having the 3 new keys under model.*
    own = {
        "model.mask_emb": torch.randn(4),
        "model.z_mean": torch.zeros(4),
        "model.z_std": torch.ones(4),
    }
    filled, n = backfill_compat_state_dict(sd, own)
    assert n == 3
    assert len(filled) == n_before + 3
    for k in ("model.mask_emb", "model.z_mean", "model.z_std"):
        assert k in filled
