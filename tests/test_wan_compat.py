"""Tests for T_B_02 Wan model fields.

- WanModel.mask_emb / z_mean / z_std exist, are [in_dim], and are in state_dict.
- WanModel.load_z_stats overwrites the buffers from disk.
"""

from __future__ import annotations

import numpy as np
import torch

from models.tools.wan_model import WanModel


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
