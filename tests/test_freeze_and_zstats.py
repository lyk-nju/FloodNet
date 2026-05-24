"""B-P0-2 (mask_emb trainable under freeze_backbone) + B-P0-1 (z_stats load).

Builds a tiny DiffForcingWanModel via a precomputed-text fixture (no T5 ckpt
needed) so the freeze + z_stats behavior is verified on the real model object.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

_TEXT_DIM = 4096   # DiffForcingWanModel hardcodes self.text_dim = 4096


@pytest.fixture
def precomputed_text(tmp_path):
    path = tmp_path / "t5.pt"
    # must include the empty-string key (CFG / dropout)
    torch.save({"embeddings": {"": torch.zeros(2, _TEXT_DIM)}, "text_dim": _TEXT_DIM}, path)
    return str(path)


def _tiny_model(precomputed_text, **kw):
    from models.diffusion_forcing_wan import DiffForcingWanModel

    return DiffForcingWanModel(
        input_dim=4, hidden_dim=64, ffn_dim=128, freq_dim=64,
        num_heads=2, num_layers=1, text_len=8, traj_encoder_in_dim=7,
        use_precomputed_text_emb=True, precomputed_text_emb_path=precomputed_text,
        **kw,
    )


# ---------------------------------------------------------------------------
# B-P0-2: mask_emb stays trainable when the backbone is frozen
# ---------------------------------------------------------------------------


def test_mask_emb_trainable_under_freeze_backbone(precomputed_text):
    m = _tiny_model(precomputed_text, freeze_backbone=True)
    assert m.model.mask_emb.requires_grad is True
    # the rest of the backbone is still frozen
    assert m.model.patch_embedding.weight.requires_grad is False
    # mask_emb is in the trainable-parameter set the optimizer/EMA collect
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    assert id(m.model.mask_emb) in trainable


def test_backbone_still_frozen_except_heads(precomputed_text):
    m = _tiny_model(precomputed_text, freeze_backbone=True)
    # controlnet + traj encoders trainable (existing behavior preserved)
    assert any(p.requires_grad for p in m.controlnet.parameters())
    assert any(p.requires_grad for p in m.traj_encoder.parameters())


# ---------------------------------------------------------------------------
# B-P0-1: load_z_stats overwrites z_std on the real model (the cfg wiring in
# CustomLightningModule.__init__ just calls this).
# ---------------------------------------------------------------------------


def test_load_z_stats_changes_z_std(precomputed_text, tmp_path):
    m = _tiny_model(precomputed_text, freeze_backbone=True)
    assert torch.allclose(m.model.z_std, torch.ones(4))   # default before load
    np.save(tmp_path / "z_mean.npy", np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    np.save(tmp_path / "z_std.npy", np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    m.model.load_z_stats(str(tmp_path))
    assert torch.allclose(m.model.z_std, torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert not torch.allclose(m.model.z_std, torch.ones(4))
