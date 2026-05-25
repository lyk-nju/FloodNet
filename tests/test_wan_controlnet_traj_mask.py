"""WanControlNet trajectory conditioning tests.

Validates the mask-after-traj_in_proj contract: even when traj_in_proj /
traj_type_embed have moved off zero-init, ControlNet must zero the projected
trajectory tokens whose `traj_token_mask` is 0, so masked / out-of-horizon
slots inject no traj signal into the residuals.

These tests inspect the projection-and-mask step directly (mirroring the
forward-time computation) rather than running the full ControlNet forward,
which depends on flash_attn_2 / CUDA-only kernels.
"""

from __future__ import annotations

import torch

from models.tools.wan_controlnet import WanControlNet


def _make_cn(seed: int = 0, traj_enc_dim: int = 128) -> WanControlNet:
    torch.manual_seed(seed)
    cn = WanControlNet(
        model_type="t2v",
        in_dim=4,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        traj_enc_dim=traj_enc_dim,
    ).eval()
    # Move traj_in_proj / traj_type_embed off zero so the mask is doing real work
    # (a zero-init projection would mask "for free").
    with torch.no_grad():
        cn.traj_in_proj.weight.normal_(std=0.02)
        cn.traj_in_proj.bias.normal_(std=0.02)
        cn.traj_type_embed.normal_(std=0.02)
    return cn


def _project_traj(cn: WanControlNet, traj_emb: torch.Tensor,
                  traj_token_mask: torch.Tensor | None) -> torch.Tensor:
    """Replay the masking step from WanControlNet.forward without running blocks."""
    traj_t = cn.traj_in_proj(traj_emb)
    traj_t = traj_t + cn.traj_type_embed
    if traj_token_mask is not None:
        tm = traj_token_mask.to(dtype=traj_t.dtype)
        if tm.dim() == 2:
            tm = tm[..., None]
        traj_t = traj_t * tm
    return traj_t


def test_traj_token_mask_kills_projected_traj_for_masked_tokens():
    """Tokens with mask=0 → projected traj output is exactly zero on those slots."""
    B, seq_len = 2, 6
    cn = _make_cn()
    traj_emb = torch.randn(B, seq_len, cn.traj_enc_dim)
    mask = torch.ones(B, seq_len)
    mask[:, 4:] = 0.0

    with torch.no_grad():
        out = _project_traj(cn, traj_emb, mask)
    assert out.shape == (B, seq_len, cn.dim)
    # Masked tokens are exactly zero — biases don't leak.
    assert torch.count_nonzero(out[:, 4:]) == 0
    # Valid tokens are non-zero (post-projection has both weight×emb and biases).
    assert torch.count_nonzero(out[:, :4]) > 0


def test_traj_token_mask_invariant_to_masked_traj_emb():
    """Two traj_emb that differ ONLY on masked tokens produce identical output."""
    B, seq_len = 2, 6
    cn = _make_cn(seed=1)
    traj_emb_a = torch.randn(B, seq_len, cn.traj_enc_dim)
    traj_emb_b = traj_emb_a.clone()
    traj_emb_b[:, 4:] = torch.randn_like(traj_emb_b[:, 4:])

    mask = torch.ones(B, seq_len)
    mask[:, 4:] = 0.0

    with torch.no_grad():
        out_a = _project_traj(cn, traj_emb_a, mask)
        out_b = _project_traj(cn, traj_emb_b, mask)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_no_traj_token_mask_lets_traj_through():
    """Without the mask the same masked-frame perturbation reaches the output."""
    B, seq_len = 2, 6
    cn = _make_cn(seed=2)
    traj_emb_a = torch.randn(B, seq_len, cn.traj_enc_dim)
    traj_emb_b = traj_emb_a.clone()
    traj_emb_b[:, 4:] = torch.randn_like(traj_emb_b[:, 4:])

    with torch.no_grad():
        out_a = _project_traj(cn, traj_emb_a, None)
        out_b = _project_traj(cn, traj_emb_b, None)
    assert (out_a - out_b).abs().max().item() > 1e-5


def test_traj_in_proj_zero_init_after_construction():
    """ControlNet's traj_in_proj must initialize to zero — masked / unmasked
    tokens should both produce zero residuals at step 0 (zero-init contract)."""
    cn = WanControlNet(
        model_type="t2v",
        in_dim=4, dim=64, ffn_dim=128, num_heads=4, num_layers=2,
        traj_enc_dim=128,
    ).eval()
    assert torch.count_nonzero(cn.traj_in_proj.weight) == 0
    assert torch.count_nonzero(cn.traj_in_proj.bias) == 0
    # traj_type_embed is also zero at init (default nn.Parameter(torch.zeros(...))).
    assert torch.count_nonzero(cn.traj_type_embed) == 0
