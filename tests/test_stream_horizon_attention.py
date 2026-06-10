"""Streaming horizon attention contract tests."""

from __future__ import annotations

import torch
import pytest

from models.tools.attention import flextraj_self_attn_bias
from models.tools.wan_controlnet import WanControlNet
from models.tools.wan_model import (
    WanCrossAttention,
    WanModel,
    WanSelfAttention,
    rope_params,
)


def _freqs_for_head_dim(head_dim: int) -> torch.Tensor:
    return torch.cat(
        [
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
        ],
        dim=1,
    )


def test_flextraj_bias_allows_traj_segment_longer_than_latent_segment():
    bias, query_valid = flextraj_self_attn_bias(
        seq_lens=torch.tensor([3]),
        n_latent=3,
        total_len=8,
        latent_lens=torch.tensor([3]),
        traj_lens=torch.tensor([5]),
        device=torch.device("cpu"),
    )

    assert bias.shape == (1, 1, 8, 8)
    assert query_valid.shape == (1, 8, 1)
    assert query_valid[0, :, 0].tolist() == [1.0] * 8
    assert bias[0, 0, 0, 7].item() == 0.0        # latent query can see future traj
    assert bias[0, 0, 3, 0].item() < -1e30       # traj query still cannot see latent


def test_flextraj_bias_blocks_arbitrary_invalid_traj_tokens():
    bias, query_valid = flextraj_self_attn_bias(
        seq_lens=torch.tensor([3]),
        n_latent=3,
        total_len=7,
        latent_lens=torch.tensor([3]),
        traj_lens=torch.tensor([4]),
        traj_token_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        device=torch.device("cpu"),
    )

    assert query_valid[0, :, 0].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert bias[0, 0, 0, 3].item() < -1e30
    assert bias[0, 0, 0, 4].item() < -1e30
    assert bias[0, 0, 0, 5].item() == 0.0
    assert bias[0, 0, 5, 5].item() == 0.0
    assert bias[0, 0, 5, 3].item() < -1e30


def test_wan_self_attention_accepts_longer_traj_segment():
    attn = WanSelfAttention(dim=12, num_heads=2).eval()
    x = torch.randn(1, 8, 12)
    out = attn(
        x,
        seq_lens=torch.tensor([3]),
        grid_sizes=torch.tensor([[3, 1, 1]]),
        freqs=_freqs_for_head_dim(6),
        latent_pad_len=3,
        traj_pad_len=5,
        traj_seq_lens=torch.tensor([5]),
    )
    assert out.shape == (1, 8, 12)


def test_wan_self_attention_is_invariant_to_masked_invalid_prefix_traj_tokens():
    torch.manual_seed(123)
    attn = WanSelfAttention(dim=12, num_heads=2).eval()
    x_a = torch.randn(1, 7, 12)
    x_b = x_a.clone()
    x_b[:, 3:5] = torch.randn_like(x_b[:, 3:5]) * 1000.0
    traj_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    kwargs = dict(
        seq_lens=torch.tensor([3]),
        grid_sizes=torch.tensor([[3, 1, 1]]),
        freqs=_freqs_for_head_dim(6),
        latent_pad_len=3,
        traj_pad_len=4,
        traj_seq_lens=torch.tensor([4]),
        traj_token_mask=traj_mask,
    )

    with torch.no_grad():
        out_a = attn(x_a, **kwargs)
        out_b = attn(x_b, **kwargs)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_wan_self_attention_without_mask_sees_invalid_prefix_traj_tokens():
    torch.manual_seed(123)
    attn = WanSelfAttention(dim=12, num_heads=2).eval()
    x_a = torch.randn(1, 7, 12)
    x_b = x_a.clone()
    x_b[:, 3:5] = torch.randn_like(x_b[:, 3:5]) * 1000.0

    kwargs = dict(
        seq_lens=torch.tensor([3]),
        grid_sizes=torch.tensor([[3, 1, 1]]),
        freqs=_freqs_for_head_dim(6),
        latent_pad_len=3,
        traj_pad_len=4,
        traj_seq_lens=torch.tensor([4]),
        traj_token_mask=None,
    )

    with torch.no_grad():
        out_a = attn(x_a, **kwargs)
        out_b = attn(x_b, **kwargs)

    assert (out_a - out_b).abs().max().item() > 1e-4


class _CaptureBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_len = None
        self.seen_latent_pad_len = None
        self.seen_traj_pad_len = None
        self.seen_traj_seq_lens = None
        self.seen_traj_token_mask = None
        self.seen_x = None

    def forward(self, x, **kwargs):
        self.seen_len = x.shape[1]
        self.seen_latent_pad_len = kwargs.get("latent_pad_len")
        self.seen_traj_pad_len = kwargs.get("traj_pad_len")
        self.seen_traj_seq_lens = kwargs.get("traj_seq_lens")
        self.seen_traj_token_mask = kwargs.get("traj_token_mask")
        self.seen_x = x.detach().clone()
        return x


def _tiny_wan_model(
    *,
    use_traj_token_mask_in_attention: bool = False,
    use_future_traj_attention: bool = False,
) -> WanModel:
    kwargs = dict(
        model_type="t2v",
        in_dim=4,
        dim=12,
        ffn_dim=24,
        freq_dim=16,
        text_dim=32,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        patch_size=(1, 1, 1),
        text_len=4,
        traj_enc_dim=8,
        use_traj_token_mask_in_attention=use_traj_token_mask_in_attention,
    )
    if use_future_traj_attention:
        kwargs["use_future_traj_attention"] = True
    model = WanModel(**kwargs).eval()
    model.blocks = torch.nn.ModuleList([_CaptureBlock()])
    return model


def test_wan_model_defaults_to_legacy_traj_pad_len():
    model = _tiny_wan_model()
    out = model(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=torch.randn(1, 5, 8),
        traj_seq_lens=torch.tensor([5]),
    )

    block = model.blocks[0]
    assert block.seen_len == 6
    assert block.seen_latent_pad_len == 3
    assert block.seen_traj_pad_len is None
    assert torch.equal(block.seen_traj_seq_lens, torch.tensor([3]))
    assert len(out) == 1
    assert out[0].shape == (4, 3, 1, 1)


def test_wan_model_opt_in_does_not_truncate_streaming_future_traj_tokens():
    model = _tiny_wan_model(use_future_traj_attention=True)
    traj_emb = torch.randn(1, 5, 8)
    out = model(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=traj_emb,
        traj_seq_lens=torch.tensor([5]),
    )

    block = model.blocks[0]
    assert block.seen_len == 8
    assert block.seen_latent_pad_len == 3
    assert block.seen_traj_pad_len == 5
    assert torch.equal(block.seen_traj_seq_lens, torch.tensor([5]))
    assert len(out) == 1
    assert out[0].shape == (4, 3, 1, 1)


def test_wan_model_defaults_missing_traj_lens_to_legacy_seq_lens():
    model = _tiny_wan_model()
    model(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=torch.randn(1, 5, 8),
        traj_seq_lens=None,
    )

    block = model.blocks[0]
    assert torch.equal(block.seen_traj_seq_lens, torch.tensor([3]))


def test_wan_model_defaults_to_projection_only_traj_token_mask():
    model = _tiny_wan_model()
    traj_mask = torch.tensor([[0.0, 1.0, 0.0, 1.0, 1.0]])
    model(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=torch.randn(1, 5, 8),
        traj_seq_lens=torch.tensor([5]),
        traj_token_mask=traj_mask,
    )

    block = model.blocks[0]
    assert block.seen_traj_token_mask is None


def test_wan_model_opt_in_threads_arbitrary_traj_token_mask_to_blocks():
    model = _tiny_wan_model(
        use_traj_token_mask_in_attention=True,
        use_future_traj_attention=True,
    )
    traj_mask = torch.tensor([[0.0, 1.0, 0.0, 1.0, 1.0]])
    model(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=torch.randn(1, 5, 8),
        traj_seq_lens=torch.tensor([5]),
        traj_token_mask=traj_mask,
    )

    block = model.blocks[0]
    assert torch.equal(block.seen_traj_token_mask, traj_mask.bool())


def test_wan_model_masks_projected_traj_tokens_before_blocks():
    model = _tiny_wan_model()
    with torch.no_grad():
        model.traj_in_proj.weight.normal_(std=0.02)
        model.traj_in_proj.bias.normal_(std=0.02)
        model.traj_type_embed.normal_(std=0.02)
    traj_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])

    model(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=torch.randn(1, 5, 8),
        traj_seq_lens=torch.tensor([5]),
        traj_token_mask=traj_mask,
    )

    block = model.blocks[0]
    traj_seen = block.seen_x[:, block.seen_latent_pad_len:]
    assert torch.count_nonzero(traj_seen[:, [1]]) == 0
    assert torch.count_nonzero(traj_seen[:, [0, 2]]) > 0


def test_wan_cross_attention_rejects_frame_aligned_flextraj_length_mismatch():
    cross = WanCrossAttention(dim=12, num_heads=2).eval()
    x = torch.randn(1, 8, 12)  # latent length 3, traj length 5
    context = torch.randn(3, 1, 12)  # frame-aligned context for only 3 tokens
    context_lens = torch.ones(3, dtype=torch.long)

    with pytest.raises(ValueError, match="frame-aligned FlexTraj"):
        cross(x, context, context_lens)


def test_wan_controlnet_defaults_to_legacy_traj_pad_len():
    cn = WanControlNet(
        model_type="t2v",
        in_dim=4,
        dim=12,
        ffn_dim=24,
        freq_dim=16,
        text_dim=32,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        patch_size=(1, 1, 1),
        text_len=4,
        traj_enc_dim=8,
    ).eval()
    cn.blocks = torch.nn.ModuleList([_CaptureBlock()])

    residuals = cn(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=torch.randn(1, 5, 8),
        traj_seq_lens=torch.tensor([5]),
        traj_token_mask=torch.ones(1, 5),
    )

    block = cn.blocks[0]
    assert block.seen_len == 6
    assert block.seen_latent_pad_len == 3
    assert block.seen_traj_pad_len is None
    assert torch.equal(block.seen_traj_seq_lens, torch.tensor([3]))
    assert len(residuals) == 1
    assert residuals[0].shape == (1, 3, 12)


def test_wan_controlnet_opt_in_keeps_future_traj_tokens_but_returns_latent_residuals():
    cn = WanControlNet(
        model_type="t2v",
        in_dim=4,
        dim=12,
        ffn_dim=24,
        freq_dim=16,
        text_dim=32,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        patch_size=(1, 1, 1),
        text_len=4,
        traj_enc_dim=8,
        use_future_traj_attention=True,
    ).eval()
    cn.blocks = torch.nn.ModuleList([_CaptureBlock()])

    residuals = cn(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=torch.randn(1, 5, 8),
        traj_seq_lens=torch.tensor([5]),
        traj_token_mask=torch.ones(1, 5),
    )

    block = cn.blocks[0]
    assert block.seen_len == 8
    assert block.seen_latent_pad_len == 3
    assert block.seen_traj_pad_len == 5
    assert torch.equal(block.seen_traj_seq_lens, torch.tensor([5]))
    assert len(residuals) == 1
    assert residuals[0].shape == (1, 3, 12)


def test_wan_controlnet_defaults_to_projection_only_traj_token_mask():
    cn = WanControlNet(
        model_type="t2v",
        in_dim=4,
        dim=12,
        ffn_dim=24,
        freq_dim=16,
        text_dim=32,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        patch_size=(1, 1, 1),
        text_len=4,
        traj_enc_dim=8,
    ).eval()
    cn.blocks = torch.nn.ModuleList([_CaptureBlock()])
    traj_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])

    cn(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=torch.randn(1, 5, 8),
        traj_seq_lens=torch.tensor([5]),
        traj_token_mask=traj_mask,
    )

    block = cn.blocks[0]
    assert block.seen_traj_token_mask is None


def test_wan_controlnet_opt_in_threads_arbitrary_traj_token_mask_to_blocks():
    cn = WanControlNet(
        model_type="t2v",
        in_dim=4,
        dim=12,
        ffn_dim=24,
        freq_dim=16,
        text_dim=32,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        patch_size=(1, 1, 1),
        text_len=4,
        traj_enc_dim=8,
        use_future_traj_attention=True,
        use_traj_token_mask_in_attention=True,
    ).eval()
    cn.blocks = torch.nn.ModuleList([_CaptureBlock()])
    traj_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])

    cn(
        [torch.randn(4, 3, 1, 1)],
        torch.zeros(1),
        [torch.randn(1, 32)],
        seq_len=3,
        traj_emb=torch.randn(1, 5, 8),
        traj_seq_lens=torch.tensor([5]),
        traj_token_mask=traj_mask,
    )

    block = cn.blocks[0]
    assert torch.equal(block.seen_traj_token_mask, traj_mask.bool())
