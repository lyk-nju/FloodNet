"""Streaming horizon attention contract tests."""

from __future__ import annotations

import torch

from models.tools.attention import flextraj_self_attn_bias
from models.tools.wan_controlnet import WanControlNet
from models.tools.wan_model import WanModel, WanSelfAttention, rope_params


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


class _CaptureBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_len = None
        self.seen_latent_pad_len = None
        self.seen_traj_pad_len = None
        self.seen_traj_seq_lens = None

    def forward(self, x, **kwargs):
        self.seen_len = x.shape[1]
        self.seen_latent_pad_len = kwargs.get("latent_pad_len")
        self.seen_traj_pad_len = kwargs.get("traj_pad_len")
        self.seen_traj_seq_lens = kwargs.get("traj_seq_lens")
        return x


def _tiny_wan_model() -> WanModel:
    model = WanModel(
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
    model.blocks = torch.nn.ModuleList([_CaptureBlock()])
    return model


def test_wan_model_does_not_truncate_streaming_future_traj_tokens():
    model = _tiny_wan_model()
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


def test_wan_model_defaults_long_traj_lens_to_full_traj_length():
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
    assert torch.equal(block.seen_traj_seq_lens, torch.tensor([5]))


def test_wan_controlnet_keeps_future_traj_tokens_but_returns_latent_residuals():
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
    assert block.seen_len == 8
    assert block.seen_latent_pad_len == 3
    assert block.seen_traj_pad_len == 5
    assert torch.equal(block.seen_traj_seq_lens, torch.tensor([5]))
    assert len(residuals) == 1
    assert residuals[0].shape == (1, 3, 12)
