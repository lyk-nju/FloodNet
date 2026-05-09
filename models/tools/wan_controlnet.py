# ControlNet-style residual branch for WanModel (1D / t2v usage).
#
# Design goals:
# - Same inputs as WanModel.forward (noisy latent + time + text + traj_emb).
# - Outputs per-layer residuals (B, seq_len, dim) to be injected into WanModel blocks.
# - Zero-initialized residual heads so initial behavior matches the backbone.

import math
import warnings
from typing import List, Optional

import torch
import torch.nn as nn

from .wan_model import (
    WanAttentionBlock,
    rope_params,
    sinusoidal_embedding_1d,
)


def _zero_linear(dim: int) -> nn.Linear:
    m = nn.Linear(dim, dim)
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


class WanControlNet(nn.Module):
    """A lightweight ControlNet branch matching WanModel's internal representations."""

    def __init__(
        self,
        *,
        model_type: str = "t2v",
        patch_size=(1, 1, 1),
        text_len: int = 512,
        in_dim: int = 256,
        dim: int = 1024,
        ffn_dim: int = 2048,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 256,  # unused, kept for interface parity
        num_heads: int = 8,
        num_layers: int = 8,
        window_size=(-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        causal: bool = False,
        traj_enc_dim: int = 0,
    ):
        super().__init__()
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.causal = causal
        self.traj_enc_dim = traj_enc_dim

        # Match WanModel embeddings.
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Blocks: same class as WanModel.
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    causal,
                )
                for _ in range(num_layers)
            ]
        )

        # FlexTraj tokens.
        if traj_enc_dim > 0:
            self.traj_in_proj = nn.Linear(traj_enc_dim, dim)
            self.traj_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.traj_in_proj = None
            self.traj_type_embed = None

        # RoPE freqs (same construction as WanModel).
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        # Per-layer residual heads (zero-init).
        self.zero_out = nn.ModuleList([_zero_linear(dim) for _ in range(num_layers)])

        # Init to match WanModel defaults for shared layers.
        self.init_weights()
        if self.traj_in_proj is not None:
            nn.init.zeros_(self.traj_in_proj.weight)
            nn.init.zeros_(self.traj_in_proj.bias)

    def init_weights(self):
        # Same init policy as WanModel (linear xavier; embeddings normal; residual heads already zero).
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in set(self.zero_out):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    @torch.no_grad()
    def init_from_backbone(self, backbone) -> None:
        """Copy matching weights from a WanModel instance."""
        result = self.load_state_dict(backbone.state_dict(), strict=False)
        if result.missing_keys:
            # Expected: ControlNet-only layers (zero_out heads, traj_in_proj) have no backbone counterpart.
            warnings.warn(
                f"init_from_backbone: {len(result.missing_keys)} ControlNet-only keys "
                f"not copied from backbone (will keep current init): {result.missing_keys}",
                stacklevel=2,
            )
        if result.unexpected_keys:
            # Keys present in backbone but absent from ControlNet — silently ignored by strict=False.
            warnings.warn(
                f"init_from_backbone: {len(result.unexpected_keys)} backbone keys have no "
                f"ControlNet counterpart (ignored): {result.unexpected_keys}",
                stacklevel=2,
            )
        # Ensure zero heads remain exactly zero.
        for m in self.zero_out:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
        y: Optional[List[torch.Tensor]] = None,
        traj_emb: Optional[torch.Tensor] = None,
        traj_seq_lens: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if self.model_type == "i2v":
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Patch embeddings (same as WanModel).
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max().item() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        latent_pad_len = None
        traj_seq_lens_attn = None
        if self.traj_in_proj is not None and traj_emb is not None:
            traj_t = self.traj_in_proj(traj_emb.to(dtype=x.dtype, device=x.device))
            traj_t = traj_t + self.traj_type_embed
            bt, tlen, _ = traj_t.shape
            if tlen < seq_len:
                traj_t = torch.cat(
                    [traj_t, traj_t.new_zeros(bt, seq_len - tlen, traj_t.size(-1))],
                    dim=1,
                )
            elif tlen > seq_len:
                traj_t = traj_t[:, :seq_len, :]
            x = torch.cat([x, traj_t], dim=1)
            if traj_seq_lens is None:
                traj_seq_lens_attn = seq_lens
            else:
                traj_seq_lens_attn = (
                    traj_seq_lens.to(device=device, dtype=torch.long).clamp(
                        min=0, max=seq_len
                    )
                )
            latent_pad_len = seq_len

        # Time embeddings (same as WanModel).
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(-1, seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            tflat = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, tflat)
                .unflatten(0, (bt, seq_len))
                .float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            if latent_pad_len is not None:
                e0_traj = torch.zeros_like(e0)
                e0 = torch.cat([e0, e0_traj], dim=1)
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Text context — deduplicate frame-aligned entries (same as WanModel).
        context_lens = torch.tensor(
            [u.size(0) for u in context], dtype=torch.long, device=device
        )
        if len(context) > 0:
            unique_map = {}
            unique_list = []
            ctx_indices = []
            for u in context:
                key = u.data_ptr()
                idx = unique_map.get(key)
                if idx is None:
                    idx = len(unique_list)
                    unique_map[key] = idx
                    unique_list.append(u)
                ctx_indices.append(idx)
            unique_stacked = torch.stack(
                [
                    torch.cat(
                        [u[: self.text_len],
                         u.new_zeros(self.text_len - min(u.size(0), self.text_len), u.size(1))]
                    )
                    for u in unique_list
                ]
            )
            unique_embedded = self.text_embedding(unique_stacked)
            if len(unique_list) < len(context):
                idx_t = torch.tensor(
                    ctx_indices, device=unique_embedded.device, dtype=torch.long
                )
                context = unique_embedded[idx_t]
            else:
                context = unique_embedded
        else:
            context = torch.empty(
                0, self.text_len, self.dim, device=device, dtype=torch.float32
            )

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            traj_seq_lens=traj_seq_lens_attn,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            latent_pad_len=latent_pad_len,
        )

        residuals: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            h_lat = x[:, :seq_len, :]
            residuals.append(self.zero_out[i](h_lat))
        return residuals

