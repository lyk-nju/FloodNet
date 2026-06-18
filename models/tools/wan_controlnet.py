import math
import warnings
import torch
import torch.nn as nn

from typing import List, Optional
from .wan_model import (
    WanAttentionBlock,
    _embed_text_context,
    _prepare_traj_attn_mask,
    rope_params,
    sinusoidal_embedding_1d,
)


class WanControlNet(nn.Module):
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
        out_dim: int = 256,  # useless, kept for interface parity
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

        # Traj-token embedding and type embedding
        if traj_enc_dim > 0:
            self.traj_in_proj = nn.Linear(traj_enc_dim, dim)
            self.traj_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.traj_in_proj = None
            self.traj_type_embed = None

        # RoPE freqs 
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        head_dim = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

        # ControlNet like zero-init.
        self.zero_out = nn.ModuleList(
            [self._make_zero_linear(dim) for _ in range(num_layers)]
        )

        # Init to match WanModel defaults for shared layers.
        self.init_weights()
        if self.traj_in_proj is not None:
            nn.init.zeros_(self.traj_in_proj.weight)
            nn.init.zeros_(self.traj_in_proj.bias)

    def init_weights(self):
        # Match WanModel init, except for explicitly zero-initialized layers.
        excluded = set(self.zero_out)
        if self.traj_in_proj is not None:
            excluded.add(self.traj_in_proj)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in excluded:
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

    @staticmethod
    def _make_zero_linear(dim: int) -> nn.Linear:
        layer = nn.Linear(dim, dim)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    @torch.no_grad()
    def init_from_backbone(self, backbone) -> None:
        """Copy matching weights from a WanModel instance."""
        result = self.load_state_dict(backbone.state_dict(), strict=False)
        if result.missing_keys:
            warnings.warn(
                f"init_from_backbone: {len(result.missing_keys)} ControlNet-only keys "
                f"not copied from backbone (will keep current init): {result.missing_keys}",
                stacklevel=2,
            )
        if result.unexpected_keys:
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
        traj_token_mask: Optional[torch.Tensor] = None,
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
        traj_pad_len = None
        traj_seq_lens_attn = None
        traj_token_mask_attn = None
        if self.traj_in_proj is not None and traj_emb is not None:
            traj_t = self.traj_in_proj(traj_emb.to(dtype=x.dtype, device=x.device))
            traj_t = traj_t + self.traj_type_embed
            # Mask after projection because proj/type-embed bias can affect invalid tokens.
            if traj_token_mask is not None:
                traj_mask = traj_token_mask.to(device=x.device, dtype=traj_t.dtype)
                if traj_mask.dim() == 2:
                    traj_mask = traj_mask[..., None]
                mask_len = traj_mask.shape[1]
                traj_len = traj_t.shape[1]
                if mask_len < traj_len:
                    traj_mask = torch.cat(
                        [
                            traj_mask,
                            traj_mask.new_zeros(traj_mask.shape[0], traj_len - mask_len, 1),
                        ],
                        dim=1,
                    )
                elif mask_len > traj_len:
                    traj_mask = traj_mask[:, :traj_len, :]
                traj_t = traj_t * traj_mask
            batch_size, traj_len, _ = traj_t.shape
            traj_pad_len = max(seq_len, int(traj_len))
            if traj_token_mask is not None:
                traj_token_mask_attn = _prepare_traj_attn_mask(
                    traj_token_mask,
                    batch_size=batch_size,
                    traj_pad_len=traj_pad_len,
                    device=x.device,
                )
            if traj_len < traj_pad_len:
                traj_t = torch.cat(
                    [
                        traj_t,
                        traj_t.new_zeros(batch_size, traj_pad_len - traj_len, traj_t.size(-1)),
                    ],
                    dim=1,
                )
            elif traj_len > traj_pad_len:
                traj_t = traj_t[:, :traj_pad_len, :]
            x = torch.cat([x, traj_t], dim=1)
            if traj_seq_lens is None:
                traj_seq_lens_attn = torch.full_like(seq_lens, int(traj_len))
            else:
                traj_seq_lens_attn = (
                    traj_seq_lens.to(device=device, dtype=torch.long).clamp(
                        min=0, max=traj_pad_len
                    )
                )
            latent_pad_len = seq_len

        # Time embeddings (same as WanModel).
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(-1, seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            batch_size = t.size(0)
            t_flat = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_flat)
                .unflatten(0, (batch_size, seq_len))
                .float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            if latent_pad_len is not None:
                e0_traj = e0.new_zeros(e0.shape[0], traj_pad_len, *e0.shape[2:])
                e0 = torch.cat([e0, e0_traj], dim=1)
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Text context.
        context, context_lens = _embed_text_context(
            self.text_embedding, context, self.text_len, self.dim, device
        )

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            traj_seq_lens=traj_seq_lens_attn,
            traj_token_mask=traj_token_mask_attn,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            latent_pad_len=latent_pad_len,
            traj_pad_len=(
                traj_pad_len if traj_pad_len is not None and traj_pad_len != seq_len else None
            ),
        )

        residuals: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            latent_hidden = x[:, :seq_len, :]
            residuals.append(self.zero_out[i](latent_hidden))
        return residuals
