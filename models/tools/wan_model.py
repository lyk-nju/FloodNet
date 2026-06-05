# This module uses modified code from Alibaba Wan Team
# Original source: https://github.com/Wan-Video/Wan2.2
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified to support stream mode for cross-attention.
# Added causal attention for self-attention (1d case)
# Added context length corrrection.

import math
import os

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention, flextraj_self_attention


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def _embed_text_context(text_embedding, context, text_len, dim, device):
    """Pad/embed text contexts while deduplicating repeated tensor references."""
    dedup_enabled = os.environ.get("FLOODNET_TEXT_CONTEXT_DEDUP", "1").lower()
    if dedup_enabled in {"0", "false", "no", "off"}:
        context_lens = torch.tensor(
            [u.size(0) for u in context], dtype=torch.long, device=device
        )
        if len(context) == 0:
            return (
                torch.empty(0, text_len, dim, device=device, dtype=torch.float32),
                context_lens,
            )
        return (
            text_embedding(
                torch.stack(
                    [
                        torch.cat(
                            [u, u.new_zeros(text_len - u.size(0), u.size(1))]
                        )
                        for u in context
                    ]
                )
            ),
            context_lens,
        )

    context_lens = torch.tensor(
        [min(u.size(0), text_len) for u in context], dtype=torch.long, device=device
    )
    if len(context) == 0:
        return (
            torch.empty(0, text_len, dim, device=device, dtype=torch.float32),
            context_lens,
        )

    unique_map = {}
    unique_list = []
    ctx_indices = []
    for u in context:
        key = (
            u.data_ptr(),
            tuple(u.size()),
            tuple(u.stride()),
            u.storage_offset(),
            u.dtype,
            u.device,
        )
        idx = unique_map.get(key)
        if idx is None:
            idx = len(unique_list)
            unique_map[key] = idx
            unique_list.append(u)
        ctx_indices.append(idx)

    unique_stacked = torch.stack(
        [
            torch.cat(
                [
                    u[:text_len],
                    u.new_zeros(text_len - min(u.size(0), text_len), u.size(1)),
                ]
            )
            for u in unique_list
        ]
    )
    unique_embedded = text_embedding(unique_stacked)
    if len(unique_list) == len(context):
        return unique_embedded, context_lens

    idx_t = torch.tensor(ctx_indices, device=unique_embedded.device, dtype=torch.long)
    return unique_embedded[idx_t], context_lens


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_concat_latent_traj(x, grid_sizes, freqs, latent_pad_len, traj_pad_len=None):
    """
    RoPE for sequences [latent_0..L || traj_0..T].

    In the standard training path T == L. Streaming can pass T > L so latent
    tokens attend to a future trajectory horizon. For the body model's 1D token
    grid (H=W=1), future traj tokens receive the next temporal RoPE positions.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    L = latent_pad_len
    T = L if traj_pad_len is None else int(traj_pad_len)
    assert x.size(1) == L + T
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        grid_len = f * h * w

        def freqs_for_grid(f_len: int) -> torch.Tensor:
            return torch.cat(
                [
                    freqs_split[0][:f_len].view(f_len, 1, 1, -1).expand(f_len, h, w, -1),
                    freqs_split[1][:h].view(1, h, 1, -1).expand(f_len, h, w, -1),
                    freqs_split[2][:w].view(1, 1, w, -1).expand(f_len, h, w, -1),
                ],
                dim=-1,
            ).reshape(f_len * h * w, 1, -1)

        freqs_i = freqs_for_grid(f)

        def apply_rope_prefix(x_prefix: torch.Tensor, freqs_prefix: torch.Tensor) -> torch.Tensor:
            # rotate first grid_len tokens; leave tail of prefix unchanged
            prefix_len = min(x_prefix.shape[0], freqs_prefix.shape[0])
            if prefix_len == 0:
                return x_prefix
            x_rot = torch.view_as_complex(
                x_prefix[:prefix_len].to(torch.float64).reshape(prefix_len, n, -1, 2)
            )
            x_rot = torch.view_as_real(x_rot * freqs_prefix[:prefix_len]).flatten(2)
            return torch.cat([x_rot, x_prefix[prefix_len:]], dim=0)

        x_lat = apply_rope_prefix(x[i, :L], freqs_i)
        traj_freqs = freqs_for_grid(T) if h == 1 and w == 1 else freqs_i
        x_traj = apply_rope_prefix(x[i, L : L + T], traj_freqs)
        output.append(torch.cat([x_lat, x_traj], dim=0))
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        causal=False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.causal = causal
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        latent_pad_len=None,
        traj_pad_len=None,
        traj_seq_lens=None,
        traj_token_mask=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            latent_pad_len: if set, x is [latent || traj] with
                L = latent_pad_len + traj_pad_len; traj segment
                starts at this index. Self-attn 使用 Task4 块稀疏（见 ``flextraj_self_attention``）。
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x_in):
            q = self.norm_q(self.q(x_in)).view(b, s, n, d)
            k = self.norm_k(self.k(x_in)).view(b, s, n, d)
            v = self.v(x_in).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if latent_pad_len is not None:
            traj_pad_len = latent_pad_len if traj_pad_len is None else int(traj_pad_len)
            assert s == latent_pad_len + traj_pad_len
            q = rope_apply_concat_latent_traj(
                q, grid_sizes, freqs, latent_pad_len, traj_pad_len
            )
            k = rope_apply_concat_latent_traj(
                k, grid_sizes, freqs, latent_pad_len, traj_pad_len
            )
        else:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)

        if latent_pad_len is not None:
            if self.window_size != (-1, -1):
                raise NotImplementedError(
                    "FlexTraj (latent||traj) with sliding-window self-attention is not supported."
                )
            x = flextraj_self_attention(
                q,
                k,
                v,
                seq_lens=seq_lens,
                n_latent=latent_pad_len,
                latent_lens=seq_lens,
                traj_lens=traj_seq_lens,
                traj_token_mask=traj_token_mask,
                causal=self.causal,
            )
        else:
            x = flash_attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
                causal=self.causal,
            )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args non-stream mode:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        Args stream mode (frame-aligned, plain):
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [BxL1, L2, C]
            context_lens(Tensor): Shape [BxL1]
        Args stream mode (frame-aligned, FlexTraj):
            x(Tensor): Shape [B, 2*L1, C]  — [latent || traj]
            context(Tensor): Shape [BxL1, L2, C]
            context_lens(Tensor): Shape [BxL1]
        """
        out_sizes = x.size()
        bq = x.size(0)
        b_ctx = context.size(0)
        n, d = self.num_heads, self.head_dim

        k = self.norm_k(self.k(context)).view(b_ctx, -1, n, d)
        v = self.v(context).view(b_ctx, -1, n, d)

        if b_ctx == bq:
            # Standard: one context per sample (no frame-alignment)
            q = self.norm_q(self.q(x)).view(bq, -1, n, d)
            x = flash_attention(q, k, v, k_lens=context_lens)
            x = x.flatten(2).view(*out_sizes)
        else:
            # Frame-aligned: b_ctx = bq * L_lat
            Lq = x.size(1)
            L_lat = b_ctx // bq
            q_all = self.norm_q(self.q(x)).view(bq, Lq, n, d)  # (bq, Lq, n, d)

            if Lq == L_lat:
                # Plain frame-aligned: each token has its own context
                q = q_all.reshape(b_ctx, 1, n, d)
                x = flash_attention(q, k, v, k_lens=context_lens)
                x = x.flatten(2).view(*out_sizes)
            else:
                # FlexTraj: x = [lat_0..L_lat ‖ traj_0..L_lat], Lq = 2*L_lat
                # Each latent token k and its paired traj token k attend to context[b*L_lat+k]
                q_lat = q_all[:, :L_lat].reshape(b_ctx, 1, n, d)
                q_traj = q_all[:, L_lat:].reshape(b_ctx, 1, n, d)
                out_lat = flash_attention(q_lat, k, v, k_lens=context_lens)
                out_traj = flash_attention(q_traj, k, v, k_lens=context_lens)
                out_lat = out_lat.flatten(2).view(bq, L_lat, -1)
                out_traj = out_traj.flatten(2).view(bq, L_lat, -1)
                x = torch.cat([out_lat, out_traj], dim=1)  # (bq, 2*L_lat, C)

        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        causal=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.causal = causal
        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps, causal
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        traj_seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        latent_pad_len=None,
        traj_pad_len=None,
        traj_token_mask=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            latent_pad_len: optional; when set, L = latent_pad_len + traj_pad_len
                (latent || traj).
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
            latent_pad_len=latent_pad_len,
            traj_pad_len=traj_pad_len,
            traj_seq_lens=traj_seq_lens,
            traj_token_mask=traj_token_mask,
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # FlexTraj论文语义：condition/trajectory tokens不应该通过 cross-attn
            # 去查询文本tokens（只允许噪声/主干去查询）。
            cross_out = self.cross_attn(self.norm3(x), context, context_lens)
            if latent_pad_len is not None:
                # x = [latent_tokens || traj_tokens], 仅更新latent部分
                cross_out[:, latent_pad_len:, :] = 0
            x = x + cross_out
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
            )
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        causal=False,
        traj_enc_dim=0,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            traj_enc_dim (`int`, *optional*, defaults to 0):
                Trajectory encoder output dim; 0 disables FlexTraj token concat.
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
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
        # embeddings
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

        # blocks
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

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        if traj_enc_dim > 0:
            self.traj_in_proj = nn.Linear(traj_enc_dim, dim)
            self.traj_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.traj_in_proj = None
            self.traj_type_embed = None

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
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

        # initialize weights
        self.init_weights()
        if self.traj_in_proj is not None:
            nn.init.zeros_(self.traj_in_proj.weight)
            nn.init.zeros_(self.traj_in_proj.bias)

        # T_B_02: history-corruption support fields (consumed by T_B_03).
        # Added AFTER init_weights() so the generic initializer doesn't touch them.
        #   mask_emb: learned replacement vector for corrupted history tokens,
        #             in the in_dim (VAE latent) space.
        #   z_mean / z_std: VAE latent per-channel stats, loaded via load_z_stats.
        # All three are persistent (in state_dict) per Done criteria; old ckpts
        # that lack them are handled by DiffForcingWanModel.load_state_dict's
        # backward-compat back-fill.
        self.mask_emb = nn.Parameter(torch.randn(self.in_dim) * 0.02)
        self.register_buffer("z_mean", torch.zeros(self.in_dim))
        self.register_buffer("z_std", torch.ones(self.in_dim))

    def load_z_stats(self, stats_dir: str) -> None:
        """Load VAE latent stats (deps/body_stats/{z_mean,z_std}.npy) into buffers."""
        import os

        import numpy as np

        z_mean = np.load(os.path.join(stats_dir, "z_mean.npy"))
        z_std = np.load(os.path.join(stats_dir, "z_std.npy"))
        self.z_mean.copy_(torch.as_tensor(z_mean, dtype=self.z_mean.dtype))
        self.z_std.copy_(torch.as_tensor(z_std, dtype=self.z_std.dtype))

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        traj_emb=None,
        traj_seq_lens=None,
        traj_token_mask=None,
        controlnet_residuals=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            traj_emb (`Tensor`, *optional*):
                Trajectory encoder output before `traj_in_proj`, shape (B, T, traj_enc_dim).
                When set, concatenated as a second half of the self-attention sequence (FlexTraj).

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
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
            traj_t = self.traj_in_proj(traj_emb.to(dtype=x.dtype))
            traj_t = traj_t + self.traj_type_embed
            bt, tlen, _ = traj_t.shape
            traj_pad_len = max(seq_len, int(tlen))
            if traj_token_mask is not None:
                traj_token_mask_attn = traj_token_mask.to(
                    device=x.device, dtype=torch.bool
                )
                if (
                    traj_token_mask_attn.dim() == 3
                    and traj_token_mask_attn.shape[-1] == 1
                ):
                    traj_token_mask_attn = traj_token_mask_attn[..., 0]
                if traj_token_mask_attn.shape[1] < traj_pad_len:
                    traj_token_mask_attn = torch.cat(
                        [
                            traj_token_mask_attn,
                            traj_token_mask_attn.new_zeros(
                                traj_token_mask_attn.shape[0],
                                traj_pad_len - traj_token_mask_attn.shape[1],
                            ),
                        ],
                        dim=1,
                    )
                elif traj_token_mask_attn.shape[1] > traj_pad_len:
                    traj_token_mask_attn = traj_token_mask_attn[:, :traj_pad_len]
            if tlen < traj_pad_len:
                traj_t = torch.cat(
                    [traj_t, traj_t.new_zeros(bt, traj_pad_len - tlen, traj_t.size(-1))],
                    dim=1,
                )
            x = torch.cat([x, traj_t], dim=1)
            if traj_seq_lens is None:
                traj_seq_lens_attn = torch.full_like(seq_lens, int(tlen))
            else:
                traj_seq_lens_attn = (
                    traj_seq_lens.to(device=seq_lens.device, dtype=torch.long)
                    .clamp(min=0, max=traj_pad_len)
                )
            latent_pad_len = seq_len

        # time embeddings
        if t.dim() == 1:  # per-sample scalar → (B, seq_len)
            t = t.unsqueeze(1).expand(-1, seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t)
                .unflatten(0, (bt, seq_len))
                .float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            if latent_pad_len is not None:
                # FlexTraj: latent tokens depend on diffusion timestep (time modulation),
                # while traj tokens are treated as known control and should not be
                # time-modulated by the diffusion step.
                e0_traj = e0.new_zeros(e0.shape[0], traj_pad_len, *e0.shape[2:])
                e0 = torch.cat([e0, e0_traj], dim=1)
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context, context_lens = _embed_text_context(
            self.text_embedding, context, self.text_len, self.dim, device
        )

        # arguments
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
            traj_pad_len=traj_pad_len,
        )

        if controlnet_residuals is not None and len(controlnet_residuals) != len(
            self.blocks
        ):
            raise ValueError(
                f"controlnet_residuals length {len(controlnet_residuals)} "
                f"!= num_layers {len(self.blocks)}"
            )

        for i, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            if controlnet_residuals is not None:
                r = controlnet_residuals[i].to(dtype=x.dtype, device=x.device)
                if r.dim() != 3 or r.size(0) != x.size(0) or r.size(1) != seq_len:
                    raise ValueError(
                        "controlnet_residuals[i] must have shape (B, seq_len, dim); "
                        f"got {tuple(r.shape)} with seq_len={seq_len}"
                    )
                # Only apply residuals to latent tokens (first seq_len tokens).
                x[:, :seq_len, :] = x[:, :seq_len, :] + r

        # head (latent tokens only)
        if latent_pad_len is not None:
            x = self.head(x[:, :seq_len], e)
        else:
            x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
