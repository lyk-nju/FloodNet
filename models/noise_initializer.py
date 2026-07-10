"""Context-conditioned residual initializer for frontier diffusion noise."""

from __future__ import annotations

import torch
from torch import nn

from models.tools.traj_encoder import TrajectoryEncoder


def _mean_pool_tokens(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.dim() != 3:
        raise ValueError(f"{name} must have shape [B,T,C], got {tuple(value.shape)}")
    if int(value.shape[1]) == 0:
        return value.new_zeros(int(value.shape[0]), int(value.shape[2]))
    return value.mean(dim=1)


def _as_batched_scalar_feature(
    value: torch.Tensor,
    *,
    batch_size: int,
    token_count: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = value.to(device=device, dtype=dtype)
    if tensor.dim() == 1:
        if int(tensor.numel()) != int(token_count):
            raise ValueError(
                f"{name} must have {token_count} entries, got {int(tensor.numel())}"
            )
        tensor = tensor.view(1, token_count).expand(batch_size, token_count)
    elif tensor.dim() == 2:
        if tuple(tensor.shape) != (batch_size, token_count):
            raise ValueError(
                f"{name} must have shape [{batch_size},{token_count}], "
                f"got {tuple(tensor.shape)}"
            )
    else:
        raise ValueError(f"{name} must be [T] or [B,T], got {tuple(tensor.shape)}")
    return tensor


class NoiseInitializer(nn.Module):
    """Predict residual corrections for earliest future frontier z_T tokens.

    The module predicts only ``delta_zT``.  The caller owns applying the residual:

    ``zT_frontier = base_zT + alpha * delta_zT``.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        text_dim: int,
        frontier_tokens: int = 5,
        hidden_dim: int = 512,
        traj_encoder: nn.Module | None = None,
        traj_emb_dim: int = 128,
        freeze_traj_encoder: bool | None = None,
        offset_scale: float = 32.0,
        num_attention_heads: int = 4,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.text_dim = int(text_dim)
        self.frontier_tokens = int(frontier_tokens)
        self.hidden_dim = int(hidden_dim)
        self.traj_emb_dim = int(traj_emb_dim)
        self.offset_scale = float(offset_scale)
        self.num_attention_heads = int(num_attention_heads)

        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.text_dim <= 0:
            raise ValueError("text_dim must be positive")
        if self.frontier_tokens <= 0:
            raise ValueError("frontier_tokens must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_attention_heads <= 0 or self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_attention_heads: "
                f"got hidden_dim={self.hidden_dim}, heads={self.num_attention_heads}"
            )

        owns_traj_encoder = traj_encoder is None
        self.traj_encoder = (
            TrajectoryEncoder(out_dim=self.traj_emb_dim)
            if owns_traj_encoder
            else traj_encoder
        )
        encoder_out_dim = int(getattr(self.traj_encoder, "out_dim", self.traj_emb_dim))
        if encoder_out_dim != self.traj_emb_dim:
            raise ValueError(
                "traj_emb_dim must match traj_encoder.out_dim: "
                f"got traj_emb_dim={self.traj_emb_dim}, encoder out_dim={encoder_out_dim}"
            )
        if freeze_traj_encoder is None:
            freeze_traj_encoder = not owns_traj_encoder
        if bool(freeze_traj_encoder):
            for parameter in self.traj_encoder.parameters():
                parameter.requires_grad_(False)

        self.history_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.active_proj = nn.Sequential(
            nn.LayerNorm(self.latent_dim + 2),
            nn.Linear(self.latent_dim + 2, self.hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.traj_proj = nn.Sequential(
            nn.LayerNorm(self.traj_emb_dim),
            nn.Linear(self.traj_emb_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.frontier_offset_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
        )
        self.frontier_base_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.position_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.memory_type_embedding = nn.Parameter(torch.zeros(4, self.hidden_dim))
        nn.init.normal_(self.memory_type_embedding, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.num_attention_heads,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_head = nn.Linear(self.hidden_dim, self.latent_dim)
        if bool(zero_init_output):
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        *,
        history_latents: torch.Tensor,
        active_latents: torch.Tensor,
        active_beta: torch.Tensor,
        active_offsets: torch.Tensor,
        text_embedding: torch.Tensor,
        traj_token_frames: torch.Tensor,
        frontier_offsets: torch.Tensor,
        frontier_base_zT: torch.Tensor,
        traj_frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``delta_zT`` with shape ``[B, frontier_tokens, latent_dim]``."""

        if history_latents.dim() != 3 or int(history_latents.shape[-1]) != self.latent_dim:
            raise ValueError(
                "history_latents must have shape "
                f"[B,H,{self.latent_dim}], got {tuple(history_latents.shape)}"
            )
        if active_latents.dim() != 3 or int(active_latents.shape[-1]) != self.latent_dim:
            raise ValueError(
                "active_latents must have shape "
                f"[B,A,{self.latent_dim}], got {tuple(active_latents.shape)}"
            )
        if text_embedding.dim() != 2 or int(text_embedding.shape[-1]) != self.text_dim:
            raise ValueError(
                "text_embedding must have shape "
                f"[B,{self.text_dim}], got {tuple(text_embedding.shape)}"
            )
        batch_size = int(history_latents.shape[0])
        active_tokens = int(active_latents.shape[1])
        if int(active_latents.shape[0]) != batch_size or int(text_embedding.shape[0]) != batch_size:
            raise ValueError("history_latents, active_latents, and text_embedding must share batch size")
        if int(traj_token_frames.shape[0]) != batch_size:
            raise ValueError(
                "traj_token_frames must share batch size with history_latents: "
                f"got {traj_token_frames.shape[0]} and {batch_size}"
            )
        if tuple(frontier_base_zT.shape) != (
            batch_size,
            self.frontier_tokens,
            self.latent_dim,
        ):
            raise ValueError(
                "frontier_base_zT must have shape "
                f"[{batch_size},{self.frontier_tokens},{self.latent_dim}], "
                f"got {tuple(frontier_base_zT.shape)}"
            )

        device = history_latents.device
        dtype = history_latents.dtype
        active_beta_b = _as_batched_scalar_feature(
            active_beta,
            batch_size=batch_size,
            token_count=active_tokens,
            device=device,
            dtype=dtype,
            name="active_beta",
        )
        active_offsets_b = _as_batched_scalar_feature(
            active_offsets,
            batch_size=batch_size,
            token_count=active_tokens,
            device=device,
            dtype=dtype,
            name="active_offsets",
        )
        frontier_offsets_t = frontier_offsets.to(device=device, dtype=dtype).reshape(-1)
        if int(frontier_offsets_t.numel()) != self.frontier_tokens:
            raise ValueError(
                "frontier_offsets must contain frontier_tokens entries: "
                f"expected {self.frontier_tokens}, got {int(frontier_offsets_t.numel())}"
            )

        def add_position_and_type(features: torch.Tensor, type_index: int) -> torch.Tensor:
            token_count = int(features.shape[1])
            if token_count == 0:
                return features
            positions = torch.linspace(
                -1.0,
                1.0,
                token_count,
                device=device,
                dtype=dtype,
            ).view(1, token_count, 1)
            return (
                features
                + self.position_proj(positions)
                + self.memory_type_embedding[type_index].to(dtype=dtype).view(1, 1, -1)
            )

        history_feat = add_position_and_type(self.history_proj(history_latents), 0)
        active_input = torch.cat(
            [
                active_latents,
                active_beta_b.unsqueeze(-1),
                (active_offsets_b / self.offset_scale).unsqueeze(-1),
            ],
            dim=-1,
        )
        active_feat = add_position_and_type(self.active_proj(active_input), 1)
        traj_emb = self.traj_encoder(
            traj_token_frames.to(device=device, dtype=dtype),
            frame_mask=(
                None
                if traj_frame_mask is None
                else traj_frame_mask.to(device=device, dtype=dtype)
            ),
        )
        traj_feat = add_position_and_type(self.traj_proj(traj_emb), 2)
        text_feat = (
            self.text_proj(text_embedding.to(device=device, dtype=dtype)).unsqueeze(1)
            + self.memory_type_embedding[3].to(dtype=dtype).view(1, 1, -1)
        )
        memory = torch.cat([history_feat, active_feat, traj_feat, text_feat], dim=1)

        memory_mask = torch.zeros(
            batch_size,
            int(memory.shape[1]),
            device=device,
            dtype=torch.bool,
        )
        if traj_frame_mask is not None:
            traj_valid = traj_frame_mask.to(device=device).reshape(
                batch_size, int(traj_frame_mask.shape[1]), -1
            ).any(dim=-1)
            traj_start = int(history_feat.shape[1]) + int(active_feat.shape[1])
            memory_mask[:, traj_start : traj_start + int(traj_feat.shape[1])] = ~traj_valid

        frontier_feat = self.frontier_offset_proj(
            (frontier_offsets_t / self.offset_scale).view(1, self.frontier_tokens, 1)
        )
        query = self.frontier_base_proj(
            frontier_base_zT.to(device=device, dtype=dtype)
        ) + frontier_feat
        attended, _ = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=memory_mask,
            need_weights=False,
        )
        token_context = query + attended
        token_context = token_context + self.fusion(token_context)
        return self.output_head(self.output_norm(token_context))


__all__ = ["NoiseInitializer"]
