# This file previously contained a fork of Wan with alternate cross-RoPE / spatial_dim layouts.
# The project now maintains a single backbone in wan_model.py (FlexTraj traj tokens, shared RoPE, etc.).
# Re-export for any legacy ``from .wan_model_cross_rope import WanModel`` imports.

from .wan_model import (
    Head,
    WanAttentionBlock,
    WanCrossAttention,
    WanLayerNorm,
    WanModel,
    WanRMSNorm,
    WanSelfAttention,
    rope_apply,
    rope_apply_concat_latent_traj,
    rope_params,
    sinusoidal_embedding_1d,
)

__all__ = [
    "Head",
    "WanAttentionBlock",
    "WanCrossAttention",
    "WanLayerNorm",
    "WanModel",
    "WanRMSNorm",
    "WanSelfAttention",
    "rope_apply",
    "rope_apply_concat_latent_traj",
    "rope_params",
    "sinusoidal_embedding_1d",
]
