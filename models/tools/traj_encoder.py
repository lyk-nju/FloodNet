"""Lightweight trajectory encoder for FlexTraj-style traj tokens (feeds WanModel.traj_in_proj).

`in_dim` is flag-gated (config `traj_encoder_in_dim`, default 4):
  - 4D legacy: [x, z, cos(path_heading), sin(path_heading)]
  - 7D (T_B_07): [x, y, z, cos(yaw), sin(yaw), fwd_delta, yaw_delta] (physical
    root yaw). The 4D path stays fully working until the flag is flipped to 7.
"""

import torch
import torch.nn as nn

# Causal-VAE frames-per-token (token k>=1 spans 4 frames; token 0 padded to 4).
_FRAMES_PER_TOKEN = 4


class LocalTrajEncoder(nn.Module):
    """Within-token local encoder over the 4 frames of a token.

    Input:  (B, T_token, 4, in_dim)  — 4 frames x in_dim feats per token
    Output: (B, T_token, in_dim)     — pooled token-level in_dim feats
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.in_dim = in_dim
        # Per-frame features are channels C=in_dim over length L=4; pool over L.
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(-2) != _FRAMES_PER_TOKEN or x.size(-1) != self.in_dim:
            raise ValueError(
                f"expected (B,T,{_FRAMES_PER_TOKEN},{self.in_dim}), got {tuple(x.shape)}"
            )
        b, t, n_frames, c = x.shape
        # (B,T,L,C) -> (B*T,C,L)
        y = x.reshape(b * t, n_frames, c).transpose(1, 2).contiguous()
        y = self.net(y)  # (B*T,in_dim,L)
        y = y.mean(dim=-1)  # (B*T,in_dim)
        return y.reshape(b, t, self.in_dim)


class TrajEncoder(nn.Module):
    """Token-aligned trajectory encoder: maps trajectory features to traj_enc_dim.

    Input:  (B, T, in_dim)  — token-level trajectory features (in_dim 4 or 7)
    Output: (B, T, out_dim) — trajectory embeddings fed to WanModel / WanControlNet
    """

    def __init__(self, in_dim=4, hidden_dim=64, out_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        if x.size(-1) != self.in_dim:
            raise ValueError(
                f"TrajEncoder configured for in_dim={self.in_dim}, got last dim "
                f"{x.size(-1)} (shape {tuple(x.shape)}); 4D legacy input is rejected "
                f"when the encoder is built for 7D."
            )
        return self.mlp(x)
