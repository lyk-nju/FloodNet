"""7D trajectory encoder for WanControlNet trajectory tokens.

Input layout: [x, y, z, cos(yaw), sin(yaw), fwd_delta, yaw_delta].
"""

import torch
import torch.nn as nn

_FRAMES_PER_TOKEN = 4
_IN_DIM = 7

LOCAL_HIDDEN_DIM = 64
LOCAL_OUT_DIM = 128
TRAJ_OUT_DIM = 128


def _masked_mean(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-token mean over the frame axis with a frame-level mask.

    y: (B*T, C, L)
    mask: (B*T, L), where 1 means valid
    returns (B*T, C)
    """
    m = mask.unsqueeze(1).to(y.dtype)         # (B*T, 1, L)
    num = (y * m).sum(dim=-1)                  # (B*T, C)
    den = m.sum(dim=-1).clamp(min=1.0)         # (B*T, 1)
    return num / den


class FrameTrajEncoder(nn.Module):
    """Within-token Conv1d encoder over the 4 frames of a token.

    Input: (B, T_token, 4, 7)
    Output: (B, T_token, 128)
    """

    def __init__(
        self,
        in_dim: int = _IN_DIM,
        hidden_dim: int = LOCAL_HIDDEN_DIM,
        out_dim: int = LOCAL_OUT_DIM,
    ):
        super().__init__()
        if in_dim != _IN_DIM:
            raise ValueError(
                f"FrameTrajEncoder is 7D-only (in_dim must be {_IN_DIM}, got {in_dim})"
            )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.dim() != 4 or x.size(-2) != _FRAMES_PER_TOKEN or x.size(-1) != self.in_dim:
            raise ValueError(
                f"expected (B,T,{_FRAMES_PER_TOKEN},{self.in_dim}), got {tuple(x.shape)}"
            )
        b, t, n_frames, c = x.shape
        if frame_mask is not None and frame_mask.shape != (b, t, n_frames):
            raise ValueError(
                f"frame_mask shape {tuple(frame_mask.shape)} != (B,T,{n_frames}) "
                f"= ({b},{t},{n_frames})"
            )
        # Zero invalid frames before Conv1d so padded values cannot leak through
        # the kernel into neighboring valid frames.
        if frame_mask is not None:
            x = x * frame_mask.to(dtype=x.dtype).unsqueeze(-1)
        # (B,T,L,C) -> (B*T,C,L)
        y = x.reshape(b * t, n_frames, c).transpose(1, 2).contiguous()
        y = self.act(self.conv1(y))
        y = self.act(self.conv2(y))             # (B*T, out_dim, L)
        if frame_mask is None:
            y = y.mean(dim=-1)                   # plain mean
        else:
            mflat = frame_mask.reshape(b * t, n_frames)
            y = _masked_mean(y, mflat)
        return y.reshape(b, t, self.out_dim)


class TokenTrajEncoder(nn.Module):
    """Token-level encoder: LayerNorm + 2-layer MLP, all width = `out_dim`.

    Input: (B, T_token, 128)
    Output: (B, T_token, 128)
    """

    def __init__(self, in_dim: int = LOCAL_OUT_DIM, hidden_dim: int = TRAJ_OUT_DIM,
                 out_dim: int = TRAJ_OUT_DIM):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_dim:
            raise ValueError(
                f"TokenTrajEncoder configured for in_dim={self.in_dim}, got last dim "
                f"{x.size(-1)} (shape {tuple(x.shape)})."
            )
        return self.mlp(self.norm(x))


class TrajectoryEncoder(nn.Module):
    """Full frame-to-token trajectory encoder exposed at the LDF model boundary."""

    def __init__(
        self,
        in_dim: int = _IN_DIM,
        frame_hidden_dim: int = LOCAL_HIDDEN_DIM,
        frame_out_dim: int = LOCAL_OUT_DIM,
        token_hidden_dim: int = TRAJ_OUT_DIM,
        out_dim: int = TRAJ_OUT_DIM,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.frame_in_dim = int(in_dim)
        self.token_in_dim = int(frame_out_dim)
        self.out_dim = int(out_dim)
        self.frame_encoder = FrameTrajEncoder(
            in_dim=in_dim,
            hidden_dim=frame_hidden_dim,
            out_dim=frame_out_dim,
        )
        self.token_encoder = TokenTrajEncoder(
            in_dim=frame_out_dim,
            hidden_dim=token_hidden_dim,
            out_dim=out_dim,
        )

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_encoder(x)

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens = self.frame_encoder(x, frame_mask=frame_mask)
        return self.encode_tokens(tokens)
