"""7D trajectory encoder for FlexTraj-style traj tokens (feeds WanControlNet.traj_in_proj).

7D input layout: [x, y, z, cos(yaw), sin(yaw), fwd_delta, yaw_delta] — physical
root yaw + frame-to-frame deltas. The 4D legacy encoder was rewritten for the
7D fine-tune; legacy ckpts have their traj-encoder / traj_in_proj weights
stripped in `utils.training.ckpt_compat.strip_legacy_traj_encoder_weights`.
"""

import torch
import torch.nn as nn

# Causal-VAE frames-per-token (token k>=1 spans 4 frames; token 0 padded to 4).
_FRAMES_PER_TOKEN = 4

# Hard-coded 7D contract — the layout this encoder is built for.
_IN_DIM = 7

# Default hidden / output widths for the 7D encoder. Kept as module-level
# constants so external code (config validation, ckpt-compat tests) can refer
# to them without import-time gymnastics.
LOCAL_HIDDEN_DIM = 64
LOCAL_OUT_DIM = 128
TRAJ_OUT_DIM = 128


def _masked_mean(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-token mean over the frame axis with a frame-level mask.

    y:    (B*T, C, L)
    mask: (B*T, L)  in {0, 1}
    returns (B*T, C)

    Tokens with zero valid frames return all-zeros (caller is expected to
    multiply by a token-level mask anyway).
    """
    m = mask.unsqueeze(1).to(y.dtype)         # (B*T, 1, L)
    num = (y * m).sum(dim=-1)                  # (B*T, C)
    den = m.sum(dim=-1).clamp(min=1.0)         # (B*T, 1)
    return num / den


class LocalTrajEncoder(nn.Module):
    """Within-token Conv1d encoder over the 4 frames of a token.

    Conv1d 7→64, k=3, padding=1 → GELU → Conv1d 64→128, k=3, padding=1 → GELU
    → masked mean pool over the 4 frames.

    Input:  (B, T_token, 4, 7)
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
                f"LocalTrajEncoder is 7D-only (in_dim must be {_IN_DIM}, got {in_dim})"
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
        # Defense-in-depth: zero invalid frames BEFORE the conv. Otherwise any
        # caller that passes frame_mask without pre-zeroing the frames would
        # leak invalid-frame values into neighbors via the kernel-size-3 conv,
        # and only the masked-mean pool below would see the mask.
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


class TrajEncoder(nn.Module):
    """Token-level encoder: LayerNorm + 2-layer MLP, all width = `out_dim`.

    Input:  (B, T_token, 128)  — output of LocalTrajEncoder
    Output: (B, T_token, 128)  — fed into WanControlNet.traj_in_proj
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
                f"TrajEncoder configured for in_dim={self.in_dim}, got last dim "
                f"{x.size(-1)} (shape {tuple(x.shape)})."
            )
        return self.mlp(self.norm(x))
