import torch
import torch.nn as nn


class FrameTrajEncoder(nn.Module):
    """Downsample trajectory frames into motion-token aligned features."""

    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 64,
        out_dim: int = 128,
    ):
        super().__init__()
        if in_dim != 7:
            raise ValueError(f"FrameTrajEncoder is 7D-only (in_dim must be 7, got {in_dim})")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.frames_per_token = 4
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Check input shape.
        if (
            x.dim() != 4
            or x.size(-2) != self.frames_per_token
            or x.size(-1) != self.in_dim
        ):
            raise ValueError(
                f"expected (B,T,{self.frames_per_token},{self.in_dim}), "
                f"got {tuple(x.shape)}"
            )
        batch_size, num_tokens, num_frames, channels = x.shape
        expected_mask_shape = (batch_size, num_tokens, num_frames)
        if frame_mask is not None and frame_mask.shape != expected_mask_shape:
            raise ValueError(
                f"frame_mask shape {tuple(frame_mask.shape)} != (B,T,{num_frames}) "
                f"= ({batch_size},{num_tokens},{num_frames})"
            )
        if frame_mask is not None:
            x = x * frame_mask.to(dtype=x.dtype).unsqueeze(-1)

        # Flatten token blocks for within-token Conv1d.
        features = (
            x.reshape(batch_size * num_tokens, num_frames, channels)
            .transpose(1, 2)
            .contiguous()
        )
        features = self.conv1(features)
        features = self.act(features)

        features = self.conv2(features)
        features = self.act(features)

        # Pool frames back to one trajectory token.
        if frame_mask is None:
            features = features.mean(dim=-1)
        else:
            flat_frame_mask = frame_mask.reshape(batch_size * num_tokens, num_frames)
            feature_mask = flat_frame_mask.unsqueeze(1).to(features.dtype)
            numerator = (features * feature_mask).sum(dim=-1)
            denominator = feature_mask.sum(dim=-1).clamp(min=1.0)
            features = numerator / denominator
        return features.reshape(batch_size, num_tokens, self.out_dim)


class TokenTrajEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 128,
        out_dim: int = 128,
    ):
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
    def __init__(
        self,
        in_dim: int = 7,
        frame_hidden_dim: int = 64,
        frame_out_dim: int = 128,
        token_hidden_dim: int = 128,
        out_dim: int = 128,
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

    def encode_traj_token(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_encoder(x)

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens = self.frame_encoder(x, frame_mask=frame_mask)
        return self.encode_traj_token(tokens)
