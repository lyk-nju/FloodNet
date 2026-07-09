"""Build typed inputs for ``models.noise_initializer.NoiseInitializer``."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch

from utils.inference.latent_state_view import StreamLatentStateView
from utils.traj_batch import FRAMES_PER_TOKEN_DEFAULT, frames_to_tokens_range


@dataclass(frozen=True)
class NoiseInitializerContext:
    """Inputs needed to predict a residual for frontier z_T tokens."""

    history_latents: torch.Tensor
    active_latents: torch.Tensor
    active_beta: torch.Tensor
    active_offsets: torch.Tensor
    text_embedding: torch.Tensor
    traj_token_frames: torch.Tensor
    frontier_offsets: torch.Tensor
    frontier_base_zT: torch.Tensor
    frontier_ids: torch.Tensor
    traj_frame_mask: torch.Tensor | None = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor | None]:
        """Return exactly the keyword arguments accepted by ``NoiseInitializer``."""

        return {
            "history_latents": self.history_latents,
            "active_latents": self.active_latents,
            "active_beta": self.active_beta,
            "active_offsets": self.active_offsets,
            "text_embedding": self.text_embedding,
            "traj_token_frames": self.traj_token_frames,
            "traj_frame_mask": self.traj_frame_mask,
            "frontier_offsets": self.frontier_offsets,
        }


def _validate_batched(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if int(tensor.shape[0]) != int(batch_size):
        raise ValueError(
            f"{name} must have batch size {batch_size}, got {int(tensor.shape[0])}"
        )


def _select_history(latents: torch.Tensor, history_tokens: int | None) -> torch.Tensor:
    if history_tokens is None:
        return latents
    count = int(history_tokens)
    if count < 0:
        raise ValueError(f"history_tokens must be >= 0, got {history_tokens}")
    if count == 0:
        return latents[:, :0]
    return latents[:, -count:]


def _default_traj_tokens(num_frames: int, frames_per_token: int) -> int:
    if num_frames <= 0:
        return 0
    return 1 + int(ceil(float(max(0, num_frames - 1)) / float(frames_per_token)))


def build_noise_initializer_context(
    view: StreamLatentStateView,
    *,
    text_embedding: torch.Tensor,
    traj_local_frames: torch.Tensor,
    traj_frame_mask: torch.Tensor | None = None,
    history_tokens: int | None = None,
    frontier_tokens: int | None = None,
    traj_start_token: int | torch.Tensor = 0,
    traj_tokens: int | None = None,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> NoiseInitializerContext:
    """Build the model-facing context for a frontier z_T residual prediction.

    ``traj_local_frames`` is expected to be window-local 7D trajectory condition
    anchored consistently with the active-window payload builder.  This helper
    only tokenizes and validates it; it does not re-anchor trajectory data.
    """

    if text_embedding.dim() != 2:
        raise ValueError(
            f"text_embedding must have shape [B,C], got {tuple(text_embedding.shape)}"
        )
    if traj_local_frames.dim() != 3 or int(traj_local_frames.shape[-1]) != 7:
        raise ValueError(
            "traj_local_frames must have shape [B,F,7], "
            f"got {tuple(traj_local_frames.shape)}"
        )

    batch_size = int(text_embedding.shape[0])
    _validate_batched("traj_local_frames", traj_local_frames, batch_size)
    _validate_batched("committed_latents", view.committed_latents, batch_size)
    _validate_batched("active_latents", view.active_latents, batch_size)
    _validate_batched("frontier_base_zT", view.frontier_base_zT, batch_size)

    frames_per_token = int(frames_per_token)
    if frames_per_token <= 0:
        raise ValueError(f"frames_per_token must be positive, got {frames_per_token}")

    requested_frontier = (
        int(view.frontier_ids.numel()) if frontier_tokens is None else int(frontier_tokens)
    )
    if requested_frontier <= 0:
        raise ValueError(f"frontier_tokens must be positive, got {requested_frontier}")
    if int(view.frontier_ids.numel()) < requested_frontier:
        raise ValueError(
            "not enough frontier tokens in StreamLatentStateView: "
            f"requested {requested_frontier}, available {int(view.frontier_ids.numel())}"
        )

    token_count = (
        _default_traj_tokens(int(traj_local_frames.shape[1]), frames_per_token)
        if traj_tokens is None
        else int(traj_tokens)
    )
    if token_count <= 0:
        raise ValueError(f"traj_tokens must be positive, got {token_count}")

    token_frames = frames_to_tokens_range(
        traj_local_frames,
        traj_start_token,
        token_count,
        frames_per_token=frames_per_token,
    )

    token_frame_mask = None
    if traj_frame_mask is not None:
        if traj_frame_mask.dim() != 2:
            raise ValueError(
                f"traj_frame_mask must have shape [B,F], got {tuple(traj_frame_mask.shape)}"
            )
        _validate_batched("traj_frame_mask", traj_frame_mask, batch_size)
        token_frame_mask = frames_to_tokens_range(
            traj_frame_mask.unsqueeze(-1).to(dtype=traj_local_frames.dtype),
            traj_start_token,
            token_count,
            frames_per_token=frames_per_token,
        ).squeeze(-1)

    return NoiseInitializerContext(
        history_latents=_select_history(view.committed_latents, history_tokens),
        active_latents=view.active_latents,
        active_beta=view.active_beta,
        active_offsets=view.active_offsets,
        text_embedding=text_embedding,
        traj_token_frames=token_frames,
        traj_frame_mask=token_frame_mask,
        frontier_offsets=view.frontier_offsets[:requested_frontier],
        frontier_base_zT=view.frontier_base_zT[:, :requested_frontier],
        frontier_ids=view.frontier_ids[:requested_frontier],
    )


__all__ = ["NoiseInitializerContext", "build_noise_initializer_context"]
