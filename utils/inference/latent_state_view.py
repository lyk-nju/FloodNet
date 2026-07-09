"""Interpret the LDF streaming latent cache by diffusion schedule state.

``DiffForcingWanModel.generated`` stores committed history, active noisy state,
and future initial noise in one tensor.  These helpers provide a small typed
view over that cache without introducing a separate runtime buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def triangular_token_beta(
    token_indices: torch.Tensor,
    *,
    current_step: int,
    dt: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute the triangular schedule beta for token indices."""

    if not torch.is_tensor(token_indices):
        token_indices = torch.as_tensor(token_indices)
    current_time = float(current_step) * float(dt)
    chunk = float(max(1, int(chunk_size)))
    return torch.clamp(
        1.0 + token_indices.to(dtype=torch.float32) / chunk - current_time,
        min=0.0,
        max=1.0,
    )


def _extract_latents(generated_preprocessed: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    if token_ids.numel() == 0:
        return generated_preprocessed.new_empty(
            int(generated_preprocessed.shape[0]),
            0,
            int(generated_preprocessed.shape[1]),
        )
    return (
        generated_preprocessed[:, :, token_ids.to(device=generated_preprocessed.device), 0, 0]
        .permute(0, 2, 1)
        .contiguous()
    )


@dataclass(frozen=True)
class StreamLatentStateView:
    """Read-only split of a stream latent cache.

    Shapes:
      - latents are ``[B, N, C]``
      - ids / beta / offsets are one-dimensional token arrays
    """

    committed_ids: torch.Tensor
    active_ids: torch.Tensor
    frontier_ids: torch.Tensor
    committed_latents: torch.Tensor
    active_latents: torch.Tensor
    frontier_base_zT: torch.Tensor
    active_beta: torch.Tensor
    frontier_beta: torch.Tensor
    active_offsets: torch.Tensor
    frontier_offsets: torch.Tensor
    commit_index: int
    current_step: int

    @classmethod
    def from_model(
        cls,
        model,
        *,
        beta_threshold: float = 0.999,
        frontier_tokens: int = 5,
        token_update_count: torch.Tensor | None = None,
        require_zero_update_count: bool = False,
        include_commit_token_in_frontier: bool = False,
    ) -> "StreamLatentStateView":
        generated = getattr(model, "generated")
        num_steps = int(getattr(model, "num_denoise_steps", 1))
        dt = float(getattr(model, "dt", 1.0 / max(1, num_steps)))
        return cls.from_generated(
            generated,
            commit_index=int(getattr(model, "commit_index", 0)),
            current_step=int(getattr(model, "current_step", 0)),
            dt=dt,
            chunk_size=int(getattr(model, "chunk_size", 1)),
            beta_threshold=float(beta_threshold),
            frontier_tokens=int(frontier_tokens),
            token_update_count=token_update_count,
            require_zero_update_count=bool(require_zero_update_count),
            include_commit_token_in_frontier=bool(include_commit_token_in_frontier),
        )

    @classmethod
    def from_generated(
        cls,
        generated_preprocessed: torch.Tensor,
        *,
        commit_index: int,
        current_step: int,
        dt: float,
        chunk_size: int,
        beta_threshold: float = 0.999,
        frontier_tokens: int = 5,
        token_update_count: torch.Tensor | None = None,
        require_zero_update_count: bool = False,
        include_commit_token_in_frontier: bool = False,
    ) -> "StreamLatentStateView":
        if generated_preprocessed.dim() != 5:
            raise ValueError(
                "generated_preprocessed must have shape [B,C,T,1,1]; "
                f"got {tuple(generated_preprocessed.shape)}"
            )
        if int(generated_preprocessed.shape[3]) != 1 or int(generated_preprocessed.shape[4]) != 1:
            raise ValueError(
                "generated_preprocessed must have singleton spatial dims; "
                f"got {tuple(generated_preprocessed.shape)}"
            )

        device = generated_preprocessed.device
        num_tokens = int(generated_preprocessed.shape[2])
        commit = max(0, min(int(commit_index), num_tokens))
        all_ids = torch.arange(num_tokens, device=device, dtype=torch.long)
        beta = triangular_token_beta(
            all_ids,
            current_step=int(current_step),
            dt=float(dt),
            chunk_size=int(chunk_size),
        ).to(device=device)

        committed_ids = all_ids[all_ids < commit]
        frontier_start = commit if bool(include_commit_token_in_frontier) else commit + 1
        frontier_mask = (all_ids >= frontier_start) & (beta >= float(beta_threshold))

        if require_zero_update_count:
            if token_update_count is None:
                raise ValueError(
                    "token_update_count is required when require_zero_update_count=True"
                )
            counts = token_update_count.to(device=device).reshape(-1)
            if int(counts.numel()) != num_tokens:
                raise ValueError(
                    "token_update_count must have one entry per generated token: "
                    f"expected {num_tokens}, got {int(counts.numel())}"
                )
            frontier_mask = frontier_mask & (counts == 0)

        frontier_ids = all_ids[frontier_mask][: max(0, int(frontier_tokens))]
        non_committed_ids = all_ids[all_ids >= commit]
        if frontier_ids.numel() > 0:
            first_frontier = int(frontier_ids[0].item())
            active_ids = non_committed_ids[non_committed_ids < first_frontier]
        else:
            active_ids = non_committed_ids[beta[non_committed_ids] < float(beta_threshold)]

        return cls(
            committed_ids=committed_ids.detach(),
            active_ids=active_ids.detach(),
            frontier_ids=frontier_ids.detach(),
            committed_latents=_extract_latents(generated_preprocessed, committed_ids).detach(),
            active_latents=_extract_latents(generated_preprocessed, active_ids).detach(),
            frontier_base_zT=_extract_latents(generated_preprocessed, frontier_ids).detach(),
            active_beta=beta[active_ids].detach(),
            frontier_beta=beta[frontier_ids].detach(),
            active_offsets=(active_ids - int(commit_index)).detach(),
            frontier_offsets=(frontier_ids - int(commit_index)).detach(),
            commit_index=int(commit_index),
            current_step=int(current_step),
        )


def inject_frontier_zT(
    generated_preprocessed: torch.Tensor,
    view: StreamLatentStateView,
    frontier_zT: torch.Tensor,
) -> torch.Tensor:
    """Return a copy of ``generated_preprocessed`` with only frontier ids replaced."""

    if generated_preprocessed.dim() != 5:
        raise ValueError(
            "generated_preprocessed must have shape [B,C,T,1,1]; "
            f"got {tuple(generated_preprocessed.shape)}"
        )
    frontier_ids = view.frontier_ids.to(device=generated_preprocessed.device, dtype=torch.long)
    expected = (
        int(generated_preprocessed.shape[0]),
        int(frontier_ids.numel()),
        int(generated_preprocessed.shape[1]),
    )
    if tuple(frontier_zT.shape) != expected:
        raise ValueError(
            "frontier_zT must have shape [B,M,C] matching view.frontier_ids; "
            f"expected {expected}, got {tuple(frontier_zT.shape)}"
        )

    out = generated_preprocessed.clone()
    if frontier_ids.numel() > 0:
        out[:, :, frontier_ids, 0, 0] = frontier_zT.permute(0, 2, 1)
    return out


__all__ = [
    "StreamLatentStateView",
    "inject_frontier_zT",
    "triangular_token_beta",
]
