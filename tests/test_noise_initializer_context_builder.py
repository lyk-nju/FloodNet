from __future__ import annotations

import pytest
import torch

from utils.inference.latent_state_view import StreamLatentStateView
from utils.training.noise_initializer.context_builder import (
    NoiseInitializerContext,
    build_noise_initializer_context,
)


def _state_view(batch_size: int = 2) -> StreamLatentStateView:
    generated = torch.arange(batch_size * 3 * 12, dtype=torch.float32).view(
        batch_size, 3, 12, 1, 1
    )
    return StreamLatentStateView.from_generated(
        generated,
        commit_index=3,
        current_step=10,
        dt=0.1,
        chunk_size=5,
        beta_threshold=0.999,
        frontier_tokens=4,
    )


def test_context_builder_returns_noise_initializer_kwargs():
    view = _state_view()
    text_embedding = torch.randn(2, 6)
    traj_local_frames = torch.randn(2, 13, 7)
    frame_mask = torch.ones(2, 13)

    context = build_noise_initializer_context(
        view,
        text_embedding=text_embedding,
        traj_local_frames=traj_local_frames,
        traj_frame_mask=frame_mask,
        history_tokens=2,
        frontier_tokens=3,
        traj_tokens=4,
    )

    assert isinstance(context, NoiseInitializerContext)
    assert context.history_latents.shape == (2, 2, 3)
    assert torch.equal(context.history_latents, view.committed_latents[:, -2:])
    assert torch.equal(context.active_latents, view.active_latents)
    assert torch.equal(context.active_beta, view.active_beta)
    assert torch.equal(context.active_offsets, view.active_offsets)
    assert torch.equal(context.text_embedding, text_embedding)
    assert context.traj_token_frames.shape == (2, 4, 4, 7)
    assert context.traj_frame_mask.shape == (2, 4, 4)
    assert torch.equal(context.frontier_base_zT, view.frontier_base_zT[:, :3])
    assert torch.equal(context.frontier_ids, view.frontier_ids[:3])
    assert torch.equal(context.frontier_offsets, view.frontier_offsets[:3])

    kwargs = context.as_model_kwargs()
    assert set(kwargs) == {
        "history_latents",
        "active_latents",
        "active_beta",
        "active_offsets",
        "text_embedding",
        "traj_token_frames",
        "traj_frame_mask",
        "frontier_offsets",
    }
    assert kwargs["frontier_offsets"].shape == (3,)


def test_context_builder_uses_existing_token_range_padding_and_masking():
    view = _state_view(batch_size=1)
    text_embedding = torch.randn(1, 6)
    traj_local_frames = torch.arange(1 * 6 * 7, dtype=torch.float32).view(1, 6, 7)
    frame_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.float32)

    context = build_noise_initializer_context(
        view,
        text_embedding=text_embedding,
        traj_local_frames=traj_local_frames,
        traj_frame_mask=frame_mask,
        frontier_tokens=2,
        traj_tokens=3,
    )

    assert torch.equal(
        context.traj_token_frames[:, 0],
        traj_local_frames[:, 0:1].expand(-1, 4, -1),
    )
    assert torch.equal(context.traj_token_frames[:, 1], traj_local_frames[:, 1:5])
    assert torch.equal(context.traj_token_frames[:, 2, :1], traj_local_frames[:, 5:6])
    assert torch.equal(context.traj_token_frames[:, 2, 1:], torch.zeros(1, 3, 7))
    assert torch.equal(
        context.traj_frame_mask,
        torch.tensor(
            [[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]],
            dtype=torch.float32,
        ),
    )


def test_context_builder_defaults_traj_tokens_to_cover_causal_frame_window():
    view = _state_view(batch_size=1)
    text_embedding = torch.randn(1, 6)
    traj_local_frames = torch.randn(1, 6, 7)

    context = build_noise_initializer_context(
        view,
        text_embedding=text_embedding,
        traj_local_frames=traj_local_frames,
        frontier_tokens=2,
    )

    assert context.traj_token_frames.shape == (1, 3, 4, 7)
    assert torch.equal(context.traj_token_frames[:, 2, :1], traj_local_frames[:, 5:6])
    assert context.traj_frame_mask is None


def test_context_builder_rejects_insufficient_frontier_tokens():
    view = _state_view(batch_size=1)

    with pytest.raises(ValueError, match="frontier"):
        build_noise_initializer_context(
            view,
            text_embedding=torch.randn(1, 6),
            traj_local_frames=torch.randn(1, 8, 7),
            frontier_tokens=5,
        )
