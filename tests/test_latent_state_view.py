from __future__ import annotations

import pytest
import torch

from utils.inference.latent_state_view import (
    StreamLatentStateView,
    inject_frontier_zT,
    triangular_token_beta,
)


class _DummyStreamModel:
    def __init__(self, generated: torch.Tensor):
        self.generated = generated
        self.commit_index = 3
        self.current_step = 10
        self.dt = 0.1
        self.chunk_size = 5


def test_triangular_token_beta_matches_stream_schedule():
    ids = torch.arange(8)

    beta = triangular_token_beta(ids, current_step=10, dt=0.1, chunk_size=5)

    assert torch.allclose(
        beta,
        torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]),
    )


def test_state_view_splits_committed_active_and_frontier_tokens():
    generated = torch.arange(1 * 2 * 12, dtype=torch.float32).view(1, 2, 12, 1, 1)

    view = StreamLatentStateView.from_generated(
        generated,
        commit_index=3,
        current_step=10,
        dt=0.1,
        chunk_size=5,
        beta_threshold=0.999,
        frontier_tokens=3,
    )

    assert view.commit_index == 3
    assert view.current_step == 10
    assert view.committed_ids.tolist() == [0, 1, 2]
    assert view.active_ids.tolist() == [3, 4]
    assert view.frontier_ids.tolist() == [5, 6, 7]
    assert view.active_offsets.tolist() == [0, 1]
    assert view.frontier_offsets.tolist() == [2, 3, 4]
    assert torch.all(view.frontier_beta >= 0.999)
    assert view.committed_latents.shape == (1, 3, 2)
    assert view.active_latents.shape == (1, 2, 2)
    assert view.frontier_base_zT.shape == (1, 3, 2)
    assert torch.equal(view.frontier_base_zT[0, 0], generated[0, :, 5, 0, 0])


def test_state_view_can_require_zero_update_count_for_frontier_tokens():
    generated = torch.zeros(1, 2, 10, 1, 1)
    token_update_count = torch.zeros(10, dtype=torch.long)
    token_update_count[5] = 1

    view = StreamLatentStateView.from_generated(
        generated,
        commit_index=3,
        current_step=10,
        dt=0.1,
        chunk_size=5,
        beta_threshold=0.999,
        frontier_tokens=3,
        token_update_count=token_update_count,
        require_zero_update_count=True,
    )

    assert view.frontier_ids.tolist() == [6, 7, 8]


def test_state_view_from_model_reads_stream_schedule_attributes():
    generated = torch.zeros(1, 2, 12, 1, 1)
    model = _DummyStreamModel(generated)

    view = StreamLatentStateView.from_model(model, frontier_tokens=2)

    assert view.frontier_ids.tolist() == [5, 6]
    assert view.active_ids.tolist() == [3, 4]


def test_inject_frontier_zT_replaces_only_frontier_tokens_and_keeps_grad():
    generated = torch.arange(1 * 2 * 8, dtype=torch.float32).view(1, 2, 8, 1, 1)
    view = StreamLatentStateView.from_generated(
        generated,
        commit_index=2,
        current_step=5,
        dt=0.1,
        chunk_size=4,
        beta_threshold=0.999,
        frontier_tokens=2,
    )
    replacement = torch.full_like(view.frontier_base_zT, -3.0).requires_grad_(True)

    injected = inject_frontier_zT(generated, view, replacement)

    frontier_ids = view.frontier_ids.tolist()
    for offset, idx in enumerate(frontier_ids):
        assert torch.equal(injected[0, :, idx, 0, 0], replacement[0, offset])
    unchanged_ids = [idx for idx in range(generated.shape[2]) if idx not in frontier_ids]
    for idx in unchanged_ids:
        assert torch.equal(injected[:, :, idx], generated[:, :, idx])
    assert injected.requires_grad


def test_inject_frontier_zT_rejects_wrong_shape():
    generated = torch.zeros(1, 2, 8, 1, 1)
    view = StreamLatentStateView.from_generated(
        generated,
        commit_index=2,
        current_step=5,
        dt=0.1,
        chunk_size=4,
        frontier_tokens=2,
    )

    with pytest.raises(ValueError, match="frontier_zT"):
        inject_frontier_zT(generated, view, torch.zeros(1, 3, 2))
