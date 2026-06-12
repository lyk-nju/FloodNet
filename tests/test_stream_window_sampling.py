from __future__ import annotations

import pytest
import torch

from utils.training.window_sampling import (
    resolve_history_tokens_max,
    sample_stream_window_indices,
)


def test_auto_history_max_accounts_for_rollout_span():
    assert resolve_history_tokens_max(
        "auto",
        context_tokens=30,
        chunk_size=5,
        rollout_span=4,
    ) == 21


def test_stream_window_sampler_respects_context_and_full_horizon_constraints():
    torch.manual_seed(0)
    token_length = torch.tensor([60, 44, 35], dtype=torch.long)

    sample = sample_stream_window_indices(
        token_length,
        context_tokens=30,
        chunk_size=5,
        rollout_span=4,
        history_tokens_min=0,
        history_tokens_max="auto",
        horizon_tokens_min=5,
        horizon_tokens_max=25,
    )

    b = sample["window_left_tokens"]
    a = sample["active_left_tokens"]
    g = sample["history_tokens"]
    h = sample["horizon_tokens"]
    c = 5
    span = 4

    assert torch.equal(b, a - g)
    assert bool((b >= 0).all())
    assert bool((g >= 0).all())
    assert bool((g <= 21).all())
    assert bool((g + c + span <= 30).all())
    assert bool((h >= 5).all())
    assert bool((h <= 25).all())
    assert bool((a + c + span + h <= token_length).all())
    assert torch.equal(sample["latent_num_tokens"], g + c + span)
    assert torch.equal(sample["traj_num_tokens"], g + c + span + h)
    assert sample["history_tokens_max_effective"] == 21
    assert sample["horizon_short_fallback"].tolist() == [False, False, False]
    assert sample["horizon_cap_clip"].tolist() == [51, 35, 26]


def test_stream_window_sampler_falls_back_below_preferred_min_for_short_clips():
    torch.manual_seed(1)
    sample = sample_stream_window_indices(
        torch.tensor([11, 12], dtype=torch.long),
        context_tokens=30,
        chunk_size=5,
        rollout_span=4,
        history_tokens_min=0,
        history_tokens_max="auto",
        horizon_tokens_min=5,
        horizon_tokens_max=25,
    )

    h = sample["horizon_tokens"]
    a = sample["active_left_tokens"]
    assert bool((h >= 1).all())
    assert bool((h <= torch.tensor([2, 3])).all())
    assert bool((a + 5 + 4 + h <= torch.tensor([11, 12])).all())
    assert sample["horizon_short_fallback"].tolist() == [True, True]
    assert sample["horizon_cap_clip"].tolist() == [2, 3]


def test_stream_window_sampler_keeps_absolute_horizon_floor_when_preferred_min_is_zero():
    torch.manual_seed(3)
    sample = sample_stream_window_indices(
        torch.tensor([10, 12], dtype=torch.long),
        context_tokens=30,
        chunk_size=5,
        rollout_span=4,
        history_tokens_min=0,
        history_tokens_max="auto",
        horizon_tokens_min=0,
        horizon_tokens_max=25,
    )

    assert bool((sample["horizon_tokens"] >= 1).all())
    assert bool((sample["horizon_tokens"] <= torch.tensor([1, 3])).all())


def test_stream_window_sampler_rejects_clips_without_any_complete_horizon():
    with pytest.raises(ValueError, match="full horizon"):
        sample_stream_window_indices(
            torch.tensor([9], dtype=torch.long),
            context_tokens=30,
            chunk_size=5,
            rollout_span=4,
            history_tokens_min=0,
            history_tokens_max="auto",
            horizon_tokens_min=5,
            horizon_tokens_max=25,
        )
