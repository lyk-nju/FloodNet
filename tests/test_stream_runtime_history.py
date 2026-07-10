"""Tests for the absolute-coordinate generated-root history."""

from __future__ import annotations

import torch
import pytest

from utils.inference.stream_runtime import GeneratedRootHistory


def test_generated_history_preserves_absolute_base_after_trim():
    history = GeneratedRootHistory.empty(dtype=torch.float32)
    history.append(torch.arange(35, dtype=torch.float32).view(5, 7), start_frame_abs=0)
    history.trim_before(3)
    assert history.base_frame_abs == 3
    assert history.next_frame_abs == 5
    assert torch.equal(history.slice_abs(3, 5), history.frames_7d)


def test_generated_history_requires_contiguous_append_and_generated_slice():
    history = GeneratedRootHistory.empty(dtype=torch.float32)
    frames = torch.zeros(2, 7, dtype=torch.float32)

    with pytest.raises(ValueError):
        history.append(frames, start_frame_abs=1)

    history.append(frames, start_frame_abs=0)
    with pytest.raises(ValueError):
        history.slice_abs(-1, 1)
    with pytest.raises(ValueError):
        history.slice_abs(1, 3)


def test_generated_history_trim_rejects_future_frame():
    history = GeneratedRootHistory.empty(dtype=torch.float32)
    history.append(torch.zeros(2, 7, dtype=torch.float32), start_frame_abs=0)

    with pytest.raises(ValueError):
        history.trim_before(3)
