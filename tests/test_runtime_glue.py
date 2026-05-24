"""Unit tests for utils/runtime_glue.py (T_C_02 dual-anchor commit-slice math)."""

from __future__ import annotations

import pytest

from utils.runtime_glue import body_window_start_commit_idx, committed_frame_slice
from utils.token_frame import token_end_frame, token_start_frame


# ---------------------------------------------------------------------------
# body_window_start_commit_idx
# ---------------------------------------------------------------------------


def test_body_window_start_default():
    assert body_window_start_commit_idx(35, body_history_tokens=20) == 15


def test_body_window_start_clamps_to_zero():
    assert body_window_start_commit_idx(5, body_history_tokens=20) == 0


def test_body_window_start_explicit_override():
    assert body_window_start_commit_idx(35, body_history_tokens=20, explicit_start=7) == 7


# ---------------------------------------------------------------------------
# committed_frame_slice (Done #5)
# ---------------------------------------------------------------------------


def test_committed_slice_body_window_local_offset():
    # body_anchor=20, head=35, committed=5 → relative_start_token = 15
    sl = committed_frame_slice(head_commit_idx=35, body_anchor_commit_idx=20,
                               committed_tokens=5)
    assert sl.start == token_start_frame(15)            # 57, NOT 0 (global misuse)
    assert sl.start == 57
    assert sl.stop == token_end_frame(19) + 1            # tokens 15..19
    assert sl.stop - sl.start == 20                      # arbitrary range = 4*committed


def test_committed_slice_at_window_start_is_prefix():
    # head == body_anchor → relative_start_token 0 → prefix length 4N-3
    sl = committed_frame_slice(head_commit_idx=20, body_anchor_commit_idx=20,
                               committed_tokens=5)
    assert sl.start == 0
    assert sl.stop == 17                                 # 4*5 - 3


def test_committed_slice_head_before_anchor_raises():
    with pytest.raises(ValueError):
        committed_frame_slice(head_commit_idx=10, body_anchor_commit_idx=20,
                              committed_tokens=5)


def test_committed_slice_nonpositive_tokens_raises():
    with pytest.raises(ValueError):
        committed_frame_slice(head_commit_idx=35, body_anchor_commit_idx=20,
                              committed_tokens=0)
