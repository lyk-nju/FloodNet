"""Unit tests for InferenceGlueTimeline (T_A_03c).

Covers G09-G13 (timeline ops, at_commit, has_reached, has_exact_state,
trim_before) and G23-G25 (pending-edit / effective_commit gating expressed
purely via timeline.has_reached + has_exact_state, no ModelManager required).
"""

from __future__ import annotations

import logging

import pytest
import torch

from utils.inference_glue import (
    InferenceGlueState,
    InferenceGlueTimeline,
)

ATOL = 1e-5


def _state(commit_idx: int, xz=(0.0, 0.0), yaw=0.0, source="commit"):
    return InferenceGlueState(
        commit_idx=commit_idx,
        world_xz=torch.tensor(xz, dtype=torch.float64),
        world_yaw=torch.tensor(yaw, dtype=torch.float64),
        source=source,
    )


def _build_timeline(commits, *, xz_fn=None, yaw_fn=None):
    """Build a timeline with states at given commit indices."""
    xz_fn = xz_fn or (lambda c: (float(c), 0.0))
    yaw_fn = yaw_fn or (lambda c: 0.0)
    initial = _state(commits[0], xz=xz_fn(commits[0]), yaw=yaw_fn(commits[0]),
                     source="init")
    timeline = InferenceGlueTimeline(initial)
    for c in commits[1:]:
        timeline.append(_state(c, xz=xz_fn(c), yaw=yaw_fn(c)))
    return timeline


# ---------------------------------------------------------------------------
# G09: head returns latest
# ---------------------------------------------------------------------------


def test_G09_head_returns_latest_appended_state():
    tl = _build_timeline([0, 5, 10, 15])
    assert tl.head.commit_idx == 15
    new_state = _state(20, xz=(20.0, 0.0))
    tl.append(new_state)
    assert tl.head.commit_idx == 20
    assert tl.head is new_state


# ---------------------------------------------------------------------------
# G10: at_commit on existing snapshot returns identical state
# ---------------------------------------------------------------------------


def test_G10_at_commit_existing_returns_same_state():
    tl = _build_timeline([0, 5, 10, 15, 20])
    for c in (0, 5, 10, 15, 20):
        s = tl.at_commit(c)
        assert s.commit_idx == c
        # Match constructed values exactly
        assert torch.allclose(s.world_xz, torch.tensor([float(c), 0.0], dtype=torch.float64),
                                atol=ATOL)


# ---------------------------------------------------------------------------
# G11: at_commit beyond head returns head + warning
# ---------------------------------------------------------------------------


def test_G11_at_commit_beyond_head_returns_head_and_logs_warning(caplog):
    tl = _build_timeline([0, 5, 10])
    with caplog.at_level(logging.WARNING):
        s = tl.at_commit(100)
    assert s.commit_idx == tl.head.commit_idx == 10
    assert any("at_commit(100)" in rec.message and "past head" in rec.message
                for rec in caplog.records), f"records: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# G12: has_reached
# ---------------------------------------------------------------------------


def test_G12_has_reached_uses_le_head_commit_idx():
    tl = _build_timeline([0, 5, 10, 15])
    # head.commit_idx = 15
    assert tl.has_reached(0)
    assert tl.has_reached(7)        # idx not in timeline but <= head; still True
    assert tl.has_reached(15)       # boundary
    assert not tl.has_reached(16)   # past head
    assert not tl.has_reached(100)


# ---------------------------------------------------------------------------
# G13: has_exact_state + trim_before invalidation
# ---------------------------------------------------------------------------


def test_G13a_has_exact_state_basic():
    tl = _build_timeline([0, 5, 10, 15])
    assert tl.has_exact_state(0)
    assert tl.has_exact_state(5)
    assert tl.has_exact_state(10)
    assert tl.has_exact_state(15)
    # Not exact (no snapshot at these)
    assert not tl.has_exact_state(3)
    assert not tl.has_exact_state(11)


def test_G13b_trim_before_invalidates_has_exact_state_below_cutoff():
    tl = _build_timeline([0, 5, 10, 15, 20])
    tl.trim_before(10)
    # 0, 5 are gone
    assert not tl.has_exact_state(0)
    assert not tl.has_exact_state(5)
    # 10, 15, 20 remain
    assert tl.has_exact_state(10)
    assert tl.has_exact_state(15)
    assert tl.has_exact_state(20)
    # has_reached(0) is still True (head has passed 0 long ago).
    # ⚠ G13 lock-in: has_reached and has_exact_state diverge after trim.
    assert tl.has_reached(0)
    assert not tl.has_exact_state(0)


def test_G13c_at_commit_after_trim_gives_fallback_with_warning_NOT_for_stale_check(caplog):
    tl = _build_timeline([0, 5, 10, 15, 20])
    tl.trim_before(10)
    with caplog.at_level(logging.WARNING):
        s = tl.at_commit(5)
    assert any("at_commit(5)" in rec.message and "trimmed" in rec.message.lower()
                for rec in caplog.records), f"records: {[r.message for r in caplog.records]}"
    # Fallback should be the earliest available (commit_idx=10).
    assert s.commit_idx == 10
    # ⚠ The caller MUST NOT use this fallback for stale check; they should use
    # has_exact_state(5) -> False instead.
    assert not tl.has_exact_state(5)


def test_trim_before_keeps_head_when_all_states_are_older():
    tl = _build_timeline([0, 5, 10])
    tl.trim_before(50)   # all states < 50
    assert len(tl) == 1
    assert tl.head.commit_idx == 10


# ---------------------------------------------------------------------------
# Append validation
# ---------------------------------------------------------------------------


def test_append_rejects_non_monotone_commit_idx():
    tl = _build_timeline([0, 5, 10])
    with pytest.raises(ValueError):
        tl.append(_state(10))   # same as head
    with pytest.raises(ValueError):
        tl.append(_state(5))    # before head


def test_init_requires_inference_glue_state():
    with pytest.raises(TypeError):
        InferenceGlueTimeline("not a state")   # type: ignore


# ---------------------------------------------------------------------------
# Gap-inside fallback
# ---------------------------------------------------------------------------


def test_at_commit_gap_inside_returns_nearest_preceding_with_warning(caplog):
    """If snapshots exist at [0, 10, 20] and we query 15, return nearest
    preceding (commit_idx=10) + warn.
    """
    tl = _build_timeline([0, 10, 20])
    with caplog.at_level(logging.WARNING):
        s = tl.at_commit(15)
    assert s.commit_idx == 10
    assert any("at_commit(15)" in rec.message and "nearest preceding" in rec.message
                for rec in caplog.records)


# ---------------------------------------------------------------------------
# G23-G25: pending-edit / effective_commit gating via timeline primitives
# (no ModelManager required; expressed purely as timeline.has_reached +
# has_exact_state behavior).
# ---------------------------------------------------------------------------


def test_G23_pending_edit_effective_commit_not_yet_reached():
    """G23: when timeline has not yet advanced to effective_commit, the
    application layer must see has_reached(effective_commit) == False
    and therefore not dispatch the Refiner.
    """
    # commit_tokens=5, edit at commit_idx=20, delay=10 → effective_commit=30
    tl = _build_timeline([0, 5, 10, 15, 20])   # head at 20
    effective_commit = 30
    assert not tl.has_reached(effective_commit)
    # Application uses this gate; if True → dispatch, if False → wait.


def test_G24_timeline_reaches_effective_commit_exact_anchor_available():
    """G24: once timeline advances to effective_commit, both has_reached and
    has_exact_state return True, and timeline.head equals the anchor used
    for canonicalize.
    """
    tl = _build_timeline([0, 5, 10, 15, 20])
    effective_commit = 30
    # advance head past effective_commit (committed_tokens batch = 10)
    tl.append(_state(25, xz=(25.0, 0.0)))
    tl.append(_state(30, xz=(30.0, 0.0)))
    assert tl.has_reached(effective_commit)
    assert tl.has_exact_state(effective_commit)
    anchor = tl.at_commit(effective_commit)
    assert anchor.commit_idx == effective_commit
    assert anchor is tl.head   # exact hit lands at current head


def test_G25_prepare_margin_zero_exact_hit_lock():
    """G25: with prepare_margin_tokens=0, the Refiner dispatch lands exactly
    at the effective_commit token boundary — has_exact_state must succeed.
    (For nonzero margins, callers may instead query at effective_commit -
    margin; but v1 spec is margin=0.)
    """
    tl = _build_timeline([0, 5, 10, 15, 20, 25, 30])
    effective_commit = 25
    assert tl.has_exact_state(effective_commit)
    anchor = tl.at_commit(effective_commit)
    assert anchor.commit_idx == 25
    # If margin had been 3 and we queried 22, has_exact_state would be False
    # (no snapshot at 22) — caller MUST NOT use that as a stale-check proxy.
    assert not tl.has_exact_state(22)
