"""Unit tests for utils/inference_glue.py InferenceGlueState + advance helper (T_A_03b).

Covers G01-G08 per docs/TODO.md §T_A_03 Unit tests:
    G01-G04: commit_idx exclusive-end semantics
    G05-G08: state advance via body-window-local delta + dual anchor + NaN protection
"""

from __future__ import annotations

import math

import torch

from utils.inference_glue import (
    InferenceGlueState,
    advance_head_from_body_window,
)

ATOL = 1e-5
PI = math.pi


def _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=0.0, source="init"):
    return InferenceGlueState(
        commit_idx=commit_idx,
        world_xz=torch.tensor(xz, dtype=torch.float64),
        world_yaw=torch.tensor(yaw, dtype=torch.float64),
        source=source,
    )


def _zero_delta_inputs():
    """Body-window-local delta inputs that correspond to zero motion."""
    return dict(
        body_output_local_xz_first=torch.tensor([0.0, 0.0], dtype=torch.float64),
        body_output_local_xz_last=torch.tensor([0.0, 0.0], dtype=torch.float64),
        body_output_local_yaw_first=torch.tensor(0.0, dtype=torch.float64),
        body_output_local_yaw_last=torch.tensor(0.0, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# G01-G04: commit_idx semantics (exclusive-end count)
# ---------------------------------------------------------------------------


def test_G01_initial_commit_idx_zero():
    s = InferenceGlueState.initial()
    assert s.commit_idx == 0


def test_G02_advance_5_tokens_yields_commit_idx_5():
    head = _make_state(commit_idx=0)
    body_anchor = head  # for this test, irrelevant since delta is zero
    new_head = advance_head_from_body_window(
        head, body_anchor, **_zero_delta_inputs(), committed_tokens=5,
    )
    assert new_head.commit_idx == 5


def test_G03_advance_another_5_tokens_yields_commit_idx_10():
    head = _make_state(commit_idx=5)
    body_anchor = _make_state(commit_idx=0)   # history0 typically earlier
    new_head = advance_head_from_body_window(
        head, body_anchor, **_zero_delta_inputs(), committed_tokens=5,
    )
    assert new_head.commit_idx == 10


def test_G04_commit_idx_is_next_start_not_last_committed():
    """G04 lock-in: commit_idx is the EXCLUSIVE END = next window start = total
    committed token count. NOT "the last committed token's index" (which would
    be commit_idx - 1).
    """
    head = _make_state(commit_idx=0)
    body_anchor = head
    after_one = advance_head_from_body_window(
        head, body_anchor, **_zero_delta_inputs(), committed_tokens=1,
    )
    # After committing exactly 1 token, the next window starts at index 1.
    # If commit_idx meant "last committed token's index", this would be 0.
    assert after_one.commit_idx == 1, (
        f"After committing 1 token, commit_idx should be 1 (next start), "
        f"got {after_one.commit_idx}"
    )


# ---------------------------------------------------------------------------
# G05-G08: state advance + dual anchor + NaN protection
# ---------------------------------------------------------------------------


def test_G05_zero_local_delta_does_not_change_world_state():
    """Zero body-local delta → world_xz/yaw unchanged, commit_idx advances."""
    head = _make_state(commit_idx=3, xz=(1.5, -0.7), yaw=0.4)
    body_anchor = _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=0.0)
    new_head = advance_head_from_body_window(
        head, body_anchor, **_zero_delta_inputs(), committed_tokens=2,
    )
    assert new_head.commit_idx == 5
    assert torch.allclose(new_head.world_xz, head.world_xz, atol=ATOL)
    assert torch.allclose(new_head.world_yaw, head.world_yaw, atol=ATOL)
    assert new_head.source == "commit"


def test_G06_body_anchor_yaw_zero_local_plus_z_delta_to_world_plus_z():
    """body_anchor.world_yaw = 0, body-local +Z delta → world +Z delta."""
    head = _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=0.0)
    body_anchor = _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=0.0)
    new_head = advance_head_from_body_window(
        head, body_anchor,
        body_output_local_xz_first=torch.tensor([0.0, 0.0], dtype=torch.float64),
        body_output_local_xz_last=torch.tensor([0.0, 1.0], dtype=torch.float64),   # +Z
        body_output_local_yaw_first=torch.tensor(0.0, dtype=torch.float64),
        body_output_local_yaw_last=torch.tensor(0.0, dtype=torch.float64),
        committed_tokens=1,
    )
    # World delta should be +Z (= (0, 1))
    assert torch.allclose(new_head.world_xz, torch.tensor([0.0, 1.0], dtype=torch.float64),
                            atol=ATOL), f"got world_xz={new_head.world_xz.tolist()}"


def test_G07_body_anchor_yaw_quarter_pi_local_plus_z_delta_to_world_plus_x():
    """body_anchor.world_yaw = +π/2 (body faces +X in world), local +Z delta →
    world +X delta. Verifies transform_xz_local_delta_to_world convention
    consistency (matches T18 in test_local_frame).
    """
    head = _make_state(commit_idx=0, xz=(2.0, -1.0), yaw=0.0)
    # ⚠ body_anchor.world_yaw is the local frame's orientation — head's yaw
    # is irrelevant for the position delta rotation.
    body_anchor = _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=PI / 2)
    new_head = advance_head_from_body_window(
        head, body_anchor,
        body_output_local_xz_first=torch.tensor([0.0, 0.0], dtype=torch.float64),
        body_output_local_xz_last=torch.tensor([0.0, 1.0], dtype=torch.float64),
        body_output_local_yaw_first=torch.tensor(0.0, dtype=torch.float64),
        body_output_local_yaw_last=torch.tensor(0.0, dtype=torch.float64),
        committed_tokens=1,
    )
    # World delta = +X (= (1, 0)); new head xz = (2+1, -1+0) = (3, -1)
    assert torch.allclose(new_head.world_xz, torch.tensor([3.0, -1.0], dtype=torch.float64),
                            atol=ATOL), f"got world_xz={new_head.world_xz.tolist()}"
    # Yaw unchanged (yaw delta = 0)
    assert torch.allclose(new_head.world_yaw, head.world_yaw, atol=ATOL)


def test_G07b_head_yaw_does_not_affect_position_delta_rotation():
    """Explicit anti-regression: ensure head.world_yaw is NOT used for delta
    rotation. Use head.world_yaw ≠ body_anchor.world_yaw, change head yaw, expect
    world delta unchanged.
    """
    body_anchor = _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=0.0)
    delta_inputs = dict(
        body_output_local_xz_first=torch.tensor([0.0, 0.0], dtype=torch.float64),
        body_output_local_xz_last=torch.tensor([0.0, 1.0], dtype=torch.float64),
        body_output_local_yaw_first=torch.tensor(0.0, dtype=torch.float64),
        body_output_local_yaw_last=torch.tensor(0.0, dtype=torch.float64),
    )

    head_a = _make_state(commit_idx=5, xz=(0.0, 0.0), yaw=0.0)
    head_b = _make_state(commit_idx=5, xz=(0.0, 0.0), yaw=PI / 2)
    out_a = advance_head_from_body_window(
        head_a, body_anchor, **delta_inputs, committed_tokens=1,
    )
    out_b = advance_head_from_body_window(
        head_b, body_anchor, **delta_inputs, committed_tokens=1,
    )
    # Position delta is body_anchor-anchored, so xz advance should be identical
    # regardless of head's yaw (= (0, 1) world).
    assert torch.allclose(out_a.world_xz, out_b.world_xz, atol=ATOL), (
        f"head yaw leaked into delta rotation: A={out_a.world_xz.tolist()} "
        f"B={out_b.world_xz.tolist()}"
    )


def test_G08_nan_delta_preserves_old_state_and_does_not_advance_commit_idx():
    """NaN/Inf in body delta → preserve old head_state (no advance, no commit)."""
    head = _make_state(commit_idx=7, xz=(1.0, 2.0), yaw=0.3)
    body_anchor = _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=0.0)
    new_head = advance_head_from_body_window(
        head, body_anchor,
        body_output_local_xz_first=torch.tensor([0.0, 0.0], dtype=torch.float64),
        body_output_local_xz_last=torch.tensor([float("nan"), 1.0], dtype=torch.float64),
        body_output_local_yaw_first=torch.tensor(0.0, dtype=torch.float64),
        body_output_local_yaw_last=torch.tensor(0.0, dtype=torch.float64),
        committed_tokens=1,
    )
    # Same state object (identity OK since we returned head unchanged)
    assert new_head.commit_idx == head.commit_idx
    assert torch.equal(new_head.world_xz, head.world_xz)
    assert torch.equal(new_head.world_yaw, head.world_yaw)


def test_G08b_inf_yaw_delta_preserves_old_state():
    head = _make_state(commit_idx=7, xz=(1.0, 2.0), yaw=0.3)
    body_anchor = _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=0.0)
    new_head = advance_head_from_body_window(
        head, body_anchor,
        body_output_local_xz_first=torch.tensor([0.0, 0.0], dtype=torch.float64),
        body_output_local_xz_last=torch.tensor([0.0, 1.0], dtype=torch.float64),
        body_output_local_yaw_first=torch.tensor(float("inf"), dtype=torch.float64),
        body_output_local_yaw_last=torch.tensor(0.0, dtype=torch.float64),
        committed_tokens=1,
    )
    assert new_head.commit_idx == head.commit_idx
    assert torch.equal(new_head.world_xz, head.world_xz)


# ---------------------------------------------------------------------------
# .initial() and .to() helpers
# ---------------------------------------------------------------------------


def test_initial_injects_device_and_dtype():
    s = InferenceGlueState.initial(xz=(1.5, -0.5), yaw=0.4,
                                    device=torch.device("cpu"), dtype=torch.float32)
    assert s.commit_idx == 0
    assert s.world_xz.dtype == torch.float32
    assert s.world_yaw.dtype == torch.float32
    assert s.world_xz.device.type == "cpu"
    assert torch.allclose(s.world_xz, torch.tensor([1.5, -0.5], dtype=torch.float32))
    assert torch.allclose(s.world_yaw, torch.tensor(0.4, dtype=torch.float32))


def test_to_cast_returns_new_state_with_new_dtype_and_preserves_fields():
    s = InferenceGlueState.initial(xz=(1.5, -0.5), yaw=0.4, dtype=torch.float32)
    casted = s.to(dtype=torch.float64)
    assert casted is not s   # new instance
    assert casted.commit_idx == s.commit_idx
    assert casted.source == s.source
    assert casted.world_xz.dtype == torch.float64
    assert casted.world_yaw.dtype == torch.float64
    # Values preserved
    assert torch.allclose(casted.world_xz.float(), s.world_xz)


def test_yaw_wrapping_after_large_advance():
    """Yaw wraps to [-pi, pi) after accumulating large deltas."""
    head = _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=PI - 0.05)
    body_anchor = _make_state(commit_idx=0, xz=(0.0, 0.0), yaw=0.0)
    new_head = advance_head_from_body_window(
        head, body_anchor,
        body_output_local_xz_first=torch.tensor([0.0, 0.0], dtype=torch.float64),
        body_output_local_xz_last=torch.tensor([0.0, 0.0], dtype=torch.float64),
        body_output_local_yaw_first=torch.tensor(0.0, dtype=torch.float64),
        body_output_local_yaw_last=torch.tensor(0.2, dtype=torch.float64),  # crosses +pi
        committed_tokens=1,
    )
    # Resulting yaw should wrap to ~ -pi + 0.15
    assert -PI <= new_head.world_yaw.item() < PI
    expected = (PI - 0.05) + 0.2 - 2 * PI   # = -pi + 0.15
    assert abs(new_head.world_yaw.item() - expected) < 1e-6
