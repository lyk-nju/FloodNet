"""Unit tests for utils/token_frame.py (T_A_01b).

Covers T01-T18 per docs/TODO.md §T_A_01b.
"""

from __future__ import annotations

import torch

from utils.token_frame import (
    FRAMES_PER_TOKEN_DEFAULT,
    frame_idx_to_token_idx,
    num_frames_for_tokens,
    prefix_len_from_tail_invalid,
    token_active_window_left_frame,
    token_body_window_left_frame,
    token_end_frame,
    token_range_to_frame_slice,
    token_start_frame,
)


# ---------------------------------------------------------------------------
# prefix_len_from_tail_invalid (B-P0-1 follow-up scaffold; not yet wired)
# ---------------------------------------------------------------------------


def test_prefix_len_tail_invalid_is_prefix_length():
    # case A: pure tail-invalid → prefix length 4
    m = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    assert prefix_len_from_tail_invalid(m).tolist() == [4]


def test_prefix_len_middle_hole_returns_full_T():
    # case B: a hole at index 2 followed by valid → cannot truncate → full T
    m = torch.tensor([[1, 1, 0, 1, 1, 0, 0]])
    assert prefix_len_from_tail_invalid(m).tolist() == [7]


def test_prefix_len_all_invalid_is_zero():
    m = torch.tensor([[0, 0, 0, 0]])
    assert prefix_len_from_tail_invalid(m).tolist() == [0]


def test_prefix_len_all_valid_is_T():
    m = torch.tensor([[1, 1, 1, 1]])
    assert prefix_len_from_tail_invalid(m).tolist() == [4]


def test_prefix_len_batched_mixed():
    m = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],   # A → 4
        [1, 1, 0, 1, 1, 0, 0],   # B (hole) → 7
        [0, 0, 0, 0, 0, 0, 0],   # all invalid → 0
        [1, 1, 1, 1, 1, 1, 1],   # all valid → 7
    ])
    assert prefix_len_from_tail_invalid(m).tolist() == [4, 7, 0, 7]


# ---------------------------------------------------------------------------
# T01-T05: token boundary
# ---------------------------------------------------------------------------


def test_T01_token_0_is_single_frame():
    assert token_start_frame(0) == 0
    assert token_end_frame(0) == 0


def test_T02_token_1_covers_frames_1_to_4():
    assert token_start_frame(1) == 1
    assert token_end_frame(1) == 4


def test_T03_token_2_covers_frames_5_to_8():
    assert token_start_frame(2) == 5
    assert token_end_frame(2) == 8


def test_T04_token_3_covers_frames_9_to_12():
    assert token_start_frame(3) == 9
    assert token_end_frame(3) == 12


def test_T05_tokens_have_no_gap_between_consecutive():
    """token_start_frame(k+1) == token_end_frame(k) + 1 for k >= 1."""
    for k in range(1, 50):
        assert token_start_frame(k + 1) == token_end_frame(k) + 1, (
            f"gap at k={k}: end({k})={token_end_frame(k)} "
            f"start({k + 1})={token_start_frame(k + 1)}"
        )


# ---------------------------------------------------------------------------
# T06: prefix frame count
# ---------------------------------------------------------------------------


def test_T06_num_frames_for_tokens_prefix_table():
    """N tokens prefix → effective frame count."""
    cases = {0: 0, 1: 1, 2: 5, 3: 9, 20: 77, 49: 193}
    for n, expected in cases.items():
        got = num_frames_for_tokens(n)
        assert got == expected, f"num_frames_for_tokens({n})={got}, expected {expected}"


# ---------------------------------------------------------------------------
# T07-T11: frame → token reverse map
# ---------------------------------------------------------------------------


def test_T07_frame_zero_maps_to_token_zero():
    assert frame_idx_to_token_idx(0) == 0


def test_T08_frames_1_to_4_map_to_token_1():
    for f in (1, 2, 3, 4):
        assert frame_idx_to_token_idx(f) == 1, f"frame {f}"


def test_T09_frames_5_to_8_map_to_token_2():
    for f in (5, 6, 7, 8):
        assert frame_idx_to_token_idx(f) == 2, f"frame {f}"


def test_T10_round_trip_token_start_frame_dominates_frame_idx():
    """token_start_frame(frame_idx_to_token_idx(f)) <= f for all f."""
    for f in range(0, 200):
        k = frame_idx_to_token_idx(f)
        start = token_start_frame(k)
        assert start <= f, f"f={f} k={k} start={start} (must be <= f)"


def test_T11_wrong_formula_regression_at_frame_4():
    """Lock the correct formula: (f-1) // 4 + 1 returns 1 at f=4. The wrong
    formula (f + 2) // 4 + 1 would return 2 at f=4."""
    assert frame_idx_to_token_idx(4) == 1
    wrong = (4 + 2) // 4 + 1
    assert wrong == 2 and frame_idx_to_token_idx(4) != wrong


# ---------------------------------------------------------------------------
# T12-T14: prefix vs arbitrary range distinction
# ---------------------------------------------------------------------------


def test_T12_prefix_range_token_0_single_frame():
    s = token_range_to_frame_slice(0, 1)
    assert s == slice(0, 1)
    assert (s.stop - s.start) == 1
    assert num_frames_for_tokens(1) == 1   # consistent with prefix


def test_T13_arbitrary_range_token_1_is_four_frames_NOT_one():
    """⚠ LOCK-IN: token_range_to_frame_slice(1, 1) covers 4 frames (token 1's
    full [1, 5) window), NOT 1 frame. num_frames_for_tokens(1) would give 1,
    which is only valid for the prefix. The two MUST diverge here, since this
    inconsistency is the bug that motivated the entire token_frame module.
    """
    s = token_range_to_frame_slice(1, 1)
    assert s == slice(1, 5)
    length = s.stop - s.start
    assert length == 4
    # The lock-in assertion: arbitrary-range length != prefix-length helper.
    assert length != num_frames_for_tokens(1), (
        f"arbitrary range len={length} matched num_frames_for_tokens(1)={num_frames_for_tokens(1)} "
        f"— if you ever see this assertion fire, someone has 'unified' the two helpers and "
        f"broken the prefix-only semantics."
    )


def test_T14_arbitrary_ranges_various_starts():
    """token_range_to_frame_slice across different starts / widths."""
    assert token_range_to_frame_slice(0, 2) == slice(0, 5)    # 4*2 - 3 = 5
    assert token_range_to_frame_slice(1, 2) == slice(1, 9)    # 4*2 = 8 wide, [1, 9)
    assert token_range_to_frame_slice(5, 20) == slice(17, 97)  # start=17, width=80


# ---------------------------------------------------------------------------
# T15-T17: active window left frame
# ---------------------------------------------------------------------------


def test_T15_active_window_left_frame_normal():
    assert token_active_window_left_frame(end_token_idx=10, chunk_size_tokens=5) == 17
    # 10 - 5 = 5, token_start_frame(5) = 17


def test_T16_active_window_left_frame_clamps_when_short():
    """end_token < chunk_size → left_token clamps to 0 → frame 0."""
    assert token_active_window_left_frame(end_token_idx=3, chunk_size_tokens=5) == 0


def test_T17_active_window_left_frame_at_token_zero():
    assert token_active_window_left_frame(end_token_idx=0, chunk_size_tokens=5) == 0


# ---------------------------------------------------------------------------
# T18: regression with legacy utils/traj_batch.py:149-150 boundary
# ---------------------------------------------------------------------------


def test_T18_legacy_traj_batch_mask_boundary_matches():
    """Legacy `utils/traj_batch.py:149-150` uses `sf = 4*k - 3; ef = 4*k + 1`
    for token k >= 1's frame range (exclusive end). Our `token_range_to_frame_slice(k, 1)`
    should produce the same `[sf, ef)`.
    """
    for k in range(1, 25):
        legacy_sf = 4 * k - 3
        legacy_ef = 4 * k + 1
        s = token_range_to_frame_slice(k, 1)
        assert s.start == legacy_sf and s.stop == legacy_ef, (
            f"k={k}: legacy=[{legacy_sf}, {legacy_ef}) ours=[{s.start}, {s.stop})"
        )


# ---------------------------------------------------------------------------
# Bonus: body window left frame (used by T_B_05 downstream)
# ---------------------------------------------------------------------------


def test_body_window_left_frame_distinct_from_active_window():
    """token_body_window_left_frame uses body_window_tokens (history+active);
    token_active_window_left_frame uses chunk_size_tokens (just active).
    They produce different left frames given the same end_token_idx.
    """
    end = 30
    chunk = 5
    body_window = 25   # history + active
    active_left = token_active_window_left_frame(end, chunk)
    body_left = token_body_window_left_frame(end, body_window)
    assert active_left != body_left, (
        f"active_left={active_left} body_left={body_left} should differ — "
        f"if they're equal here, body/active mixing is hidden."
    )
    # Sanity: body_left should be <= active_left because body window is wider.
    assert body_left <= active_left


def test_default_frames_per_token_is_4():
    assert FRAMES_PER_TOKEN_DEFAULT == 4
