"""Unit tests for utils/training/horizon_sched.py (T_B_04).

Covers the 3 Done-criteria tests from docs/TODO.md §T_B_04 + schedule bands.
"""

from __future__ import annotations

import random

import torch

from utils.token_frame import num_frames_for_tokens, token_start_frame
from utils.training.horizon_sched import (
    apply_horizon_mask_tokens,
    sample_random_horizon_tokens,
)

_CFG = {
    "enabled": True,
    "inference_horizon_tokens": 20,
    "early_min_horizon_tokens": 20,
    "early_max_horizon_ratio": 1.0,
    "late_min_horizon_tokens": 10,
    "late_max_horizon_tokens": 20,
    "p_exact_inference_horizon": 0.5,
    "warmup_ratio": 0.5,
}


# ---------------------------------------------------------------------------
# apply_horizon_mask_tokens (T_B_04 #1, #2)
# ---------------------------------------------------------------------------


def test_horizon_equal_clip_tokens_is_noop():
    """#1: horizon = clip_tokens → mask unchanged (cutoff at/after mask end)."""
    clip_tokens = 20
    T_frame = num_frames_for_tokens(clip_tokens)   # 77
    mask = torch.ones(2, T_frame)
    out = apply_horizon_mask_tokens(mask.clone(), active_end_token=0,
                                     horizon_tokens=clip_tokens)
    assert torch.equal(out, torch.ones(2, T_frame))


def test_cutoff_frame_formula_active_end_zero():
    """#2: active_end=0, horizon=20 → cutoff_frame = token_start_frame(20) = 77."""
    T_frame = 200
    mask = torch.ones(1, T_frame)
    apply_horizon_mask_tokens(mask, active_end_token=0, horizon_tokens=20)
    cutoff = token_start_frame(20)   # 4*20 - 3 = 77
    assert cutoff == 77
    assert mask[0, :cutoff].all()            # before cutoff: visible
    assert not mask[0, cutoff:].any()        # at/after cutoff: masked


def test_cutoff_frame_formula_active_end_nonzero():
    """#2: active_end=5, horizon=20 → cutoff_frame = token_start_frame(25) = 97."""
    T_frame = 200
    mask = torch.ones(1, T_frame)
    apply_horizon_mask_tokens(mask, active_end_token=5, horizon_tokens=20)
    cutoff = token_start_frame(25)   # 4*25 - 3 = 97
    assert cutoff == 97
    assert mask[0, :cutoff].all()
    assert not mask[0, cutoff:].any()


def test_cutoff_uses_token_start_not_token_end():
    """Guard the 77 vs 80 boundary: cutoff must be token_start_frame(E+H),
    NOT (E+H)*frames_per_token (= token_end_frame)."""
    from utils.token_frame import token_end_frame
    eh = 20
    assert token_start_frame(eh) == 77
    assert token_end_frame(eh) == 80          # the WRONG value
    mask = torch.ones(1, 100)
    apply_horizon_mask_tokens(mask, active_end_token=0, horizon_tokens=eh)
    # frame 77,78,79 must be masked (they belong to token 20) — the bug would
    # leave them visible by cutting at 80.
    assert not mask[0, 77:80].any()


def test_mask_unchanged_when_cutoff_beyond_length():
    mask = torch.ones(1, 50)
    apply_horizon_mask_tokens(mask, active_end_token=10, horizon_tokens=40)
    assert mask.all()   # cutoff_frame far beyond 50


# ---------------------------------------------------------------------------
# sample_random_horizon_tokens (T_B_04 #3 + bands)
# ---------------------------------------------------------------------------


def test_early_phase_samples_within_clip_range():
    rng = random.Random(0)
    clip = 40
    for _ in range(200):
        h = sample_random_horizon_tokens(0, 100, clip, _CFG, rng=rng)
        # early lower = max(20, clip//2=20) = 20; upper = clip*1.0 = 40
        assert 20 <= h <= 40


def test_late_phase_exact_probability_approx_p_exact():
    """#3: late schedule hits inference_horizon_tokens with prob ≈ p_exact."""
    rng = random.Random(123)
    n = 20000
    hits = sum(
        sample_random_horizon_tokens(90, 100, 40, _CFG, rng=rng) == 20
        for _ in range(n)
    )
    frac = hits / n
    # p_exact=0.5 exact-hits; the non-exact branch U[10,20] also includes 20
    # (prob 0.5 * 1/11), so observed frac ≈ 0.5 + 0.5/11 ≈ 0.545.
    assert 0.50 <= frac <= 0.59, f"frac={frac}"


def test_late_phase_values_in_10_20():
    rng = random.Random(7)
    for _ in range(500):
        h = sample_random_horizon_tokens(95, 100, 40, _CFG, rng=rng)
        assert 10 <= h <= 20


def test_warmup_boundary_uses_late_branch_at_exactly_half():
    # progress == warmup_ratio (0.5) → late branch (progress < warmup is False)
    rng = random.Random(1)
    vals = {sample_random_horizon_tokens(50, 100, 40, _CFG, rng=rng) for _ in range(300)}
    assert max(vals) <= 20   # late branch caps at 20, early would allow up to 40
