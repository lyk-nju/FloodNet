"""Unit tests for utils/training/history_corruption.py (T_B_03).

Maps to the 5 Done-criteria tests in docs/TODO.md §T_B_03 + the gate logic.
"""

from __future__ import annotations

import math

import torch

from utils.training.history_corruption import (
    apply_history_corruption,
    sample_focus_ratio,
    should_apply_corruption,
)


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Gate: should_apply_corruption
# ---------------------------------------------------------------------------


def test_disabled_gate_returns_false():
    assert should_apply_corruption(0, 100, {"enabled": False}) is False


def test_apply_prob_zero_never_applies():
    # T_B_03 #1: apply_prob=0 → no-op (gate False every time).
    cfg = {"enabled": True, "apply_prob": 0.0}
    assert all(not should_apply_corruption(i, 100, cfg) for i in range(50))


def test_apply_prob_one_always_applies():
    cfg = {"enabled": True, "apply_prob": 1.0}
    assert all(should_apply_corruption(i, 100, cfg) for i in range(50))


def test_curriculum_progress_thresholds():
    """apply_prob=None + curriculum: early/mid/late by progress thirds.
    Use prob 0 vs 1 sentinels to make the gate deterministic per band."""
    cfg = {
        "enabled": True, "apply_prob": None,
        "curriculum": {"enabled": True, "early_prob": 0.0, "mid_prob": 0.0, "late_prob": 1.0},
    }
    # early third (progress < 1/3): prob 0 → never
    assert not should_apply_corruption(10, 100, cfg)
    # late third (progress >= 2/3): prob 1 → always
    assert should_apply_corruption(80, 100, cfg)


def test_curriculum_disabled_returns_false():
    cfg = {"enabled": True, "apply_prob": None, "curriculum": {"enabled": False}}
    assert should_apply_corruption(50, 100, cfg) is False


# ---------------------------------------------------------------------------
# T_B_03 #5: focus_ratio cosine distribution mean ≈ 2/π
# ---------------------------------------------------------------------------


def test_focus_ratio_mean_approx_two_over_pi():
    g = _gen(123)
    vals = [sample_focus_ratio(generator=g) for _ in range(20000)]
    mean = sum(vals) / len(vals)
    assert abs(mean - 2.0 / math.pi) < 0.02, f"mean={mean}, expected ≈ {2/math.pi:.4f}"
    # range sanity: cos(π/2·u) ∈ [0, 1]
    assert all(0.0 <= v <= 1.0 + 1e-6 for v in vals)


# ---------------------------------------------------------------------------
# apply_history_corruption
# ---------------------------------------------------------------------------


def _setup(B=2, T=30, D=4, chunk_size=5):
    clean = torch.randn(B, T, D, generator=_gen(7), dtype=torch.float64)
    mask_emb = torch.full((D,), 7.0, dtype=torch.float64)   # distinctive sentinel
    z_std = torch.ones(D, dtype=torch.float64)
    # active window right boundary; ctx_end = end - chunk_size
    end_indices = torch.tensor([T, T])
    return clean, mask_emb, z_std, chunk_size, end_indices


def test_active_window_region_never_modified():
    """T_B_03 #2: with corruption applied, history region may change but the
    active window [ctx_end, T) is identical to the input."""
    clean, mask_emb, z_std, cs, end_idx = _setup()
    out = apply_history_corruption(
        clean, end_idx, mask_emb=mask_emb, z_std=z_std, chunk_size=cs,
        alpha_mask=0.3, alpha_noisy=0.3, generator=_gen(1),
    )
    B, T, D = clean.shape
    for b in range(B):
        ctx_end = int(end_idx[b]) - cs
        # active window region preserved
        assert torch.equal(out[b, ctx_end:, :], clean[b, ctx_end:, :])
    # something in history changed (with these seeds n_mask+n_noisy>0)
    assert not torch.equal(out, clean)


def test_ctx_end_nonpositive_is_noop():
    """T_B_03 #1 variant: when end-chunk_size<=0 there's no history → unchanged."""
    clean, mask_emb, z_std, cs, _ = _setup(T=10, chunk_size=20)
    end_idx = torch.tensor([10, 10])   # ctx_end = 10 - 20 = -10 <= 0
    out = apply_history_corruption(
        clean, end_idx, mask_emb=mask_emb, z_std=z_std, chunk_size=cs,
        generator=_gen(1),
    )
    assert torch.equal(out, clean)


def test_mask_tokens_equal_mask_emb_broadcast():
    """T_B_03 #3: masked tokens become exactly mask_emb (broadcast over D)."""
    clean, mask_emb, z_std, cs, end_idx = _setup(B=1, T=40, D=4, chunk_size=5)
    # alpha_mask=1.0, alpha_noisy=0.0 → all focus tokens are masked.
    out = apply_history_corruption(
        clean, end_idx, mask_emb=mask_emb, z_std=z_std, chunk_size=cs,
        alpha_mask=1.0, alpha_noisy=0.0, generator=_gen(3),
    )
    ctx_end = int(end_idx[0]) - cs
    # Rows that changed in the history region must equal mask_emb exactly.
    changed = (out[0, :ctx_end] != clean[0, :ctx_end]).any(dim=-1)
    n_changed = int(changed.sum())
    assert n_changed > 0
    for i in range(ctx_end):
        if changed[i]:
            assert torch.equal(out[0, i, :], mask_emb)


def test_noisy_tokens_are_clean_plus_gaussian():
    """T_B_03 #4: noisy tokens = clean + N(0, (sigma_factor·z_std)²)."""
    B, T, D, cs = 1, 200, 4, 5
    clean = torch.zeros(B, T, D, dtype=torch.float64)   # clean=0 → out = pure noise
    mask_emb = torch.zeros(D, dtype=torch.float64)
    z_std = torch.full((D,), 2.0, dtype=torch.float64)
    end_idx = torch.tensor([T])
    sigma_factor = 0.05
    out = apply_history_corruption(
        clean, end_idx, mask_emb=mask_emb, z_std=z_std, chunk_size=cs,
        alpha_mask=0.0, alpha_noisy=1.0, noise_sigma_factor=sigma_factor,
        generator=_gen(11),
    )
    ctx_end = T - cs
    changed = (out[0, :ctx_end] != clean[0, :ctx_end]).any(dim=-1)
    noisy_rows = out[0, :ctx_end][changed]
    assert noisy_rows.shape[0] > 50   # enough samples for a std estimate
    # Empirical std per channel ≈ sigma_factor * z_std = 0.05 * 2.0 = 0.1
    emp_std = noisy_rows.std(dim=0, unbiased=False)
    expected = sigma_factor * z_std
    assert torch.allclose(emp_std, expected, rtol=0.25), (
        f"emp_std={emp_std.tolist()} expected≈{expected.tolist()}"
    )


def test_returns_new_tensor_input_unchanged():
    clean, mask_emb, z_std, cs, end_idx = _setup()
    clean_copy = clean.clone()
    _ = apply_history_corruption(
        clean, end_idx, mask_emb=mask_emb, z_std=z_std, chunk_size=cs,
        generator=_gen(1),
    )
    assert torch.equal(clean, clean_copy)   # input not mutated


def test_rejects_non_3d_input():
    import pytest
    with pytest.raises(ValueError):
        apply_history_corruption(
            torch.zeros(5, 4), torch.tensor([5]),
            mask_emb=torch.zeros(4), z_std=torch.ones(4), chunk_size=1,
        )
