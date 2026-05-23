"""Unit tests for models/root_refiner.py (T_A_05).

Covers T01-T08 per docs/TODO.md §T_A_05 Unit tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.root_refiner import RootRefiner
from utils.token_frame import num_frames_for_tokens


def _make_inputs(model: RootRefiner, B: int = 2, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    text_emb = torch.randn(B, model.text_emb_dim, generator=g)
    xz_path = torch.randn(B, model.n_path, 2, generator=g)
    path_mask = torch.ones(B, model.n_path, dtype=torch.bool)
    path_stats = torch.randn(B, model.path_stats_dim, generator=g)
    current_motion = torch.randn(B, model.n_hist, 5, generator=g)
    history_mask = torch.ones(B, model.n_hist, dtype=torch.bool)
    return dict(
        text_emb=text_emb,
        xz_path=xz_path,
        path_mask=path_mask,
        path_stats=path_stats,
        current_motion=current_motion,
        history_mask=history_mask,
    )


# ---------------------------------------------------------------------------
# T01-T02: shape
# ---------------------------------------------------------------------------


def test_T01_forward_shape_num_token_logits_and_waypoints():
    model = RootRefiner()
    B = 3
    out = model(**_make_inputs(model, B=B))
    K = model.max_tokens - model.min_tokens + 1
    assert out["num_token_logits"].shape == (B, K)
    assert out["waypoints"].shape == (B, num_frames_for_tokens(model.max_tokens), 7)


def test_T02_max_frames_equals_num_frames_for_tokens_max():
    model = RootRefiner(max_tokens=49, min_tokens=4)
    assert model.max_frames == num_frames_for_tokens(49)
    assert model.max_frames == 193


# ---------------------------------------------------------------------------
# T03-T05: attention masking
# ---------------------------------------------------------------------------


def test_T03_history_padding_does_not_leak_into_output():
    """history_mask = only last slot True (full-plan); padding region values
    must NOT affect the model output (proves src_key_padding_mask works for history).
    """
    torch.manual_seed(0)
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                          max_tokens=8, min_tokens=2)
    model.eval()
    B = 2
    inputs = _make_inputs(model, B=B)
    # Full-plan style mask: only last history slot is valid.
    inputs["history_mask"] = torch.zeros(B, model.n_hist, dtype=torch.bool)
    inputs["history_mask"][:, -1] = True
    # Build a perturbed twin: same valid slot, but garbage in padding region.
    inputs_perturbed = {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}
    inputs_perturbed["current_motion"][:, :-1] = torch.randn(
        B, model.n_hist - 1, 5, generator=torch.Generator().manual_seed(99),
    ) * 100.0
    with torch.no_grad():
        out_a = model(**inputs)
        out_b = model(**inputs_perturbed)
    assert torch.allclose(
        out_a["waypoints"], out_b["waypoints"], atol=1e-5,
    ), "history padding leaked into output (mask not effective)"
    assert torch.allclose(out_a["num_token_logits"], out_b["num_token_logits"], atol=1e-5)


def test_T04_path_padding_does_not_leak_into_output():
    """path_mask = first half True, rest False → padding region values must NOT
    affect output."""
    torch.manual_seed(1)
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                          max_tokens=8, min_tokens=2)
    model.eval()
    B = 2
    inputs = _make_inputs(model, B=B)
    half = model.n_path // 2
    inputs["path_mask"] = torch.zeros(B, model.n_path, dtype=torch.bool)
    inputs["path_mask"][:, :half] = True
    inputs_perturbed = {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}
    inputs_perturbed["xz_path"][:, half:] = torch.randn(
        B, model.n_path - half, 2, generator=torch.Generator().manual_seed(123),
    ) * 100.0
    with torch.no_grad():
        out_a = model(**inputs)
        out_b = model(**inputs_perturbed)
    assert torch.allclose(out_a["waypoints"], out_b["waypoints"], atol=1e-5)
    assert torch.allclose(out_a["num_token_logits"], out_b["num_token_logits"], atol=1e-5)


def test_T05_specials_and_queries_are_never_masked():
    """Internal: src_key_padding_mask never has True at CLS / text / stats /
    query positions, regardless of path / history masks.

    Verified by checking the mask builder via a forward-pass instrumentation:
    we construct extreme mask inputs (all-False on path + history) and confirm
    the model still runs (CLS / text / stats / queries keep attention alive).
    """
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                          max_tokens=8, min_tokens=2)
    model.eval()
    B = 1
    inputs = _make_inputs(model, B=B)
    inputs["path_mask"] = torch.zeros(B, model.n_path, dtype=torch.bool)
    inputs["history_mask"] = torch.zeros(B, model.n_hist, dtype=torch.bool)
    with torch.no_grad():
        out = model(**inputs)
    # Output must be finite (no NaN from full-mask attention).
    assert torch.isfinite(out["num_token_logits"]).all()
    assert torch.isfinite(out["waypoints"]).all()


# ---------------------------------------------------------------------------
# T06: heading unit-norm
# ---------------------------------------------------------------------------


def test_T06_heading_channels_unit_norm():
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                          max_tokens=8, min_tokens=2)
    out = model(**_make_inputs(model, B=2))
    head = out["waypoints"][..., 3:5]
    norms = head.pow(2).sum(-1)   # [B, max_frames]
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"heading norms out of unit range: min={norms.min().item()}, "
        f"max={norms.max().item()}"
    )


# ---------------------------------------------------------------------------
# T07: backward
# ---------------------------------------------------------------------------


def test_T07_backward_produces_finite_gradients():
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                          max_tokens=8, min_tokens=2)
    out = model(**_make_inputs(model, B=2))
    loss = out["num_token_logits"].sum() + out["waypoints"].sum()
    loss.backward()
    bad = 0
    for n, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                bad += 1
                print(f"NaN/Inf grad in {n}")
    assert bad == 0, f"{bad} parameters have non-finite gradients"


# ---------------------------------------------------------------------------
# parameter budget
# ---------------------------------------------------------------------------


def test_default_model_under_50M_params():
    """Done criteria: < 50M params at default sizes."""
    model = RootRefiner()
    n = model.count_parameters()
    assert n < 50_000_000, f"param count {n} >= 50M"
    # Sanity lower bound — default d_model=256, n_layers=6 should be at least ~3M.
    assert n > 3_000_000, f"param count {n} unexpectedly small ({n})"


# ---------------------------------------------------------------------------
# T08: tiny-batch overfit (mandatory)
# ---------------------------------------------------------------------------


def test_T08_tiny_batch_overfit():
    """Overfit a tiny synthetic batch: losses should drop significantly.

    Catches dataset / target / loss-mask bugs at the integration level. Uses a
    small model + small batch + reduced step count to keep runtime ~10-30s.
    """
    torch.manual_seed(0)
    # Compact model: d_model=64, n_layers=2, max_tokens=8 → max_frames=29.
    model = RootRefiner(
        d_model=64, n_layers=2, n_heads=4, ff_dim=128,
        max_tokens=8, min_tokens=2, n_hist=8, n_path=16,
        text_emb_dim=16, path_stats_dim=3, dropout=0.0,
    )
    model.train()
    B = 4
    K = model.max_tokens - model.min_tokens + 1
    F_max = model.max_frames

    g = torch.Generator().manual_seed(7)
    # Fixed batch (will be reused every step — true overfit setup).
    text_emb = torch.randn(B, model.text_emb_dim, generator=g)
    xz_path = torch.randn(B, model.n_path, 2, generator=g)
    path_mask = torch.ones(B, model.n_path, dtype=torch.bool)
    path_stats = torch.randn(B, model.path_stats_dim, generator=g)
    current_motion = torch.randn(B, model.n_hist, 5, generator=g)
    history_mask = torch.ones(B, model.n_hist, dtype=torch.bool)

    # Random but fixed targets.
    target_num_tokens_class = torch.randint(0, K, (B,), generator=g)
    # Build valid 7D waypoints: cos/sin on unit circle.
    target_yaw = torch.randn(B, F_max, generator=g) * 0.5
    target_waypoints = torch.zeros(B, F_max, 7)
    target_waypoints[..., :3] = torch.randn(B, F_max, 3, generator=g) * 0.3
    target_waypoints[..., 3] = torch.cos(target_yaw)
    target_waypoints[..., 4] = torch.sin(target_yaw)
    target_waypoints[..., 5:7] = torch.randn(B, F_max, 2, generator=g) * 0.2
    # (target_mask is intentionally all-True so we skip mask gating in this
    # overfit test; the dataset-level test_T06 already locks the strict
    # target_mask.sum() == num_frames_for_tokens(num_tokens) relationship.)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    initial_losses = []
    final_losses = []

    n_steps = 200
    for step in range(n_steps):
        out = model(
            text_emb=text_emb, xz_path=xz_path, path_mask=path_mask,
            path_stats=path_stats, current_motion=current_motion,
            history_mask=history_mask,
        )
        # Loss: num_token CE + xyz SmoothL1 + heading cosine + fwd_delta + yaw_delta
        l_num = F.cross_entropy(out["num_token_logits"], target_num_tokens_class)
        l_xyz = F.smooth_l1_loss(
            out["waypoints"][..., :3], target_waypoints[..., :3],
        )
        cos_term = (out["waypoints"][..., 3:5] * target_waypoints[..., 3:5]).sum(-1)
        l_heading = (1.0 - cos_term).mean()
        l_fwd = F.smooth_l1_loss(out["waypoints"][..., 5], target_waypoints[..., 5])
        l_yaw = F.smooth_l1_loss(out["waypoints"][..., 6], target_waypoints[..., 6])
        loss = l_num + 5.0 * l_xyz + 1.0 * l_heading + 0.5 * l_fwd + 0.5 * l_yaw

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step < 20:
            initial_losses.append(loss.item())
        if step >= n_steps - 20:
            final_losses.append(loss.item())

    initial = sum(initial_losses) / len(initial_losses)
    final = sum(final_losses) / len(final_losses)
    assert final < initial * 0.3, (
        f"tiny-batch overfit did not converge: initial loss avg={initial:.4f}, "
        f"final loss avg={final:.4f} (need final < 0.3 * initial)"
    )
    # Also check num_token argmax accuracy on the overfit batch.
    with torch.no_grad():
        pred_class = model(
            text_emb=text_emb, xz_path=xz_path, path_mask=path_mask,
            path_stats=path_stats, current_motion=current_motion,
            history_mask=history_mask,
        )["num_token_logits"].argmax(dim=-1)
    acc = (pred_class == target_num_tokens_class).float().mean().item()
    assert acc >= 0.5, (
        f"num_token overfit accuracy too low: {acc:.2f} (>= 0.5 expected)"
    )
