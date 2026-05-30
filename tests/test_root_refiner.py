"""Unit tests for models/root_refiner.py (T_A_05).

Covers T01-T08 per docs/TODO.md §T_A_05 Unit tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.root_refiner import PathCondFrameDecoder, RootRefiner
from utils.token_frame import num_frames_for_tokens


def _make_inputs(model: RootRefiner, B: int = 2, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    text_emb = torch.randn(B, model.text_emb_dim, generator=g)
    path = torch.randn(B, model.n_path, 2, generator=g)
    path_valid_mask = torch.ones(B, model.n_path, dtype=torch.bool)
    path_features = torch.randn(B, model.path_features_dim, generator=g)
    history_motion = torch.randn(B, model.n_hist, 5, generator=g)
    history_mask = torch.ones(B, model.n_hist, dtype=torch.bool)
    return dict(
        text_emb=text_emb,
        path=path,
        path_valid_mask=path_valid_mask,
        path_features=path_features,
        history_motion=history_motion,
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
    assert out["waypoints"].shape == (B, num_frames_for_tokens(model.max_tokens), 5)


def test_T02_max_frames_equals_num_frames_for_tokens_max():
    model = RootRefiner(max_tokens=49, min_tokens=4)
    assert model.max_frames == num_frames_for_tokens(49)
    assert model.max_frames == 193


def test_forward_with_and_without_num_tokens():
    """Duration-first/trajectory-second: forward must work both with a GT
    num_tokens (teacher-forced horizon) and without (argmax-driven inference)."""
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                          max_tokens=8, min_tokens=2)
    inputs = _make_inputs(model, B=3)
    K = model.num_token_logits_dim
    Fm = model.max_frames

    model.train()
    nt = torch.tensor([2, 5, 8])
    out = model(**inputs, num_tokens=nt)
    assert out["num_token_logits"].shape == (3, K)
    assert out["waypoints"].shape == (3, Fm, 5)
    assert torch.equal(out["used_num_tokens"], nt)   # teacher-forced horizon

    model.eval()
    out2 = model(**inputs)                              # no num_tokens → use pred
    assert out2["waypoints"].shape == (3, Fm, 5)
    assert torch.equal(out2["used_num_tokens"], out2["pred_num_tokens"])

    # Teacher-forcing is gated on num_tokens PRESENCE, not train/eval mode: passing
    # num_tokens in eval() must still teacher-force (needed for val + oracle eval).
    out3 = model(**inputs, num_tokens=nt)
    assert torch.equal(out3["used_num_tokens"], nt)


def test_pred_num_tokens_in_range():
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                          max_tokens=8, min_tokens=2)
    model.eval()
    out = model(**_make_inputs(model, B=4))
    pnt = out["pred_num_tokens"]
    assert int(pnt.min()) >= model.min_tokens
    assert int(pnt.max()) <= model.max_tokens


def test_both_decoder_types_build_and_output_effective_frames():
    """simple and path_cond decoders both return effective [B, num_frames_for_tokens, 5]."""
    for dt in ("simple", "path_cond"):
        model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                             max_tokens=8, min_tokens=2, n_hist=8, n_path=16,
                             text_emb_dim=16, decoder_type=dt, decoder_width=48)
        out = model(**_make_inputs(model, B=2))
        assert out["waypoints"].shape == (2, num_frames_for_tokens(8), 5), dt
        # heading unit-norm holds for both.
        norms = out["waypoints"][..., 3:5].pow(2).sum(-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), dt


def test_path_cond_zeroed_for_degenerate_path():
    """PathCondFrameDecoder: a sample whose path has < 2 valid points gets an
    all-zero path condition (decoder falls back to the token-latent plan)."""
    dec = PathCondFrameDecoder(d_model=32, max_tokens=8, n_path=16, width=48,
                                token_res_depth=1, frame_res_depth=2)
    xz = torch.randn(2, 16, 2)
    # sample 0: degenerate (0 valid points); sample 1: valid (all 16).
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[1] = True
    cond = dec._build_path_cond(xz, mask, torch.tensor([4, 8]))
    assert cond.shape == (2, dec.max_frames, 6)
    assert float(cond[0].abs().max()) == 0.0           # degenerate → zeroed
    assert float(cond[1].abs().max()) > 0.0            # valid → nonzero


def test_path_cond_decoder_no_leak_from_post_horizon():
    """Valid-horizon waypoints must be INVARIANT to the post-horizon path
    condition. frame_valid masking + per-block re-mask stop the dilated frame
    convs (kernel/bias) from leaking the future path condition into boundary
    valid frames."""
    torch.manual_seed(0)
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                         max_tokens=8, min_tokens=2, decoder_type="path_cond",
                         decoder_width=48)
    model.eval()
    inputs = _make_inputs(model, B=1)
    N = 3
    veff = num_frames_for_tokens(N)
    dec = model.frame_decoder
    orig = dec._build_path_cond

    def perturbed(xz, pm, cnt):
        c = orig(xz, pm, cnt).clone()
        c[:, veff:] = c[:, veff:] + torch.randn_like(c[:, veff:]) * 50.0   # garbage past horizon
        return c

    with torch.no_grad():
        out1 = model(**inputs, num_tokens=torch.tensor([N]))["waypoints"]
        dec._build_path_cond = perturbed
        out2 = model(**inputs, num_tokens=torch.tensor([N]))["waypoints"]
    assert torch.allclose(out1[:, :veff], out2[:, :veff], atol=1e-6), (
        "post-horizon path condition leaked into valid frames"
    )


def test_output_is_5d_and_boundary_assembles_physical_7d():
    """Model emits NORMALIZED 5D = [x,y,z,cos,sin]; the 7D physical contract is
    assembled at the boundary by `build_physical_7d_from_normalized_5d`
    (unnormalize xyz → unit heading → derive deltas IN PHYSICAL SPACE).

    Locks: (a) model output is 5D; (b) heading channels are unit-norm;
    (c) the boundary helper produces a 7D where re-deriving deltas from its
    own [:5] reproduces the full 7D (i.e. deltas are an exact function of
    the physical xz + heading); (d) the boundary helper unnormalizes xyz
    when wp_mean/std are provided (no-op when None)."""
    from utils.motion_process import (
        append_traj_deltas_5d_to_7d,
        build_physical_7d_from_normalized_5d,
    )
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                        max_tokens=8, min_tokens=2)
    model.eval()
    with torch.no_grad():
        wp = model(**_make_inputs(model, B=2), num_tokens=torch.tensor([3, 5]))["waypoints"]
    # (a) 5D contract.
    assert wp.shape[-1] == 5
    # (b) heading unit-norm.
    norms = wp[..., 3:5].pow(2).sum(-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # (c) boundary helper: physical-7D [:5] re-derived gives back the full 7D,
    #     i.e. deltas are an exact function of physical xz + heading.
    phys7 = build_physical_7d_from_normalized_5d(wp, None, None)
    assert phys7.shape[-1] == 7
    rederived = append_traj_deltas_5d_to_7d(phys7[..., :5])
    assert torch.allclose(rederived, phys7, atol=1e-6)
    # frame-0 deltas exactly zero (no preceding frame).
    assert torch.count_nonzero(phys7[:, 0, 5:7]) == 0

    # (d) unnormalize path: xyz get *std + mean, cos/sin pass through.
    wp_mean = torch.tensor([10.0, 0.5, -3.0, 0.0, 0.0, 0.0, 0.0])
    wp_std = torch.tensor([2.0, 0.5, 4.0, 1.0, 1.0, 1.0, 1.0])
    norm_idx = torch.tensor([0, 1, 2])
    phys = build_physical_7d_from_normalized_5d(wp, wp_mean, wp_std, norm_idx)
    expected_xyz = wp[..., :3] * wp_std[:3] + wp_mean[:3]
    assert torch.allclose(phys[..., :3], expected_xyz, atol=1e-6)
    assert torch.allclose(phys[..., 3:5], wp[..., 3:5], atol=1e-6)  # cos/sin untouched


def test_path_cond_tangent_is_unit_or_zero():
    """Tangent channels [2:4] are eps-safe unit vectors on valid frames (never NaN)."""
    dec = PathCondFrameDecoder(d_model=32, max_tokens=8, n_path=16, width=48,
                                token_res_depth=1, frame_res_depth=2)
    xz = torch.randn(1, 16, 2)
    cond = dec._build_path_cond(xz, torch.ones(1, 16, dtype=torch.bool), torch.tensor([8]))
    tan = cond[..., 2:4]
    assert torch.isfinite(tan).all()
    norms = tan.pow(2).sum(-1)
    # unit (valid tangent) or ~0 (degenerate segment) — never NaN/Inf.
    assert ((norms - 1.0).abs() < 1e-4).logical_or(norms < 1e-4).all()


def test_invalid_tail_token_hidden_zeroed_before_decoder():
    """Plan tokens past used_num_tokens must be ZEROED before the Conv1d frame
    decoder — otherwise the conv (kernel 3) mixes their garbage hiddens into the
    boundary VALID waypoints (cross-token leak)."""
    model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                          max_tokens=8, min_tokens=2)
    model.train()
    captured = {}
    model.frame_decoder.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("x", args[0].detach().clone())
    )
    nt = torch.tensor([3, 5])   # chosen horizons
    model(**_make_inputs(model, B=2), num_tokens=nt)
    x = captured["x"]           # [2, max_tokens, D] = plan_token_hidden fed to decoder
    for b, n in enumerate(nt.tolist()):
        assert torch.count_nonzero(x[b, n:]) == 0, f"sample {b}: tail (>= {n}) not zeroed"
        assert torch.count_nonzero(x[b, :n]) > 0, f"sample {b}: valid head wrongly zeroed"


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
    inputs_perturbed["history_motion"][:, :-1] = torch.randn(
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
    """Perturbing the padded path region:
    - num_token_logits is invariant for BOTH decoders (Stage-1 cond_transformer
      key-masks the padded path tokens).
    - waypoints are invariant for the SIMPLE decoder (it never reads path), but
      the path_cond decoder INTENTIONALLY consumes the full path geometry at
      Stage 3, so its waypoints are NOT asserted invariant here. (In real data
      path_valid_mask is all-True for a valid path or whole-degenerate; a partial
      mask is synthetic — the decoder only zeroes the cond when valid points < 2.)"""
    for dt, waypoints_invariant in (("simple", True), ("path_cond", False)):
        torch.manual_seed(1)
        model = RootRefiner(d_model=64, n_layers=2, n_heads=4, ff_dim=128,
                             max_tokens=8, min_tokens=2, decoder_type=dt, decoder_width=48)
        model.eval()
        B = 2
        inputs = _make_inputs(model, B=B)
        half = model.n_path // 2
        inputs["path_valid_mask"] = torch.zeros(B, model.n_path, dtype=torch.bool)
        inputs["path_valid_mask"][:, :half] = True
        inputs_perturbed = {k: v.clone() if isinstance(v, torch.Tensor) else v
                             for k, v in inputs.items()}
        inputs_perturbed["path"][:, half:] = torch.randn(
            B, model.n_path - half, 2, generator=torch.Generator().manual_seed(123),
        ) * 100.0
        with torch.no_grad():
            out_a = model(**inputs)
            out_b = model(**inputs_perturbed)
        assert torch.allclose(
            out_a["num_token_logits"], out_b["num_token_logits"], atol=1e-5), dt
        if waypoints_invariant:
            assert torch.allclose(out_a["waypoints"], out_b["waypoints"], atol=1e-5), dt


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
    inputs["path_valid_mask"] = torch.zeros(B, model.n_path, dtype=torch.bool)
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
        text_emb_dim=16, path_features_dim=5, dropout=0.0,
    )
    model.train()
    B = 4
    K = model.max_tokens - model.min_tokens + 1
    F_max = model.max_frames

    g = torch.Generator().manual_seed(7)
    # Fixed batch (will be reused every step — true overfit setup).
    text_emb = torch.randn(B, model.text_emb_dim, generator=g)
    path = torch.randn(B, model.n_path, 2, generator=g)
    path_valid_mask = torch.ones(B, model.n_path, dtype=torch.bool)
    path_features = torch.randn(B, model.path_features_dim, generator=g)
    history_motion = torch.randn(B, model.n_hist, 5, generator=g)
    history_mask = torch.ones(B, model.n_hist, dtype=torch.bool)

    # Random but fixed targets — built as 5D (same as model output) so the
    # overfit loss is exactly what `train_refiner._compute_loss` does in
    # same-space derivation mode.
    target_num_tokens_class = torch.randint(0, K, (B,), generator=g)
    target_num_tokens = target_num_tokens_class + model.min_tokens   # teacher-forced horizon
    target_yaw = torch.randn(B, F_max, generator=g) * 0.5
    target_waypoints = torch.zeros(B, F_max, 5)
    target_waypoints[..., :3] = torch.randn(B, F_max, 3, generator=g) * 0.3
    target_waypoints[..., 3] = torch.cos(target_yaw)
    target_waypoints[..., 4] = torch.sin(target_yaw)
    # (target_mask is intentionally all-True so we skip mask gating in this
    # overfit test; the dataset-level test_T06 already locks the strict
    # target_mask.sum() == num_frames_for_tokens(num_tokens) relationship.)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    initial_losses = []
    final_losses = []

    n_steps = 200
    for step in range(n_steps):
        out = model(
            text_emb=text_emb, path=path, path_valid_mask=path_valid_mask,
            path_features=path_features, history_motion=history_motion,
            history_mask=history_mask, num_tokens=target_num_tokens,
        )
        # Loss: num_token CE + xyz SmoothL1 + heading cosine + same-space
        # derived fwd_delta / yaw_delta (channels 5, 6 of the 7D produced by
        # `append_traj_deltas_5d_to_7d` from the same-space 5D).
        from utils.motion_process import append_traj_deltas_5d_to_7d
        l_num = F.cross_entropy(out["num_token_logits"], target_num_tokens_class)
        l_xyz = F.smooth_l1_loss(
            out["waypoints"][..., :3], target_waypoints[..., :3],
        )
        cos_term = (out["waypoints"][..., 3:5] * target_waypoints[..., 3:5]).sum(-1)
        l_heading = (1.0 - cos_term).mean()
        pred7 = append_traj_deltas_5d_to_7d(out["waypoints"])
        gt7 = append_traj_deltas_5d_to_7d(target_waypoints)
        l_fwd = F.smooth_l1_loss(pred7[..., 5], gt7[..., 5])
        l_yaw = F.smooth_l1_loss(pred7[..., 6], gt7[..., 6])
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
            text_emb=text_emb, path=path, path_valid_mask=path_valid_mask,
            path_features=path_features, history_motion=history_motion,
            history_mask=history_mask,
        )["num_token_logits"].argmax(dim=-1)
    acc = (pred_class == target_num_tokens_class).float().mean().item()
    assert acc >= 0.5, (
        f"num_token overfit accuracy too low: {acc:.2f} (>= 0.5 expected)"
    )
