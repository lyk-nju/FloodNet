"""Unit tests for the frame-based RootRefiner model contract."""

from __future__ import annotations

import torch

from models.root_refiner import RootDurationHead, RootPlanDecoder, RootRefiner


def _model() -> RootRefiner:
    return RootRefiner(
        d_model=64,
        n_layers=2,
        n_layers_cond=1,
        n_layers_root=1,
        n_heads=4,
        ff_dim=128,
        dropout=0.0,
        max_frames=29,
        min_frames=5,
        n_path=16,
        n_hist=8,
        text_emb_dim=16,
        path_features_dim=5,
    )


def _make_inputs(model: RootRefiner, batch_size: int = 2, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "text_emb": torch.randn(batch_size, model.text_emb_dim, generator=g),
        "path": torch.randn(batch_size, model.n_path, 2, generator=g),
        "path_valid_mask": torch.ones(batch_size, model.n_path, dtype=torch.bool),
        "path_control_mask": torch.ones(batch_size, model.n_path, dtype=torch.bool),
        "path_features": torch.randn(batch_size, model.path_features_dim, generator=g),
        "path_features_raw": torch.rand(batch_size, model.path_features_dim, generator=g) + 1.0,
        "history_motion": torch.randn(batch_size, model.n_hist, 5, generator=g),
        "history_mask": torch.ones(batch_size, model.n_hist, dtype=torch.bool),
    }


def test_forward_shape_and_frame_duration_contract():
    model = _model()
    num_frames = torch.tensor([5, 17])

    out = model(**_make_inputs(model), num_frames=num_frames)

    assert isinstance(model.duration_head, RootDurationHead)
    assert isinstance(model.root_decoder, RootPlanDecoder)
    assert torch.equal(out["used_frames"], num_frames)
    assert out["pred_frames"].shape == (2,)
    assert out["pred_frames_float"].shape == (2,)
    assert out["waypoints"].shape == (2, model.max_frames, 5)
    assert out["future_waypoints"].shape == (2, model.max_frames, 5)
    assert torch.equal(
        out["frame_mask"],
        torch.arange(model.max_frames).unsqueeze(0) < num_frames.unsqueeze(1),
    )
    assert "num_token_logits" not in out
    assert "used_num_tokens" not in out


def test_inference_uses_predicted_frame_count_in_range():
    model = _model().eval()
    out = model(**_make_inputs(model, batch_size=4))

    assert torch.equal(out["used_frames"], out["pred_frames"])
    assert int(out["pred_frames"].min()) >= model.min_frames
    assert int(out["pred_frames"].max()) <= model.max_frames


def test_heading_is_unit_norm_on_valid_frames_and_zeroed_after_horizon():
    model = _model().eval()
    num_frames = torch.tensor([5, 17])
    out = model(**_make_inputs(model), num_frames=num_frames)

    valid = out["frame_mask"]
    heading_norm = out["waypoints"][..., 3:5].pow(2).sum(-1)
    assert torch.allclose(
        heading_norm[valid],
        torch.ones_like(heading_norm[valid]),
        atol=1e-5,
    )
    assert torch.count_nonzero(out["waypoints"][~valid]) == 0


def test_history_padding_does_not_leak_into_output():
    model = _model().eval()
    inputs = _make_inputs(model)
    inputs["history_mask"] = torch.zeros(2, model.n_hist, dtype=torch.bool)
    inputs["history_mask"][:, -1] = True

    perturbed = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    perturbed["history_motion"][:, :-1] = torch.randn(
        2,
        model.n_hist - 1,
        5,
        generator=torch.Generator().manual_seed(99),
    ) * 100.0

    with torch.no_grad():
        out_a = model(**inputs, num_frames=torch.tensor([17, 17]))
        out_b = model(**perturbed, num_frames=torch.tensor([17, 17]))

    assert torch.allclose(out_a["waypoints"], out_b["waypoints"], atol=1e-5)
    assert torch.allclose(out_a["pred_log_pace"], out_b["pred_log_pace"], atol=1e-5)


def test_backward_produces_finite_gradients():
    model = _model()
    out = model(**_make_inputs(model), num_frames=torch.tensor([17, 21]))
    loss = out["pred_log_pace"].sum() + out["waypoints"].sum()
    loss.backward()

    bad = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and not torch.isfinite(param.grad).all()
    ]
    assert not bad


def test_default_model_under_parameter_budget():
    n_params = RootRefiner().count_parameters()

    assert n_params < 50_000_000
    assert n_params > 3_000_000
