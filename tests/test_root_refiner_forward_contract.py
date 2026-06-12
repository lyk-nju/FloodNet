from __future__ import annotations

import torch
import pytest

from models.root_refiner import PathCondFrameDecoder, RootRefiner


def _model() -> RootRefiner:
    return RootRefiner(
        d_model=64,
        n_layers=2,
        n_layers_cond=1,
        n_layers_token=1,
        n_heads=4,
        ff_dim=128,
        dropout=0.0,
        max_tokens=8,
        min_tokens=2,
        n_path=16,
        n_hist=8,
        text_emb_dim=32,
        path_features_dim=5,
        decoder_type="simple",
    )


def _inputs(model: RootRefiner, batch_size: int = 3) -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "text_emb": torch.randn(batch_size, model.text_emb_dim, generator=g),
        "path": torch.randn(batch_size, model.n_path, 2, generator=g),
        "path_valid_mask": torch.ones(batch_size, model.n_path, dtype=torch.bool),
        "path_control_mask": torch.ones(batch_size, model.n_path, dtype=torch.bool),
        "path_features": torch.randn(batch_size, 5, generator=g),
        "path_features_raw": torch.rand(batch_size, 5, generator=g) + 1.0,
        "history_motion": torch.randn(batch_size, model.n_hist, 5, generator=g),
        "history_mask": torch.ones(batch_size, model.n_hist, dtype=torch.bool),
        "sample_mode": ["full", "sliding", "full"][:batch_size],
    }


def test_root_refiner_accepts_new_forward_contract_and_returns_used_tokens():
    model = _model()
    inputs = _inputs(model)
    num_tokens = torch.tensor([2, 4, 8])

    out = model(**inputs, num_tokens=num_tokens)

    assert out["num_token_logits"].shape == (3, model.max_tokens - model.min_tokens + 1)
    assert out["expected_num_tokens_cls"].shape == (3,)
    assert out["pred_num_tokens_cls"].shape == (3,)
    assert out["pred_log_pace"].shape == (3,)
    assert out["pred_num_tokens_pace"].shape == (3,)
    assert torch.equal(out["used_num_tokens"], num_tokens)
    assert torch.equal(out["pred_num_tokens"], out["pred_num_tokens_pace"])
    assert out["pred_num_tokens"].shape == (3,)
    assert out["waypoints"].shape == (3, model.max_frames, 5)


def test_root_refiner_rejects_legacy_forward_aliases():
    model = _model()
    inputs = _inputs(model)
    legacy_inputs = {
        "text_emb": inputs["text_emb"],
        "xz_path": inputs["path"],
        "path_mask": inputs["path_valid_mask"],
        "path_stats": inputs["path_features"],
        "current_motion": inputs["history_motion"],
        "history_mask": inputs["history_mask"],
    }

    with pytest.raises(TypeError, match="unexpected keyword argument 'xz_path'"):
        model(**legacy_inputs)


def test_root_refiner_inference_uses_predicted_duration():
    model = _model()
    inputs = _inputs(model)

    out = model(**inputs)

    assert torch.equal(out["used_num_tokens"], out["pred_num_tokens_pace"])
    assert torch.equal(out["pred_num_tokens"], out["pred_num_tokens_pace"])
    assert out["used_num_tokens"].min() >= model.min_tokens
    assert out["used_num_tokens"].max() <= model.max_tokens


def test_path_control_mask_changes_condition_encoding():
    model = _model().eval()
    inputs = _inputs(model)
    inputs["path_control_mask"] = torch.zeros_like(inputs["path_control_mask"])
    out_without_controls = model(**inputs)

    inputs["path_control_mask"] = torch.ones_like(inputs["path_control_mask"])
    out_with_controls = model(**inputs)

    assert not torch.allclose(
        out_without_controls["num_token_logits"],
        out_with_controls["num_token_logits"],
    )


def test_root_refiner_rejects_unknown_path_mode():
    model = _model()
    inputs = _inputs(model)
    inputs["path_mode"] = ["dense_path", "unknown_mode", "goal_point"]

    with pytest.raises(ValueError, match="unknown path_mode"):
        model(**inputs)


def test_path_cond_frame_decoder_accepts_new_path_condition_names():
    decoder = PathCondFrameDecoder(d_model=16, max_tokens=4, n_path=8, width=24)
    token_hidden = torch.randn(2, 4, 16)
    path = torch.randn(2, 8, 2)
    path_valid_mask = torch.ones(2, 8, dtype=torch.bool)
    used_num_tokens = torch.tensor([2, 4])

    out = decoder(
        token_hidden,
        path=path,
        path_valid_mask=path_valid_mask,
        used_num_tokens=used_num_tokens,
    )

    assert out.shape == (2, decoder.max_frames, 5)


def test_path_cond_frame_decoder_offsets_path_hint_prefix():
    decoder = PathCondFrameDecoder(d_model=16, max_tokens=4, n_path=8, width=24)
    path = torch.zeros(1, 8, 2)
    path[0, :, 0] = torch.linspace(10.0, 17.0, 8)
    path_valid_mask = torch.ones(1, 8, dtype=torch.bool)
    used_num_tokens = torch.tensor([4])
    offset = torch.tensor([5])

    cond = decoder._build_path_cond(
        path,
        path_valid_mask,
        used_num_tokens,
        offset_start_frames=offset,
    )

    assert torch.allclose(cond[0, :5], torch.zeros_like(cond[0, :5]))
    assert torch.allclose(cond[0, 5, :2], path[0, 0], atol=1e-5)


def test_pace_head_gets_clean_raw_path_features_skip():
    """Duration v2: pace logits must depend on path_features_raw through the raw
    physical feature branch, so the head sees an undiluted length summary."""
    model = _model()
    inputs = _inputs(model)
    inputs["path_features_raw"].requires_grad_(True)

    out = model(**inputs)
    out["pred_log_pace"].sum().backward()

    grads = [
        p.grad for p in model.pace_feature_proj.parameters() if p.grad is not None
    ]
    assert grads, "pace_feature_proj received no gradient — raw feature path not wired"
    assert any(g.abs().sum() > 0 for g in grads)


def test_sample_mode_changes_pace_head_not_condition_classifier():
    model = _model().eval()
    inputs = _inputs(model)
    inputs["sample_mode"] = ["full"] * inputs["text_emb"].shape[0]
    full_out = model(**inputs)
    inputs["sample_mode"] = ["sliding"] * inputs["text_emb"].shape[0]
    sliding_out = model(**inputs)

    assert not torch.allclose(full_out["pred_log_pace"], sliding_out["pred_log_pace"])
    assert torch.allclose(full_out["num_token_logits"], sliding_out["num_token_logits"])
