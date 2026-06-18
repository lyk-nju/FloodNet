from __future__ import annotations

import inspect
import torch
import pytest

from models.root_refiner import RootDurationHead, RootRefiner


def _model() -> RootRefiner:
    return RootRefiner(
        d_model=64,
        n_layers=2,
        n_layers_cond=1,
        n_layers_root=1,
        n_heads=4,
        ff_dim=128,
        dropout=0.0,
        max_frames=24,
        min_frames=2,
        n_path=16,
        n_hist=8,
        text_emb_dim=32,
        path_features_dim=5,
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
    }


def test_root_refiner_accepts_frame_forward_contract_and_returns_future_waypoints():
    model = _model()
    inputs = _inputs(model)
    num_frames = torch.tensor([2, 12, 24])

    out = model(**inputs, num_frames=num_frames)

    assert out["pred_log_pace"].shape == (3,)
    assert out["pred_frames_float"].shape == (3,)
    effective_length = (
        inputs["path_features_raw"][:, 0].clamp_min(0.0)
        + inputs["path_features_raw"][:, 3].clamp_min(0.0)
    )
    expected_float = out["pred_log_pace"].clamp(-8.0, 8.0).exp() * effective_length
    assert torch.allclose(out["pred_frames_float"], expected_float)
    assert torch.equal(out["used_frames"], num_frames)
    assert torch.equal(out["pred_frames"], out["pred_frames_pace"])
    assert out["pred_frames"].shape == (3,)
    assert out["future_waypoints"].shape == (3, model.max_frames, 5)
    assert out["waypoints"].shape == (3, model.max_frames, 5)
    assert torch.equal(out["frame_mask"], torch.arange(model.max_frames)[None] < num_frames[:, None])
    assert "used_num_tokens" not in out


def test_root_duration_head_is_explicit_module_and_outputs_frame_duration():
    g = torch.Generator().manual_seed(1)
    head = RootDurationHead(
        d_model=64,
        path_features_dim=5,
        min_frames=2,
        max_frames=24,
        dropout=0.0,
        pace_text_dim=16,
    )
    inputs = {
        "cls_summary": torch.randn(3, 64, generator=g),
        "path_summary": torch.randn(3, 64, generator=g),
        "history_summary": torch.randn(3, 64, generator=g),
        "text_summary": torch.randn(3, 64, generator=g),
        "path_features": torch.randn(3, 5, generator=g),
        "path_features_raw": torch.rand(3, 5, generator=g) + 1.0,
    }

    out = head(**inputs)

    assert out["pred_log_pace"].shape == (3,)
    assert out["pred_frames_float"].shape == (3,)
    effective_length = (
        inputs["path_features_raw"][:, 0].clamp_min(0.0)
        + inputs["path_features_raw"][:, 3].clamp_min(0.0)
    )
    expected_float = out["pred_log_pace"].clamp(-8.0, 8.0).exp() * effective_length
    assert torch.allclose(out["pred_frames_float"], expected_float)
    assert torch.equal(out["pred_frames"], out["pred_frames_pace"])
    assert out["pred_frames"].min() >= 2
    assert out["pred_frames"].max() <= 24


def test_root_refiner_uses_explicit_duration_head():
    model = _model()

    assert isinstance(model.duration_head, RootDurationHead)
    assert not hasattr(model, "pace_head")


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

    assert torch.equal(out["used_frames"], out["pred_frames_pace"])
    assert torch.equal(out["pred_frames"], out["pred_frames_pace"])
    assert out["used_frames"].min() >= model.min_frames
    assert out["used_frames"].max() <= model.max_frames


def test_path_control_mask_changes_condition_encoding():
    model = _model().eval()
    inputs = _inputs(model)
    inputs["path_control_mask"] = torch.zeros_like(inputs["path_control_mask"])
    out_without_controls = model(**inputs)

    inputs["path_control_mask"] = torch.ones_like(inputs["path_control_mask"])
    out_with_controls = model(**inputs)

    assert not torch.allclose(
        out_without_controls["pred_log_pace"],
        out_with_controls["pred_log_pace"],
    )


def test_root_refiner_forward_contract_excludes_metadata_modes():
    signature = inspect.signature(RootRefiner.forward)

    assert "path_mode" not in signature.parameters
    assert "sample_mode" not in signature.parameters


def test_pace_head_gets_clean_raw_path_features_skip():
    """Duration v2: pace logits must depend on path_features_raw through the raw
    physical feature branch, so the head sees an undiluted length summary."""
    model = _model()
    inputs = _inputs(model)
    inputs["path_features_raw"].requires_grad_(True)

    out = model(**inputs)
    out["pred_log_pace"].sum().backward()

    grads = [
        p.grad for p in model.duration_head.raw_feature_proj.parameters()
        if p.grad is not None
    ]
    assert grads, "duration_head.raw_feature_proj received no gradient"
    assert any(g.abs().sum() > 0 for g in grads)

