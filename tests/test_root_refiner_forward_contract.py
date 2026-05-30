from __future__ import annotations

import torch

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
        "history_motion": torch.randn(batch_size, model.n_hist, 5, generator=g),
        "history_mask": torch.ones(batch_size, model.n_hist, dtype=torch.bool),
    }


def test_root_refiner_accepts_new_forward_contract_and_returns_used_tokens():
    model = _model()
    inputs = _inputs(model)
    num_tokens = torch.tensor([2, 4, 8])

    out = model(**inputs, num_tokens=num_tokens)

    assert out["num_token_logits"].shape == (3, model.max_tokens - model.min_tokens + 1)
    assert out["expected_num_tokens"].shape == (3,)
    assert torch.equal(out["used_num_tokens"], num_tokens)
    assert out["pred_num_tokens"].shape == (3,)
    assert out["waypoints"].shape == (3, model.max_frames, 5)


def test_root_refiner_inference_uses_predicted_duration():
    model = _model()
    inputs = _inputs(model)

    out = model(**inputs)

    assert torch.equal(out["used_num_tokens"], out["pred_num_tokens"])
    assert out["used_num_tokens"].min() >= model.min_tokens
    assert out["used_num_tokens"].max() <= model.max_tokens


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
