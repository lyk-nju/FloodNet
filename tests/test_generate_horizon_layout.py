from __future__ import annotations

from types import SimpleNamespace

import torch

from models.diffusion_forcing_wan import DiffForcingWanModel
from utils.ldf_condition import LDFCondition


def _make_generate_harness():
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model._dummy_param = torch.nn.Parameter(torch.zeros(()))
    model.input_dim = 2
    model.chunk_size = 1
    model.noise_steps = 1
    model.time_embedding_scale = 1.0
    model.prediction_type = "vel"
    model.param_dtype = torch.float32
    model.recorded = SimpleNamespace(
        seq_lens=[],
        t_lens=[],
        noisy_lens=[],
        text_context_lens=[],
        traj_lens=[],
    )

    def get_noise_levels(device, seq_len, time_steps):
        return torch.zeros(time_steps.shape[0], seq_len, device=device)

    def denoise(
        noisy_input,
        t_scaled,
        text_cond_ctx,
        text_null_ctx,
        traj_emb,
        traj_seq_lens,
        seq_len,
        batch_size,
        traj_token_mask=None,
    ):
        model.recorded.seq_lens.append(int(seq_len))
        model.recorded.t_lens.append(int(t_scaled.shape[1]))
        model.recorded.noisy_lens.append(int(noisy_input[0].shape[1]))
        model.recorded.text_context_lens.append(len(text_cond_ctx))
        model.recorded.traj_lens.append(int(traj_emb.shape[1]))
        return [torch.zeros_like(noisy_input[0])]

    model._get_noise_levels = get_noise_levels
    model.preprocess = lambda x: x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    model.postprocess = lambda x: x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
    model._denoise_with_cfg = denoise
    return model


def test_generate_keeps_latent_seq_len_separate_from_future_traj():
    model = _make_generate_harness()
    feature_len = 3
    latent_len = feature_len + model.chunk_size
    traj_len = 5
    text = torch.zeros(1, 1)
    condition = LDFCondition(
        text_context=[text for _ in range(latent_len)],
        text_null_context=[text],
        traj_emb=torch.zeros(1, traj_len, 2),
        traj_seq_lens=torch.tensor([traj_len]),
        traj_token_mask=torch.ones(1, traj_len),
        seq_len=latent_len,
        attn_len=traj_len,
    )
    batch = {
        "feature_length": torch.tensor([feature_len]),
        "feature": torch.zeros(1, feature_len, 2),
        "text": ["walk"],
    }

    model.generate(batch, condition=condition, num_denoise_steps=1)

    assert model.recorded.traj_lens
    assert model.recorded.traj_lens == [traj_len] * len(model.recorded.traj_lens)
    assert model.recorded.seq_lens == [latent_len] * len(model.recorded.seq_lens)
    assert model.recorded.t_lens == [latent_len] * len(model.recorded.t_lens)
    assert model.recorded.text_context_lens == [latent_len] * len(
        model.recorded.text_context_lens
    )


def test_stream_generate_keeps_latent_seq_len_separate_from_future_traj():
    model = _make_generate_harness()
    feature_len = 3
    latent_len = feature_len + model.chunk_size
    traj_len = 5
    text = torch.zeros(1, 1)
    condition = LDFCondition(
        text_context=[text for _ in range(latent_len)],
        text_null_context=[text],
        traj_emb=torch.zeros(1, traj_len, 2),
        traj_seq_lens=torch.tensor([traj_len]),
        traj_token_mask=torch.ones(1, traj_len),
        seq_len=latent_len,
        attn_len=traj_len,
    )
    batch = {
        "feature_length": torch.tensor([feature_len]),
        "feature": torch.zeros(1, feature_len, 2),
        "text": ["walk"],
    }

    list(model.stream_generate(batch, condition=condition, num_denoise_steps=1))

    assert model.recorded.traj_lens
    assert model.recorded.traj_lens == [traj_len] * len(model.recorded.traj_lens)
    assert model.recorded.seq_lens == [latent_len] * len(model.recorded.seq_lens)
    assert model.recorded.t_lens == [latent_len] * len(model.recorded.t_lens)
    assert model.recorded.text_context_lens == [latent_len] * len(
        model.recorded.text_context_lens
    )
