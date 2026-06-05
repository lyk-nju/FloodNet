from __future__ import annotations

from types import SimpleNamespace

import torch

from models.diffusion_forcing_wan import DiffForcingWanModel


class _NoopTrajBuffer:
    def update(self, x, commit_index, device):
        pass


def _make_stream_step_harness(
    *,
    seq_len=4,
    chunk_size=1,
    num_denoise_steps=1,
    commit_index=0,
    current_step=0,
    generated_len=32,
):
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model._dummy_param = torch.nn.Parameter(torch.zeros(()))
    model.batch_size = 1
    model.seq_len = seq_len
    model.chunk_size = chunk_size
    model.num_denoise_steps = num_denoise_steps
    model.dt = 1.0 / float(num_denoise_steps)
    model.current_step = current_step
    model.commit_index = commit_index
    model.input_dim = 2
    model.generated = torch.zeros(1, 2, generated_len, 1, 1)
    model._traj_buf = _NoopTrajBuffer()
    model.use_text_cond = True
    model.param_dtype = torch.float32
    model.time_embedding_scale = 1.0
    model.prediction_type = "vel"
    model.text_condition_list = [[torch.zeros(1, 1) for _ in range(commit_index)]]
    model.recorded = SimpleNamespace(
        seq_lens=[],
        model_sls=[],
        window_starts=[],
        traj_lens=[],
        noisy_lens=[],
        t_lens=[],
        text_context_lens=[],
    )

    def encode_text_with_cache(text_list, device):
        return [torch.zeros(1, 1, device=device) for _ in text_list]

    def build_direct(x, model_sl, window_start_token, device, traj_sl=None):
        payload_start = int(x.get("traj_start_token", window_start_token))
        if payload_start > window_start_token:
            raise ValueError("payload starts after window")
        crop_tokens = max(0, window_start_token - payload_start)
        payload_tokens = int(x.get("traj_num_tokens", model_sl))
        out_tokens = max(model_sl, payload_tokens - crop_tokens)
        model.recorded.model_sls.append(int(model_sl))
        model.recorded.window_starts.append(int(window_start_token))
        model.recorded.traj_lens.append(int(out_tokens))
        return (
            torch.zeros(1, out_tokens, 2, device=device),
            torch.tensor([out_tokens], dtype=torch.long, device=device),
            torch.ones(1, out_tokens, device=device),
        )

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
        model.recorded.noisy_lens.append(int(noisy_input[0].shape[1]))
        model.recorded.t_lens.append(int(t_scaled.shape[1]))
        model.recorded.text_context_lens.append(len(text_cond_ctx))
        return [torch.zeros_like(noisy_input[0])]

    model.encode_text_with_cache = encode_text_with_cache
    model._build_stream_direct_traj_condition = build_direct
    model._denoise_with_cfg = denoise
    model.postprocess = lambda x: x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
    return model


def test_stream_generate_step_uses_future_traj_length_for_denoise_attention():
    model = _make_stream_step_harness()
    step_input = {
        "text": ["walk"],
        "traj_cond_7d_frame": torch.zeros(1, 12, 7),
        "traj_cond_frame_mask": torch.ones(1, 12),
        "traj_start_token": 0,
        "traj_abs_start_token": 0,
        "traj_num_tokens": 3,
    }

    model.stream_generate_step(step_input, first_chunk=True)

    assert model.recorded.seq_lens == [3]
    assert model.recorded.text_context_lens == [3]


def test_stream_generate_step_reuses_direct_7d_payload_across_chunk_substeps():
    model = _make_stream_step_harness(
        seq_len=30,
        chunk_size=5,
        num_denoise_steps=5,
        commit_index=40,
        current_step=40,
        generated_len=96,
    )
    step_input = {
        "text": ["walk"],
        "traj_cond_7d_frame": torch.zeros(1, 216, 7),
        "traj_cond_frame_mask": torch.ones(1, 216),
        "traj_start_token": 11,
        "traj_abs_start_token": 11,
        "traj_num_tokens": 54,
    }

    model.stream_generate_step(step_input, first_chunk=False)

    assert model.recorded.window_starts == [11, 12, 13, 14, 15]
    assert model.recorded.model_sls == [30, 30, 30, 30, 30]
    assert model.recorded.noisy_lens == [30, 30, 30, 30, 30]
    assert model.recorded.traj_lens == [54, 53, 52, 51, 50]
    assert model.recorded.seq_lens == model.recorded.traj_lens
    assert model.recorded.t_lens == model.recorded.traj_lens
    assert model.recorded.text_context_lens == model.recorded.traj_lens
