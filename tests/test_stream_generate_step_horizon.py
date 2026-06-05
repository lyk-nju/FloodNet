from __future__ import annotations

from types import SimpleNamespace

import torch

from models.diffusion_forcing_wan import DiffForcingWanModel


class _NoopTrajBuffer:
    def update(self, x, commit_index, device):
        pass


def _make_stream_step_harness():
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model._dummy_param = torch.nn.Parameter(torch.zeros(()))
    model.batch_size = 1
    model.seq_len = 4
    model.chunk_size = 1
    model.num_denoise_steps = 1
    model.dt = 1.0
    model.current_step = 0
    model.commit_index = 0
    model.input_dim = 2
    model.generated = torch.zeros(1, 2, 9, 1, 1)
    model._traj_buf = _NoopTrajBuffer()
    model.use_text_cond = True
    model.param_dtype = torch.float32
    model.time_embedding_scale = 1.0
    model.prediction_type = "vel"
    model.text_condition_list = [[]]
    model.recorded = SimpleNamespace(seq_lens=[])

    def encode_text_with_cache(text_list, device):
        return [torch.zeros(1, 1, device=device) for _ in text_list]

    def build_direct(x, model_sl, window_start_token, device, traj_sl=None):
        return (
            torch.zeros(1, 3, 2, device=device),
            torch.tensor([3], dtype=torch.long, device=device),
            torch.ones(1, 3, device=device),
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
        model.recorded.text_context_len = len(text_cond_ctx)
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
    assert model.recorded.text_context_len == 3
