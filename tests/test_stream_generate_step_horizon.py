from __future__ import annotations

import torch

from types import SimpleNamespace
from models.diffusion_forcing_wan import DiffForcingWanModel
from utils.conditions.ldf import LDFCondition


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
    model.traj_encoder = torch.nn.Identity()
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

    def build_direct(
        x,
        model_sl,
        window_start_token,
        device,
        *,
        batch_size,
        traj_encoder,
        traj_sl=None,
    ):
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
    model._patched_build_direct = build_direct
    model._denoise_with_cfg = denoise
    model.postprocess = lambda x: x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
    return model


def _condition_provider(model, step_input):
    def provider(*, end_index, model_sl, window_start_token, time_steps, device):
        traj_emb, traj_seq_lens, traj_token_mask = (
            model._patched_build_direct(
                step_input,
                model_sl,
                window_start_token,
                device,
                batch_size=1,
                traj_encoder=model.traj_encoder,
            )
        )
        attn_sl = max(model_sl, int(traj_emb.shape[1]))
        text = model.encode_text_with_cache(["walk"], device)[0]
        text_null = model.encode_text_with_cache([""], device)
        return LDFCondition(
            text_context=[text for _ in range(model_sl)],
            text_null_context=text_null,
            traj_emb=traj_emb,
            traj_seq_lens=traj_seq_lens,
            traj_token_mask=traj_token_mask,
            seq_len=model_sl,
            attn_len=attn_sl,
        )

    return provider


def test_stream_generate_step_keeps_latent_length_separate_from_future_traj():
    model = _make_stream_step_harness()
    step_input = {
        "text": ["walk"],
        "traj_cond_7d_frame": torch.zeros(1, 12, 7),
        "traj_cond_frame_mask": torch.ones(1, 12),
        "traj_start_token": 0,
        "traj_abs_start_token": 0,
        "traj_num_tokens": 3,
    }

    model.stream_generate_step(
        {},
        first_chunk=True,
        condition=_condition_provider(model, step_input),
    )

    assert model.recorded.model_sls == [1]
    assert model.recorded.traj_lens == [3]
    assert model.recorded.seq_lens == model.recorded.model_sls
    assert model.recorded.t_lens == model.recorded.model_sls
    assert model.recorded.text_context_lens == model.recorded.model_sls


def test_init_generated_accepts_stream_initial_buffer():
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model.noise_steps = 2
    model.chunk_size = 1
    model.input_dim = 2
    model.preprocess = lambda x: x.permute(0, 2, 1)[:, :, :, None, None]

    initial_generated = torch.arange(10, dtype=torch.float32).view(1, 5, 2)

    model.init_generated(
        2,
        batch_size=1,
        num_denoise_steps=2,
        initial_generated=initial_generated,
    )

    assert model.generated.requires_grad is False
    assert torch.equal(
        model.generated.squeeze(-1).squeeze(-1).permute(0, 2, 1),
        initial_generated,
    )


def test_stream_generate_step_can_backpropagate_to_initial_buffer():
    model = _make_stream_step_harness(
        seq_len=2,
        chunk_size=1,
        num_denoise_steps=2,
        generated_len=5,
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
        return [noisy_input[0] * 0.1]

    model._denoise_with_cfg = denoise
    initial_generated = torch.randn(1, 5, 2, requires_grad=True)
    model.generated = initial_generated.permute(0, 2, 1)[:, :, :, None, None]

    with torch.enable_grad():
        output = DiffForcingWanModel.stream_generate_step.__wrapped__(
            model,
            {},
            first_chunk=True,
            condition=_condition_provider(model, {"text": ["walk"]}),
        )
        loss = output["generated"].pow(2).sum()
        loss.backward()

    assert initial_generated.grad is not None
    assert torch.isfinite(initial_generated.grad).all()


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

    model.stream_generate_step(
        {},
        first_chunk=False,
        condition=_condition_provider(model, step_input),
    )

    assert model.recorded.window_starts == [11, 12, 13, 14, 15]
    assert model.recorded.model_sls == [30, 30, 30, 30, 30]
    assert model.recorded.noisy_lens == [30, 30, 30, 30, 30]
    assert model.recorded.traj_lens == [54, 53, 52, 51, 50]
    assert model.recorded.seq_lens == model.recorded.model_sls
    assert model.recorded.t_lens == model.recorded.model_sls
    assert model.recorded.text_context_lens == model.recorded.model_sls


def test_stream_generate_step_calls_projection_hook_after_denoise_update():
    model = _make_stream_step_harness(
        seq_len=4,
        chunk_size=2,
        num_denoise_steps=2,
        commit_index=0,
        current_step=0,
        generated_len=8,
    )

    def denoise(*args, **kwargs):
        return [torch.ones(2, 1, 1, 1)]

    model._denoise_with_cfg = denoise
    calls = []

    def projection_hook(**kwargs):
        calls.append(
            {
                "commit_index": kwargs["commit_index"],
                "start_index": kwargs["start_index"],
                "end_index": kwargs["end_index"],
                "current_step": kwargs["current_step"],
                "generated_value": float(
                    kwargs["model"].generated[0, 0, kwargs["start_index"], 0, 0]
                ),
            }
        )

    model.stream_generate_step(
        {},
        first_chunk=True,
        condition=_condition_provider(model, {"text": ["walk"]}),
        projection_callback=projection_hook,
    )

    assert calls == [
        {
            "commit_index": 0,
            "start_index": 0,
            "end_index": 1,
            "current_step": 0,
            "generated_value": 0.5,
        },
        {
            "commit_index": 0,
            "start_index": 0,
            "end_index": 2,
            "current_step": 1,
            "generated_value": 1.0,
        },
    ]


def test_stream_generate_step_projection_hook_receives_time_consistent_state():
    model = _make_stream_step_harness(
        seq_len=4,
        chunk_size=1,
        num_denoise_steps=10,
        commit_index=0,
        current_step=8,
        generated_len=8,
    )
    model.generated[0, :, 0, 0, 0] = torch.tensor([1.0, 2.0])

    def denoise(*args, **kwargs):
        return [torch.tensor([[[[0.5]]], [[[-0.5]]]])]

    model._denoise_with_cfg = denoise
    calls = []

    def projection_hook(**kwargs):
        calls.append(
            {
                "current_step": kwargs["current_step"],
                "x_before": kwargs["x_beta_before_update"].detach().clone(),
                "predicted_vel": kwargs["predicted_vel"].detach().clone(),
                "beta_before": kwargs["beta_before"].detach().clone(),
                "beta_after": kwargs["beta_after"].detach().clone(),
                "x_after": kwargs["x_after_velocity_update"].detach().clone(),
            }
        )

    model.stream_generate_step(
        {},
        first_chunk=False,
        condition=_condition_provider(model, {"text": ["walk"]}),
        projection_callback=projection_hook,
    )

    call = next(item for item in calls if item["current_step"] == 8)
    assert call["current_step"] == 8
    assert torch.allclose(call["x_before"][:, 0, 0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(call["predicted_vel"][:, 0, 0], torch.tensor([0.5, -0.5]))
    assert torch.isclose(call["beta_before"], torch.tensor(0.2))
    assert torch.isclose(call["beta_after"], torch.tensor(0.1))
    assert torch.allclose(call["x_after"][:, 0, 0], torch.tensor([1.05, 1.95]))
