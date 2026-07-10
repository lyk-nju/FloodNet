from __future__ import annotations

import torch
from torch import nn

from utils.inference.stream_runtime import KernelStepResult
from utils.inference.stream_generator import StreamGenerator


class _KernelLdf(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.batch_size = 1
        self.chunk_size = 1
        self.commit_index = 3
        self.current_step = 7
        self.latent_buffer_start_commit_abs = 32
        self.latent_buffer_epoch = 2
        self.num_denoise_steps = 10
        self.text_condition_list = [[]]

    def stream_generate_step(self, step_input, *, first_chunk, condition):
        self.seen_step_input = step_input
        self.seen_first_chunk = first_chunk
        self.seen_condition = condition
        self.commit_index += 1
        return {"generated": torch.tensor([[[1.0, 2.0]]])}


def test_kernel_consumes_exact_payload_without_route_or_timeline_access():
    model = _KernelLdf()
    generator = StreamGenerator(ldf_model=model, device="cpu")
    payload = {"traj_cond_7d_frame": torch.zeros(1, 5, 7)}
    generator.build_root_plan_stream_payload = lambda **_: (_ for _ in ()).throw(
        AssertionError("kernel must not build route payloads")
    )
    generator.build_ldf_condition_provider = lambda step_input, **_: (
        "provider",
        step_input,
    )

    result = generator.generate_token(
        "walk",
        payload,
        first_chunk=False,
        num_denoise_steps=12,
    )

    assert isinstance(result, KernelStepResult)
    assert result.actual_payload is payload
    assert result.raw_latent.shape == (1, 2)
    assert result.local_commit_before == 3
    assert result.local_commit_after == 4
    assert result.latent_buffer_start_commit_abs == 32
    assert result.latent_buffer_epoch == 2
    assert model.num_denoise_steps == 12
    assert model.seen_step_input["text"] == ["walk"]
    assert model.seen_step_input["traj_cond_7d_frame"] is payload["traj_cond_7d_frame"]


def test_kernel_rejects_multi_token_generation_result():
    model = _KernelLdf()
    generator = StreamGenerator(ldf_model=model, device="cpu")
    generator.build_ldf_condition_provider = lambda *_, **__: None

    def generate_many(*_, **__):
        model.commit_index += 2
        return {"generated": torch.zeros(1, 2, 2)}

    model.stream_generate_step = generate_many

    try:
        generator.generate_token("walk", None, first_chunk=True)
    except ValueError as error:
        assert "exactly one committed token" in str(error)
    else:
        raise AssertionError("multi-token kernel result must be rejected")


def test_kernel_normalizes_commit_indices_to_post_roll_buffer():
    model = _KernelLdf()
    model.commit_index = 3
    model.latent_buffer_start_commit_abs = 0
    model.latent_buffer_epoch = 0
    generator = StreamGenerator(ldf_model=model, device="cpu")
    generator.build_ldf_condition_provider = lambda *_, **__: None

    def generate_and_roll(*_, **__):
        model.commit_index = 2
        model.latent_buffer_start_commit_abs = 2
        model.latent_buffer_epoch = 1
        return {"generated": torch.zeros(1, 1, 2)}

    model.stream_generate_step = generate_and_roll
    result = generator.generate_token("walk", None, first_chunk=False)

    assert result.local_commit_before == 1
    assert result.local_commit_after == 2
    assert result.latent_buffer_start_commit_abs == 2
    assert result.latent_buffer_epoch == 1
