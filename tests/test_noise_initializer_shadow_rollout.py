from __future__ import annotations

import torch
from torch import nn

from utils.training.noise_initializer.context_builder import NoiseInitializerContext
from utils.training.noise_initializer.shadow_rollout import (
    run_residual_shadow_rollout,
)


class _FakeInitializer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, **kwargs):
        return torch.ones_like(kwargs["frontier_base_zT"]) * self.scale


class _FakeFrozenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.generated = torch.arange(1 * 2 * 8, dtype=torch.float32).view(1, 2, 8, 1, 1)
        self.commit_index = 2
        self.current_step = 5


class _FakeFrozenVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(3.0))
        self.cache_restored = False

    def snapshot_cache(self):
        return {"cache": "snapshot"}

    def restore_cache(self, state):
        self.cache_restored = state == {"cache": "snapshot"}


def _context(model: _FakeFrozenModel) -> NoiseInitializerContext:
    return NoiseInitializerContext(
        history_latents=torch.zeros(1, 2, 2),
        active_latents=torch.zeros(1, 1, 2),
        active_beta=torch.tensor([0.5]),
        active_offsets=torch.tensor([0]),
        text_embedding=torch.zeros(1, 4),
        traj_token_frames=torch.zeros(1, 2, 4, 7),
        traj_frame_mask=torch.ones(1, 2, 4),
        frontier_offsets=torch.tensor([3, 4]),
        frontier_base_zT=(
            model.generated[:, :, torch.tensor([5, 6]), 0, 0]
            .permute(0, 2, 1)
            .contiguous()
        ),
        frontier_ids=torch.tensor([5, 6]),
    )


def _rollout_fn(model, *, rollout_tokens: int, first_chunk: bool):
    del first_chunk
    frontier = model.generated[:, :, 5 : 5 + rollout_tokens, 0, 0].permute(0, 2, 1)
    model.commit_index += int(rollout_tokens)
    model.current_step += int(rollout_tokens)
    return frontier * model.weight


def test_shadow_rollout_injects_only_frontier_and_restores_state():
    model = _FakeFrozenModel()
    vae = _FakeFrozenVae()
    initializer = _FakeInitializer()
    original_generated = model.generated.clone()

    result = run_residual_shadow_rollout(
        model=model,
        vae=vae,
        initializer=initializer,
        context=_context(model),
        rollout_fn=_rollout_fn,
        alpha=2.0,
        rollout_tokens=2,
        first_chunk=False,
    )

    assert torch.equal(model.generated, original_generated)
    assert model.commit_index == 2
    assert model.current_step == 5
    assert vae.cache_restored
    assert torch.equal(result.injected_generated[:, :, :5], original_generated[:, :, :5])
    assert torch.equal(result.injected_generated[:, :, 7:], original_generated[:, :, 7:])
    assert torch.equal(
        result.injected_generated[:, :, 5:7, 0, 0].permute(0, 2, 1),
        result.frontier_zT,
    )


def test_shadow_rollout_freezes_ldf_vae_and_backprops_to_initializer_only():
    model = _FakeFrozenModel()
    vae = _FakeFrozenVae()
    initializer = _FakeInitializer()

    result = run_residual_shadow_rollout(
        model=model,
        vae=vae,
        initializer=initializer,
        context=_context(model),
        rollout_fn=_rollout_fn,
        alpha=1.0,
        rollout_tokens=2,
    )
    loss = result.shadow_latents.pow(2).mean()
    loss.backward()

    assert initializer.scale.grad is not None
    assert model.weight.grad is None
    assert vae.weight.grad is None
    assert not model.weight.requires_grad
    assert not vae.weight.requires_grad
