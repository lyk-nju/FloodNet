from __future__ import annotations

import torch
from torch import nn

from utils.training.noise_initializer.context_builder import NoiseInitializerContext
from utils.training.noise_initializer.lightning_module import (
    NoiseInitializerLightningModule,
)
from utils.training.noise_initializer.losses import anchored_root_xz_loss
from utils.training.noise_initializer.shadow_rollout import clip_delta_to_base_norm


class _FakeInitializer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, **kwargs):
        return torch.ones_like(kwargs["frontier_base_zT"]) * self.scale


class _FakeFrozenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.5))
        self.generated = torch.arange(1 * 2 * 8, dtype=torch.float32).view(1, 2, 8, 1, 1)
        self.commit_index = 2
        self.current_step = 5


class _FakeFrozenVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))


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
    return model.generated[:, :, 5 : 5 + rollout_tokens, 0, 0].permute(0, 2, 1) * model.weight


def _decode_latents_fn(vae, latents, **kwargs):
    del kwargs
    return latents[0, :, :2] * vae.weight.detach()


def test_noise_initializer_lightning_optimizer_updates_initializer_only():
    initializer = _FakeInitializer()
    model = _FakeFrozenModel()
    vae = _FakeFrozenVae()
    module = NoiseInitializerLightningModule(
        cfg={"optimizer": {"lr": 1e-3}},
        initializer=initializer,
        ldf_model=model,
        vae=vae,
        rollout_fn=_rollout_fn,
        decode_latents_fn=_decode_latents_fn,
    )

    optimizer = module.configure_optimizers()
    optimized_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}

    assert optimized_ids == {id(initializer.scale)}
    assert id(model.weight) not in optimized_ids
    assert id(vae.weight) not in optimized_ids


def test_noise_initializer_lightning_training_step_backprops_to_initializer():
    initializer = _FakeInitializer()
    model = _FakeFrozenModel()
    vae = _FakeFrozenVae()
    module = NoiseInitializerLightningModule(
        cfg={
            "optimizer": {"lr": 1e-3},
            "loss": {"lambda_vel": 0.0, "lambda_delta": 0.01},
            "rollout": {"alpha": 1.0, "loss_horizon_tokens": 2},
        },
        initializer=initializer,
        ldf_model=model,
        vae=vae,
        rollout_fn=_rollout_fn,
        decode_latents_fn=_decode_latents_fn,
    )
    batch = {
        "context": _context(model),
        "target_xz": torch.zeros(2, 2),
        "target_mask": torch.ones(2),
        "history_frames": 0,
        "first_chunk": False,
    }

    loss = module.training_step(batch, 0)
    loss.backward()

    assert loss.requires_grad
    assert initializer.scale.grad is not None
    assert model.weight.grad is None
    assert vae.weight.grad is None


def test_noise_initializer_lightning_delta_regularization_uses_raw_delta_before_clip():
    initializer = _FakeInitializer()
    initializer.scale.data.fill_(10.0)
    model = _FakeFrozenModel()
    vae = _FakeFrozenVae()
    module = NoiseInitializerLightningModule(
        cfg={
            "optimizer": {"lr": 1e-3},
            "loss": {"lambda_vel": 0.0, "lambda_delta": 0.01},
            "rollout": {
                "alpha": 1.0,
                "loss_horizon_tokens": 2,
                "max_delta_norm_ratio": 0.1,
            },
        },
        initializer=initializer,
        ldf_model=model,
        vae=vae,
        rollout_fn=_rollout_fn,
        decode_latents_fn=_decode_latents_fn,
    )
    batch = {
        "context": _context(model),
        "target_xz": torch.zeros(2, 2),
        "target_mask": torch.ones(2),
        "history_frames": 0,
        "first_chunk": False,
    }

    loss = module.training_step(batch, 0)

    raw_delta = torch.ones_like(batch["context"].frontier_base_zT) * initializer.scale.detach()
    clipped_delta, _ = clip_delta_to_base_norm(
        raw_delta,
        batch["context"].frontier_base_zT,
        max_delta_norm_ratio=0.1,
    )
    zt = batch["context"].frontier_base_zT + clipped_delta
    pred_xz = (zt * model.weight.detach() * vae.weight.detach())[0]
    traj_loss, _ = anchored_root_xz_loss(
        pred_xz,
        batch["target_xz"],
        batch["target_mask"],
        history_frames=0,
        lambda_vel=0.0,
    )
    expected = traj_loss + 0.01 * raw_delta.pow(2).mean()

    assert torch.allclose(loss.detach(), expected.detach())


def test_noise_initializer_lightning_passes_generated_anchor_xz_to_loss():
    initializer = _FakeInitializer()
    model = _FakeFrozenModel()
    vae = _FakeFrozenVae()
    module = NoiseInitializerLightningModule(
        cfg={
            "optimizer": {"lr": 1e-3},
            "loss": {
                "lambda_vel": 0.0,
                "lambda_delta": 0.0,
                "anchor_mode": "generated_anchor_abs",
            },
            "rollout": {"alpha": 1.0, "loss_horizon_tokens": 2},
        },
        initializer=initializer,
        ldf_model=model,
        vae=vae,
        rollout_fn=_rollout_fn,
        decode_latents_fn=_decode_latents_fn,
    )
    generated_anchor_xz = torch.tensor([100.0, 100.0])
    batch = {
        "context": _context(model),
        "target_xz": torch.zeros(2, 2),
        "target_mask": torch.ones(2),
        "history_frames": 0,
        "first_chunk": False,
        "generated_anchor_xz": generated_anchor_xz,
    }

    loss = module.training_step(batch, 0)

    zt = batch["context"].frontier_base_zT + initializer.scale.detach()
    pred_xz = (zt * model.weight.detach() * vae.weight.detach())[0]
    expected, _ = anchored_root_xz_loss(
        pred_xz,
        batch["target_xz"],
        batch["target_mask"],
        history_frames=0,
        lambda_vel=0.0,
        anchor_mode="generated_anchor_abs",
        generated_anchor_xz=generated_anchor_xz,
    )
    assert torch.allclose(loss.detach(), expected.detach())
