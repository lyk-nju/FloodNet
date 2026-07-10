from __future__ import annotations

import pytest
import torch

from models.noise_initializer import NoiseInitializer
from models.tools.traj_encoder import TrajectoryEncoder


def _inputs(batch_size: int = 2, latent_dim: int = 4, text_dim: int = 6):
    return {
        "history_latents": torch.randn(batch_size, 5, latent_dim),
        "active_latents": torch.randn(batch_size, 3, latent_dim),
        "active_beta": torch.rand(batch_size, 3),
        "active_offsets": torch.tensor([0, 1, 2]),
        "text_embedding": torch.randn(batch_size, text_dim),
        "traj_token_frames": torch.randn(batch_size, 7, 4, 7),
        "traj_frame_mask": torch.ones(batch_size, 7, 4),
        "frontier_offsets": torch.tensor([3, 4, 5]),
        "frontier_base_zT": torch.randn(batch_size, 3, latent_dim),
    }


def test_noise_initializer_outputs_frontier_delta_shape():
    model = NoiseInitializer(
        latent_dim=4,
        text_dim=6,
        frontier_tokens=3,
        hidden_dim=32,
        traj_encoder=TrajectoryEncoder(out_dim=8),
        traj_emb_dim=8,
    )

    delta = model(**_inputs())

    assert delta.shape == (2, 3, 4)


def test_noise_initializer_freezes_reused_traj_encoder_by_default():
    traj_encoder = TrajectoryEncoder(out_dim=8)
    model = NoiseInitializer(
        latent_dim=4,
        text_dim=6,
        frontier_tokens=3,
        hidden_dim=32,
        traj_encoder=traj_encoder,
        traj_emb_dim=8,
        freeze_traj_encoder=True,
        zero_init_output=False,
    )

    delta = model(**_inputs())
    loss = delta.pow(2).mean()
    loss.backward()

    assert all(not parameter.requires_grad for parameter in traj_encoder.parameters())
    assert all(parameter.grad is None for parameter in traj_encoder.parameters())
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("traj_encoder.")
    )


def test_noise_initializer_trains_new_traj_encoder_by_default():
    model = NoiseInitializer(
        latent_dim=4,
        text_dim=6,
        frontier_tokens=3,
        hidden_dim=32,
        traj_emb_dim=8,
        zero_init_output=False,
    )

    assert any(parameter.requires_grad for parameter in model.traj_encoder.parameters())


def test_noise_initializer_zero_init_starts_as_noop_delta():
    model = NoiseInitializer(
        latent_dim=4,
        text_dim=6,
        frontier_tokens=3,
        hidden_dim=32,
        traj_encoder=TrajectoryEncoder(out_dim=8),
        traj_emb_dim=8,
        zero_init_output=True,
    )

    delta = model(**_inputs())

    assert torch.equal(delta, torch.zeros_like(delta))


def test_noise_initializer_accepts_unbatched_active_beta():
    model = NoiseInitializer(
        latent_dim=4,
        text_dim=6,
        frontier_tokens=3,
        hidden_dim=32,
        traj_encoder=TrajectoryEncoder(out_dim=8),
        traj_emb_dim=8,
    )
    inputs = _inputs()
    inputs["active_beta"] = torch.rand(3)

    delta = model(**inputs)

    assert delta.shape == (2, 3, 4)


def test_noise_initializer_rejects_wrong_frontier_offset_count():
    model = NoiseInitializer(
        latent_dim=4,
        text_dim=6,
        frontier_tokens=3,
        hidden_dim=32,
        traj_encoder=TrajectoryEncoder(out_dim=8),
        traj_emb_dim=8,
    )
    inputs = _inputs()
    inputs["frontier_offsets"] = torch.tensor([3, 4])

    with pytest.raises(ValueError, match="frontier_offsets"):
        model(**inputs)


def test_noise_initializer_output_depends_on_frontier_base_zT():
    torch.manual_seed(7)
    model = NoiseInitializer(
        latent_dim=4,
        text_dim=6,
        frontier_tokens=3,
        hidden_dim=32,
        traj_encoder=TrajectoryEncoder(out_dim=8),
        traj_emb_dim=8,
        zero_init_output=False,
    ).eval()
    inputs = _inputs(batch_size=1)
    shifted = dict(inputs)
    shifted["frontier_base_zT"] = inputs["frontier_base_zT"] + 2.0

    delta_a = model(**inputs)
    delta_b = model(**shifted)

    assert not torch.allclose(delta_a, delta_b)


def test_noise_initializer_preserves_trajectory_token_order():
    torch.manual_seed(11)
    model = NoiseInitializer(
        latent_dim=4,
        text_dim=6,
        frontier_tokens=3,
        hidden_dim=32,
        traj_encoder=TrajectoryEncoder(out_dim=8),
        traj_emb_dim=8,
        zero_init_output=False,
    ).eval()
    inputs = _inputs(batch_size=1)
    reversed_inputs = dict(inputs)
    reversed_inputs["traj_token_frames"] = inputs["traj_token_frames"].flip(1)
    reversed_inputs["traj_frame_mask"] = inputs["traj_frame_mask"].flip(1)

    delta_forward = model(**inputs)
    delta_reversed = model(**reversed_inputs)

    assert not torch.allclose(delta_forward, delta_reversed)
