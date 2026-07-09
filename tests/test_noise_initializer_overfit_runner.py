from __future__ import annotations

import torch

from utils.training.noise_initializer.text_encoder import (
    resolve_noise_initializer_text_encoder,
)
from utils.training.noise_initializer.config_validate import (
    validate_noise_initializer_overfit_config,
)
from utils.training.noise_initializer.context_builder import NoiseInitializerContext
from utils.training.noise_initializer.overfit_runner import (
    _build_initializer_context_for_commit,
    apply_initializer_to_stream_state,
    encode_initializer_text_embedding,
    should_train_commit,
    slice_future_traj_frames,
)


class _FakeTextModel:
    def encode_text_with_cache(self, texts, device):
        assert texts == ["walk"]
        return [torch.arange(12, dtype=torch.float32, device=device).view(3, 4)]


class _FakeTextRollout:
    def get_text_for_commit_index(self, commit_index: int) -> str:
        del commit_index
        return "walk"


class _FakeStreamModel(_FakeTextModel):
    def __init__(self):
        self.generated = torch.arange(1 * 2 * 12, dtype=torch.float32).view(1, 2, 12, 1, 1)
        self.commit_index = 3
        self.current_step = 10
        self.dt = 0.1
        self.chunk_size = 5
        self.num_denoise_steps = 10


def test_encode_initializer_text_embedding_mean_pools_sequence_context():
    embedding = encode_initializer_text_embedding(
        _FakeTextModel(),
        "walk",
        torch.device("cpu"),
    )

    assert embedding.shape == (1, 4)
    assert torch.equal(embedding[0], torch.tensor([4.0, 5.0, 6.0, 7.0]))


def test_encode_initializer_text_embedding_prefers_precomputed_text_encoder(tmp_path):
    cache_path = tmp_path / "text_embeddings.pt"
    torch.save(
        {
            "embeddings": {
                "walk": torch.arange(12, dtype=torch.float32).view(3, 4),
            },
            "text_dim": 4,
        },
        cache_path,
    )
    text_encoder = resolve_noise_initializer_text_encoder(
        {
            "text_encoder": {
                "type": "precomputed_t5_pool",
                "precomputed_text_emb_path": str(cache_path),
                "pooling": "first",
            }
        },
        text_emb_dim=4,
    )

    embedding = encode_initializer_text_embedding(
        model=None,
        text="walk",
        device=torch.device("cpu"),
        text_encoder=text_encoder,
    )

    assert embedding.shape == (1, 4)
    assert torch.equal(embedding[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_should_train_commit_respects_optimize_every_tokens():
    assert should_train_commit(0, optimize_every_tokens=5)
    assert not should_train_commit(4, optimize_every_tokens=5)
    assert should_train_commit(5, optimize_every_tokens=5)
    assert should_train_commit(3, optimize_every_tokens=1)


def test_slice_future_traj_frames_uses_commit_frame_and_pads():
    traj = torch.arange(1 * 6 * 7, dtype=torch.float32).view(1, 6, 7)

    sliced = slice_future_traj_frames(
        traj,
        commit_index=1,
        traj_horizon_tokens=2,
        frames_per_token=4,
    )

    assert sliced.shape == (1, 9, 7)
    assert torch.equal(sliced[:, :2], traj[:, 4:6])
    assert torch.equal(sliced[:, 2:], torch.zeros(1, 7, 7))


def test_context_for_commit_prefers_runtime_traj_payload_over_sample_gt():
    model = _FakeStreamModel()
    sample_batch = {
        "traj_cond_7d": torch.zeros(1, 12, 7),
        "traj_cond_mask": torch.ones(1, 12),
    }
    runtime_payload = {
        "traj_cond_7d_frame": torch.full((1, 9, 7), 7.0),
        "traj_cond_frame_mask": torch.ones(1, 9),
    }

    context = _build_initializer_context_for_commit(
        model=model,
        sample_batch=sample_batch,
        text_rollout=_FakeTextRollout(),
        commit_index=3,
        device=torch.device("cpu"),
        text_encoder=None,
        history_tokens=2,
        frontier_tokens=2,
        traj_horizon_tokens=2,
        frames_per_token=4,
        beta_threshold=0.999,
        traj_payload=runtime_payload,
    )

    assert torch.equal(context.traj_token_frames, torch.full((1, 2, 4, 7), 7.0))


def test_apply_initializer_to_stream_state_updates_frontier_before_real_rollin():
    model = _FakeStreamModel()
    context = NoiseInitializerContext(
        history_latents=torch.zeros(1, 2, 2),
        active_latents=torch.zeros(1, 1, 2),
        active_beta=torch.tensor([0.5]),
        active_offsets=torch.tensor([0]),
        text_embedding=torch.zeros(1, 4),
        traj_token_frames=torch.zeros(1, 2, 4, 7),
        traj_frame_mask=torch.ones(1, 2, 4),
        frontier_offsets=torch.tensor([2, 3]),
        frontier_base_zT=(
            model.generated[:, :, torch.tensor([5, 6]), 0, 0]
            .permute(0, 2, 1)
            .contiguous()
        ),
        frontier_ids=torch.tensor([5, 6]),
    )
    original = model.generated.clone()

    class Initializer(torch.nn.Module):
        def forward(self, **kwargs):
            return torch.ones_like(kwargs["frontier_base_zT"]) * 0.5

    diagnostics = apply_initializer_to_stream_state(
        model=model,
        initializer=Initializer(),
        context=context,
        alpha=2.0,
    )

    assert torch.equal(model.generated[:, :, :5], original[:, :, :5])
    assert torch.equal(model.generated[:, :, 7:], original[:, :, 7:])
    assert torch.equal(
        model.generated[:, :, 5:7, 0, 0].permute(0, 2, 1),
        context.frontier_base_zT + 1.0,
    )
    assert diagnostics["applied"] is True
    assert diagnostics["delta_norm"] > 0.0


def test_validate_noise_initializer_overfit_config_requires_positive_frontier_tokens():
    cfg = {
        "ckpt": "model.ckpt",
        "meta_path": "meta.txt",
        "model": {"params": {"latent_dim": 4, "text_dim": 4096, "frontier_tokens": 0}},
    }

    try:
        validate_noise_initializer_overfit_config(cfg)
    except ValueError as exc:
        assert "frontier_tokens" in str(exc)
    else:
        raise AssertionError("expected frontier_tokens validation error")


def test_validate_noise_initializer_overfit_config_requires_precomputed_text_path():
    cfg = {
        "ckpt": "model.ckpt",
        "meta_path": "meta.txt",
        "model": {"params": {"latent_dim": 4, "text_dim": 4096, "frontier_tokens": 5}},
        "text_encoder": {"type": "precomputed_t5_pool"},
    }

    try:
        validate_noise_initializer_overfit_config(cfg)
    except ValueError as exc:
        assert "precomputed_text_emb_path" in str(exc)
    else:
        raise AssertionError("expected precomputed text path validation error")
