from __future__ import annotations

import pytest
import torch

from utils.token_frame import num_frames_for_tokens, num_tokens_for_frame_len, token_start_frame
from utils.training.ldf.sample_creator import SampleCreator


def _make_motion263(batch_size: int, num_frames: int) -> torch.Tensor:
    motion = torch.zeros(batch_size, num_frames, 263, dtype=torch.float32)
    for b in range(batch_size):
        motion[b, :, 1] = 0.1 * (b + 1)
        motion[b, :, 2] = 0.05 * (b + 1)
        motion[b, :, 3] = 1.0 + 0.1 * b
    return motion


class _RecordingVAE:
    def __init__(self, latent_dim: int = 3):
        self.latent_dim = int(latent_dim)
        self.encode_inputs: list[torch.Tensor] = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.encode_inputs.append(x.detach().clone())
        n_tokens = num_tokens_for_frame_len(int(x.shape[1]))
        values = torch.arange(
            n_tokens * self.latent_dim,
            device=x.device,
            dtype=x.dtype,
        ).view(1, n_tokens, self.latent_dim)
        return values.expand(x.shape[0], -1, -1).clone()


def test_sample_creator_fixed_prefix_routes_token_and_7d_traj():
    token = torch.zeros(2, 5, 4)
    traj7 = torch.randn(2, 20, 7)
    traj_xyz = torch.randn(2, 20, 3)
    batch = {
        "token": token,
        "token_length": torch.tensor([5, 4]),
        "traj_cond_7d": traj7,
        "traj_cond": traj_xyz,
        "traj": torch.randn(2, 20, 3),
        "traj_length": torch.tensor([20, 18]),
        "traj_cond_mask": torch.ones(2, 20),
    }

    out = SampleCreator(
        sample_policy="fixed_window",
        end_tokens=torch.tensor([5, 4]),
    ).create(batch)

    assert torch.equal(out["feature"], token)
    assert torch.equal(out["feature_length"], batch["token_length"])
    assert out["traj_features"].shape == (2, num_frames_for_tokens(5), 7)
    assert torch.equal(out["traj_features"][0], traj7[0, : num_frames_for_tokens(5)])
    assert torch.equal(out["traj"][0], traj_xyz[0, : num_frames_for_tokens(5)])
    assert out["traj_num_tokens"].tolist() == [5, 4]
    assert out["traj_features_length"].tolist() == [5, 4]
    assert out["traj_mask"][0].sum().item() == num_frames_for_tokens(5)
    assert out["traj_mask"][1].sum().item() == num_frames_for_tokens(4)


def test_sample_creator_prefix_window_uses_full_future_traj():
    token = torch.arange(10 * 4, dtype=torch.float32).view(1, 10, 4)
    traj_tokens = 10
    traj_frames = num_frames_for_tokens(traj_tokens)
    traj7 = torch.arange(traj_frames * 7, dtype=torch.float32).view(1, traj_frames, 7)
    batch = {
        "token": token,
        "token_length": torch.tensor([10]),
        "traj_cond_7d": traj7,
        "traj_cond": torch.zeros(1, traj_frames, 3),
        "traj_length": torch.tensor([traj_frames]),
        "traj_cond_mask": torch.ones(1, traj_frames),
    }

    out = SampleCreator(
        context_tokens=6,
        sample_policy="fixed_window",
        min_history_tokens=1,
        end_tokens=torch.tensor([6]),
    ).create(batch)

    assert out["feature"].shape == (1, 6, 4)
    assert out["feature_length"].tolist() == [6]
    assert torch.equal(out["feature"], token[:, :6])
    assert out["traj_features"].shape == (1, traj_frames, 7)
    assert out["traj_num_tokens"].tolist() == [traj_tokens]
    assert out["traj_features_length"].tolist() == [traj_tokens]
    assert out["traj_start_token"].tolist() == [0]
    assert torch.equal(out["traj_features"], traj7[:, :traj_frames])


def test_sample_creator_prefix_does_not_expose_legacy_token_mask_to_traj_path():
    token = torch.arange(10 * 4, dtype=torch.float32).view(1, 10, 4)
    traj_frames = num_frames_for_tokens(10)
    batch = {
        "token": token,
        "token_length": torch.tensor([10]),
        "token_mask": torch.ones(1, 10),
        "traj_cond_7d": torch.zeros(1, traj_frames, 7),
        "traj_cond": torch.zeros(1, traj_frames, 3),
        "traj_length": torch.tensor([traj_frames]),
        "traj_cond_mask": torch.ones(1, traj_frames),
    }

    out = SampleCreator(
        sample_policy="fixed_window",
        end_tokens=torch.tensor([5]),
    ).create(batch)

    assert "token_mask" not in out
    assert out["latent_token_mask"].shape == (1, 5)
    assert torch.allclose(out["latent_token_mask"], torch.ones(1, 5))


def test_sample_creator_full_policy_preserves_precomputed_latents_and_traj():
    token = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4)
    traj_frames = num_frames_for_tokens(6)
    traj7 = torch.arange(2 * traj_frames * 7, dtype=torch.float32).view(
        2,
        traj_frames,
        7,
    )
    token_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    traj_mask = torch.ones(2, traj_frames)
    batch = {
        "token": token,
        "token_length": torch.tensor([6, 4]),
        "token_mask": token_mask,
        "token_text_end": torch.tensor([6, 4]),
        "traj_cond_7d": traj7,
        "traj_cond": torch.zeros(2, traj_frames, 3),
        "traj_length": torch.tensor([traj_frames, num_frames_for_tokens(4)]),
        "traj_cond_mask": traj_mask,
    }

    out = SampleCreator(window_policy="full").create(batch)

    assert torch.equal(out["feature"], token)
    assert torch.equal(out["feature_length"], batch["token_length"])
    assert torch.equal(out["token"], token)
    assert torch.equal(out["token_length"], batch["token_length"])
    assert torch.equal(out["feature_text_end"], batch["token_text_end"])
    assert torch.equal(out["token_mask"], token_mask)
    assert torch.equal(out["traj_features"], traj7)
    assert torch.equal(out["traj_length"], batch["traj_length"])
    assert torch.equal(out["traj_mask"], traj_mask)
    assert "_window_local_sample_policy" not in out
    assert "latent_token_mask" not in out
    assert "traj_num_tokens" not in out


def test_sample_creator_prefix_window_does_not_cap_active_right_by_context_tokens():
    token = torch.arange(10 * 4, dtype=torch.float32).view(1, 10, 4)
    traj_frames = num_frames_for_tokens(10)
    batch = {
        "token": token,
        "token_length": torch.tensor([10]),
        "traj_cond_7d": torch.zeros(1, traj_frames, 7),
        "traj_cond": torch.zeros(1, traj_frames, 3),
        "traj_length": torch.tensor([traj_frames]),
        "traj_cond_mask": torch.ones(1, traj_frames),
    }

    out = SampleCreator(
        context_tokens=3,
        sample_policy="fixed_window",
        min_history_tokens=1,
        end_tokens=torch.tensor([10]),
    ).create(batch)

    assert out["feature_length"].tolist() == [10]
    assert out["traj_num_tokens"].tolist() == [10]


def test_sample_creator_default_prefix_samples_active_right_from_token_length(monkeypatch):
    token = torch.arange(5 * 4, dtype=torch.float32).view(1, 5, 4)
    traj_frames = num_frames_for_tokens(5)
    batch = {
        "token": token,
        "token_length": torch.tensor([5]),
        "traj_cond_7d": torch.zeros(1, traj_frames, 7),
        "traj_cond": torch.zeros(1, traj_frames, 3),
        "traj_length": torch.tensor([traj_frames]),
        "traj_cond_mask": torch.ones(1, traj_frames),
    }
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None: torch.full(
            size, int(low), device=device, dtype=torch.long
        ),
    )

    out = SampleCreator().create(batch)

    assert out["feature_length"].tolist() == [1]
    assert out["feature"].shape == (1, 1, 4)
    assert out["traj_num_tokens"].tolist() == [5]


def test_sample_creator_prefix_respects_dynamic_min_prefix_tokens(monkeypatch):
    token = torch.arange(8 * 4, dtype=torch.float32).view(1, 8, 4)
    traj_frames = num_frames_for_tokens(8)
    batch = {
        "token": token,
        "token_length": torch.tensor([8]),
        "traj_cond_7d": torch.zeros(1, traj_frames, 7),
        "traj_cond": torch.zeros(1, traj_frames, 3),
        "traj_length": torch.tensor([traj_frames]),
        "traj_cond_mask": torch.ones(1, traj_frames),
    }
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None: torch.full(
            size,
            int(low),
            device=device,
            dtype=torch.long,
        ),
    )

    out = SampleCreator(min_prefix_tokens=5).create(batch)

    assert out["feature_length"].tolist() == [5]
    assert out["feature"].shape == (1, 5, 4)
    assert torch.equal(out["feature"], token[:, :5])


def test_sample_creator_prefix_errors_when_token_length_below_min_prefix_tokens():
    token = torch.arange(4 * 4, dtype=torch.float32).view(1, 4, 4)
    traj_frames = num_frames_for_tokens(4)
    batch = {
        "token": token,
        "token_length": torch.tensor([4]),
        "traj_cond_7d": torch.zeros(1, traj_frames, 7),
        "traj_cond": torch.zeros(1, traj_frames, 3),
        "traj_length": torch.tensor([traj_frames]),
        "traj_cond_mask": torch.ones(1, traj_frames),
    }

    with pytest.raises(ValueError, match="min_prefix_tokens"):
        SampleCreator(min_prefix_tokens=5).create(batch)


def test_sample_creator_prefix_errors_when_end_tokens_below_min_prefix_tokens():
    token = torch.arange(6 * 4, dtype=torch.float32).view(1, 6, 4)
    traj_frames = num_frames_for_tokens(6)
    batch = {
        "token": token,
        "token_length": torch.tensor([6]),
        "traj_cond_7d": torch.zeros(1, traj_frames, 7),
        "traj_cond": torch.zeros(1, traj_frames, 3),
        "traj_length": torch.tensor([traj_frames]),
        "traj_cond_mask": torch.ones(1, traj_frames),
    }

    with pytest.raises(ValueError, match="min_prefix_tokens"):
        SampleCreator(
            sample_policy="fixed_window",
            end_tokens=torch.tensor([4]),
            min_prefix_tokens=5,
        ).create(batch)


def test_sample_creator_prefix_window_allows_short_samples_without_horizon_config():
    token = torch.arange(4 * 4, dtype=torch.float32).view(1, 4, 4)
    traj_frames = num_frames_for_tokens(4)
    batch = {
        "token": token,
        "token_length": torch.tensor([4]),
        "traj_cond_7d": torch.zeros(1, traj_frames, 7),
        "traj_cond": torch.zeros(1, traj_frames, 3),
        "traj_length": torch.tensor([traj_frames]),
        "traj_cond_mask": torch.ones(1, traj_frames),
    }

    out = SampleCreator(
        context_tokens=30,
        sample_policy="fixed_window",
        min_history_tokens=1,
        end_tokens=torch.tensor([4]),
    ).create(batch)

    assert out["feature_length"].tolist() == [4]
    assert out["traj_num_tokens"].tolist() == [4]
    assert out["traj_features_length"].tolist() == [4]


def test_sample_creator_prefix_traj_length_uses_valid_source_frames():
    token = torch.zeros(2, 11, 4)
    traj_frames = num_frames_for_tokens(11)
    traj7 = torch.ones(2, traj_frames, 7)
    traj_xyz = torch.ones(2, traj_frames, 3)
    batch = {
        "token": token,
        "token_length": torch.tensor([11, 5]),
        "traj_cond_7d": traj7,
        "traj_cond": traj_xyz,
        "traj_length": torch.tensor([40, 15]),
        "traj_cond_mask": torch.ones(2, traj_frames),
    }

    out = SampleCreator(
        sample_policy="fixed_window",
        end_tokens=torch.tensor([11, 5]),
    ).create(batch)

    assert out["traj_num_tokens"].tolist() == [11, 5]
    assert out["traj_features_length"].tolist() == [11, 5]
    assert out["traj_length"].tolist() == [40, 15]
    assert out["traj_cond_mask"][0].sum().item() == 40
    assert out["traj_cond_mask"][1].sum().item() == 15


def test_sample_creator_stream_batch_online_encodes_motion_window():
    token = torch.full((1, 40, 3), -999.0)
    raw = _make_motion263(batch_size=1, num_frames=200)
    raw[0, :, 10] = torch.arange(200, dtype=raw.dtype)
    batch = {
        "token": token,
        "token_length": torch.tensor([40]),
        "feature": raw,
        "feature_length": torch.tensor([200]),
        "text": ["walk"],
    }
    vae = _RecordingVAE(latent_dim=3)

    out = SampleCreator(
        stream_enabled=True,
        context_tokens=30,
        horizon_tokens=0,
        window_sampling={
            "enabled": True,
            "history_tokens_min": 0,
            "history_tokens_max": "auto",
            "horizon_tokens_min": 5,
            "horizon_tokens_max": 25,
        },
        chunk_size=5,
        rollout_span=4,
        active_left_tokens=torch.tensor([10]),
        history_tokens=torch.tensor([3]),
        sampled_horizon_tokens=torch.tensor([7]),
    ).create(batch, vae=vae)

    global_start_token = 7
    latent_tokens = 12
    traj_tokens = 19
    global_start_frame = token_start_frame(global_start_token)

    assert len(vae.encode_inputs) == 1
    assert vae.encode_inputs[0].shape == (1, num_frames_for_tokens(latent_tokens), 263)
    assert float(vae.encode_inputs[0][0, 0, 10].item()) == float(global_start_frame)
    assert out["feature_length"].tolist() == [latent_tokens]
    assert out["_window_global_start_token"].tolist() == [global_start_token]
    assert out["_window_local_latent_start_token"].tolist() == [0]
    assert "_window_local_latent_source" not in out
    assert out["traj_start_token"].tolist() == [0]
    assert out["traj_num_tokens"].tolist() == [traj_tokens]
    assert out["traj_length"].tolist() == [num_frames_for_tokens(traj_tokens)]
    assert torch.allclose(out["traj_features"][0, 0, [0, 2]], torch.zeros(2))
    assert torch.allclose(out["traj_features"][0, 0, 3:5], torch.tensor([1.0, 0.0]))
