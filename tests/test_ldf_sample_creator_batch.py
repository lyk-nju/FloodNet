from __future__ import annotations

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


def test_sample_creator_full_batch_routes_token_and_7d_traj():
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

    out = SampleCreator().create(batch)

    assert out["feature"] is token
    assert torch.equal(out["feature_length"], batch["token_length"])
    assert out["traj_features"] is traj7
    assert out["traj"] is traj_xyz
    assert out["traj_mask"] is batch["traj_cond_mask"]


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
