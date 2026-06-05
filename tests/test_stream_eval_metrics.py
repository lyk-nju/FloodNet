from __future__ import annotations

import math

import numpy as np
import torch

from eval.runtime.metrics import (
    build_stream_eval_summary,
    compute_heading_path_error_deg,
    compute_lateral_velocity_ratio,
    compute_root_jitter,
    compute_yaw_error,
    estimate_body_yaw,
)
from eval.ldf.stream_metrics import run_stream_generate_step_sample


def test_compute_yaw_error_wraps_angles():
    pred = np.array([math.pi - 0.1], dtype=np.float32)
    target = np.array([-math.pi + 0.1], dtype=np.float32)

    err = compute_yaw_error(pred, target)

    assert abs(err - 0.2) < 1e-5


def test_compute_root_jitter_is_zero_for_constant_velocity():
    root = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0],
         [4.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    assert compute_root_jitter(root) == 0.0


def test_estimate_body_yaw_matches_physical_yaw_convention():
    motion = np.zeros((5, 263), dtype=np.float32)
    motion[0, 0] = -math.pi / 4.0

    yaw = estimate_body_yaw(motion)

    expected = np.array([0.0, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0])
    assert np.allclose(yaw, expected, atol=1e-5)


def test_lateral_velocity_ratio_zero_for_forward_plus_z_motion():
    motion = np.zeros((6, 263), dtype=np.float32)
    motion[:, 2] = 1.0

    assert compute_lateral_velocity_ratio(motion) < 1e-6


def test_heading_path_error_uses_project_yaw_convention_for_plus_z_path():
    motion = np.zeros((6, 263), dtype=np.float32)
    motion[:, 2] = 1.0
    target = np.zeros((6, 3), dtype=np.float32)
    target[:, 2] = np.arange(6, dtype=np.float32)

    assert compute_heading_path_error_deg(motion, target) < 1e-5


def test_build_stream_eval_summary_uses_stream_metric_keys():
    pred = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = pred.copy()
    target[-1, 0] = 4.0

    summary = build_stream_eval_summary(
        pred,
        target,
        pred_yaw=np.zeros(4, dtype=np.float32),
        target_yaw=np.zeros(4, dtype=np.float32),
    )

    assert summary["stream/root_ADE"] == 0.25
    assert summary["stream/root_FDE"] == 1.0
    assert summary["stream/yaw_error"] == 0.0
    assert "stream/jitter" in summary
    assert summary["stream/num_frames"] == 4


class _FakeStepModel:
    input_dim = 4
    noise_steps = 1

    def __init__(self):
        self.payloads = []
        self.commit_index = 0
        self.chunk_size = 1

    def init_generated(self, history_length, batch_size, num_denoise_steps):
        self.history_length = history_length
        self.batch_size = batch_size
        self.num_denoise_steps = num_denoise_steps
        self.commit_index = 0

    def stream_generate_step(self, step_input, first_chunk=True):
        self.payloads.append(dict(step_input))
        self.commit_index += 1
        return {"generated": [torch.zeros(1, self.input_dim)]}


class _FakeRollingStepModel(_FakeStepModel):
    def stream_generate_step(self, step_input, first_chunk=True):
        out = super().stream_generate_step(step_input, first_chunk=first_chunk)
        if self.commit_index == 3:
            self.commit_index = 1
        return out


class _FakeStepVAE:
    def clear_cache(self):
        pass

    def stream_decode(self, latent, first_chunk=True):
        return torch.zeros(1, 1, 263)


def test_ldf_stream_generate_step_uses_direct_7d_payload_when_available():
    traj7 = torch.zeros(1, 5, 7, dtype=torch.float32)
    traj7[0, :, 2] = torch.arange(5, dtype=torch.float32)
    traj7[0, :, 3] = 1.0
    sample_batch = {
        "name": ["sample"],
        "dataset": ["HumanML3D"],
        "text": ["walk"],
        "token": torch.zeros(1, 2, 4, dtype=torch.float32),
        "token_length": torch.tensor([2], dtype=torch.long),
        "feature_length": torch.tensor([5], dtype=torch.long),
        "traj_cond_7d": traj7,
        "traj_cond": traj7[..., :3].clone(),
        "traj": traj7[..., :3].clone(),
        "traj_length": torch.tensor([5], dtype=torch.long),
        "traj_cond_mask": torch.ones(1, 5, dtype=torch.float32),
        "traj_mask": torch.ones(1, 5, dtype=torch.float32),
        "token_mask": torch.ones(1, 2, dtype=torch.float32),
    }
    model = _FakeStepModel()

    run_stream_generate_step_sample(
        model=model,
        vae=_FakeStepVAE(),
        sample_batch=sample_batch,
        device=torch.device("cpu"),
        history_length=2,
        num_denoise_steps=1,
        traj_horizon_tokens=1,
    )

    assert model.payloads
    assert all("traj_cond_7d_frame" in payload for payload in model.payloads)
    assert all("traj_cond_frame_mask" in payload for payload in model.payloads)
    assert all("traj_features" not in payload for payload in model.payloads)
    assert all("traj" not in payload for payload in model.payloads)


def test_ldf_stream_generate_step_separates_local_and_absolute_commit_after_roll():
    traj7 = torch.zeros(1, 13, 7, dtype=torch.float32)
    traj7[0, :, 2] = torch.arange(13, dtype=torch.float32)
    traj7[0, :, 3] = 1.0
    sample_batch = {
        "name": ["sample"],
        "dataset": ["HumanML3D"],
        "text": ["walk"],
        "token": torch.zeros(1, 4, 4, dtype=torch.float32),
        "token_length": torch.tensor([4], dtype=torch.long),
        "feature_length": torch.tensor([13], dtype=torch.long),
        "traj_cond_7d": traj7,
        "traj_cond": traj7[..., :3].clone(),
        "traj": traj7[..., :3].clone(),
        "traj_length": torch.tensor([13], dtype=torch.long),
        "traj_cond_mask": torch.ones(1, 13, dtype=torch.float32),
        "traj_mask": torch.ones(1, 13, dtype=torch.float32),
        "token_mask": torch.ones(1, 4, dtype=torch.float32),
    }
    model = _FakeRollingStepModel()

    run_stream_generate_step_sample(
        model=model,
        vae=_FakeStepVAE(),
        sample_batch=sample_batch,
        device=torch.device("cpu"),
        history_length=2,
        num_denoise_steps=1,
        traj_horizon_tokens=0,
    )

    assert model.payloads[-1]["traj_start_token"] == 0
    assert model.payloads[-1]["traj_abs_start_token"] == 2
