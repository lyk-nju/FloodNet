from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from eval.ldf.conditioning import (
    LdfEvalStreamConditioner,
    build_gt_rootplan_from_batch,
)
from utils.training.ldf.validation_conditioning import (
    prepare_ldf_eval_model_batch,
)
from metrics.traj import _compute_deterministic_fwd_ctrl_loss_sample
from utils.token_frame import num_frames_for_tokens


def _make_7d_batch() -> dict:
    yaw0 = math.pi / 2.0
    yaw1 = math.pi / 2.0
    traj7 = torch.zeros(1, 5, 7, dtype=torch.float32)
    traj7[0, :, 0] = torch.tensor([10.0, 10.0, 10.0, 11.0, 12.0])
    traj7[0, :, 1] = 0.5
    traj7[0, :, 2] = torch.tensor([2.0, 3.0, 4.0, 4.0, 4.0])
    traj7[0, :, 3] = math.cos(yaw0)
    traj7[0, :, 4] = math.sin(yaw0)
    traj7[0, 3:, 3] = math.cos(yaw1)
    traj7[0, 3:, 4] = math.sin(yaw1)
    traj7[0, 1:, 5] = 1.0

    return {
        "name": ["sample"],
        "text": ["walk"],
        "token": torch.zeros(1, 2, 4, dtype=torch.float32),
        "token_length": torch.tensor([2], dtype=torch.long),
        "feature_length": torch.tensor([5], dtype=torch.long),
        "traj_cond_7d": traj7.clone(),
        "traj_cond": traj7[..., :3].clone(),
        "traj": traj7[..., :3].clone(),
        "traj_length": torch.tensor([5], dtype=torch.long),
        "traj_cond_mask": torch.ones(1, 5, dtype=torch.float32),
        "traj_mask": torch.ones(1, 5, dtype=torch.float32),
        "token_mask": torch.ones(1, 2, dtype=torch.float32),
    }


def test_prepare_ldf_eval_model_batch_canonicalizes_7d_to_clip_start():
    batch = _make_7d_batch()

    model_batch = prepare_ldf_eval_model_batch(batch, torch.device("cpu"))
    traj = model_batch["traj_features"]

    assert traj.shape == (1, 5, 7)
    assert torch.allclose(traj[0, 0, [0, 2]], torch.zeros(2), atol=1e-5)
    assert torch.allclose(traj[0, 0, 3:5], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(traj[0, :, 1], batch["traj_cond_7d"][0, :, 1])
    assert torch.allclose(traj[0, :, 5:7], batch["traj_cond_7d"][0, :, 5:7])


def test_prepare_ldf_eval_model_batch_does_not_inject_no_traj_condition():
    batch = {
        "name": ["sample"],
        "text": ["walk"],
        "token": torch.zeros(1, 2, 4, dtype=torch.float32),
        "token_length": torch.tensor([2], dtype=torch.long),
        "feature_length": torch.tensor([5], dtype=torch.long),
    }

    model_batch = prepare_ldf_eval_model_batch(batch, torch.device("cpu"))

    assert "traj_features" not in model_batch
    assert "traj" not in model_batch
    assert "traj_mask" not in model_batch


def test_prepare_ldf_eval_model_batch_without_model_is_deterministic(monkeypatch):
    batch = _make_7d_batch()
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare_ldf_eval_model_batch should not sample randomly")
        ),
    )

    model_batch = prepare_ldf_eval_model_batch(batch, torch.device("cpu"))

    assert model_batch["feature_length"].tolist() == batch["token_length"].tolist()
    assert model_batch["traj_num_tokens"].tolist() == batch["token_length"].tolist()


def test_prepare_ldf_eval_model_batch_prefix_window_uses_full_future_traj(monkeypatch):
    traj_tokens = 5
    traj_frames = num_frames_for_tokens(traj_tokens)
    batch = {
        "name": ["sample"],
        "text": ["walk"],
        "token": torch.zeros(1, traj_tokens, 4, dtype=torch.float32),
        "token_length": torch.tensor([traj_tokens], dtype=torch.long),
        "feature_length": torch.tensor([traj_frames], dtype=torch.long),
        "traj_cond_7d": torch.zeros(1, traj_frames, 7, dtype=torch.float32),
        "traj_cond": torch.zeros(1, traj_frames, 3, dtype=torch.float32),
        "traj_length": torch.tensor([traj_frames], dtype=torch.long),
        "traj_cond_mask": torch.ones(1, traj_frames, dtype=torch.float32),
    }
    model = SimpleNamespace(ldf_window_context_tokens=3, ldf_window_horizon_tokens=99)

    monkeypatch.setattr(
        "utils.training.ldf.validation_conditioning.prepare_generate_condition",
        lambda model_arg, model_batch, device: {"prepared": True},
    )
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("offline eval should use fixed full-prefix R=T")
        ),
    )

    model_batch = prepare_ldf_eval_model_batch(batch, torch.device("cpu"), model=model)

    assert model_batch["feature_length"].tolist() == [traj_tokens]
    assert model_batch["traj_num_tokens"].tolist() == [traj_tokens]
    assert model_batch["traj_features_length"].tolist() == [traj_tokens]
    assert model_batch["traj_features"].shape[1] == traj_frames
    assert model_batch["ldf_condition"] == {"prepared": True}


def test_build_gt_rootplan_from_batch_uses_first_frame_anchor():
    batch = _make_7d_batch()

    root_plan = build_gt_rootplan_from_batch(batch, token_dt=0.2)

    assert root_plan.valid_frames == 5
    assert root_plan.num_tokens_pred == 2
    assert torch.allclose(root_plan.anchor_world_xz.cpu(), torch.tensor([10.0, 2.0]))
    assert abs(float(root_plan.anchor_world_yaw.cpu()) - math.pi / 2.0) < 1e-5
    assert root_plan.waypoints_local_7d.shape == (5, 7)
    assert torch.allclose(
        root_plan.waypoints_local_7d[0, [0, 2]].cpu(),
        torch.zeros(2),
        atol=1e-5,
    )
    assert torch.allclose(
        root_plan.waypoints_local_7d[0, 3:5].cpu(),
        torch.tensor([1.0, 0.0]),
        atol=1e-5,
    )


def test_ldf_eval_stream_conditioner_builds_direct_7d_payload():
    batch = _make_7d_batch()
    conditioner = LdfEvalStreamConditioner(
        batch,
        history_length=2,
        traj_horizon_tokens=1,
        token_dt=0.2,
        device=torch.device("cpu"),
    )

    payload = conditioner.build_step_payload(
        local_commit_index=0,
        absolute_commit_index=0,
        chunk_size=1,
    )

    assert payload is not None
    assert "traj_cond_7d_frame" in payload
    assert "traj_cond_frame_mask" in payload
    assert "traj_start_token" in payload
    assert "traj" not in payload
    assert payload["traj_cond_7d_frame"].shape[-1] == 7
    assert payload["traj_cond_frame_mask"].any()


def test_ldf_eval_stream_conditioner_extends_gt_tail_for_future_horizon():
    batch = _make_7d_batch()
    conditioner = LdfEvalStreamConditioner(
        batch,
        history_length=2,
        traj_horizon_tokens=2,
        token_dt=0.2,
        device=torch.device("cpu"),
    )

    payload = conditioner.build_step_payload(
        local_commit_index=0,
        absolute_commit_index=0,
        chunk_size=1,
    )

    assert payload is not None
    traj = payload["traj_cond_7d_frame"][0]
    mask = payload["traj_cond_frame_mask"][0].bool()
    assert traj.shape[0] == 9
    assert bool(mask.all())
    assert torch.allclose(traj[5:], traj[4:5].expand_as(traj[5:]))


class _RecordingForwardModel:
    chunk_size = 1

    def __init__(self):
        self.calls = []

    def __call__(self, model_batch):
        self.calls.append(dict(model_batch))
        return {}


def test_forward_control_loss_accepts_eval_model_batch_builder():
    model = _RecordingForwardModel()
    sample_batch = {
        "name": ["sample"],
        "token_length": torch.tensor([1], dtype=torch.long),
    }

    def builder(batch, device, model=None):
        return {"built_by_eval_helper": True}

    _compute_deterministic_fwd_ctrl_loss_sample(
        model=model,
        sample_batch=sample_batch,
        vae=None,
        device=torch.device("cpu"),
        train_mode=3,
        model_batch_builder=builder,
    )

    assert model.calls
    assert model.calls[0]["built_by_eval_helper"] is True
    assert "_time_steps_override" in model.calls[0]
