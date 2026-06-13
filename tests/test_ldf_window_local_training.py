from __future__ import annotations

import torch
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from utils.token_frame import token_range_to_frame_slice, token_start_frame
from utils.traj_batch import encode_traj_batch
from utils.training.self_forcing import SelfForcingTrainer
import utils.training.self_forcing as sf_mod
from utils.training.window_local import (
    build_window_local_model_batch,
    build_window_local_traj_batch,
)
from utils.training.self_forcing import (
    _collect_window_local_metrics,
    _splice_window_local_pred_to_prefix,
)


def _make_motion263(batch_size: int, num_frames: int) -> torch.Tensor:
    motion = torch.zeros(batch_size, num_frames, 263, dtype=torch.float32)
    for b in range(batch_size):
        motion[b, :, 1] = 0.1 * (b + 1)
        motion[b, :, 2] = 0.05 * (b + 1)
        motion[b, :, 3] = 1.0 + 0.1 * b
    return motion


class _MeanLocalTrajEncoder(torch.nn.Module):
    def forward(
        self,
        feats_4: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frame_mask is None:
            return feats_4.mean(dim=2)
        weights = frame_mask.unsqueeze(-1).to(dtype=feats_4.dtype)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (feats_4 * weights).sum(dim=2) / denom


def test_window_local_traj_batch_uses_prefix_and_arbitrary_frame_lengths():
    raw = _make_motion263(batch_size=2, num_frames=40)
    out = build_window_local_traj_batch(
        raw_feature_263=raw,
        raw_feature_length=torch.tensor([40, 40]),
        start_tokens=torch.tensor([0, 5]),
        num_tokens=torch.tensor([2, 2]),
    )

    assert out["traj_features"].shape == (2, 8, 7)
    assert out["traj_cond_mask"][0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert out["traj_cond_mask"][1].tolist() == [1] * 8
    assert out["traj_length"].tolist() == [5, 8]
    assert out["traj_num_tokens"].tolist() == [2, 2]
    assert out["traj_start_token"].tolist() == [0, 5]


def test_window_local_traj_batch_recovers_from_window_origin():
    raw = _make_motion263(batch_size=1, num_frames=40)
    out = build_window_local_traj_batch(
        raw_feature_263=raw,
        raw_feature_length=torch.tensor([40]),
        start_tokens=torch.tensor([5]),
        num_tokens=torch.tensor([2]),
    )
    traj = out["traj_features"][0]

    assert torch.allclose(traj[0, [0, 2]], torch.zeros(2))
    assert torch.allclose(traj[0, 3:5], torch.tensor([1.0, 0.0]))
    assert torch.allclose(traj[0, 5:7], torch.zeros(2))


def test_window_local_traj_batch_masks_unavailable_future_tail():
    raw = _make_motion263(batch_size=1, num_frames=25)
    start = 5
    num_tokens = 4
    expected = token_range_to_frame_slice(start, num_tokens)
    assert token_start_frame(start) < 25 < expected.stop

    out = build_window_local_traj_batch(
        raw_feature_263=raw,
        raw_feature_length=torch.tensor([25]),
        start_tokens=torch.tensor([start]),
        num_tokens=torch.tensor([num_tokens]),
    )

    available = 25 - expected.start
    expected_len = expected.stop - expected.start
    assert out["traj_features"].shape == (1, expected_len, 7)
    assert out["traj_length"].tolist() == [available]
    assert out["traj_cond_mask"][0, :available].sum().item() == available
    assert out["traj_cond_mask"][0, available:].sum().item() == 0


def test_window_local_traj_batch_rejects_invalid_origin():
    raw = _make_motion263(batch_size=1, num_frames=10)
    with pytest.raises(ValueError, match="valid origin"):
        build_window_local_traj_batch(
            raw_feature_263=raw,
            raw_feature_length=torch.tensor([10]),
            start_tokens=torch.tensor([5]),
            num_tokens=torch.tensor([2]),
        )


def test_window_local_traj_batch_rejects_non_263_raw_motion():
    traj7 = torch.zeros(1, 20, 7)
    with pytest.raises(ValueError, match="raw 263D"):
        build_window_local_traj_batch(
            raw_feature_263=traj7,
            raw_feature_length=torch.tensor([20]),
            start_tokens=torch.tensor([0]),
            num_tokens=torch.tensor([2]),
        )


def test_window_local_traj_batch_rejects_raw_length_past_tensor_frames():
    raw = _make_motion263(batch_size=1, num_frames=20)
    with pytest.raises(ValueError, match="raw_feature_length"):
        build_window_local_traj_batch(
            raw_feature_263=raw,
            raw_feature_length=torch.tensor([40]),
            start_tokens=torch.tensor([0]),
            num_tokens=torch.tensor([10]),
        )


def test_window_local_model_batch_pads_latent_to_attention_len_but_keeps_valid_len():
    token = torch.arange(2 * 10 * 3, dtype=torch.float32).view(2, 10, 3)
    raw = _make_motion263(batch_size=2, num_frames=80)
    batch = {
        "token": token,
        "token_length": torch.tensor([10, 10]),
        "feature": raw,
        "feature_length": torch.tensor([80, 80]),
        "text": ["walk", "run"],
    }

    out = build_window_local_model_batch(
        batch,
        context_tokens=4,
        horizon_tokens=2,
        start_tokens=torch.tensor([1, 3]),
    )

    assert out["feature"].shape == (2, 6, 3)
    assert out["feature_length"].tolist() == [4, 4]
    assert torch.allclose(out["feature"][0, :4], token[0, 1:5])
    assert torch.allclose(out["feature"][1, :4], token[1, 3:7])
    assert torch.count_nonzero(out["feature"][:, 4:]) == 0
    assert out["traj_start_token"].tolist() == [1, 3]
    assert out["traj_num_tokens"].tolist() == [6, 6]
    assert out["traj_features_length"].tolist() == [6, 6]
    assert out["_window_local_traj"] is True


def test_window_local_model_batch_requires_raw_263_feature_not_traj_or_latent():
    token = torch.arange(1 * 10 * 3, dtype=torch.float32).view(1, 10, 3)
    batch = {
        "token": token,
        "token_length": torch.tensor([10]),
        "feature": torch.zeros(1, 80, 7),
        "feature_length": torch.tensor([80]),
        "text": ["walk"],
    }

    with pytest.raises(ValueError, match="raw 263D"):
        build_window_local_model_batch(
            batch,
            context_tokens=4,
            horizon_tokens=2,
            start_tokens=torch.tensor([1]),
        )


def test_window_local_model_batch_rejects_raw_feature_length_past_tensor_frames():
    token = torch.arange(1 * 10 * 3, dtype=torch.float32).view(1, 10, 3)
    raw = _make_motion263(batch_size=1, num_frames=20)
    batch = {
        "token": token,
        "token_length": torch.tensor([10]),
        "feature": raw,
        "feature_length": torch.tensor([40]),
        "text": ["walk"],
    }

    with pytest.raises(ValueError, match="raw_feature_length"):
        build_window_local_model_batch(
            batch,
            context_tokens=4,
            horizon_tokens=2,
            start_tokens=torch.tensor([0]),
        )


def test_window_local_model_batch_fixed_window_uses_explicit_right_boundary():
    token = torch.arange(2 * 10 * 3, dtype=torch.float32).view(2, 10, 3)
    raw = _make_motion263(batch_size=2, num_frames=80)
    batch = {
        "token": token,
        "token_length": torch.tensor([10, 10]),
        "feature": raw,
        "feature_length": torch.tensor([80, 80]),
        "text": ["walk", "run"],
    }

    out = build_window_local_model_batch(
        batch,
        context_tokens=4,
        horizon_tokens=1,
        sample_policy="fixed_window",
        min_history_tokens=2,
        end_tokens=torch.tensor([7, 3]),
    )

    assert out["_window_local_sample_policy"] == "fixed_window"
    assert out["traj_start_token"].tolist() == [3, 0]
    assert out["feature_length"].tolist() == [4, 3]
    assert torch.allclose(out["feature"][0, :4], token[0, 3:7])
    assert torch.allclose(out["feature"][1, :3], token[1, 0:3])
    assert out["traj_num_tokens"].tolist() == [5, 4]


def test_window_local_model_batch_rejects_variable_history_shorter_than_min_history():
    token = torch.arange(1 * 6 * 3, dtype=torch.float32).view(1, 6, 3)
    raw = _make_motion263(batch_size=1, num_frames=40)
    batch = {
        "token": token,
        "token_length": torch.tensor([6]),
        "feature": raw,
        "feature_length": torch.tensor([40]),
        "text": ["walk"],
    }

    with pytest.raises(ValueError, match="min_history_tokens"):
        build_window_local_model_batch(
            batch,
            context_tokens=4,
            horizon_tokens=1,
            sample_policy="variable_history",
            min_history_tokens=4,
            start_tokens=torch.tensor([4]),
        )


def test_window_local_model_batch_rejects_nonpositive_context_and_negative_horizon():
    token = torch.arange(1 * 6 * 3, dtype=torch.float32).view(1, 6, 3)
    raw = _make_motion263(batch_size=1, num_frames=40)
    batch = {
        "token": token,
        "token_length": torch.tensor([6]),
        "feature": raw,
        "feature_length": torch.tensor([40]),
        "text": ["walk"],
    }

    with pytest.raises(ValueError, match="context_tokens"):
        build_window_local_model_batch(
            batch,
            context_tokens=0,
            horizon_tokens=1,
        )
    with pytest.raises(ValueError, match="horizon_tokens"):
        build_window_local_model_batch(
            batch,
            context_tokens=4,
            horizon_tokens=-1,
        )


def test_window_local_model_batch_crops_segmented_text_to_local_window():
    token = torch.arange(1 * 10 * 3, dtype=torch.float32).view(1, 10, 3)
    raw = _make_motion263(batch_size=1, num_frames=80)
    batch = {
        "token": token,
        "token_length": torch.tensor([10]),
        "feature": raw,
        "feature_length": torch.tensor([80]),
        "text": [["walk", "run", "turn"]],
        "token_text_end": [[3, 7, 10]],
    }

    out = build_window_local_model_batch(
        batch,
        context_tokens=4,
        horizon_tokens=2,
        start_tokens=torch.tensor([2]),
    )

    assert out["text"] == [["walk", "run"]]
    assert out["token_text_end"] == [[1, 4]]
    assert out["feature_text_end"] == [[1, 4]]


def test_window_local_model_batch_v2_uses_active_left_history_and_horizon_metadata():
    token = torch.arange(2 * 40 * 3, dtype=torch.float32).view(2, 40, 3)
    raw = _make_motion263(batch_size=2, num_frames=200)
    batch = {
        "token": token,
        "token_length": torch.tensor([40, 40]),
        "feature": raw,
        "feature_length": torch.tensor([200, 200]),
        "text": ["walk", "run"],
    }

    out = build_window_local_model_batch(
        batch,
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
        active_left_tokens=torch.tensor([10, 0]),
        history_tokens=torch.tensor([3, 0]),
        sampled_horizon_tokens=torch.tensor([7, 5]),
    )

    assert out["_window_local_sample_policy"] == "active_left"
    assert out["_window_local_latent_start_token"].tolist() == [7, 0]
    assert out["_window_sampling_active_left_token"].tolist() == [10, 0]
    assert out["_window_sampling_history_tokens"].tolist() == [3, 0]
    assert out["_window_sampling_horizon_tokens"].tolist() == [7, 5]
    assert out["_window_sampling_horizon_cap_clip"].tolist() == [31, 31]
    assert out["_window_sampling_horizon_short_fallback"].tolist() == [False, False]
    assert out["_window_sampling_rollout_span"] == 4
    assert out["_window_sampling_history_tokens_max_effective"] == 21

    assert out["feature_length"].tolist() == [12, 9]
    assert out["traj_num_tokens"].tolist() == [19, 14]
    assert torch.allclose(out["feature"][0, :12], token[0, 7:19])
    assert torch.allclose(out["feature"][1, :9], token[1, 0:9])
    assert torch.count_nonzero(out["feature"][0, 12:]) == 0
    assert torch.count_nonzero(out["feature"][1, 9:]) == 0


def test_window_local_v2_future_horizon_survives_traj_token_mask_when_source_has_token_mask():
    token = torch.arange(1 * 40 * 3, dtype=torch.float32).view(1, 40, 3)
    raw = _make_motion263(batch_size=1, num_frames=200)
    batch = {
        "token": token,
        "token_length": torch.tensor([40]),
        "token_mask": torch.ones(1, 40),
        "feature": raw,
        "feature_length": torch.tensor([200]),
        "text": ["walk"],
    }

    out = build_window_local_model_batch(
        batch,
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
    )

    latent_len = int(out["feature_length"][0].item())
    traj_len = int(out["traj_num_tokens"][0].item())
    assert latent_len == 12
    assert traj_len == 19
    assert "token_mask" not in out
    assert out["latent_token_mask"].shape == (1, traj_len)
    assert torch.count_nonzero(out["latent_token_mask"][0, :latent_len]) == latent_len
    assert torch.count_nonzero(out["latent_token_mask"][0, latent_len:]) == 0

    _, traj_token_mask = encode_traj_batch(
        out,
        traj_len,
        "cpu",
        _MeanLocalTrajEncoder(),
        torch.nn.Identity(),
        return_token_mask=True,
    )

    assert traj_token_mask is not None
    assert traj_token_mask.shape == (1, traj_len)
    assert torch.allclose(traj_token_mask[0, :traj_len], torch.ones(traj_len))
    assert torch.count_nonzero(traj_token_mask[0, latent_len:traj_len]) == (
        traj_len - latent_len
    )


def test_window_local_model_batch_fills_uncovered_text_tail_with_empty_text():
    token = torch.arange(1 * 6 * 3, dtype=torch.float32).view(1, 6, 3)
    raw = _make_motion263(batch_size=1, num_frames=40)
    batch = {
        "token": token,
        "token_length": torch.tensor([6]),
        "feature": raw,
        "feature_length": torch.tensor([40]),
        "text": [["walk"]],
        "token_text_end": [[2]],
    }

    out = build_window_local_model_batch(
        batch,
        context_tokens=4,
        horizon_tokens=1,
        start_tokens=torch.tensor([0]),
    )

    assert out["text"] == [["walk", ""]]
    assert out["token_text_end"] == [[2, 4]]


def test_window_local_model_batch_rejects_malformed_segmented_text_schedule():
    token = torch.arange(1 * 8 * 3, dtype=torch.float32).view(1, 8, 3)
    raw = _make_motion263(batch_size=1, num_frames=60)
    batch = {
        "token": token,
        "token_length": torch.tensor([8]),
        "feature": raw,
        "feature_length": torch.tensor([60]),
        "text": [["walk", "run"]],
        "token_text_end": [[3]],
    }

    with pytest.raises(ValueError, match="text/end schedule mismatch"):
        build_window_local_model_batch(
            batch,
            context_tokens=4,
            horizon_tokens=1,
            start_tokens=torch.tensor([0]),
        )

    batch["token_text_end"] = [[5, 4]]
    with pytest.raises(ValueError, match="monotonic"):
        build_window_local_model_batch(
            batch,
            context_tokens=4,
            horizon_tokens=1,
            start_tokens=torch.tensor([0]),
        )


def test_shifted_local_time_steps_match_global_prefix_beta_schedule():
    from models.diffusion_forcing_wan import DiffForcingWanModel
    from utils.training.self_forcing import shifted_local_time_steps

    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model.chunk_size = 4

    start_tokens = torch.tensor([5, 0], dtype=torch.long)
    local_end_indices = torch.tensor([7, 3], dtype=torch.long)
    phase_offset = torch.tensor([0.125, 0.5], dtype=torch.float32)

    local_time = shifted_local_time_steps(
        local_end_indices,
        start_tokens=start_tokens,
        chunk_size=model.chunk_size,
        phase_offset=phase_offset,
    )
    global_end_indices = start_tokens + local_end_indices
    global_time = (
        (global_end_indices.to(dtype=torch.float32) - 1.0) / model.chunk_size
        + phase_offset
    )
    expected_local_time = global_time - start_tokens.to(dtype=torch.float32) / model.chunk_size
    assert torch.allclose(local_time, expected_local_time)

    full_beta = model._get_noise_levels(
        torch.device("cpu"),
        int(global_end_indices.max().item()),
        global_time,
    )
    local_beta = model._get_noise_levels(
        torch.device("cpu"),
        int(local_end_indices.max().item()),
        local_time,
    )

    for b in range(start_tokens.numel()):
        start = int(start_tokens[b].item())
        length = int(local_end_indices[b].item())
        assert torch.allclose(
            local_beta[b, :length],
            full_beta[b, start:start + length],
        )


def test_training_step_uses_window_local_model_batch_when_enabled():
    token = torch.arange(1 * 10 * 3, dtype=torch.float32).view(1, 10, 3)
    raw = _make_motion263(batch_size=1, num_frames=80)
    batch = {
        "token": token,
        "token_length": torch.tensor([10]),
        "feature": raw,
        "feature_length": torch.tensor([80]),
        "traj_cond_7d": torch.zeros(1, 80, 7),
        "traj_length": torch.tensor([80]),
        "text": ["walk"],
    }

    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "stream_training": {
            "enabled": True,
            "context_tokens": 4,
            "horizon_tokens": 2,
        },
    }.get(key, default)
    module = SimpleNamespace(cfg=cfg, trainer=None)
    trainer = SelfForcingTrainer.__new__(SelfForcingTrainer)
    trainer._module = module
    trainer._preconditions_checked = False
    trainer._self_forcing_step = MagicMock(return_value=torch.tensor(3.0))

    out = trainer.training_step(batch)

    assert float(out.item()) == 3.0
    loss_batch, model_batch = trainer._self_forcing_step.call_args.args
    assert "traj_cond_7d" in loss_batch
    assert "traj_length" in loss_batch
    assert model_batch["_window_local_traj"] is True
    assert model_batch["feature"].shape == (1, 6, 3)
    assert model_batch["feature_length"].tolist() == [4]
    assert model_batch["traj_num_tokens"].tolist() == [6]


def test_splice_window_local_pred_to_prefix_preserves_prefix_context():
    token = torch.arange(1 * 8 * 2, dtype=torch.float32).view(1, 8, 2)
    local_pred = [torch.full((3, 2), 100.0, requires_grad=True)]
    batch = {
        "token": token,
        "_window_local_latent_start_token": torch.tensor([2]),
    }

    spliced = _splice_window_local_pred_to_prefix(local_pred, batch, torch.device("cpu"))

    assert len(spliced) == 1
    assert spliced[0].shape == (5, 2)
    assert torch.allclose(spliced[0][:2], token[0, :2])
    assert torch.allclose(spliced[0][2:], local_pred[0])
    assert spliced[0].requires_grad


def test_splice_window_local_pred_to_prefix_rejects_invalid_start_shape_and_range():
    token = torch.arange(1 * 4 * 2, dtype=torch.float32).view(1, 4, 2)
    local_pred = [torch.full((2, 2), 100.0, requires_grad=True)]

    with pytest.raises(ValueError, match="one value per pred"):
        _splice_window_local_pred_to_prefix(
            local_pred,
            {
                "token": token,
                "_window_local_latent_start_token": torch.tensor([1, 2]),
            },
            torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="exceeds original token length"):
        _splice_window_local_pred_to_prefix(
            local_pred,
            {
                "token": token,
                "_window_local_latent_start_token": torch.tensor([5]),
            },
            torch.device("cpu"),
        )


def test_splice_window_local_pred_to_prefix_rejects_pred_that_exceeds_window_or_prefix():
    token = torch.arange(1 * 8 * 2, dtype=torch.float32).view(1, 8, 2)

    with pytest.raises(ValueError, match="exceeds window-local valid length"):
        _splice_window_local_pred_to_prefix(
            [torch.full((4, 2), 100.0, requires_grad=True)],
            {
                "token": token,
                "_window_local_latent_start_token": torch.tensor([2]),
                "_window_local_latent_valid_len": torch.tensor([3]),
            },
            torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="extends past original token length"):
        _splice_window_local_pred_to_prefix(
            [torch.full((2, 2), 100.0, requires_grad=True)],
            {
                "token": token[:, :4],
                "_window_local_latent_start_token": torch.tensor([3]),
            },
            torch.device("cpu"),
        )


def test_body_aux_wrapper_splices_window_local_pred_before_decode(monkeypatch):
    token = torch.arange(1 * 8 * 2, dtype=torch.float32).view(1, 8, 2)
    local_pred = [torch.full((3, 2), 100.0, requires_grad=True)]
    batch = {
        "token": token,
        "_window_local_latent_start_token": torch.tensor([2]),
        "traj_cond_7d": torch.zeros(1, 40, 7),
        "traj_length": torch.tensor([40]),
    }
    captured = {}

    def fake_compute_body_aux_loss(pred_list, *args, **kwargs):
        captured["pred"] = pred_list
        captured["window_start_tokens"] = kwargs.get("window_start_tokens")
        return torch.tensor(1.25), {"root_xz": 1.25}

    monkeypatch.setattr(sf_mod, "compute_body_aux_loss", fake_compute_body_aux_loss)
    module = SimpleNamespace(device=torch.device("cpu"), vae=None, model=SimpleNamespace(chunk_size=1))

    loss, terms = sf_mod._compute_body_aux_loss(
        local_pred,
        batch,
        module,
        sample_loss_mask=None,
        ba_cfg={"weights": {}},
    )

    assert float(loss.item()) == 1.25
    assert terms == {"root_xz": 1.25}
    assert captured["pred"][0].shape == (5, 2)
    assert torch.allclose(captured["pred"][0][:2], token[0, :2])
    assert torch.allclose(captured["pred"][0][2:], local_pred[0])
    assert captured["window_start_tokens"].tolist() == [2]


def test_collect_window_local_metrics_summarizes_sampling_contract():
    batch = {
        "_window_local_traj": True,
        "_window_local_latent_start_token": torch.tensor([3, 0]),
        "_window_local_latent_valid_len": torch.tensor([4, 3]),
        "_window_local_sample_policy": "fixed_window",
        "traj_num_tokens": torch.tensor([6, 5]),
    }

    metrics = _collect_window_local_metrics(batch)

    assert metrics["stream_training/enabled"] == 1.0
    assert metrics["stream_training/sample_policy_fixed_window"] == 1.0
    assert metrics["stream_training/window_start_mean"] == 1.5
    assert metrics["stream_training/window_len_mean"] == 3.5
    assert metrics["stream_training/traj_tokens_mean"] == 5.5


def test_self_forcing_step_logs_window_local_metrics(monkeypatch):
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.1)
    captured = {}
    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "self_forcing_grad_clip": 1.0,
    }.get(key, default)
    module = SimpleNamespace(
        model=SimpleNamespace(parameters=lambda: [param]),
        cfg=cfg,
        trainer=None,
        optimizers=lambda: optimizer,
        lr_schedulers=lambda: None,
        manual_backward=lambda loss: loss.backward(),
        _log_step_metrics=lambda *args, **kwargs: captured.update(kwargs),
    )
    trainer = SelfForcingTrainer.__new__(SelfForcingTrainer)
    trainer._module = module
    trainer._grad_clip_val = None
    trainer._last_replace_diff = None
    trainer._last_corruption_applied = 0.0
    trainer._last_horizon_tokens = -1.0
    trainer._last_sample_loss_mask = None
    trainer._last_body_aux_terms = None
    trainer._last_window_local_rollout_metrics = {
        "stream_training/active_history_len_mean": 4.0,
        "stream_training/active_history_len_min": 4.0,
        "stream_training/active_history_len_max": 4.0,
        "stream_training/active_abs_end_mean": 7.0,
    }
    trainer._build_runtime_metrics = lambda: (
        SimpleNamespace(progress=0.0),
        {
            "self_forcing/phase_step": 1.0,
            "self_forcing/enabled": 1.0,
            "self_forcing/active": 1.0,
        },
    )
    trainer._log_metrics = lambda metrics: None
    trainer._run_rollout = MagicMock(
        return_value=({"loss": param * 0.0 + 1.0}, 1)
    )
    trainer._compute_losses = MagicMock(
        return_value=(param * 0.0 + 1.0, param * 0.0 + 1.0, None)
    )
    model_batch = {
        "_window_local_traj": True,
        "_window_local_latent_start_token": torch.tensor([3, 0]),
        "_window_local_latent_valid_len": torch.tensor([4, 3]),
        "_window_local_sample_policy": "fixed_window",
        "traj_num_tokens": torch.tensor([6, 5]),
    }

    trainer._self_forcing_step({}, model_batch)

    extra = captured["extra_metrics"]
    assert extra["stream_training/enabled"] == 1.0
    assert extra["stream_training/sample_policy_fixed_window"] == 1.0
    assert extra["stream_training/window_len_mean"] == 3.5
    assert extra["stream_training/active_history_len_mean"] == 4.0
    assert extra["stream_training/active_abs_end_mean"] == 7.0


def test_forward_single_window_returns_pred_latents_for_window_local_traj_features():
    from models.diffusion_forcing_wan import DiffForcingWanModel

    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model.chunk_size = 1
    model.prediction_type = "vel"
    model.time_embedding_scale = 1.0
    model.add_noise = lambda clean, noise_level: (clean, torch.zeros_like(clean))
    model.preprocess = lambda x: x.permute(0, 2, 1)[:, :, :, None, None]
    model._get_noise_levels = lambda device, seq_len, time_steps: torch.zeros(
        time_steps.shape[0], seq_len, device=device
    )
    model._controlnet_forward = lambda *args, **kwargs: None

    def fake_backbone(noisy_input, *args, **kwargs):
        return [torch.zeros_like(noisy_input[0])]

    model.model = fake_backbone
    x = {
        "feature_length": torch.tensor([2]),
        "traj_features": torch.zeros(1, 8, 7),
        "_window_local_traj": True,
    }

    out = model._forward_single_window(
        x,
        clean_feature=torch.ones(1, 2, 3),
        time_steps=torch.tensor([2.0]),
        all_text_context=[torch.zeros(1)],
        traj_emb=torch.zeros(1, 2, 3),
        traj_seq_lens=torch.tensor([2]),
        traj_dropped=False,
    )

    assert out["pred_x0_latent_list"] is not None
    assert out["pred_x0_latent_list"][0].shape == (2, 3)


def test_stream_training_default_full_prefix_smoke_with_real_tiny_model(tmp_path, monkeypatch):
    from models.diffusion_forcing_wan import DiffForcingWanModel
    import models.tools.wan_model as wan_model_mod

    def cpu_flash_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):
        outs = []
        for b in range(q.shape[0]):
            q_len = int(q_lens[b].item()) if q_lens is not None else q.shape[1]
            k_len = int(k_lens[b].item()) if k_lens is not None else k.shape[1]
            qb = q[b:b + 1, :q_len].transpose(1, 2)
            kb = k[b:b + 1, :k_len].transpose(1, 2)
            vb = v[b:b + 1, :k_len].transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb)
            out = out.transpose(1, 2)
            if q_len < q.shape[1]:
                pad = q.new_zeros(1, q.shape[1] - q_len, q.shape[2], v.shape[-1])
                out = torch.cat([out, pad], dim=1)
            outs.append(out)
        return torch.cat(outs, dim=0)

    monkeypatch.setattr(wan_model_mod, "flash_attention", cpu_flash_attention)

    text_dim = 4096
    text_path = tmp_path / "text.pt"
    torch.save(
        {
            "embeddings": {
                "": torch.zeros(2, text_dim),
                "walk": torch.ones(2, text_dim) * 0.01,
            },
            "text_dim": text_dim,
        },
        text_path,
    )
    model = DiffForcingWanModel(
        input_dim=3,
        hidden_dim=32,
        ffn_dim=64,
        freq_dim=32,
        num_heads=2,
        num_layers=1,
        text_len=4,
        chunk_size=1,
        traj_encoder_in_dim=7,
        traj_out_dim=32,
        traj_dropout=0.0,
        text_dropout=0.0,
        freeze_backbone=False,
        use_precomputed_text_emb=True,
        precomputed_text_emb_path=str(text_path),
        self_forcing_enabled=True,
        self_forcing_k_schedule=((0.0, 1),),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    raw = _make_motion263(batch_size=1, num_frames=80)
    batch = {
        "token": torch.randn(1, 8, 3),
        "token_length": torch.tensor([8]),
        "feature": raw,
        "feature_length": torch.tensor([80]),
        "traj_cond_7d": torch.zeros(1, 80, 7),
        "traj_length": torch.tensor([80]),
        "text": ["walk"],
    }
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace(params={"control_loss_weight": 0.0})
    cfg.get = lambda key, default=None: {
        "stream_training": {
            "enabled": True,
            "context_tokens": 4,
            "min_history_tokens": 1,
            "horizon_tokens": 2,
        },
        "anchor_canonicalize": {"enabled": True},
        "horizon_sim": {"enabled": False},
        "history_corruption": {},
        "body_aux_loss": {"enabled": True},
        "self_forcing_disable_replace": False,
        "self_forcing_grad_clip": 1.0,
    }.get(key, default)
    module = SimpleNamespace(
        model=model,
        cfg=cfg,
        trainer=None,
        global_step=0,
        _resume_step_offset=0,
        optimizers=lambda: optimizer,
        lr_schedulers=lambda: None,
        manual_backward=lambda loss: loss.backward(),
        _log_step_metrics=lambda *args, **kwargs: None,
    )
    trainer = SelfForcingTrainer(module)

    loss = trainer.training_step(batch)

    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_stream_training_full_prefix_motion_aux_smoke_with_real_tiny_model(tmp_path, monkeypatch):
    from models.diffusion_forcing_wan import DiffForcingWanModel
    import models.tools.wan_model as wan_model_mod
    from utils.token_frame import num_frames_for_tokens

    def cpu_flash_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):
        outs = []
        for b in range(q.shape[0]):
            q_len = int(q_lens[b].item()) if q_lens is not None else q.shape[1]
            k_len = int(k_lens[b].item()) if k_lens is not None else k.shape[1]
            qb = q[b:b + 1, :q_len].transpose(1, 2)
            kb = k[b:b + 1, :k_len].transpose(1, 2)
            vb = v[b:b + 1, :k_len].transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb)
            out = out.transpose(1, 2)
            if q_len < q.shape[1]:
                pad = q.new_zeros(1, q.shape[1] - q_len, q.shape[2], v.shape[-1])
                out = torch.cat([out, pad], dim=1)
            outs.append(out)
        return torch.cat(outs, dim=0)

    class DummyVAE:
        def decode(self, latents):
            frames = num_frames_for_tokens(int(latents.shape[1]))
            out = latents.new_zeros(latents.shape[0], frames, 263)
            out[..., 3] = 1.0
            return out + latents.sum() * 0.0

    monkeypatch.setattr(wan_model_mod, "flash_attention", cpu_flash_attention)
    text_dim = 4096
    text_path = tmp_path / "text.pt"
    torch.save(
        {
            "embeddings": {
                "": torch.zeros(2, text_dim),
                "walk": torch.ones(2, text_dim) * 0.01,
            },
            "text_dim": text_dim,
        },
        text_path,
    )
    model = DiffForcingWanModel(
        input_dim=3,
        hidden_dim=32,
        ffn_dim=64,
        freq_dim=32,
        num_heads=2,
        num_layers=1,
        text_len=4,
        chunk_size=1,
        traj_encoder_in_dim=7,
        traj_out_dim=32,
        traj_dropout=0.0,
        text_dropout=0.0,
        freeze_backbone=False,
        use_precomputed_text_emb=True,
        precomputed_text_emb_path=str(text_path),
        self_forcing_enabled=True,
        self_forcing_k_schedule=((0.0, 1),),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    raw = _make_motion263(batch_size=1, num_frames=80)
    batch = {
        "token": torch.randn(1, 8, 3),
        "token_length": torch.tensor([8]),
        "feature": raw,
        "feature_length": torch.tensor([80]),
        "traj_cond_7d": torch.zeros(1, 80, 7),
        "traj_length": torch.tensor([80]),
        "text": ["walk"],
    }
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace(params={"control_loss_weight": 0.1})
    cfg.get = lambda key, default=None: {
        "stream_training": {
            "enabled": True,
            "context_tokens": 4,
            "min_history_tokens": 1,
            "horizon_tokens": 2,
        },
        "anchor_canonicalize": {"enabled": True},
        "horizon_sim": {"enabled": False},
        "history_corruption": {},
        "body_aux_loss": {"enabled": True, "weights": {}},
        "self_forcing_disable_replace": False,
        "self_forcing_grad_clip": 1.0,
    }.get(key, default)
    module = SimpleNamespace(
        model=model,
        cfg=cfg,
        device=torch.device("cpu"),
        vae=DummyVAE(),
        trainer=None,
        global_step=0,
        _resume_step_offset=0,
        optimizers=lambda: optimizer,
        lr_schedulers=lambda: None,
        manual_backward=lambda loss: loss.backward(),
        _log_step_metrics=lambda *args, **kwargs: None,
    )
    trainer = SelfForcingTrainer(module)

    loss = trainer.training_step(batch)

    assert torch.isfinite(loss)
    assert loss.requires_grad
