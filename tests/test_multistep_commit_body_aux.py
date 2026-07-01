from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from utils.training.self_forcing import RolloutPlan, SelfForcingTrainer


_BODY_AUX_WEIGHTS = {
    "root_xz": 2.0,
    "root_y": 0.1,
    "heading": 0.5,
    "fwd_delta": 0.1,
    "yaw_delta": 0.1,
    "end_xz": 1.0,
}

_STALE_COMMIT_WEIGHTS = {
    "root_xz": 99.0,
    "root_y": 99.0,
    "heading": 99.0,
    "fwd_delta": 99.0,
    "yaw_delta": 99.0,
    "end_xz": 99.0,
}


def _cfg(
    commit_enabled: bool,
    disable_replace: bool = True,
    strict_valid_commits: bool = False,
    legacy_commit_overrides: bool = False,
):
    commit_cfg = {
        "enabled": commit_enabled,
        "decode_mode": "single",
        "replace_final_body_aux": True,
        "include_final_step": True,
        "reduction": "step_mean",
        "strict_valid_commits": strict_valid_commits,
    }
    if legacy_commit_overrides:
        commit_cfg["weight"] = 7.0
        commit_cfg["weights"] = _STALE_COMMIT_WEIGHTS

    def get(key, default=None):
        values = {
            "anchor_canonicalize": {"enabled": False},
            "body_aux_loss": {
                "enabled": True,
                "heading_form": "cosine",
                "weights": _BODY_AUX_WEIGHTS,
            },
            "history_corruption": {},
            "horizon_sim": {"enabled": False},
            "self_forcing_disable_replace": disable_replace,
            "multistep_commit_body_aux": commit_cfg,
        }
        return values.get(key, default)

    return SimpleNamespace(get=get)


def _trainer(
    commit_enabled: bool,
    k: int = 3,
    disable_replace: bool = True,
    chunk_size: int = 1,
    strict_valid_commits: bool = False,
    legacy_commit_overrides: bool = False,
):
    batch = 1
    seq_len = 5
    hidden = 4
    feature = torch.zeros(batch, seq_len, hidden)
    model = MagicMock(name="model")
    model.chunk_size = chunk_size
    model.self_forcing_stride_tokens = 1
    model.self_forcing_k_schedule = [(0.0, k)]
    model._decide_text_dropout.return_value = torch.zeros(batch, dtype=torch.bool)
    model._prepare_text_context.return_value = None
    model._decide_traj_dropout.return_value = False
    model._prepare_traj_condition.return_value = (None, None, False, None)

    grad_flags = []
    seen_features = []

    def forward(_model_batch, current_feature, *args, **kwargs):
        grad_flags.append(torch.is_grad_enabled())
        seen_features.append(current_feature.detach().clone())
        pred = torch.ones(seq_len, hidden, requires_grad=torch.is_grad_enabled())
        repl = torch.full((seq_len, hidden), 2.0)
        return {
            "loss": pred.sum() * 0.0,
            "pred_x0_latent_list": [pred],
            "x0_latent_list": [repl],
        }

    model._forward_single_window.side_effect = forward
    module = SimpleNamespace(
        model=model,
        cfg=_cfg(
            commit_enabled,
            disable_replace,
            strict_valid_commits,
            legacy_commit_overrides,
        ),
    )
    trainer = SelfForcingTrainer.__new__(SelfForcingTrainer)
    trainer._module = module
    trainer._last_replace_diff = None
    trainer._last_sample_loss_mask = None
    trainer._last_horizon_tokens = -1.0
    trainer._last_corruption_applied = 0.0
    trainer.plan_rollout = MagicMock(
        return_value=RolloutPlan(
            effective_k=k,
            start_end_indices=torch.tensor([1], dtype=torch.long),
            phase_offset=torch.tensor([0.0]),
        )
    )
    model_batch = {
        "feature": feature,
        "feature_length": torch.tensor([seq_len], dtype=torch.long),
    }
    return trainer, model_batch, grad_flags, seen_features


def test_default_rollout_keeps_non_final_steps_no_grad():
    trainer, model_batch, grad_flags, _ = _trainer(commit_enabled=False, k=3)

    final_result, k = trainer._run_rollout(model_batch, progress=1.0)

    assert k == 3
    assert final_result is not None
    assert grad_flags == [False, False, True]


def test_commit_aux_rollout_enables_grad_for_every_step_and_records_commits():
    trainer, model_batch, grad_flags, _ = _trainer(commit_enabled=True, k=3)

    final_result, k = trainer._run_rollout(model_batch, progress=1.0)

    assert k == 3
    assert final_result is not None
    assert grad_flags == [True, True, True]
    records = trainer._last_commit_token_records
    assert [int(r.local_commit_idx) for r in records] == [0, 1, 2]
    assert [int(r.global_commit_idx) for r in records] == [0, 1, 2]
    assert all(r.pred_token.requires_grad for r in records)


def test_replacement_uses_detached_x0_latent_list_not_pred_x0():
    trainer, model_batch, _, seen_features = _trainer(
        commit_enabled=True, k=2, disable_replace=False
    )

    trainer._run_rollout(model_batch, progress=1.0)

    assert torch.equal(seen_features[1][0, 0], torch.full((4,), 2.0))
    assert not seen_features[1].requires_grad


def test_commit_aux_rejects_chunk_size_greater_than_one_for_first_version():
    trainer, model_batch, _, _ = _trainer(commit_enabled=True, k=2, chunk_size=2)

    try:
        trainer._run_rollout(model_batch, progress=1.0)
    except NotImplementedError as exc:
        assert "chunk_size == 1" in str(exc)
    else:
        raise AssertionError("expected chunk_size > 1 to be rejected")


def test_full_prefix_splice_assembly_decodes_only_to_global_commit_prefix():
    from utils.token_frame import token_range_to_frame_slice

    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    batch = {
        "token": torch.zeros(1, 6, 2),
        "token_length": torch.tensor([6]),
        "traj_cond_7d": torch.zeros(1, 24, 7),
        "traj_length": torch.tensor([24]),
    }
    model_batch = {
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([2]),
        "_window_local_latent_start_token": torch.tensor([99]),
        "feature": torch.zeros(1, 4, 2),
        "feature_length": torch.tensor([4]),
    }
    record = torch.ones(2, requires_grad=True)
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=0,
            local_commit_idx=1,
            global_commit_idx=3,
            pred_token=record,
        )
    ]

    decoded_latents, commit_masks, commit_positions, sample_indices, window_starts = (
        trainer._assemble_commit_decode_inputs(batch, model_batch)
    )

    assert decoded_latents[0].shape[0] == 4
    assert torch.equal(decoded_latents[0][3], record)
    assert decoded_latents[0][3].requires_grad
    decoded_latents[0][2].sum().backward(retain_graph=True)
    assert record.grad is None or torch.equal(record.grad, torch.zeros_like(record))
    record.grad = None
    decoded_latents[0][3].sum().backward(retain_graph=True)
    assert torch.equal(record.grad, torch.ones_like(record))
    assert commit_positions.tolist() == [0]
    assert sample_indices.tolist() == [0]
    assert window_starts.tolist() == [0]
    sl = token_range_to_frame_slice(3, 1)
    assert commit_masks.shape[0] == 1
    assert commit_masks[0, sl.start:sl.stop].sum().item() == 4
    assert commit_masks[0, :sl.start].sum().item() == 0
    assert commit_masks[0, sl.stop:].sum().item() == 0


def test_local_decode_assembly_decodes_only_to_local_commit_prefix():
    from utils.token_frame import token_range_to_frame_slice

    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    batch = {
        "token": torch.zeros(1, 6, 2),
        "token_length": torch.tensor([6]),
        "traj_cond_7d": torch.zeros(1, 24, 7),
        "traj_length": torch.tensor([24]),
    }
    model_batch = {
        "_window_local_body_aux_mode": "local_decode",
        "_window_global_start_token": torch.tensor([5]),
        "_window_local_latent_start_token": torch.tensor([0]),
        "feature": torch.zeros(1, 4, 2),
        "feature_length": torch.tensor([4]),
    }
    record = torch.ones(2, requires_grad=True)
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=0,
            local_commit_idx=1,
            global_commit_idx=6,
            pred_token=record,
        )
    ]

    decoded_latents, commit_masks, commit_positions, sample_indices, window_starts = (
        trainer._assemble_commit_decode_inputs(batch, model_batch)
    )

    assert decoded_latents[0].shape[0] == 2
    assert torch.equal(decoded_latents[0][1], record)
    assert decoded_latents[0][1].requires_grad
    decoded_latents[0][0].sum().backward(retain_graph=True)
    assert record.grad is None or torch.equal(record.grad, torch.zeros_like(record))
    record.grad = None
    decoded_latents[0][1].sum().backward(retain_graph=True)
    assert torch.equal(record.grad, torch.ones_like(record))
    assert commit_positions.tolist() == [0]
    assert sample_indices.tolist() == [0]
    assert window_starts.tolist() == [5]
    sl = token_range_to_frame_slice(1, 1)
    assert commit_masks[0, sl.start:sl.stop].sum().item() == 4
    assert commit_masks[0, :sl.start].sum().item() == 0
    assert commit_masks[0, sl.stop:].sum().item() == 0


def test_local_prefix_assembly_preserves_sample_indices_and_expands_scalar_starts():
    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    batch = {
        "token": torch.zeros(3, 6, 2),
        "token_length": torch.tensor([6, 6, 6]),
        "traj_cond_7d": torch.zeros(3, 24, 7),
        "traj_length": torch.tensor([24, 24, 24]),
    }
    model_batch = {
        "_window_local_body_aux_mode": "local_decode",
        "_window_global_start_token": torch.tensor([2]),
        "feature": torch.zeros(3, 4, 2),
        "feature_length": torch.tensor([4, 4, 4]),
    }
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=2,
            local_commit_idx=1,
            global_commit_idx=3,
            pred_token=torch.ones(2, requires_grad=True),
        )
    ]

    _, _, commit_positions, sample_indices, window_starts = (
        trainer._assemble_commit_decode_inputs(batch, model_batch)
    )

    assert commit_positions.tolist() == [0]
    assert sample_indices.tolist() == [2]
    assert window_starts.tolist() == [2]


def test_prefix_assembly_does_not_include_future_gt_suffix():
    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    batch = {
        "token": torch.zeros(1, 10, 2),
        "token_length": torch.tensor([10]),
        "traj_cond_7d": torch.zeros(1, 40, 7),
        "traj_length": torch.tensor([40]),
    }
    batch["token"][0, 4:, :] = 99.0
    model_batch = {
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([2]),
        "feature": torch.zeros(1, 4, 2),
        "feature_length": torch.tensor([4]),
    }
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=0,
            local_commit_idx=1,
            global_commit_idx=3,
            pred_token=torch.ones(2, requires_grad=True),
        )
    ]

    decoded_latents, _, _, _, _ = trainer._assemble_commit_decode_inputs(
        batch, model_batch
    )

    assert decoded_latents[0].shape[0] == 4
    assert not torch.any(decoded_latents[0] == 99.0)


def test_commit_aux_replaces_final_body_aux(monkeypatch):
    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    trainer._module.device = torch.device("cpu")
    trainer._module.vae = MagicMock()
    trainer._module.cfg.model = SimpleNamespace(params={"control_loss_weight": 5.0})
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=0,
            local_commit_idx=0,
            global_commit_idx=0,
            pred_token=torch.ones(4, requires_grad=True),
        )
    ]
    final_step_result = {
        "loss": torch.tensor(2.0, requires_grad=True),
        "pred_x0_latent_list": [torch.ones(1, 4, requires_grad=True)],
    }
    batch = {
        "token": torch.zeros(1, 1, 4),
        "token_length": torch.tensor([1]),
        "traj_cond_7d": torch.zeros(1, 1, 7),
        "traj_length": torch.tensor([1]),
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([0]),
    }
    model_batch = {
        "feature": torch.zeros(1, 1, 4),
        "feature_length": torch.tensor([1]),
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([0]),
    }
    monkeypatch.setattr(
        "utils.training.self_forcing._compute_body_aux_loss",
        MagicMock(side_effect=AssertionError("final body aux should be skipped")),
    )
    monkeypatch.setattr(
        "utils.training.self_forcing.compute_body_aux_loss_on_commit_masks",
        MagicMock(return_value=(torch.tensor(0.5), {"root_xz": 0.5})),
        raising=False,
    )
    monkeypatch.setattr(
        trainer,
        "_decode_commit_latents",
        MagicMock(return_value=[torch.zeros(1, 263)]),
        raising=False,
    )

    total, diff, control = trainer._compute_losses(
        final_step_result, batch, model_batch
    )

    assert torch.isclose(total, torch.tensor(4.5))
    assert torch.isclose(diff, torch.tensor(2.0))
    assert torch.isclose(control, torch.tensor(0.5))
    assert trainer._last_commit_body_aux_valid_count == 1


def test_commit_aux_uses_body_aux_weights_and_ignores_multistep_overrides(monkeypatch):
    trainer, _, _, _ = _trainer(
        commit_enabled=True, k=1, legacy_commit_overrides=True
    )
    trainer._module.device = torch.device("cpu")
    trainer._module.vae = MagicMock()
    trainer._module.cfg.model = SimpleNamespace(params={"control_loss_weight": 5.0})
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=0,
            local_commit_idx=0,
            global_commit_idx=0,
            pred_token=torch.ones(4, requires_grad=True),
        )
    ]
    final_step_result = {
        "loss": torch.tensor(2.0, requires_grad=True),
        "pred_x0_latent_list": [torch.ones(1, 4, requires_grad=True)],
    }
    batch = {
        "token": torch.zeros(1, 1, 4),
        "token_length": torch.tensor([1]),
        "traj_cond_7d": torch.zeros(1, 1, 7),
        "traj_length": torch.tensor([1]),
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([0]),
    }
    model_batch = {
        "feature": torch.zeros(1, 1, 4),
        "feature_length": torch.tensor([1]),
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([0]),
    }
    captured = {}

    def fake_commit_helper(*args, **kwargs):
        captured["weights"] = args[7]
        return torch.tensor(0.5), {"valid_count": 1.0}

    monkeypatch.setattr(
        "utils.training.self_forcing.compute_body_aux_loss_on_commit_masks",
        MagicMock(side_effect=fake_commit_helper),
    )
    monkeypatch.setattr(
        trainer,
        "_decode_commit_latents",
        MagicMock(return_value=[torch.zeros(1, 263)]),
    )

    total, _, _ = trainer._compute_losses(final_step_result, batch, model_batch)

    assert captured["weights"] == _BODY_AUX_WEIGHTS
    assert torch.isclose(total, torch.tensor(4.5))


def test_commit_aux_none_does_not_fallback_to_final_body_aux(monkeypatch):
    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    trainer._module.device = torch.device("cpu")
    trainer._module.vae = MagicMock()
    trainer._module.cfg.model = SimpleNamespace(params={"control_loss_weight": 5.0})
    trainer._last_commit_token_records = []
    final_step_result = {
        "loss": torch.tensor(2.0, requires_grad=True),
        "pred_x0_latent_list": [torch.ones(1, 4, requires_grad=True)],
    }
    batch = {
        "token": torch.zeros(1, 1, 4),
        "token_length": torch.tensor([1]),
        "traj_cond_7d": torch.zeros(1, 1, 7),
        "traj_length": torch.tensor([1]),
    }
    model_batch = {
        "feature": torch.zeros(1, 1, 4),
        "feature_length": torch.tensor([1]),
    }
    monkeypatch.setattr(
        "utils.training.self_forcing._compute_body_aux_loss",
        MagicMock(side_effect=AssertionError("must not fallback")),
    )

    total, diff, control = trainer._compute_losses(
        final_step_result, batch, model_batch
    )

    assert torch.isclose(total, torch.tensor(2.0))
    assert torch.isclose(diff, torch.tensor(2.0))
    assert control is None
    assert trainer._last_commit_body_aux_valid_count == 0


def test_commit_aux_none_raises_in_strict_mode(monkeypatch):
    trainer, _, _, _ = _trainer(
        commit_enabled=True, k=1, strict_valid_commits=True
    )
    trainer._module.device = torch.device("cpu")
    trainer._module.vae = MagicMock()
    trainer._module.cfg.model = SimpleNamespace(params={"control_loss_weight": 5.0})
    trainer._last_commit_token_records = []
    final_step_result = {
        "loss": torch.tensor(2.0, requires_grad=True),
        "pred_x0_latent_list": [torch.ones(1, 4, requires_grad=True)],
    }
    batch = {
        "token": torch.zeros(1, 1, 4),
        "token_length": torch.tensor([1]),
        "traj_cond_7d": torch.zeros(1, 1, 7),
        "traj_length": torch.tensor([1]),
    }
    model_batch = {
        "feature": torch.zeros(1, 1, 4),
        "feature_length": torch.tensor([1]),
    }
    monkeypatch.setattr(
        "utils.training.self_forcing._compute_body_aux_loss",
        MagicMock(side_effect=AssertionError("must not fallback")),
    )

    try:
        trainer._compute_losses(final_step_result, batch, model_batch)
    except RuntimeError as exc:
        assert "no valid commit body aux loss" in str(exc)
    else:
        raise AssertionError("expected strict mode to reject missing commit loss")
