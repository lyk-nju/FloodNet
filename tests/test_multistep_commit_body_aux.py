from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from utils.training.self_forcing import RolloutPlan, SelfForcingTrainer


def _cfg(
    commit_enabled: bool,
    disable_replace: bool = True,
    strict_valid_commits: bool = False,
):
    def get(key, default=None):
        values = {
            "anchor_canonicalize": {"enabled": False},
            "history_corruption": {},
            "horizon_sim": {"enabled": False},
            "self_forcing_disable_replace": disable_replace,
            "multistep_commit_body_aux": {
                "enabled": commit_enabled,
                "decode_mode": "single",
                "replace_final_body_aux": True,
                "include_final_step": True,
                "reduction": "step_mean",
                "strict_valid_commits": strict_valid_commits,
                "weight": 1.0,
                "weights": {
                    "root_xz": 1.0,
                    "root_y": 0.0,
                    "heading": 0.2,
                    "fwd_delta": 0.05,
                    "yaw_delta": 0.05,
                    "end_xz": 0.0,
                },
            },
        }
        return values.get(key, default)

    return SimpleNamespace(get=get)


def _trainer(
    commit_enabled: bool,
    k: int = 3,
    disable_replace: bool = True,
    chunk_size: int = 1,
    strict_valid_commits: bool = False,
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
        cfg=_cfg(commit_enabled, disable_replace, strict_valid_commits),
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
