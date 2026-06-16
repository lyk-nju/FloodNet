"""Regression: SelfForcingTrainer._run_rollout must

  1. unpack `prepare_traj_condition`'s 4-tuple (B-P0-1: added traj_token_mask),
  2. thread `traj_token_mask` through every `PreparedCondition` consumed by
     `run_training_window` —
     both the no_grad rollout steps (0..K-2) and the supervised final step.

Before the fix the trainer unpacked 3 values and silently dropped the mask, so
SF training raised `ValueError: too many values to unpack (expected 3)` the
moment self-forcing engaged.

The test bypasses the real model: `_run_rollout` only touches the
`SelfForcingTrainer._module.{model, cfg}` surface, so a SimpleNamespace + a
handful of MagicMocks reproduce the exact call shape this regression cares
about — no torch ckpts, no T5, no DDP.
"""

from __future__ import annotations

import torch
import utils.training.ldf.self_forcing as sf_mod

from types import SimpleNamespace
from unittest.mock import MagicMock
from utils.training.ldf.conditioning import PreparedCondition
from utils.training.ldf.self_forcing import RolloutPlan, SelfForcingTrainer


_HIDDEN = 8
_BATCH = 1
_SEQ_LEN = 4
_CHUNK = 1
_K = 3


def _make_trainer():
    """Build the smallest object graph SelfForcingTrainer._run_rollout uses."""
    feature = torch.zeros(_BATCH, _SEQ_LEN, _HIDDEN)
    feature_length = torch.tensor([_SEQ_LEN], dtype=torch.long)

    # Sentinel mask: distinguishable in `assert_called_with` and easy to assert
    # by-identity below.
    sentinel_mask = torch.ones(_BATCH, _SEQ_LEN)

    # `prepare_traj_condition` is the 4-tuple producer the rewrite added.
    traj_emb_sentinel = torch.zeros(_BATCH, _SEQ_LEN, _HIDDEN)
    traj_seq_lens_sentinel = torch.tensor([_SEQ_LEN], dtype=torch.long)

    model = MagicMock(name="model")
    model.chunk_size = _CHUNK
    model._test_text_context = None
    model._test_text_dropped_flags = torch.zeros(_BATCH, dtype=torch.bool)
    model._test_traj_emb = traj_emb_sentinel
    model._test_traj_seq_lens = traj_seq_lens_sentinel
    model._test_traj_token_mask = sentinel_mask

    # `run_training_window` returns enough structure for the rollout-step
    # replacement code to no-op (replace_idx < 0 because chunk_size=1 and
    # end_indices start at 1 → replace_idx = 0; pred_seq must be shape-(T,H)).
    fake_x0 = [torch.zeros(_SEQ_LEN, _HIDDEN) for _ in range(_BATCH)]
    model.training_window_step = MagicMock(
        return_value={
            "loss": torch.tensor(0.0),
            "x0_latent_list": fake_x0,
        }
    )

    cfg = SimpleNamespace()
    cfg.self_forcing = SimpleNamespace(k_schedule=[(0.0, _K)], stride_tokens=1)
    cfg.get = lambda key, default=None: {
        "anchor_canonicalize": {"enabled": False},
        "horizon_sim": {"enabled": False},
        "history_corruption": {},
        "self_forcing_disable_replace": True,   # skip the substitute-token branch
    }.get(key, default)

    module = SimpleNamespace(model=model, cfg=cfg)
    trainer = SelfForcingTrainer.__new__(SelfForcingTrainer)
    trainer._module = module
    trainer._last_replace_diff = None
    trainer._last_sample_loss_mask = None
    trainer._last_horizon_tokens = -1.0
    trainer._last_corruption_applied = 0.0

    # Force a deterministic 3-step rollout: every step has end_index >= 1 so
    # plan.start_end_indices = [1] → end_indices at step k is [1+k].
    trainer.plan_rollout = MagicMock(
        return_value=RolloutPlan(
            effective_k=_K,
            start_end_indices=torch.tensor([1], dtype=torch.long),
            phase_offset=torch.tensor([0.0]),
        )
    )

    model_batch = {"feature": feature, "feature_length": feature_length}
    return trainer, model, model_batch, sentinel_mask


def _patch_run_training_window(monkeypatch, model):
    monkeypatch.setattr(sf_mod, "run_training_window", model.training_window_step)
    model.prepare_text_condition_step = MagicMock(
        return_value=(model._test_text_context, model._test_text_dropped_flags)
    )
    model.sample_traj_dropout_step = MagicMock(return_value=False)
    model.prepare_traj_condition_step = MagicMock(
        return_value=(
            model._test_traj_emb,
            model._test_traj_seq_lens,
            False,
            model._test_traj_token_mask,
        )
    )
    monkeypatch.setattr(
        sf_mod, "prepare_text_condition", model.prepare_text_condition_step
    )
    monkeypatch.setattr(
        sf_mod, "sample_traj_dropout", model.sample_traj_dropout_step
    )
    monkeypatch.setattr(
        sf_mod, "prepare_traj_condition", model.prepare_traj_condition_step
    )
    return model.training_window_step


def test_run_rollout_threads_traj_token_mask_to_every_step(monkeypatch):
    """Every `run_training_window` call (K rollout steps) must see the same
    `traj_token_mask` instance produced by `prepare_traj_condition`."""
    trainer, model, model_batch, sentinel_mask = _make_trainer()
    window_step = _patch_run_training_window(monkeypatch, model)

    final_result, k = trainer._run_rollout(model_batch, progress=1.0)

    assert k == _K
    assert final_result is not None
    assert window_step.call_count == _K

    for i, call in enumerate(window_step.call_args_list):
        condition = call.args[4]
        assert isinstance(condition, PreparedCondition)
        # Identity check — the trainer must pass the exact mask returned by
        # `prepare_traj_condition`, not silently re-derive or None it out.
        assert condition.traj_token_mask is sentinel_mask, (
            f"step {i}: traj_token_mask is not the value from prepare_traj_condition"
        )


def test_run_rollout_unpacks_four_tuple_from_traj_condition_helper(monkeypatch):
    """Direct guard: `_run_rollout` must call `prepare_traj_condition` exactly
    once and accept its 4-tuple return without raising. A regression to a
    3-tuple unpack would manifest here as `ValueError: too many values to
    unpack`."""
    trainer, model, model_batch, _ = _make_trainer()
    _patch_run_training_window(monkeypatch, model)

    trainer._run_rollout(model_batch, progress=1.0)

    assert model.prepare_traj_condition_step.call_count == 1


def test_run_rollout_canonicalizes_7d_traj_by_default(monkeypatch):
    trainer, _, model_batch, _ = _make_trainer()
    _patch_run_training_window(monkeypatch, trainer._module.model)
    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "horizon_sim": {"enabled": False},
        "history_corruption": {},
        "self_forcing_disable_replace": True,
    }.get(key, default)
    trainer._module.cfg = cfg
    traj_features = torch.zeros(_BATCH, _SEQ_LEN, 7)
    canon = torch.ones_like(traj_features)
    called = {}

    def fake_canonicalize(traj_cond_7d, *args, **kwargs):
        called["traj_cond_7d"] = traj_cond_7d
        return canon, torch.ones(_BATCH)

    monkeypatch.setattr(sf_mod, "apply_body_window_canonicalize", fake_canonicalize)

    trainer._run_rollout(
        {
            **model_batch,
            "traj_features": traj_features,
            "traj_length": torch.tensor([_SEQ_LEN], dtype=torch.long),
        },
        progress=1.0,
    )

    assert called["traj_cond_7d"] is traj_features
    assert torch.equal(trainer._last_sample_loss_mask, torch.ones(_BATCH))


def test_run_rollout_anchor_canonicalize_ablation_override_disables(monkeypatch):
    trainer, _, model_batch, _ = _make_trainer()
    _patch_run_training_window(monkeypatch, trainer._module.model)
    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "anchor_canonicalize": {"enabled": False},
        "horizon_sim": {"enabled": False},
        "history_corruption": {},
        "self_forcing_disable_replace": True,
    }.get(key, default)
    trainer._module.cfg = cfg
    canonicalize = MagicMock()
    monkeypatch.setattr(sf_mod, "apply_body_window_canonicalize", canonicalize)

    trainer._run_rollout(
        {
            **model_batch,
            "traj_features": torch.zeros(_BATCH, _SEQ_LEN, 7),
            "traj_length": torch.tensor([_SEQ_LEN], dtype=torch.long),
        },
        progress=1.0,
    )

    canonicalize.assert_not_called()


def test_plan_rollout_respects_ldf_training_min_history_tokens():
    model = MagicMock(name="model")
    model.chunk_size = 1
    cfg = SimpleNamespace()
    cfg.self_forcing = SimpleNamespace(k_schedule=[(0.0, 1)], stride_tokens=1)
    cfg.get = lambda key, default=None: {
        "ldf_training": {
            "window_policy": "rolling",
            "min_history_tokens": 4,
        },
    }.get(key, default)
    module = SimpleNamespace(model=model, cfg=cfg)
    trainer = SelfForcingTrainer.__new__(SelfForcingTrainer)
    trainer._module = module

    for _ in range(20):
        plan = trainer.plan_rollout(
            torch.tensor([8], dtype=torch.long),
            torch.device("cpu"),
            progress=1.0,
        )
        assert int(plan.start_end_indices[0].item()) >= 4


def test_plan_rollout_right_aligns_fixed_window_policy():
    model = MagicMock(name="model")
    model.chunk_size = 1
    cfg = SimpleNamespace()
    cfg.self_forcing = SimpleNamespace(k_schedule=[(0.0, 3)], stride_tokens=1)
    cfg.get = lambda key, default=None: {
        "ldf_training": {
            "window_policy": "rolling",
            "sample_policy": "fixed_window",
            "min_history_tokens": 4,
        },
    }.get(key, default)
    module = SimpleNamespace(model=model, cfg=cfg)
    trainer = SelfForcingTrainer.__new__(SelfForcingTrainer)
    trainer._module = module

    plan = trainer.plan_rollout(
        torch.tensor([8], dtype=torch.long),
        torch.device("cpu"),
        progress=1.0,
    )

    assert plan.effective_k == 3
    assert plan.start_end_indices.tolist() == [6]


def test_plan_rollout_uses_window_sampling_history_as_step0_active_right():
    model = MagicMock(name="model")
    model.chunk_size = 5
    cfg = SimpleNamespace()
    cfg.self_forcing = SimpleNamespace(k_schedule=[(0.0, 5)], stride_tokens=1)
    cfg.get = lambda key, default=None: {
        "ldf_training": {
            "window_policy": "rolling",
            "window_sampling": {"enabled": True},
        },
    }.get(key, default)
    module = SimpleNamespace(model=model, cfg=cfg)
    trainer = SelfForcingTrainer.__new__(SelfForcingTrainer)
    trainer._module = module
    model_batch = {
        "_window_local_traj": True,
        "_window_sampling_history_tokens": torch.tensor([0, 7], dtype=torch.long),
    }

    plan = trainer.plan_rollout(
        torch.tensor([9, 16], dtype=torch.long),
        torch.device("cpu"),
        progress=1.0,
        model_batch=model_batch,
    )

    assert plan.effective_k == 5
    assert plan.start_end_indices.tolist() == [5, 12]


def test_plan_rollout_prefix_aligns_final_active_right_to_latent_end(monkeypatch):
    model = MagicMock(name="model")
    model.chunk_size = 5
    cfg = SimpleNamespace()
    cfg.self_forcing = SimpleNamespace(k_schedule=[(0.0, 5)], stride_tokens=1)
    cfg.get = lambda key, default=None: {
        "ldf_training": {
            "window_policy": "prefix",
        },
    }.get(key, default)
    module = SimpleNamespace(model=model, cfg=cfg)
    trainer = SelfForcingTrainer.__new__(SelfForcingTrainer)
    trainer._module = module
    model_batch = {
        "_window_local_sample_policy": "prefix",
    }
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None: torch.full(
            size, int(low), device=device, dtype=torch.long
        ),
    )

    plan = trainer.plan_rollout(
        torch.tensor([12], dtype=torch.long),
        torch.device("cpu"),
        progress=1.0,
        model_batch=model_batch,
    )

    rollout_span = (plan.effective_k - 1) * cfg.self_forcing.stride_tokens
    assert plan.effective_k == 5
    assert plan.start_end_indices.tolist() == [8]
    assert (plan.start_end_indices + rollout_span).tolist() == [12]


def test_run_rollout_adds_window_start_to_horizon_active_end(monkeypatch):
    trainer, model, model_batch, _ = _make_trainer()
    _patch_run_training_window(monkeypatch, model)

    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "anchor_canonicalize": {"enabled": False},
        "horizon_sim": {
            "enabled": True,
            "warmup_ratio": 0.0,
            "p_exact_inference_horizon": 1.0,
            "inference_horizon_tokens": 2,
        },
        "history_corruption": {},
        "self_forcing_disable_replace": True,
    }.get(key, default)
    trainer._module.cfg = cfg
    trainer.plan_rollout = MagicMock(
        return_value=RolloutPlan(
            effective_k=2,
            start_end_indices=torch.tensor([2], dtype=torch.long),
            phase_offset=torch.tensor([0.0]),
        )
    )
    model_batch = {
        **model_batch,
        "feature": torch.zeros(_BATCH, 8, _HIDDEN),
        "feature_length": torch.tensor([4], dtype=torch.long),
        "traj_start_token": torch.tensor([5], dtype=torch.long),
    }

    trainer._run_rollout(model_batch, progress=1.0)

    kwargs = model.prepare_traj_condition_step.call_args.kwargs
    assert kwargs["horizon_tokens"] == 2
    assert kwargs["horizon_active_end"].tolist() == [8]


def test_run_rollout_window_sampling_rebuilds_traj_condition_each_step_with_fixed_horizon(
    monkeypatch,
):
    trainer, model, model_batch, _ = _make_trainer()
    _patch_run_training_window(monkeypatch, model)

    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "ldf_training": {
            "window_policy": "rolling",
            "window_sampling": {"enabled": True},
        },
        "anchor_canonicalize": {"enabled": False},
        "horizon_sim": {"enabled": False},
        "history_corruption": {},
        "self_forcing_disable_replace": True,
    }.get(key, default)
    trainer._module.cfg = cfg
    trainer.plan_rollout = MagicMock(
        return_value=RolloutPlan(
            effective_k=3,
            start_end_indices=torch.tensor([2], dtype=torch.long),
            phase_offset=torch.tensor([0.0]),
        )
    )
    model_batch = {
        **model_batch,
        "feature": torch.zeros(_BATCH, 8, _HIDDEN),
        "feature_length": torch.tensor([8], dtype=torch.long),
        "_window_local_traj": True,
        "_window_local_latent_start_token": torch.tensor([5], dtype=torch.long),
        "_window_sampling_horizon_tokens": torch.tensor([4], dtype=torch.long),
    }

    trainer._run_rollout(model_batch, progress=1.0)

    assert model.prepare_traj_condition_step.call_count == 3
    horizon_active_ends = [
        call.kwargs["horizon_active_end"].tolist()
        for call in model.prepare_traj_condition_step.call_args_list
    ]
    horizon_tokens = [
        call.kwargs["horizon_tokens"].tolist()
        for call in model.prepare_traj_condition_step.call_args_list
    ]
    assert horizon_active_ends == [[7], [8], [9]]
    assert horizon_tokens == [[4], [4], [4]]
    assert trainer._last_horizon_tokens == 4.0


def test_run_rollout_uses_ldf_training_horizon_when_horizon_sim_disabled(
    monkeypatch,
):
    trainer, model, model_batch, _ = _make_trainer()
    _patch_run_training_window(monkeypatch, model)

    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "ldf_training": {
            "window_policy": "rolling",
            "horizon_tokens": 2,
        },
        "anchor_canonicalize": {"enabled": False},
        "horizon_sim": {"enabled": False},
        "history_corruption": {},
        "self_forcing_disable_replace": True,
    }.get(key, default)
    trainer._module.cfg = cfg
    trainer.plan_rollout = MagicMock(
        return_value=RolloutPlan(
            effective_k=2,
            start_end_indices=torch.tensor([2], dtype=torch.long),
            phase_offset=torch.tensor([0.0]),
        )
    )
    model_batch = {
        **model_batch,
        "feature": torch.zeros(_BATCH, 8, _HIDDEN),
        "feature_length": torch.tensor([4], dtype=torch.long),
        "traj_start_token": torch.tensor([5], dtype=torch.long),
    }

    trainer._run_rollout(model_batch, progress=1.0)

    kwargs = model.prepare_traj_condition_step.call_args.kwargs
    assert kwargs["horizon_tokens"] == 2
    assert kwargs["horizon_active_end"].tolist() == [8]
    assert trainer._last_horizon_tokens == 2.0


def test_run_rollout_clamps_sampled_horizon_to_ldf_training_horizon(monkeypatch):
    trainer, model, model_batch, _ = _make_trainer()
    _patch_run_training_window(monkeypatch, model)

    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "ldf_training": {
            "window_policy": "rolling",
            "horizon_tokens": 2,
        },
        "anchor_canonicalize": {"enabled": False},
        "horizon_sim": {
            "enabled": True,
            "warmup_ratio": 0.0,
            "p_exact_inference_horizon": 1.0,
            "inference_horizon_tokens": 50,
        },
        "history_corruption": {},
        "self_forcing_disable_replace": True,
    }.get(key, default)
    trainer._module.cfg = cfg
    trainer.plan_rollout = MagicMock(
        return_value=RolloutPlan(
            effective_k=2,
            start_end_indices=torch.tensor([2], dtype=torch.long),
            phase_offset=torch.tensor([0.0]),
        )
    )
    model_batch = {
        **model_batch,
        "feature": torch.zeros(_BATCH, 8, _HIDDEN),
        "feature_length": torch.tensor([4], dtype=torch.long),
        "traj_start_token": torch.tensor([5], dtype=torch.long),
    }

    trainer._run_rollout(model_batch, progress=1.0)

    kwargs = model.prepare_traj_condition_step.call_args.kwargs
    assert kwargs["horizon_tokens"] == 2
    assert kwargs["horizon_active_end"].tolist() == [8]
    assert trainer._last_horizon_tokens == 2.0


def test_run_rollout_records_window_local_active_history_metrics(monkeypatch):
    trainer, _, model_batch, _ = _make_trainer()
    _patch_run_training_window(monkeypatch, trainer._module.model)
    model_batch = {
        **model_batch,
        "_window_local_traj": True,
        "_window_local_latent_start_token": torch.tensor([5], dtype=torch.long),
        "_window_local_latent_valid_len": torch.tensor([4], dtype=torch.long),
        "_window_sampling_horizon_cap_clip": torch.tensor([2], dtype=torch.long),
        "_window_sampling_horizon_short_fallback": torch.tensor([True]),
    }

    trainer._run_rollout(model_batch, progress=1.0)

    metrics = trainer._last_window_local_rollout_metrics
    assert metrics["ldf_training/active_history_len_mean"] == 3.0
    assert metrics["ldf_training/active_history_len_min"] == 3.0
    assert metrics["ldf_training/active_history_len_max"] == 3.0
    assert metrics["ldf_training/active_abs_end_mean"] == 8.0
    assert metrics["ldf_training/horizon_cap_clip_mean"] == 2.0
    assert metrics["ldf_training/horizon_short_fallback_rate"] == 1.0


def test_run_rollout_replacement_diff_uses_clean_state_under_corruption(monkeypatch):
    """History corruption is an input view, not the committed latent state.

    If replacement diff is measured against the corrupted view, a masked/noisy
    history token can make self-forcing diagnostics depend on the corruption
    overlay instead of the clean committed latent it replaces.
    """
    trainer, model, model_batch, _ = _make_trainer()
    window_step = _patch_run_training_window(monkeypatch, model)
    trainer.plan_rollout = MagicMock(
        return_value=RolloutPlan(
            effective_k=2,
            start_end_indices=torch.tensor([1], dtype=torch.long),
            phase_offset=torch.tensor([0.0]),
        )
    )
    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "anchor_canonicalize": {"enabled": False},
        "horizon_sim": {"enabled": False},
        "history_corruption": {
            "enabled": True,
            "apply_prob": 1.0,
        },
        "self_forcing_disable_replace": False,
    }.get(key, default)
    trainer._module.cfg = cfg

    monkeypatch.setattr(sf_mod, "should_apply_corruption", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        sf_mod,
        "apply_history_corruption",
        lambda clean_feature, *args, **kwargs: clean_feature + 10.0,
    )

    first_pred = [torch.zeros(_SEQ_LEN, _HIDDEN) for _ in range(_BATCH)]
    first_pred[0][0, :] = 2.0
    final_pred = [torch.zeros(_SEQ_LEN, _HIDDEN) for _ in range(_BATCH)]
    window_step.side_effect = [
        {"loss": torch.tensor(0.0), "x0_latent_list": first_pred},
        {"loss": torch.tensor(0.0), "x0_latent_list": final_pred},
    ]

    trainer._run_rollout(model_batch, progress=1.0)

    assert trainer._last_replace_diff == 2.0
