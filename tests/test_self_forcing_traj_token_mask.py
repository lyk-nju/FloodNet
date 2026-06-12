"""Regression: SelfForcingTrainer._run_rollout must

  1. unpack `_prepare_traj_condition`'s 4-tuple (B-P0-1: added traj_token_mask),
  2. thread `traj_token_mask` into every `_forward_single_window` call —
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import utils.training.self_forcing as sf_mod
from utils.training.self_forcing import RolloutPlan, SelfForcingTrainer


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

    # `_prepare_traj_condition` is the 4-tuple producer the rewrite added.
    traj_emb_sentinel = torch.zeros(_BATCH, _SEQ_LEN, _HIDDEN)
    traj_seq_lens_sentinel = torch.tensor([_SEQ_LEN], dtype=torch.long)

    model = MagicMock(name="model")
    model.chunk_size = _CHUNK
    # K-step schedule: progress=1.0 → K=3 (table: [(0.0, 3)] is enough).
    model.self_forcing_k_schedule = [(0.0, _K)]
    model._decide_text_dropout.return_value = torch.zeros(_BATCH, dtype=torch.bool)
    model._prepare_text_context.return_value = None
    model._decide_traj_dropout.return_value = False
    model._prepare_traj_condition.return_value = (
        traj_emb_sentinel,
        traj_seq_lens_sentinel,
        False,                # traj_dropped (already passed via override)
        sentinel_mask,        # traj_token_mask — the field this test guards
    )

    # `_forward_single_window` returns enough structure for the rollout-step
    # replacement code to no-op (replace_idx < 0 because chunk_size=1 and
    # end_indices start at 1 → replace_idx = 0; pred_seq must be shape-(T,H)).
    fake_x0 = [torch.zeros(_SEQ_LEN, _HIDDEN) for _ in range(_BATCH)]
    model._forward_single_window.return_value = {
        "loss": torch.tensor(0.0),
        "x0_latent_list": fake_x0,
    }

    cfg = SimpleNamespace()
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


def test_run_rollout_threads_traj_token_mask_to_every_step():
    """Every `_forward_single_window` call (K rollout steps) must see the same
    `traj_token_mask` instance produced by `_prepare_traj_condition`."""
    trainer, model, model_batch, sentinel_mask = _make_trainer()

    final_result, k = trainer._run_rollout(model_batch, progress=1.0)

    assert k == _K
    assert final_result is not None
    assert model._forward_single_window.call_count == _K

    for i, call in enumerate(model._forward_single_window.call_args_list):
        kwargs = call.kwargs
        assert "traj_token_mask" in kwargs, (
            f"step {i}: traj_token_mask was dropped from _forward_single_window kwargs"
        )
        # Identity check — the trainer must pass the exact mask returned by
        # `_prepare_traj_condition`, not silently re-derive or None it out.
        assert kwargs["traj_token_mask"] is sentinel_mask, (
            f"step {i}: traj_token_mask is not the value from _prepare_traj_condition"
        )


def test_run_rollout_unpacks_four_tuple_from_prepare_traj_condition():
    """Direct guard: `_run_rollout` must call `_prepare_traj_condition` exactly
    once and accept its 4-tuple return without raising. A regression to a
    3-tuple unpack would manifest here as `ValueError: too many values to
    unpack`."""
    trainer, model, model_batch, _ = _make_trainer()

    trainer._run_rollout(model_batch, progress=1.0)

    assert model._prepare_traj_condition.call_count == 1


def test_plan_rollout_respects_stream_training_min_history_tokens():
    model = MagicMock(name="model")
    model.chunk_size = 1
    model.self_forcing_k_schedule = [(0.0, 1)]
    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "stream_training": {
            "enabled": True,
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
    model.self_forcing_k_schedule = [(0.0, 3)]
    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "stream_training": {
            "enabled": True,
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
    model.self_forcing_k_schedule = [(0.0, 5)]
    model.self_forcing_stride_tokens = 1
    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "stream_training": {
            "enabled": True,
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


def test_run_rollout_adds_window_start_to_horizon_active_end():
    trainer, model, model_batch, _ = _make_trainer()

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

    kwargs = model._prepare_traj_condition.call_args.kwargs
    assert kwargs["horizon_tokens"] == 2
    assert kwargs["horizon_active_end"].tolist() == [8]


def test_run_rollout_window_sampling_rebuilds_traj_condition_each_step_with_fixed_horizon():
    trainer, model, model_batch, _ = _make_trainer()

    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "stream_training": {
            "enabled": True,
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

    assert model._prepare_traj_condition.call_count == 3
    horizon_active_ends = [
        call.kwargs["horizon_active_end"].tolist()
        for call in model._prepare_traj_condition.call_args_list
    ]
    horizon_tokens = [
        call.kwargs["horizon_tokens"].tolist()
        for call in model._prepare_traj_condition.call_args_list
    ]
    assert horizon_active_ends == [[7], [8], [9]]
    assert horizon_tokens == [[4], [4], [4]]
    assert trainer._last_horizon_tokens == 4.0


def test_run_rollout_uses_stream_training_horizon_when_horizon_sim_disabled():
    trainer, model, model_batch, _ = _make_trainer()

    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "stream_training": {
            "enabled": True,
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

    kwargs = model._prepare_traj_condition.call_args.kwargs
    assert kwargs["horizon_tokens"] == 2
    assert kwargs["horizon_active_end"].tolist() == [8]
    assert trainer._last_horizon_tokens == 2.0


def test_run_rollout_clamps_sampled_horizon_to_stream_training_horizon():
    trainer, model, model_batch, _ = _make_trainer()

    cfg = SimpleNamespace()
    cfg.get = lambda key, default=None: {
        "stream_training": {
            "enabled": True,
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

    kwargs = model._prepare_traj_condition.call_args.kwargs
    assert kwargs["horizon_tokens"] == 2
    assert kwargs["horizon_active_end"].tolist() == [8]
    assert trainer._last_horizon_tokens == 2.0


def test_run_rollout_records_window_local_active_history_metrics():
    trainer, _, model_batch, _ = _make_trainer()
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
    assert metrics["stream_training/active_history_len_mean"] == 3.0
    assert metrics["stream_training/active_history_len_min"] == 3.0
    assert metrics["stream_training/active_history_len_max"] == 3.0
    assert metrics["stream_training/active_abs_end_mean"] == 8.0
    assert metrics["stream_training/horizon_cap_clip_mean"] == 2.0
    assert metrics["stream_training/horizon_short_fallback_rate"] == 1.0


def test_run_rollout_replacement_diff_uses_clean_state_under_corruption(monkeypatch):
    """History corruption is an input view, not the committed latent state.

    If replacement diff is measured against the corrupted view, a masked/noisy
    history token can make self-forcing diagnostics depend on the corruption
    overlay instead of the clean committed latent it replaces.
    """
    trainer, model, model_batch, _ = _make_trainer()
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
    model._forward_single_window.side_effect = [
        {"loss": torch.tensor(0.0), "x0_latent_list": first_pred},
        {"loss": torch.tensor(0.0), "x0_latent_list": final_pred},
    ]

    trainer._run_rollout(model_batch, progress=1.0)

    assert trainer._last_replace_diff == 2.0
