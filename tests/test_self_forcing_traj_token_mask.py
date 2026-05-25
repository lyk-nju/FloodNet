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
