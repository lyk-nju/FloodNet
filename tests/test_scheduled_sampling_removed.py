"""Tests for T_B_01: scheduled_sampling dead-code removal + deprecation.

Verifies:
- warn_scheduled_sampling_deprecated semantics (warn only on nonzero, once/proc);
- SelfForcingTrainer no longer exposes _scheduled_sampling_step;
- the model still constructs the (ignored, backward-compat) constructor param
  path without raising, exercised via the helper rather than the full model
  (full model needs deps/data not available here).
"""

from __future__ import annotations

import warnings

import models.diffusion_forcing_wan as dfw
from models.diffusion_forcing_wan import warn_scheduled_sampling_deprecated
from utils.training.self_forcing import SelfForcingTrainer


def test_zero_value_does_not_warn():
    dfw._SCHEDULED_SAMPLING_WARNED = False
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        emitted = warn_scheduled_sampling_deprecated(0.0)
    assert emitted is False
    assert len(rec) == 0


def test_nonzero_value_warns_once_per_process():
    dfw._SCHEDULED_SAMPLING_WARNED = False
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        first = warn_scheduled_sampling_deprecated(0.5)
        second = warn_scheduled_sampling_deprecated(0.5)
    assert first is True
    assert second is False          # once-per-process
    assert len(rec) == 1
    assert issubclass(rec[0].category, DeprecationWarning)
    assert "scheduled_sampling_prob" in str(rec[0].message)
    assert "deprecated" in str(rec[0].message).lower()


def test_none_and_garbage_values_are_safe():
    dfw._SCHEDULED_SAMPLING_WARNED = False
    assert warn_scheduled_sampling_deprecated(None) is False
    assert warn_scheduled_sampling_deprecated("not a number") is False


def test_self_forcing_trainer_no_longer_has_scheduled_sampling_step():
    """The dead scheduled-sampling single-step path is removed."""
    assert not hasattr(SelfForcingTrainer, "_scheduled_sampling_step")
    # The real K-step rollout entrypoint is retained.
    assert hasattr(SelfForcingTrainer, "_self_forcing_step")
    assert hasattr(SelfForcingTrainer, "training_step")


def test_model_module_exposes_deprecation_helper():
    """The helper is importable + the once-flag exists at module scope."""
    assert callable(warn_scheduled_sampling_deprecated)
    assert hasattr(dfw, "_SCHEDULED_SAMPLING_WARNED")
