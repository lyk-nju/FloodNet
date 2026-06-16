"""Scheduled sampling has been fully removed from the LDF path."""

from __future__ import annotations

import inspect

import models.diffusion_forcing_wan as dfw
from models.diffusion_forcing_wan import DiffForcingWanModel
from utils.training.ldf.self_forcing import SelfForcingTrainer


def test_model_constructor_has_no_scheduled_sampling_param():
    params = inspect.signature(DiffForcingWanModel.__init__).parameters
    assert "scheduled_sampling_prob" not in params


def test_self_forcing_trainer_no_longer_has_scheduled_sampling_step():
    assert not hasattr(SelfForcingTrainer, "_scheduled_sampling_step")
    assert hasattr(SelfForcingTrainer, "_self_forcing_step")
    assert hasattr(SelfForcingTrainer, "training_step")


def test_model_module_exposes_no_scheduled_sampling_helpers():
    names = dir(dfw)
    assert not any("scheduled_sampling" in name for name in names)
