"""LDF training utilities."""

from .sample_creator import SampleCreator, StreamSample
from .self_forcing_config import (
    self_forcing_enabled,
    self_forcing_k_schedule,
    self_forcing_stride_tokens,
    validate_self_forcing_runtime_config,
)
from .self_forcing import RolloutPlan, SelfForcingTrainer, resolve_sf_runtime
from .validation_eval_runtime import (
    build_generation_eval_cfg,
    build_probe_loaders,
    build_test_probe_tags,
    build_val_dataloaders,
    control_loss_train_mode,
    get_test_probe_tags,
    resolve_test_probe_tag,
    t2m_metric_enabled,
    validation_repeat_count,
)


def compute_control_loss_xz(*args, **kwargs):
    from .losses import compute_control_loss_xz as _compute_control_loss_xz

    return _compute_control_loss_xz(*args, **kwargs)


__all__ = [
    "RolloutPlan",
    "SampleCreator",
    "SelfForcingTrainer",
    "StreamSample",
    "build_generation_eval_cfg",
    "build_probe_loaders",
    "build_test_probe_tags",
    "build_val_dataloaders",
    "compute_control_loss_xz",
    "control_loss_train_mode",
    "get_test_probe_tags",
    "resolve_sf_runtime",
    "resolve_test_probe_tag",
    "self_forcing_enabled",
    "self_forcing_k_schedule",
    "self_forcing_stride_tokens",
    "t2m_metric_enabled",
    "validate_self_forcing_runtime_config",
    "validation_repeat_count",
]
