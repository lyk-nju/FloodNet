"""LDF training utilities."""

from .model_batch import _copy_trajectory_fields, prepare_model_input
from .sample_creator import SampleCreator, StreamSample
from .self_forcing import RolloutPlan, SelfForcingTrainer, resolve_sf_runtime
from .test_probes import (
    build_probe_loaders,
    build_test_probe_tags,
    build_val_dataloaders,
)
from .validation_eval_runtime import (
    build_generation_eval_cfg,
    control_loss_train_mode,
    get_test_probe_tags,
    resolve_test_probe_tag,
    t2m_metric_enabled,
    validation_repeat_count,
)
from .window_local import build_window_local_model_batch, build_window_local_traj_batch


def compute_control_loss_xz(*args, **kwargs):
    from .control_loss import compute_control_loss_xz as _compute_control_loss_xz

    return _compute_control_loss_xz(*args, **kwargs)


__all__ = [
    "RolloutPlan",
    "SampleCreator",
    "SelfForcingTrainer",
    "StreamSample",
    "_copy_trajectory_fields",
    "build_generation_eval_cfg",
    "build_probe_loaders",
    "build_test_probe_tags",
    "build_val_dataloaders",
    "build_window_local_model_batch",
    "build_window_local_traj_batch",
    "compute_control_loss_xz",
    "control_loss_train_mode",
    "get_test_probe_tags",
    "prepare_model_input",
    "resolve_sf_runtime",
    "resolve_test_probe_tag",
    "t2m_metric_enabled",
    "validation_repeat_count",
]
