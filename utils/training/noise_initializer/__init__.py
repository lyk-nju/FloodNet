"""Training utilities for the learned frontier noise initializer."""

from .config_validate import validate_noise_initializer_overfit_config
from .context_builder import NoiseInitializerContext, build_noise_initializer_context
from .free_delta import FreeDeltaOptimizationResult, optimize_free_delta
from .lightning_module import NoiseInitializerLightningModule
from .losses import (
    anchored_root_xz_loss,
    delta_zT_l2_regularization,
    masked_mean,
)
from .overfit_runner import run_single_sample_overfit
from .shadow_rollout import run_residual_shadow_rollout
from .text_encoder import resolve_noise_initializer_text_encoder

__all__ = [
    "NoiseInitializerContext",
    "NoiseInitializerLightningModule",
    "FreeDeltaOptimizationResult",
    "anchored_root_xz_loss",
    "build_noise_initializer_context",
    "delta_zT_l2_regularization",
    "masked_mean",
    "optimize_free_delta",
    "run_residual_shadow_rollout",
    "run_single_sample_overfit",
    "resolve_noise_initializer_text_encoder",
    "validate_noise_initializer_overfit_config",
]
