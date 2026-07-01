"""T_B_10: config rollup + 4D/7D traj-dim consistency guard. Local (no data)."""

from __future__ import annotations

import pytest

from pathlib import Path
from omegaconf import OmegaConf
from utils.training.ldf.config_validate import (
    validate_7d_requires_self_forcing,
    validate_ldf_training_config,
    validate_traj_dim_consistency,
)

_LDF = Path(__file__).resolve().parent.parent / "configs" / "ldf.yaml"


# ---------------------------------------------------------------------------
# Done criteria: ldf.yaml loads without raise + every new section is readable
# ---------------------------------------------------------------------------


def test_ldf_yaml_loads():
    cfg = OmegaConf.load(_LDF)
    assert cfg is not None


def test_all_new_sections_present_and_readable():
    cfg = OmegaConf.load(_LDF)
    # T_B_03
    assert cfg.history_corruption.enabled is True
    assert "curriculum" in cfg.history_corruption
    # LDF training is always windowed. Rolling/prefix sampling lives under
    # ldf_training, not a separate stream-training task config.
    assert cfg.ldf_training.formulation == "windowed"
    assert cfg.ldf_training.window_policy == "prefix"
    assert "context_tokens" not in cfg.ldf_training
    assert "horizon_tokens" not in cfg.ldf_training
    assert "window_sampling" not in cfg.ldf_training
    assert "min_history_tokens" not in cfg.ldf_training
    assert "stream_training" not in cfg
    assert "horizon_sim" not in cfg
    assert "scheduled_sampling_prob" not in cfg.model.params
    assert "self_forcing_enabled" not in cfg.model.params
    assert "self_forcing_stride_tokens" not in cfg.model.params
    assert "self_forcing_detach_between_steps" not in cfg.model.params
    assert "self_forcing_k_schedule" not in cfg.model.params
    assert cfg.self_forcing.enabled is True
    assert cfg.self_forcing.stride_tokens == 1
    assert cfg.self_forcing.detach_between_steps is True
    assert cfg.self_forcing.k_schedule[0] == [0.0, 5]
    assert "anchor_move_in_rollout" not in cfg.ldf_training
    assert "latent_source" not in cfg.ldf_training
    assert "motion_aux_loss" not in cfg.ldf_training
    assert "t2m_metric" not in cfg
    assert cfg.validation.t2m_metric is True
    assert list(cfg.validation.t2m_generation_modes) == ["stream_generate_step"]
    assert cfg.validation.eval_generation_mode == "stream_generate_step"
    assert cfg.validation.eval_stream_history_length == 30
    assert cfg.validation.eval_stream_traj_horizon_tokens == 20
    assert "val_repeat" not in cfg
    assert cfg.validation.val_repeat == 1
    # T_B_05 is now a hard default in the self-forcing path; only ablations
    # should add anchor_canonicalize.enabled=false as an override.
    assert "anchor_canonicalize" not in cfg
    # T_B_06 (body_aux_loss subsumes the design's older heading_loss section)
    assert cfg.body_aux_loss.enabled is True
    assert cfg.body_aux_loss.heading_form in ("cosine", "smooth_l1")
    assert "control_loss_train_mode" not in cfg
    assert cfg.body_aux_loss.control_loss_train_mode == 1
    for k in ("root_xz", "root_y", "heading", "fwd_delta", "yaw_delta", "end_xz"):
        assert k in cfg.body_aux_loss.weights
    # T_B_07 / T_B_09 flags — shipped ldf.yaml is now on the 7D path.
    assert cfg.data.traj_feat_dim == 7
    assert cfg.model.params.traj_encoder_in_dim == 7


def test_default_config_is_7d_and_consistent():
    """Shipped ldf.yaml is on the 7D path (post-rewrite default)."""
    cfg = OmegaConf.load(_LDF)
    assert validate_traj_dim_consistency(cfg) == 7


# ---------------------------------------------------------------------------
# Consistency guard
# ---------------------------------------------------------------------------


def _cfg(data_dim, model_dim):
    return OmegaConf.create(
        {"data": {"traj_feat_dim": data_dim},
         "model": {"params": {"traj_encoder_in_dim": model_dim}}}
    )


def test_4d_now_rejected():
    """The 4D legacy encoder was removed → a 4D config fails fast."""
    with pytest.raises(ValueError):
        validate_traj_dim_consistency(_cfg(4, 4))


def test_both_7_ok():
    assert validate_traj_dim_consistency(_cfg(7, 7)) == 7


def test_mismatch_raises():
    with pytest.raises(ValueError):
        validate_traj_dim_consistency(_cfg(7, 4))
    with pytest.raises(ValueError):
        validate_traj_dim_consistency(_cfg(4, 7))


def test_unsupported_dim_raises():
    with pytest.raises(ValueError):
        validate_traj_dim_consistency(_cfg(5, 5))


def test_missing_flags_default_to_7():
    assert validate_traj_dim_consistency(OmegaConf.create({})) == 7


def test_flip_to_7d_overlay_is_consistent():
    """Simulate the 7D fine-tune override (CLI / overlay): flip BOTH flags."""
    cfg = OmegaConf.load(_LDF)
    OmegaConf.update(cfg, "data.traj_feat_dim", 7)
    OmegaConf.update(cfg, "model.params.traj_encoder_in_dim", 7)
    assert validate_traj_dim_consistency(cfg) == 7


# ---------------------------------------------------------------------------
# 7D => self-forcing guard (heading supervision + canonicalize are SF-only)
# ---------------------------------------------------------------------------


def _cfg_sf(model_dim, sf):
    return OmegaConf.create(
        {
            "model": {"params": {"traj_encoder_in_dim": model_dim}},
            "self_forcing": {"enabled": sf},
        }
    )


def test_7d_without_self_forcing_raises():
    with pytest.raises(ValueError, match="self_forcing.enabled"):
        validate_7d_requires_self_forcing(_cfg_sf(7, False))


def test_7d_with_self_forcing_ok():
    validate_7d_requires_self_forcing(_cfg_sf(7, True))   # no raise


def test_non_7d_dim_is_unaffected_by_sf_guard():
    """The SF guard only fires for dim==7; any other explicit dim returns early.

    (An EMPTY config now defaults to 7D, so it correctly requires SF — see
    test_empty_config_requires_self_forcing.)"""
    validate_7d_requires_self_forcing(_cfg_sf(4, False))   # dim != 7 → no raise


def test_empty_config_requires_self_forcing():
    """Defaults are 7D now, so an empty config must require self-forcing."""
    with pytest.raises(ValueError, match="self_forcing.enabled"):
        validate_7d_requires_self_forcing(OmegaConf.create({}))


def test_shipped_ldf_passes_sf_guard():
    """Shipped ldf.yaml is 7D with self_forcing.enabled=true, so the guard passes."""
    cfg = OmegaConf.load(_LDF)
    validate_7d_requires_self_forcing(cfg)   # no raise


def test_shipped_ldf_training_config_valid():
    cfg = OmegaConf.load(_LDF)
    validate_ldf_training_config(cfg)


def test_ldf_training_accepts_full_precomputed_latent_policy():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "full",
        },
    })

    validate_ldf_training_config(cfg)


def test_ldf_training_rejects_unknown_t2m_generation_mode():
    cfg = OmegaConf.create({
        "validation": {
            "t2m_generation_modes": ["generate", "full_latent"],
        },
        "model": {"params": {"chunk_size": 5}},
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "prefix",
        },
    })

    with pytest.raises(ValueError, match="t2m_generation_modes"):
        validate_ldf_training_config(cfg)


def test_shipped_ldf_does_not_expose_async_eval_gate():
    cfg = OmegaConf.load(_LDF)
    assert "test_mode" not in cfg.validation
    assert "stream_eval" not in cfg.validation


def test_shipped_history_corruption_only_exposes_main_knobs():
    cfg = OmegaConf.load(_LDF)
    hc = cfg.history_corruption

    assert set(hc.keys()) == {"enabled", "z_stats_dir", "curriculum"}
    assert set(hc.curriculum.keys()) == {"early_prob", "mid_prob", "late_prob"}


def test_ldf_training_accepts_window_sampling_auto_history():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "self_forcing": {
            "k_schedule": [[0.0, 5]],
            "stride_tokens": 1,
        },
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
            "window_sampling": {
                "enabled": True,
                "history_tokens_min": 0,
                "history_tokens_max": "auto",
                "horizon_tokens_min": 5,
                "horizon_tokens_max": 25,
            },
            "anchor_move_in_rollout": False,
        },
    })

    validate_ldf_training_config(cfg)


def test_ldf_training_prefix_rejects_context_horizon_and_window_sampling():
    for field, value in (
        ("context_tokens", 30),
        ("horizon_tokens", 25),
        ("window_sampling", {"enabled": False}),
        ("min_history_tokens", 5),
    ):
        cfg = OmegaConf.create({
            "model": {"params": {"chunk_size": 5}},
            "ldf_training": {
                "formulation": "windowed",
                "window_policy": "prefix",
                field: value,
            },
        })

        with pytest.raises(ValueError, match=field):
            validate_ldf_training_config(cfg)


def test_ldf_training_window_sampling_rejects_invalid_horizon_range():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
            "window_sampling": {
                "enabled": True,
                "history_tokens_min": 0,
                "history_tokens_max": "auto",
                "horizon_tokens_min": 25,
                "horizon_tokens_max": 5,
            },
        },
    })

    with pytest.raises(ValueError, match="horizon_tokens"):
        validate_ldf_training_config(cfg)


def test_ldf_training_rolling_requires_window_sampling_enabled():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
        },
    })

    with pytest.raises(ValueError, match="window_sampling.enabled"):
        validate_ldf_training_config(cfg)


def test_ldf_training_rejects_disabled_rolling_window_sampling():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
            "window_sampling": {"enabled": False},
        },
    })

    with pytest.raises(ValueError, match="window_sampling.enabled"):
        validate_ldf_training_config(cfg)


def test_ldf_training_window_sampling_accepts_fixed_window_sample_policy():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "self_forcing": {
            "k_schedule": [[0.0, 5]],
            "stride_tokens": 1,
        },
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
            "sample_policy": "fixed_window",
            "window_sampling": {
                "enabled": True,
                "history_tokens_min": 0,
                "history_tokens_max": "auto",
                "horizon_tokens_min": 5,
                "horizon_tokens_max": 25,
            },
        },
    })
    validate_ldf_training_config(cfg)


def test_ldf_training_rejects_unknown_sample_policy():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "self_forcing": {
            "k_schedule": [[0.0, 5]],
            "stride_tokens": 1,
        },
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
            "window_sampling": {
                "enabled": True,
                "history_tokens_min": 0,
                "history_tokens_max": "auto",
                "horizon_tokens_min": 5,
                "horizon_tokens_max": 25,
            },
            "sample_policy": "middle_window",
        },
    })
    with pytest.raises(ValueError, match="sample_policy"):
        validate_ldf_training_config(cfg)


def test_ldf_training_rejects_exposed_motion_aux_loss():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
            "min_history_tokens": 8,
            "motion_aux_loss": "full_prefix",
        },
    })
    with pytest.raises(ValueError, match="motion_aux_loss"):
        validate_ldf_training_config(cfg)


def test_ldf_training_rejects_removed_latent_source():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
            "min_history_tokens": 8,
            "latent_source": "online_encode",
        },
    })
    with pytest.raises(ValueError, match="latent_source"):
        validate_ldf_training_config(cfg)


def test_ldf_training_rejects_anchor_move_in_rollout_until_supported():
    cfg = OmegaConf.create({
        "model": {"params": {"chunk_size": 5}},
        "self_forcing": {
            "k_schedule": [[0.0, 5]],
            "stride_tokens": 1,
        },
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
            "window_sampling": {
                "enabled": True,
                "history_tokens_min": 0,
                "history_tokens_max": "auto",
                "horizon_tokens_min": 5,
                "horizon_tokens_max": 25,
            },
            "anchor_move_in_rollout": True,
        },
    })
    with pytest.raises(ValueError, match="anchor_move_in_rollout"):
        validate_ldf_training_config(cfg)


def test_stream_training_block_is_rejected_by_ldf_validator():
    cfg = OmegaConf.create({
        "ldf_training": {
            "formulation": "windowed",
            "window_policy": "rolling",
            "context_tokens": 30,
        },
        "stream_training": {
            "enabled": True,
            "context_tokens": 30,
        },
    })
    with pytest.raises(ValueError, match="stream_training"):
        validate_ldf_training_config(cfg)
