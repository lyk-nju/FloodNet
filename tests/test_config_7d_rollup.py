"""T_B_10: config rollup + 4D/7D traj-dim consistency guard. Local (no data)."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from utils.training.config_validate import validate_traj_dim_consistency

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
    # T_B_04
    assert cfg.horizon_sim.enabled is True
    assert cfg.horizon_sim.inference_horizon_tokens == 20
    # T_B_05
    assert cfg.anchor_canonicalize.enabled is True
    assert cfg.anchor_canonicalize.mode == "full"
    # T_B_06 (body_aux_loss subsumes the design's older heading_loss section)
    assert cfg.body_aux_loss.enabled is True
    assert cfg.body_aux_loss.heading_form in ("cosine", "smooth_l1")
    for k in ("root_xz", "root_y", "heading", "fwd_delta", "yaw_delta"):
        assert k in cfg.body_aux_loss.weights
    # T_B_07 / T_B_09 flags (default 4D)
    assert cfg.data.traj_feat_dim == 4
    assert cfg.model.params.traj_encoder_in_dim == 4


def test_default_config_is_4d_and_consistent():
    """Shipped ldf.yaml stays on the legacy 4D path (conservative default)."""
    cfg = OmegaConf.load(_LDF)
    assert validate_traj_dim_consistency(cfg) == 4


# ---------------------------------------------------------------------------
# Consistency guard
# ---------------------------------------------------------------------------


def _cfg(data_dim, model_dim):
    return OmegaConf.create(
        {"data": {"traj_feat_dim": data_dim},
         "model": {"params": {"traj_encoder_in_dim": model_dim}}}
    )


def test_both_4_ok():
    assert validate_traj_dim_consistency(_cfg(4, 4)) == 4


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


def test_missing_flags_default_to_4():
    assert validate_traj_dim_consistency(OmegaConf.create({})) == 4


def test_flip_to_7d_overlay_is_consistent():
    """Simulate the 7D fine-tune override (CLI / overlay): flip BOTH flags."""
    cfg = OmegaConf.load(_LDF)
    OmegaConf.update(cfg, "data.traj_feat_dim", 7)
    OmegaConf.update(cfg, "model.params.traj_encoder_in_dim", 7)
    assert validate_traj_dim_consistency(cfg) == 7
