"""T_B_12: body fine-tune ablation config deltas (local; the runs need data)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from utils.training.ablation import (
    apply_overrides,
    body_ablation_overrides,
    overrides_to_cli,
)
from utils.training.config_validate import validate_traj_dim_consistency

_LDF = Path(__file__).resolve().parent.parent / "configs" / "ldf.yaml"
_NAMES = {
    "all_on", "no_corruption", "no_horizon_sim",
    "no_anchor_canonical", "no_heading_loss", "no_7d",
}


def test_six_named_ablations():
    assert set(body_ablation_overrides()) == _NAMES


def _applied(name):
    cfg = OmegaConf.load(_LDF)
    return apply_overrides(cfg, body_ablation_overrides()[name])


def test_every_ablation_is_traj_dim_consistent():
    # no_7d → 4D; the other five → 7D. All must pass the consistency guard.
    assert validate_traj_dim_consistency(_applied("no_7d")) == 4
    for name in _NAMES - {"no_7d"}:
        assert validate_traj_dim_consistency(_applied(name)) == 7


def test_all_on_keeps_every_subitem_enabled():
    c = _applied("all_on")
    assert c.history_corruption.enabled is True
    assert c.horizon_sim.enabled is True
    assert c.anchor_canonicalize.enabled is True
    assert c.body_aux_loss.enabled is True
    assert c.body_aux_loss.weights.heading > 0


def test_each_no_x_disables_exactly_its_subitem():
    c = _applied("no_corruption")
    assert c.history_corruption.enabled is False
    assert c.horizon_sim.enabled is True and c.anchor_canonicalize.enabled is True

    c = _applied("no_horizon_sim")
    assert c.horizon_sim.enabled is False
    assert c.history_corruption.enabled is True

    c = _applied("no_anchor_canonical")
    assert c.anchor_canonicalize.enabled is False
    assert c.history_corruption.enabled is True


def test_no_heading_loss_zeroes_heading_but_keeps_aux_enabled():
    c = _applied("no_heading_loss")
    # body_aux_loss must STAY enabled in 7D (guard), heading weight zeroed.
    assert c.body_aux_loss.enabled is True
    assert float(c.body_aux_loss.weights.heading) == 0.0
    assert float(c.body_aux_loss.weights.root_xz) > 0   # other terms intact


def test_no_7d_flips_both_traj_flags_to_4():
    c = _applied("no_7d")
    assert c.data.traj_feat_dim == 4
    assert c.model.params.traj_encoder_in_dim == 4


def test_overrides_to_cli_formats_bools_lowercase():
    cli = overrides_to_cli({"a.b": False, "c": 7, "d.e": True})
    assert "a.b=false" in cli
    assert "c=7" in cli
    assert "d.e=true" in cli
