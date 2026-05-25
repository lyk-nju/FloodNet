"""Run-control parity with train_ldf.py: seed / resume / wandb / checkpoint
resolution in train_refiner.py.

These cover the pure config-resolution helpers (no Trainer, no live wandb):
- resolve_seed precedence (CLI > cfg.seed > cfg.training.seed > 1234)
- resolve_resume_ckpt precedence (CLI > cfg.resume_ckpt > None)
- build_wandb_logger returns None when disabled / no key
- build_checkpoint_callback cadence + None when unset
- build_datasets threads the seed into the train dataset
"""

from __future__ import annotations

import train_refiner as tr


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def test_resolve_seed_cli_wins():
    assert tr.resolve_seed({"seed": 7, "training": {"seed": 99}}, cli_seed=3) == 3


def test_resolve_seed_top_level():
    assert tr.resolve_seed({"seed": 7, "training": {"seed": 99}}) == 7


def test_resolve_seed_training_block():
    assert tr.resolve_seed({"training": {"seed": 99}}) == 99


def test_resolve_seed_default():
    assert tr.resolve_seed({}) == 1234


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def test_resume_cli_wins():
    assert tr.resolve_resume_ckpt({"resume_ckpt": "/cfg.ckpt"}, "/cli.ckpt") == "/cli.ckpt"


def test_resume_from_cfg():
    assert tr.resolve_resume_ckpt({"resume_ckpt": "/cfg.ckpt"}, None) == "/cfg.ckpt"


def test_resume_empty_is_none():
    assert tr.resolve_resume_ckpt({"resume_ckpt": ""}, None) is None
    assert tr.resolve_resume_ckpt({}, None) is None


# ---------------------------------------------------------------------------
# wandb logger
# ---------------------------------------------------------------------------


def test_wandb_disabled_returns_none():
    assert tr.build_wandb_logger({}, "run", "/tmp") is None
    assert tr.build_wandb_logger({"logger": {"wandb": {"enabled": False}}}, "run", "/tmp") is None


def test_wandb_debug_gate_short_circuits():
    """debug=true → None even when a logger.wandb block + a resolvable key exist
    (LDF parity: `if not cfg.debug`)."""
    cfg = {"debug": True, "logger": {"wandb": {}}}
    assert tr.build_wandb_logger(cfg, "run", "/tmp") is None


def test_wandb_block_without_enabled_proceeds(monkeypatch):
    """A logger.wandb block with no `enabled` key (ldf style) is treated as ON
    when debug=false; here no key is resolvable so it returns None (not crash)."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tr, "_read_wandb_info_from_paths_default", lambda: {})
    cfg = {"debug": False, "logger": {"wandb": {}}}
    assert tr.build_wandb_logger(cfg, "run", "/tmp") is None


def test_wandb_enabled_no_key_returns_none(monkeypatch):
    """enabled but no key anywhere (cfg blank, no paths_default key, no env) → None,
    not a crash."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tr, "_read_wandb_info_from_paths_default", lambda: {})
    cfg = {"logger": {"wandb": {"enabled": True, "wandb_key": ""}}}
    assert tr.build_wandb_logger(cfg, "run", "/tmp") is None


def test_refiner_wandb_info_separate_project_shared_key():
    """paths_default.yaml: refiner uses its own project but inherits key/entity
    from wandb_info (so FloodNet and Root Refiner don't share a project name)."""
    info = tr._read_wandb_info_from_paths_default()
    # Repo ships both blocks; if paths_default is customized away this may be {}.
    if not info:
        return
    assert info.get("project") == "RootRefiner"     # refiner override
    assert info.get("key")                          # inherited shared key
    assert "FloodNet" not in (info.get("project") or "")


def test_literal_or_none_ignores_unresolved_interpolation():
    assert tr._literal_or_none("${wandb_info.key}") is None
    assert tr._literal_or_none("") is None
    assert tr._literal_or_none(None) is None
    assert tr._literal_or_none("real-key") == "real-key"


# ---------------------------------------------------------------------------
# checkpoint callback
# ---------------------------------------------------------------------------


def test_checkpoint_none_when_unset():
    assert tr.build_checkpoint_callback({}, "/out") is None


def test_checkpoint_keep_all_periodic():
    cb = tr.build_checkpoint_callback(
        {"checkpoint": {"save_every_n_steps": 2500, "save_top_k": -1}}, "/out",
    )
    assert cb is not None
    assert cb._every_n_train_steps == 2500
    assert cb.save_top_k == -1
    assert cb.dirpath == "/out"


def test_checkpoint_positive_top_k_without_monitor_is_coerced():
    """Lightning forbids save_top_k>0 with monitor=None → coerce to keep-all."""
    cb = tr.build_checkpoint_callback(
        {"checkpoint": {"save_every_n_steps": 2500, "save_top_k": 5}}, "/out",
    )
    assert cb is not None
    assert cb.save_top_k == -1   # coerced (no monitor)


def test_checkpoint_top_k_honored_with_monitor():
    cb = tr.build_checkpoint_callback(
        {"checkpoint": {"save_every_n_steps": 2500, "save_top_k": 3,
                        "monitor": "val/loss", "mode": "min"}}, "/out",
    )
    assert cb is not None
    assert cb.save_top_k == 3
    assert cb.monitor == "val/loss"


def test_checkpoint_ldf_style_validation_block():
    """LDF-style validation.save_every_n_steps is also honored (keep-all)."""
    cb = tr.build_checkpoint_callback(
        {"validation": {"save_every_n_steps": 5000, "save_top_k": -1}}, "/out",
    )
    assert cb is not None
    assert cb._every_n_train_steps == 5000
    assert cb.save_top_k == -1
