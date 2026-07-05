from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils.training.ldf.validation_eval_runtime import build_val_dataloaders


_ROOT = Path(__file__).resolve().parent.parent


def test_default_ldf_configs_do_not_expose_async_or_stream_eval_gate():
    for name in ("ldf.yaml", "ldf_test.yaml"):
        cfg = OmegaConf.load(_ROOT / "configs" / name)

        assert "test_mode" not in cfg.validation
        assert "stream_eval" not in cfg.validation
        assert "t2m_metric" not in cfg
        assert cfg.validation.t2m_metric is True


def test_validation_dataloaders_always_include_test_probes():
    cfg = OmegaConf.create({"validation": {"test_mode": "async"}})
    val_loader = object()
    probe_loaders = [object(), object()]

    loaders = build_val_dataloaders(cfg, val_loader, probe_loaders)

    assert loaders == [val_loader] + probe_loaders


def test_generation_eval_cfg_exposes_stream_best_of_k_defaults_and_overrides():
    from utils.training.ldf.validation_eval_runtime import build_generation_eval_cfg

    default_cfg = build_generation_eval_cfg(OmegaConf.create({"validation": {}}))

    assert default_cfg["stream_best_of_k"] == 1
    assert default_cfg["stream_best_of_k_score"] == "xz"
    assert default_cfg["stream_best_of_k_xz_weight"] == 1.0
    assert default_cfg["stream_best_of_k_fde_weight"] == 1.0
    assert default_cfg["stream_best_of_k_cont_weight"] == 0.0
    assert default_cfg["stream_best_of_k_vel_weight"] == 0.5
    assert default_cfg["stream_best_of_k_rel_margin"] == 0.10
    assert default_cfg["stream_best_of_k_abs_margin"] == 0.03
    assert default_cfg["stream_best_of_k_cont_tol"] == 0.03
    assert default_cfg["stream_best_of_k_force_candidate0"] is False
    assert default_cfg["stream_best_of_k_switch_cooldown_steps"] == 0

    custom_cfg = build_generation_eval_cfg(
        OmegaConf.create(
            {
                "validation": {
                    "eval_stream_best_of_k": 4,
                    "eval_stream_best_of_k_score": "xz",
                    "eval_stream_best_of_k_xz_weight": 0.5,
                    "eval_stream_best_of_k_fde_weight": 2.0,
                    "eval_stream_best_of_k_cont_weight": 0.25,
                    "eval_stream_best_of_k_vel_weight": 0.75,
                    "eval_stream_best_of_k_rel_margin": 0.20,
                    "eval_stream_best_of_k_abs_margin": 0.05,
                    "eval_stream_best_of_k_cont_tol": 0.07,
                    "eval_stream_best_of_k_force_candidate0": True,
                    "eval_stream_best_of_k_switch_cooldown_steps": 2,
                }
            }
        )
    )

    assert custom_cfg["stream_best_of_k"] == 4
    assert custom_cfg["stream_best_of_k_score"] == "xz"
    assert custom_cfg["stream_best_of_k_xz_weight"] == 0.5
    assert custom_cfg["stream_best_of_k_fde_weight"] == 2.0
    assert custom_cfg["stream_best_of_k_cont_weight"] == 0.25
    assert custom_cfg["stream_best_of_k_vel_weight"] == 0.75
    assert custom_cfg["stream_best_of_k_rel_margin"] == 0.20
    assert custom_cfg["stream_best_of_k_abs_margin"] == 0.05
    assert custom_cfg["stream_best_of_k_cont_tol"] == 0.07
    assert custom_cfg["stream_best_of_k_force_candidate0"] is True
    assert custom_cfg["stream_best_of_k_switch_cooldown_steps"] == 2


def test_stream_generate_step_accepts_stream_best_of_k_selector_config_kwargs():
    from inspect import signature

    from eval.ldf.stream_generation import (
        StreamBestOfKConfig,
        run_stream_generate_step_sample,
    )

    expected_kwargs = {
        "best_of_k_vel_weight": 0.75,
        "best_of_k_rel_margin": 0.20,
        "best_of_k_abs_margin": 0.05,
        "best_of_k_cont_tol": 0.07,
        "best_of_k_force_candidate0": True,
        "best_of_k_switch_cooldown_steps": 2,
    }
    params = signature(run_stream_generate_step_sample).parameters

    for name in expected_kwargs:
        assert name in params

    cfg = StreamBestOfKConfig.from_values(
        k=4,
        vel_weight=expected_kwargs["best_of_k_vel_weight"],
        rel_margin=expected_kwargs["best_of_k_rel_margin"],
        abs_margin=expected_kwargs["best_of_k_abs_margin"],
        cont_tol=expected_kwargs["best_of_k_cont_tol"],
        force_candidate0=expected_kwargs["best_of_k_force_candidate0"],
        switch_cooldown_steps=expected_kwargs[
            "best_of_k_switch_cooldown_steps"
        ],
    )

    assert cfg.vel_weight == 0.75
    assert cfg.rel_margin == 0.20
    assert cfg.abs_margin == 0.05
    assert cfg.cont_tol == 0.07
    assert cfg.force_candidate0 is True
    assert cfg.switch_cooldown_steps == 2


def test_validation_stream_generation_preserves_best_of_k_summary(monkeypatch):
    from utils.training.ldf import validation_generation
    from utils.training.ldf.t2m_generation_modes import T2M_STREAM_GENERATE_STEP

    expected_summary = {"k": 5, "switch_count": 2, "step_count": 46}

    def _fake_stream_generate_step_sample(**kwargs):
        return {
            "latent_stream": torch.zeros(1, 4),
            "decoded_feature": torch.zeros(4, 263),
            "stream_best_of_k": expected_summary,
        }

    monkeypatch.setattr(
        validation_generation,
        "run_stream_generate_step_sample",
        _fake_stream_generate_step_sample,
    )

    output = validation_generation._run_validation_generation_mode(
        model=object(),
        model_batch={"text": ["walk"]},
        generation_mode=T2M_STREAM_GENERATE_STEP,
        vae=object(),
        sample_batch={"name": ["sample"]},
        device=torch.device("cpu"),
        stream_best_of_k=5,
    )

    assert output["stream_best_of_k"] is expected_summary


def test_training_package_no_longer_exports_async_eval_helpers():
    import utils.training as training

    assert not hasattr(training, "is_async_eval")
    assert not hasattr(training, "emit_eval_request")
    assert not hasattr(training, "emit_resume_eval")
    assert not hasattr(training, "launch_eval_watcher")


def test_validation_eval_uses_validation_function_names():
    from utils.training.ldf.validation_generation import run_validation_generation_eval
    from utils.training.ldf.validation_summary import (
        process_validation_generation_results,
    )

    assert callable(run_validation_generation_eval)
    assert callable(process_validation_generation_results)


def test_eval_validation_wrappers_preserve_legacy_imports():
    from eval.eval_runner import run_validation_generation_eval
    from eval.eval_summary import process_validation_generation_results

    assert callable(run_validation_generation_eval)
    assert callable(process_validation_generation_results)


def test_train_ldf_does_not_import_eval_package():
    train_source = (_ROOT / "train_ldf.py").read_text()

    assert "from eval." not in train_source
    assert "import eval." not in train_source


def test_ldf_lightning_module_export_and_train_alias_match():
    from train_ldf import CustomLightningModule
    from utils.training.ldf.lightning_module import LDFLightningModule

    assert CustomLightningModule is LDFLightningModule


def test_ldf_lightning_module_method_names_are_clear():
    from utils.training.ldf.lightning_module import LDFLightningModule

    assert "_log_step" in LDFLightningModule.__dict__
    assert "_log_step_metrics" not in LDFLightningModule.__dict__
    assert "load_checkpoint" in LDFLightningModule.__dict__
    assert "on_load_checkpoint" in LDFLightningModule.__dict__
    assert "finish_validation_epoch" in LDFLightningModule.__dict__
    assert "on_validation_epoch_end" in LDFLightningModule.__dict__
    assert "on_train_batch_end" not in LDFLightningModule.__dict__


def test_t2m_metric_enabled_is_validation_scoped():
    from utils.training.ldf.validation_eval_runtime import t2m_metric_enabled

    assert t2m_metric_enabled(
        OmegaConf.create({"validation": {"t2m_metric": True}})
    ) is True
    assert t2m_metric_enabled(
        OmegaConf.create({"validation": {"t2m_metric": False}})
    ) is False
    assert t2m_metric_enabled(
        OmegaConf.create({"t2m_metric": True})
    ) is False


def test_val_repeat_is_validation_scoped():
    from utils.training.ldf.validation_eval_runtime import validation_repeat_count

    assert validation_repeat_count(
        OmegaConf.create({"validation": {"val_repeat": 3}})
    ) == 3
    assert validation_repeat_count(OmegaConf.create({})) == 1
    assert validation_repeat_count(OmegaConf.create({"val_repeat": 5})) == 1


def test_control_loss_train_mode_is_body_aux_scoped():
    from utils.training.ldf.validation_eval_runtime import control_loss_train_mode

    assert control_loss_train_mode(
        OmegaConf.create({"body_aux_loss": {"control_loss_train_mode": 6}})
    ) == 6
    assert control_loss_train_mode(OmegaConf.create({})) == 3
    assert control_loss_train_mode(
        OmegaConf.create({"control_loss_train_mode": 1})
    ) == 3
