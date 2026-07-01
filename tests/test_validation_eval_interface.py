from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from utils.training.test_probes import build_val_dataloaders


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


def test_training_package_no_longer_exports_async_eval_helpers():
    import utils.training as training

    assert not hasattr(training, "is_async_eval")
    assert not hasattr(training, "emit_eval_request")
    assert not hasattr(training, "emit_resume_eval")
    assert not hasattr(training, "launch_eval_watcher")


def test_validation_eval_uses_validation_function_names():
    from eval.eval_runner import run_validation_generation_eval
    from eval.eval_summary import process_validation_generation_results

    assert callable(run_validation_generation_eval)
    assert callable(process_validation_generation_results)


def test_generation_eval_defaults_to_stream_generate_and_allows_generate():
    from utils.training.validation_eval_runtime import build_generation_eval_cfg

    default_cfg = OmegaConf.create({"validation": {}})
    offline_cfg = OmegaConf.create(
        {"validation": {"eval_generation_mode": "generate"}}
    )

    assert build_generation_eval_cfg(default_cfg)["generation_mode"] == "stream_generate"
    assert build_generation_eval_cfg(offline_cfg)["generation_mode"] == "generate"


def test_validation_generation_mode_uses_stream_generate_when_configured():
    from eval.eval_runner import _run_validation_generation_mode
    import torch

    class FakeModel:
        def __init__(self):
            self.generate_called = False
            self.stream_generate_called = False

        def generate(self, model_batch, num_denoise_steps=None):
            self.generate_called = True
            return {
                "generated": [torch.tensor([[99.0]])],
                "text": ["offline"],
            }

        def stream_generate(self, model_batch, num_denoise_steps=None):
            self.stream_generate_called = True
            yield {
                "generated": [torch.tensor([[1.0], [2.0]])],
                "text": ["stream"],
            }

    model = FakeModel()
    model_batch = {
        "feature": torch.zeros(1, 2, 1),
        "feature_length": torch.tensor([2]),
        "text": ["stream"],
    }

    out = _run_validation_generation_mode(
        model,
        model_batch,
        "stream_generate",
    )

    assert model.stream_generate_called
    assert not model.generate_called
    assert out["text"] == ["stream"]
    assert torch.equal(out["generated"][0], torch.tensor([[1.0], [2.0]]))


def test_t2m_metric_enabled_is_validation_scoped():
    from utils.training.validation_eval_runtime import t2m_metric_enabled

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
    from utils.training.validation_eval_runtime import validation_repeat_count

    assert validation_repeat_count(
        OmegaConf.create({"validation": {"val_repeat": 3}})
    ) == 3
    assert validation_repeat_count(OmegaConf.create({})) == 1
    assert validation_repeat_count(OmegaConf.create({"val_repeat": 5})) == 1


def test_control_loss_train_mode_is_body_aux_scoped():
    from utils.training.validation_eval_runtime import control_loss_train_mode

    assert control_loss_train_mode(
        OmegaConf.create({"body_aux_loss": {"control_loss_train_mode": 6}})
    ) == 6
    assert control_loss_train_mode(OmegaConf.create({})) == 3
    assert control_loss_train_mode(
        OmegaConf.create({"control_loss_train_mode": 1})
    ) == 3
