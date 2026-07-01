import pytest
import torch
from omegaconf import OmegaConf
from contextlib import contextmanager

from utils.training.ldf.t2m_generation import run_t2m_generation_mode
from utils.training.ldf.t2m_generation_modes import resolve_t2m_generation_modes
from utils.training.ldf.validation_generation import _run_validation_generation_mode


def test_generation_metrics_resolves_modes_from_project_config_wrapper():
    from eval.ldf.generation_metrics import _resolve_eval_t2m_generation_modes

    class ConfigWrapper:
        config = OmegaConf.create({
            "validation": {
                "t2m_generation_modes": ["generate", "stream_generate"],
            },
        })

    assert _resolve_eval_t2m_generation_modes(ConfigWrapper()) == (
        "generate",
        "stream_generate",
    )


def test_t2m_generation_modes_default_to_generate():
    cfg = OmegaConf.create({})

    assert resolve_t2m_generation_modes(cfg) == ("generate",)


def test_t2m_generation_modes_accept_list_and_scalar_both_alias():
    list_cfg = OmegaConf.create({
        "validation": {
            "t2m_generation_modes": [
                "generate",
                "stream_generate",
                "stream_generate_step",
            ],
        },
    })
    scalar_cfg = OmegaConf.create({
        "validation": {
            "t2m_generation_mode": "both",
        },
    })

    assert resolve_t2m_generation_modes(list_cfg) == (
        "generate",
        "stream_generate",
        "stream_generate_step",
    )
    assert resolve_t2m_generation_modes(scalar_cfg) == (
        "generate",
        "stream_generate",
    )


def test_t2m_generation_modes_reject_unknown_mode():
    cfg = OmegaConf.create({
        "validation": {
            "t2m_generation_modes": ["generate", "full_latent"],
        },
    })

    with pytest.raises(ValueError, match="t2m_generation_modes"):
        resolve_t2m_generation_modes(cfg)


def test_stream_t2m_generation_mode_concatenates_chunks_to_full_latent():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def stream_generate(self, model_batch, num_denoise_steps=None):
            self.calls.append(num_denoise_steps)
            yield {
                "generated": [
                    torch.tensor([[1.0], [2.0]]),
                    torch.tensor([[10.0]]),
                ],
                "text": ["walk", "run"],
            }
            yield {
                "generated": [
                    torch.tensor([[3.0], [4.0]]),
                    None,
                ],
                "text": ["walk", "run"],
            }
            yield {
                "generated": [
                    None,
                    torch.tensor([[11.0], [12.0], [13.0]]),
                ],
                "text": ["walk", "run"],
            }

    model = FakeModel()
    model_batch = {
        "feature": torch.zeros(2, 4, 1),
        "feature_length": torch.tensor([3, 4]),
        "text": ["walk", "run"],
    }

    out = run_t2m_generation_mode(
        model,
        model_batch,
        "stream_generate",
        num_denoise_steps=7,
    )

    assert model.calls == [7]
    assert out["text"] == ["walk", "run"]
    assert torch.equal(out["generated"][0], torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.equal(
        out["generated"][1],
        torch.tensor([[10.0], [11.0], [12.0], [13.0]]),
    )


def test_validation_generation_mode_uses_stream_generate_when_configured():
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


def test_validation_generation_mode_uses_stream_generate_step_payload(monkeypatch):
    import utils.training.ldf.validation_generation as validation_generation

    calls = {}

    def fake_stream_step_sample(
        *,
        model,
        vae,
        sample_batch,
        device,
        history_length,
        num_denoise_steps,
        traj_horizon_tokens,
        token_dt,
        frames_per_token,
    ):
        calls.update(
            {
                "model": model,
                "vae": vae,
                "sample_batch": sample_batch,
                "device": device,
                "history_length": history_length,
                "num_denoise_steps": num_denoise_steps,
                "traj_horizon_tokens": traj_horizon_tokens,
                "token_dt": token_dt,
                "frames_per_token": frames_per_token,
            }
        )
        return {
            "latent_stream": torch.tensor([[1.0], [2.0]]),
            "decoded_feature": torch.tensor([[10.0], [20.0]]),
        }

    monkeypatch.setattr(
        validation_generation,
        "run_stream_generate_step_sample",
        fake_stream_step_sample,
    )
    model = object()
    vae = object()
    sample_batch = {"text": ["turn left"]}
    device = torch.device("cpu")
    model_batch = {
        "feature": torch.zeros(1, 2, 1),
        "feature_length": torch.tensor([2]),
        "text": ["turn left"],
    }

    out = validation_generation._run_validation_generation_mode(
        model,
        model_batch,
        "stream_generate_step",
        vae=vae,
        sample_batch=sample_batch,
        device=device,
        stream_history_length=12,
        stream_traj_horizon_tokens=9,
        stream_token_dt=0.25,
        stream_frames_per_token=5,
        num_denoise_steps=4,
    )

    assert calls == {
        "model": model,
        "vae": vae,
        "sample_batch": sample_batch,
        "device": device,
        "history_length": 12,
        "num_denoise_steps": 4,
        "traj_horizon_tokens": 9,
        "token_dt": 0.25,
        "frames_per_token": 5,
    }
    assert out["text"] == ["turn left"]
    assert torch.equal(out["generated"][0], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(out["decoded_feature"][0], torch.tensor([[10.0], [20.0]]))


def test_t2m_update_metrics_passes_eval_condition_mode(monkeypatch):
    import utils.training.ldf.lightning_module as lightning_module
    from utils.training.ldf.lightning_module import LDFLightningModule

    condition_modes = []

    def fake_prepare_model_batch(batch, device, model=None, condition_mode=None):
        condition_modes.append(condition_mode)
        batch_size = len(batch["name"])
        return {
            "feature": torch.zeros(batch_size, 2, 1),
            "feature_length": torch.tensor([2] * batch_size),
            "text": list(batch["text"]),
        }

    def fake_stream_step_mode(
        model,
        model_batch,
        generation_mode,
        *,
        vae,
        sample_batch,
        device,
        stream_history_length,
        stream_traj_horizon_tokens,
        stream_token_dt,
        stream_frames_per_token,
        num_denoise_steps,
    ):
        return {
            "generated": [torch.zeros(2, 1)],
            "decoded_feature": [torch.zeros(5, 263)],
            "text": ["walk"],
        }

    monkeypatch.setattr(
        lightning_module,
        "prepare_ldf_eval_model_batch",
        fake_prepare_model_batch,
    )
    monkeypatch.setattr(
        lightning_module,
        "_run_validation_generation_mode",
        fake_stream_step_mode,
    )
    monkeypatch.setattr(
        lightning_module,
        "_build_windowed_metric_ground_truth",
        lambda batch, model_batch: (
            torch.zeros(1, 2, 1),
            torch.tensor([2]),
            torch.zeros(1, 5, 263),
            torch.tensor([5]),
        ),
    )

    class FakeEMA:
        @contextmanager
        def average_parameters(self, params):
            yield

    class FakeModel:
        def parameters(self):
            return []

    class FakeVAE:
        def decode(self, latent):
            return torch.zeros(latent.shape[0], 5, 263)

    class FakeMetric:
        def __init__(self):
            self.updates = []

        def update(self, **kwargs):
            self.updates.append(kwargs)

    module = LDFLightningModule.__new__(LDFLightningModule)
    torch.nn.Module.__init__(module)
    module.t2m_enabled = True
    module.t2m_generation_modes = ("stream_generate_step",)
    module.t2m_metrics = {"stream_generate_step": FakeMetric()}
    module.cfg = OmegaConf.create({
        "validation": {
            "eval_condition_mode": "legacy_raw",
            "eval_generation_mode": "stream_generate_step",
        },
        "metrics": {"t2m": {"fid_target": "original"}},
    })
    module.ema = FakeEMA()
    module.model = FakeModel()
    module.vae = FakeVAE()
    monkeypatch.setattr(
        LDFLightningModule,
        "device",
        property(lambda self: torch.device("cpu")),
        raising=False,
    )
    batch = {
        "name": ["000021"],
        "text": ["walk"],
        "text_tokens": [["walk/VERB"]],
    }

    module.update_metrics(batch)

    assert condition_modes == ["legacy_raw", "legacy_raw"]


def test_validation_artifact_extracts_xz_from_7d_traj_features():
    from utils.training.ldf.validation_generation import _extract_target_traj_xz

    sample_batch = {
        "traj_features": torch.tensor(
            [[
                [1.0, 100.0, 2.0, 1.0, 0.0, 0.1, 0.0],
                [3.0, 200.0, 4.0, 0.0, 1.0, 0.1, 0.2],
            ]]
        ),
    }

    traj_xz = _extract_target_traj_xz(sample_batch, feat_len=2)

    assert torch.equal(
        torch.from_numpy(traj_xz),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
