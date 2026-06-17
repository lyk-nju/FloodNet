import pytest
import torch
from omegaconf import OmegaConf

from eval.ldf.t2m_generation import run_t2m_generation_mode
from utils.training.ldf.t2m_generation_modes import resolve_t2m_generation_modes


def test_t2m_generation_modes_default_to_generate():
    cfg = OmegaConf.create({})

    assert resolve_t2m_generation_modes(cfg) == ("generate",)


def test_t2m_generation_modes_accept_list_and_scalar_both_alias():
    list_cfg = OmegaConf.create({
        "validation": {
            "t2m_generation_modes": ["generate", "stream_generate"],
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
