from __future__ import annotations

from types import SimpleNamespace

import torch

from eval.ldf.generation_metrics import _select_t2m_reference
from utils.token_frame import num_frames_for_tokens


class _FakeVAE:
    def __init__(self):
        self.seen = None

    def decode(self, token):
        self.seen = token.detach().clone()
        frames = token.shape[1] * 10
        return torch.full((token.shape[0], frames, 263), 7.0, device=token.device)


def test_select_t2m_reference_uses_windowed_original_feature():
    raw_feature = torch.arange(40 * 263, dtype=torch.float32).view(1, 40, 263)
    batch = {
        "feature": raw_feature,
        "feature_length": torch.tensor([40]),
        "token": torch.zeros(1, 6, 4),
        "token_length": torch.tensor([6]),
    }
    model_batch = {
        "token": torch.zeros(1, 3, 4),
        "token_length": torch.tensor([3]),
        "feature_length": torch.tensor([3]),
        "_window_global_start_token": torch.tensor([0]),
    }

    ref, length = _select_t2m_reference(
        batch,
        model_batch,
        0,
        vae=_FakeVAE(),
        device=torch.device("cpu"),
        cfg=SimpleNamespace(metrics=SimpleNamespace(t2m=SimpleNamespace(fid_target="original"))),
    )

    expected_len = num_frames_for_tokens(3)
    assert length == expected_len
    assert torch.equal(ref, raw_feature[0, :expected_len])


def test_select_t2m_reference_uses_windowed_tokens_for_vae_target():
    full_token = torch.arange(6 * 4, dtype=torch.float32).view(1, 6, 4)
    window_token = full_token[:, :2].clone()
    batch = {
        "feature": torch.zeros(1, 40, 263),
        "feature_length": torch.tensor([40]),
        "token": full_token,
        "token_length": torch.tensor([6]),
    }
    model_batch = {
        "token": window_token,
        "token_length": torch.tensor([2]),
        "feature_length": torch.tensor([2]),
        "_window_global_start_token": torch.tensor([0]),
    }
    vae = _FakeVAE()

    ref, length = _select_t2m_reference(
        batch,
        model_batch,
        0,
        vae=vae,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(metrics=SimpleNamespace(t2m=SimpleNamespace(fid_target="vae"))),
    )

    assert length == 20
    assert ref.shape == (20, 263)
    assert torch.equal(vae.seen, window_token)
