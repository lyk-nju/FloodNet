from __future__ import annotations

import random

import numpy as np
import torch

from models.diffusion_forcing_wan import DiffForcingWanModel
from models.vae_wan_1d import VAEWanModel
from utils.inference.stream_runtime.snapshots import (
    restore_rng_state,
    snapshot_rng_state,
)


def _model_harness() -> DiffForcingWanModel:
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model.generated = torch.arange(24, dtype=torch.float32).view(1, 2, 12, 1, 1)
    model.commit_index = 4
    model.current_step = 7
    model.seq_len = 5
    model.batch_size = 1
    model.num_denoise_steps = 10
    model.dt = 0.1
    model.cfg_scale_text = 1.25
    model.cfg_scale_traj = 3.0
    model.text_condition_list = [[torch.tensor([1.0])]]
    model.latent_buffer_start_commit_abs = 20
    model.latent_buffer_epoch = 2
    model._traj_buf = None
    return model


def test_model_formal_snapshot_restores_stream_and_rolling_metadata():
    model = _model_harness()
    state = model.snapshot_stream_state()

    model.generated.zero_()
    model.commit_index = 99
    model.current_step = 101
    model.text_condition_list[0][0].zero_()
    model.cfg_scale_text = 9.0
    model.latent_buffer_start_commit_abs = 999
    model.latent_buffer_epoch = 10
    model.restore_stream_state(state)

    assert torch.equal(
        model.generated,
        torch.arange(24, dtype=torch.float32).view(1, 2, 12, 1, 1),
    )
    assert model.commit_index == 4
    assert model.current_step == 7
    assert torch.equal(model.text_condition_list[0][0], torch.tensor([1.0]))
    assert model.cfg_scale_text == 1.25
    metadata = model.stream_buffer_metadata()
    assert metadata.start_commit_abs == 20
    assert metadata.epoch == 2
    assert metadata.local_commit_index == 4


def test_rng_snapshot_replays_python_numpy_and_torch_cpu():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    state = snapshot_rng_state(devices=[])

    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_vae_formal_snapshot_restores_encoder_and_decoder_caches():
    vae = VAEWanModel.__new__(VAEWanModel)
    torch.nn.Module.__init__(vae)
    core = type("Core", (), {})()
    core._conv_num = 1
    core._conv_idx = [3]
    core._feat_map = [torch.tensor([1.0])]
    core._enc_conv_num = 1
    core._enc_conv_idx = [4]
    core._enc_feat_map = [torch.tensor([2.0])]
    vae.model = core

    state = vae.snapshot_stream_state()
    core._conv_idx[0] = 99
    core._feat_map[0].fill_(99.0)
    core._enc_conv_idx[0] = 88
    core._enc_feat_map[0].fill_(88.0)
    vae.restore_stream_state(state)

    assert core._conv_idx == [3]
    assert torch.equal(core._feat_map[0], torch.tensor([1.0]))
    assert core._enc_conv_idx == [4]
    assert torch.equal(core._enc_feat_map[0], torch.tensor([2.0]))
