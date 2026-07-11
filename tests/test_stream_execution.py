from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.inference.stream_execution import (
    RootFeedbackConfig,
    StreamCommitEvent,
    decode_token_with_root_feedback,
    restore_ldf_stream_state,
    restore_recovery_state,
    restore_vae_stream_state,
    snapshot_ldf_stream_state,
    snapshot_recovery_state,
    snapshot_vae_stream_state,
)
from utils.inference.timeline import RootFrameState


class _FakeVaeModel:
    def __init__(self):
        self._conv_num = 1
        self._conv_idx = [3]
        self._feat_map = [torch.tensor([1.0])]
        self._enc_conv_num = 1
        self._enc_conv_idx = [4]
        self._enc_feat_map = [torch.tensor([2.0])]


class _FakeVae:
    def __init__(self):
        self.model = _FakeVaeModel()
        self.decode_calls = []
        self.encode_calls = []

    def stream_decode(self, latent, first_chunk=True):
        self.decode_calls.append((latent.detach().clone(), bool(first_chunk)))
        self.model._conv_idx[0] += 1
        self.model._feat_map[0] += 1.0
        frames = 1 if first_chunk else 4
        return torch.full(
            (1, frames, 263),
            float(len(self.decode_calls)),
            dtype=torch.float32,
        )

    def stream_encode(self, motion, first_chunk=True):
        self.encode_calls.append((motion.detach().clone(), bool(first_chunk)))
        self.model._enc_conv_idx[0] += 1
        self.model._enc_feat_map[0] += 1.0
        return torch.full((1, 1, 2), 7.0, dtype=motion.dtype)


class _FakeLdf:
    def __init__(self):
        self.generated = torch.zeros(1, 2, 8, 1, 1)
        self.commit_index = 4
        self.current_step = 2
        self.text_condition_list = [[torch.tensor([1.0])]]

    def preprocess(self, latent):
        return latent.permute(0, 2, 1)[:, :, :, None, None]


def _payload(num_frames=12):
    traj = torch.zeros(1, num_frames, 7)
    traj[..., 1] = 1.0
    traj[..., 3] = 1.0
    traj[..., 2] = torch.arange(num_frames, dtype=torch.float32) * 0.1
    return {
        "traj_cond_7d_frame": traj,
        "traj_cond_frame_mask": torch.ones(1, num_frames),
        "traj_abs_start_token": 0,
    }


def test_root_feedback_config_rejects_invalid_alpha():
    assert RootFeedbackConfig().enabled is False
    with pytest.raises(ValueError, match="xz_blend_alpha"):
        RootFeedbackConfig(enabled=True, xz_blend_alpha=1.1)


def test_stream_commit_event_is_immutable():
    event = StreamCommitEvent(
        local_commit_before=0,
        absolute_commit_before=0,
        absolute_commit_after=1,
        latent_token=torch.zeros(1, 2),
        decoded_motion_chunk=torch.zeros(1, 263),
        joint_frames=np.zeros((1, 22, 3), dtype=np.float32),
        generated_root_traj7=torch.zeros(1, 7),
        timeline_state=RootFrameState.initial(dtype=torch.float32),
        traj_payload=None,
        root_feedback_applied=False,
        debug={},
    )

    with pytest.raises(FrozenInstanceError):
        event.absolute_commit_after = 2


def test_stream_state_snapshots_restore_ldf_vae_and_recovery():
    ldf = _FakeLdf()
    vae = _FakeVae()
    recovery = SimpleNamespace(
        r_pos_accum=np.array([1.0, 2.0, 3.0]),
        r_rot_ang_accum=0.25,
        prev_linear_vel=np.array([0.1, 0.2]),
    )
    ldf_state = snapshot_ldf_stream_state(ldf)
    vae_state = snapshot_vae_stream_state(vae)
    recovery_state = snapshot_recovery_state(recovery)

    ldf.generated.fill_(9.0)
    ldf.commit_index = 99
    ldf.text_condition_list[0][0].fill_(9.0)
    vae.model._conv_idx[0] = 99
    vae.model._feat_map[0].fill_(99.0)
    vae.model._enc_conv_idx[0] = 88
    vae.model._enc_feat_map[0].fill_(88.0)
    recovery.r_pos_accum[:] = 9.0
    recovery.r_rot_ang_accum = 9.0

    restore_ldf_stream_state(ldf, ldf_state)
    restore_vae_stream_state(vae, vae_state)
    restore_recovery_state(recovery, recovery_state)

    assert torch.equal(ldf.generated, torch.zeros_like(ldf.generated))
    assert ldf.commit_index == 4
    assert torch.equal(ldf.text_condition_list[0][0], torch.tensor([1.0]))
    assert vae.model._conv_idx == [3]
    assert torch.equal(vae.model._feat_map[0], torch.tensor([1.0]))
    assert vae.model._enc_conv_idx == [4]
    assert torch.equal(vae.model._enc_feat_map[0], torch.tensor([2.0]))
    assert np.allclose(recovery.r_pos_accum, [1.0, 2.0, 3.0])
    assert recovery.r_rot_ang_accum == 0.25


def test_disabled_root_feedback_formally_decodes_once():
    model = _FakeLdf()
    vae = _FakeVae()
    latent = torch.tensor([[2.0, 3.0]])

    result = decode_token_with_root_feedback(
        model=model,
        vae=vae,
        latent_token=latent,
        traj_payload=_payload(),
        generated_frame_count=0,
        local_commit_index=3,
        first_chunk=False,
        config=RootFeedbackConfig(enabled=False),
        device=torch.device("cpu"),
    )

    assert result.applied is False
    assert len(vae.decode_calls) == 1
    assert not vae.encode_calls
    assert torch.equal(result.latent_token, latent)


def test_enabled_root_feedback_reencodes_and_writes_corrected_token():
    model = _FakeLdf()
    vae = _FakeVae()
    latent = torch.tensor([[2.0, 3.0]])

    result = decode_token_with_root_feedback(
        model=model,
        vae=vae,
        latent_token=latent,
        traj_payload=_payload(),
        generated_frame_count=1,
        local_commit_index=3,
        first_chunk=False,
        config=RootFeedbackConfig(enabled=True, xz_blend_alpha=1.0),
        device=torch.device("cpu"),
    )

    assert result.applied is True
    assert len(vae.decode_calls) == 2
    assert len(vae.encode_calls) == 1
    assert torch.equal(result.latent_token, torch.full((1, 2), 7.0))
    assert torch.equal(model.generated[0, :, 3, 0, 0], torch.tensor([7.0, 7.0]))
    assert torch.all(result.decoded_motion_chunk == 2.0)


def test_root_feedback_only_replaces_partial_valid_prefix():
    model = _FakeLdf()
    vae = _FakeVae()
    payload = _payload()
    payload["traj_cond_frame_mask"][0, 3:] = False

    result = decode_token_with_root_feedback(
        model=model,
        vae=vae,
        latent_token=torch.tensor([[2.0, 3.0]]),
        traj_payload=payload,
        generated_frame_count=0,
        local_commit_index=3,
        first_chunk=False,
        config=RootFeedbackConfig(enabled=True, xz_blend_alpha=1.0),
        device=torch.device("cpu"),
    )

    corrected_before_encode = vae.encode_calls[0][0][0]
    assert result.applied is True
    assert torch.equal(corrected_before_encode[2:], torch.ones_like(corrected_before_encode[2:]))


def test_root_feedback_ignores_invalid_padding_frames():
    model = _FakeLdf()
    vae = _FakeVae()
    payload = _payload()
    payload["traj_cond_frame_mask"].zero_()

    result = decode_token_with_root_feedback(
        model=model,
        vae=vae,
        latent_token=torch.tensor([[2.0, 3.0]]),
        traj_payload=payload,
        generated_frame_count=0,
        local_commit_index=3,
        first_chunk=False,
        config=RootFeedbackConfig(enabled=True, xz_blend_alpha=1.0),
        device=torch.device("cpu"),
    )

    assert result.applied is False
    assert result.debug["reason"] == "invalid_route_frames"
    assert not vae.encode_calls
