import numpy as np
import pytest
import torch
from torch import nn

from utils.inference.stream_execution import RootFeedbackConfig, StreamCommitEvent
from utils.inference.stream_generator import StreamGenerator


class _FakeLdf(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.generated = torch.zeros(1, 2, 8, 1, 1)
        self.commit_index = 0
        self.current_step = 0
        self.chunk_size = 1
        self.text_condition_list = [[]]

    def preprocess(self, latent):
        return latent.permute(0, 2, 1)[:, :, :, None, None]

    def stream_generate_step(self, step_input, *, first_chunk, condition):
        del step_input, first_chunk, condition
        token = torch.tensor([[[1.0, 2.0]]])
        self.generated[:, :, self.commit_index : self.commit_index + 1] = self.preprocess(token)
        self.commit_index += 1
        return {"generated": token}


class _FakeVaeModel:
    def __init__(self):
        self._conv_num = 1
        self._conv_idx = [0]
        self._feat_map = [torch.tensor([0.0])]
        self._enc_conv_num = 1
        self._enc_conv_idx = [0]
        self._enc_feat_map = [torch.tensor([0.0])]


class _FakeVae:
    def __init__(self):
        self.model = _FakeVaeModel()

    def stream_decode(self, latent, first_chunk=True):
        del latent
        self.model._conv_idx[0] += 1
        self.model._feat_map[0] += 1.0
        frames = 1 if first_chunk else 4
        decoded = torch.zeros(1, frames, 263)
        decoded[..., 2] = 0.1
        decoded[..., 3] = 1.0
        return decoded

    def stream_encode(self, motion, first_chunk=True):
        del motion, first_chunk
        return torch.tensor([[[1.0, 2.0]]])

    def clear_cache(self):
        self.model._conv_idx = [0]
        self.model._feat_map = [torch.tensor([0.0])]
        self.model._enc_conv_idx = [0]
        self.model._enc_feat_map = [torch.tensor([0.0])]


class _FakeRecovery:
    def __init__(self, *, fail=False):
        self.fail = bool(fail)
        self.calls = 0
        self.r_pos_accum = np.zeros(3, dtype=np.float32)
        self.r_rot_ang_accum = 0.0

    def reset(self):
        self.calls = 0
        self.r_pos_accum = np.zeros(3, dtype=np.float32)
        self.r_rot_ang_accum = 0.0

    def process_frame(self, frame):
        del frame
        if self.fail:
            raise RuntimeError("recovery failed")
        self.calls += 1
        self.r_pos_accum[2] += 0.1
        joints = np.zeros((22, 3), dtype=np.float32)
        joints[:, 2] = self.r_pos_accum[2]
        return joints


def _generator(*, recovery=None):
    generator = StreamGenerator(
        ldf_model=_FakeLdf(),
        vae=_FakeVae(),
        motion_recovery=recovery or _FakeRecovery(),
        device="cpu",
        history_length=4,
        traj_horizon_tokens=2,
    )
    generator.build_ldf_condition_provider = lambda *args, **kwargs: None
    generator.reset_execution_state()
    return generator


def test_execute_step_owns_decode_recovery_history_and_timeline_commit():
    generator = _generator()

    event = generator.execute_step(text="walk")

    assert isinstance(event, StreamCommitEvent)
    assert event.local_commit_before == 0
    assert event.absolute_commit_before == 0
    assert event.absolute_commit_after == 1
    assert event.latent_token.shape == (1, 2)
    assert event.decoded_motion_chunk.shape == (1, 263)
    assert event.joint_frames.shape == (1, 22, 3)
    assert event.generated_root_traj7.shape == (1, 7)
    assert event.timeline_state.commit_idx == 1
    assert generator.timeline.head.commit_idx == 1
    assert generator.generated_frame_count == 1
    assert generator.first_chunk is False
    assert event.root_feedback_applied is False


def test_execute_step_uses_owned_generated_history_for_active_window_payload():
    generator = _generator()
    seen = {}

    def build_payload(**kwargs):
        seen.update(kwargs)
        return None

    generator.build_root_plan_stream_payload = build_payload

    generator.execute_step(text="walk")

    history = seen["generated_history_traj7"]
    assert history.shape == (1, 7)
    assert seen["absolute_commit_index"] == 0


def test_execute_step_rolls_back_all_owned_state_when_recovery_fails():
    recovery = _FakeRecovery(fail=True)
    generator = _generator(recovery=recovery)
    ldf_before = generator.ldf_model.generated.clone()
    vae_idx_before = list(generator.vae.model._conv_idx)
    history_before = generator.generated_history_traj7.clone()

    with pytest.raises(RuntimeError, match="recovery failed"):
        generator.execute_step(text="walk")

    assert generator.ldf_model.commit_index == 0
    assert torch.equal(generator.ldf_model.generated, ldf_before)
    assert generator.vae.model._conv_idx == vae_idx_before
    assert recovery.calls == 0
    assert generator.timeline.head.commit_idx == 0
    assert generator.generated_frame_count == 0
    assert generator.first_chunk is True
    assert torch.equal(generator.generated_history_traj7, history_before)


def test_execute_step_rolls_back_when_ldf_generation_fails_after_mutation():
    generator = _generator()
    ldf_before = generator.ldf_model.generated.clone()
    text_before = [list(items) for items in generator.ldf_model.text_condition_list]
    history_before = generator.generated_history_traj7.clone()

    def fail_after_mutation(step_input, *, first_chunk, condition):
        del step_input, first_chunk, condition
        generator.ldf_model.generated.fill_(9.0)
        generator.ldf_model.commit_index = 1
        generator.ldf_model.current_step = 7
        raise RuntimeError("ldf generation failed")

    generator.ldf_model.stream_generate_step = fail_after_mutation

    with pytest.raises(RuntimeError, match="ldf generation failed"):
        generator.execute_step(text="walk")

    assert generator.ldf_model.commit_index == 0
    assert generator.ldf_model.current_step == 0
    assert torch.equal(generator.ldf_model.generated, ldf_before)
    assert generator.ldf_model.text_condition_list == text_before
    assert generator.timeline.head.commit_idx == 0
    assert generator.generated_frame_count == 0
    assert generator.first_chunk is True
    assert torch.equal(generator.generated_history_traj7, history_before)


def test_execute_step_rolls_back_when_vae_decode_fails_after_cache_mutation():
    generator = _generator()
    ldf_before = generator.ldf_model.generated.clone()
    vae_idx_before = list(generator.vae.model._conv_idx)
    vae_feat_before = [item.clone() for item in generator.vae.model._feat_map]
    history_before = generator.generated_history_traj7.clone()

    def fail_after_cache_mutation(latent, first_chunk=True):
        del latent, first_chunk
        generator.vae.model._conv_idx[0] += 5
        generator.vae.model._feat_map[0].add_(5.0)
        raise RuntimeError("vae decode failed")

    generator.vae.stream_decode = fail_after_cache_mutation

    with pytest.raises(RuntimeError, match="vae decode failed"):
        generator.execute_step(text="walk")

    assert generator.ldf_model.commit_index == 0
    assert torch.equal(generator.ldf_model.generated, ldf_before)
    assert generator.vae.model._conv_idx == vae_idx_before
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(generator.vae.model._feat_map, vae_feat_before)
    )
    assert generator.timeline.head.commit_idx == 0
    assert generator.generated_frame_count == 0
    assert generator.first_chunk is True
    assert torch.equal(generator.generated_history_traj7, history_before)


def test_execute_step_rolls_back_when_root_feedback_encode_fails():
    generator = _generator()
    generator.configure_execution(
        root_feedback=RootFeedbackConfig(enabled=True, xz_blend_alpha=1.0)
    )
    ldf_before = generator.ldf_model.generated.clone()
    decoder_idx_before = list(generator.vae.model._conv_idx)
    encoder_idx_before = list(generator.vae.model._enc_conv_idx)
    history_before = generator.generated_history_traj7.clone()
    traj = torch.zeros(1, 2, 7)
    traj[..., 1] = 1.0
    traj[..., 3] = 1.0

    def fail_after_encoder_cache_mutation(motion, first_chunk=True):
        del motion, first_chunk
        generator.vae.model._enc_conv_idx[0] += 5
        generator.vae.model._enc_feat_map[0].add_(5.0)
        raise RuntimeError("root feedback encode failed")

    generator.vae.stream_encode = fail_after_encoder_cache_mutation

    with pytest.raises(RuntimeError, match="root feedback encode failed"):
        generator.execute_step(
            text="walk",
            traj_input={
                "traj_cond_7d_frame": traj,
                "traj_abs_start_token": 0,
            },
        )

    assert generator.ldf_model.commit_index == 0
    assert torch.equal(generator.ldf_model.generated, ldf_before)
    assert generator.vae.model._conv_idx == decoder_idx_before
    assert generator.vae.model._enc_conv_idx == encoder_idx_before
    assert generator.timeline.head.commit_idx == 0
    assert generator.generated_frame_count == 0
    assert generator.first_chunk is True
    assert torch.equal(generator.generated_history_traj7, history_before)


def test_configure_execution_updates_root_feedback_policy():
    generator = _generator()

    generator.configure_execution(
        root_feedback=RootFeedbackConfig(enabled=True, xz_blend_alpha=0.25)
    )

    assert generator.root_feedback_config.enabled is True
    assert generator.root_feedback_config.xz_blend_alpha == 0.25
