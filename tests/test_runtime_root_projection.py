from __future__ import annotations

import inspect
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf

from eval.runtime.root_projection import (
    LateDenoiseRootProjectionConfig,
    TimeConsistentRootGuidanceConfig,
    _compute_time_consistent_projection_residual,
    _encode_chunk_token,
    _estimate_clean_latent_from_velocity,
    _map_clean_delta_to_flow_state,
    apply_late_denoise_root_projection,
    apply_time_consistent_root_guidance,
    build_time_consistent_root_guidance_callback,
    select_current_token_target,
)


class _FakeModel:
    def __init__(self):
        self.generated = torch.zeros(1, 2, 8, 1, 1)
        self.num_denoise_steps = 10
        self.dt = 0.1
        self.chunk_size = 5
        self.input_dim = 2

    def postprocess(self, x):
        return x.permute(0, 2, 1, 3, 4).contiguous().view(x.size(0), x.size(2), -1)

    def preprocess(self, x):
        return x.permute(0, 2, 1)[:, :, :, None, None]


class _FakeVaeModel:
    def __init__(self):
        self._conv_num = 1
        self._conv_idx = [3]
        self._feat_map = [torch.tensor([1.0, 2.0])]
        self._enc_conv_num = 1
        self._enc_conv_idx = [4]
        self._enc_feat_map = [torch.tensor([3.0, 4.0])]


class _FakeVae:
    def __init__(self):
        self.model = _FakeVaeModel()
        self.encoded_motion = None

    def stream_decode(self, latent, first_chunk=True):
        self.model._conv_idx[0] = 99
        self.model._feat_map[0].fill_(99.0)
        del latent, first_chunk
        return torch.zeros(1, 4, 263, dtype=torch.float32)

    def stream_encode(self, motion, first_chunk=True):
        self.model._enc_conv_idx[0] = 88
        self.model._enc_feat_map[0].fill_(88.0)
        self.encoded_motion = motion.detach().clone()
        del first_chunk
        # Make the written latent easy to assert.
        return torch.full((1, 1, 2), 7.0, dtype=motion.dtype, device=motion.device)

    def encode(self, motion):
        self.model._enc_conv_idx[0] = 88
        self.model._enc_feat_map[0].fill_(88.0)
        self.encoded_motion = motion.detach().clone()
        return torch.full((1, 1, 2), 7.0, dtype=motion.dtype, device=motion.device)


def _target_traj(num_frames: int, *, x0: float = 0.0, dx: float = 0.2):
    traj = torch.zeros(num_frames, 7, dtype=torch.float32)
    traj[:, 0] = x0 + torch.arange(num_frames, dtype=torch.float32) * dx
    traj[:, 1] = 1.0
    traj[:, 3] = 1.0
    return traj


def test_select_current_token_target_uses_payload_token_offset():
    step_input = {
        "traj_cond_7d_frame": _target_traj(16, x0=10.0).unsqueeze(0),
        "traj_cond_frame_mask": torch.ones(1, 16),
        "traj_start_token": 4,
    }

    target = select_current_token_target(
        step_input,
        commit_index=5,
        frames_per_token=4,
        chunk_frames=4,
    )

    assert target is not None
    assert target.start_frame == 4
    assert torch.allclose(target.traj_7d[4:9, 0], torch.tensor([10.8, 11.0, 11.2, 11.4, 11.6]))


def test_late_projection_blends_current_token_latent_and_restores_vae_cache():
    model = _FakeModel()
    vae = _FakeVae()
    original_decode_cache = (
        list(vae.model._conv_idx),
        [item.clone() for item in vae.model._feat_map],
    )
    original_encode_cache = (
        list(vae.model._enc_conv_idx),
        [item.clone() for item in vae.model._enc_feat_map],
    )
    step_input = {
        "traj_cond_7d_frame": _target_traj(8).unsqueeze(0),
        "traj_cond_frame_mask": torch.ones(1, 8),
        "traj_start_token": 0,
    }
    cfg = LateDenoiseRootProjectionConfig(
        enabled=True,
        alpha=1.0,
        projection_start_step=6,
        latent_blend_gain=1.0,
        final_step_gain=1.0,
        max_delta_per_frame=1.0,
        max_delta_per_chunk=4.0,
        frames_per_token=4,
        frame_ramp=False,
    )

    result = apply_late_denoise_root_projection(
        model=model,
        vae=vae,
        step_input=step_input,
        first_chunk=True,
        commit_index=0,
        current_step=9,
        start_index=0,
        end_index=5,
        noise_level_full=torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8]]),
        config=cfg,
    )

    assert result.applied is True
    assert result.reason == "applied"
    assert torch.allclose(model.generated[0, :, 0, 0, 0], torch.tensor([7.0, 7.0]))
    assert vae.model._conv_idx == original_decode_cache[0]
    assert torch.allclose(vae.model._feat_map[0], original_decode_cache[1][0])
    assert vae.model._enc_conv_idx == original_encode_cache[0]
    assert torch.allclose(vae.model._enc_feat_map[0], original_encode_cache[1][0])
    # The root rewrite uses full root channels. With zero yaw and target dx=0.2,
    # the first rewritten local X delta should match the target displacement.
    assert vae.encoded_motion is not None
    assert torch.isclose(vae.encoded_motion[0, 0, 1], torch.tensor(0.2), atol=1e-5)


def test_late_projection_encodes_projected_token_without_stream_encoder_state():
    class _OfflineEncodeOnlyVae:
        def __init__(self):
            self.encoded_motion = None

        def stream_encode(self, motion, first_chunk=True):
            del motion, first_chunk
            raise AssertionError("late projection should not depend on stream_encode cache")

        def encode(self, motion):
            self.encoded_motion = motion.detach().clone()
            return torch.full((1, 1, 2), 3.0, dtype=motion.dtype, device=motion.device)

    vae = _OfflineEncodeOnlyVae()
    corrected = torch.zeros(4, 263)

    encoded = _encode_chunk_token(
        _FakeModel(),
        vae,
        corrected,
        first_chunk=False,
    )

    assert torch.allclose(encoded, torch.full((1, 2), 3.0))
    assert vae.encoded_motion is not None
    assert tuple(vae.encoded_motion.shape) == (1, 4, 263)


def test_time_consistent_guidance_uses_project_beta_formula():
    x_beta = torch.tensor([[[1.0]], [[2.0]]])
    predicted_vel = torch.tensor([[[3.0]], [[-4.0]]])
    beta_before = torch.tensor(0.25)
    beta_after = torch.tensor(0.15)
    delta_clean = torch.tensor([[[2.0]], [[-2.0]]])

    z_hat = _estimate_clean_latent_from_velocity(
        x_beta,
        predicted_vel,
        beta_before,
    )
    x_guided = _map_clean_delta_to_flow_state(
        torch.zeros_like(x_beta),
        delta_clean,
        beta_after,
        guidance_strength=0.5,
    )

    assert torch.allclose(z_hat, x_beta + beta_before * predicted_vel)
    assert torch.allclose(x_guided, 0.5 * (1.0 - beta_after) * delta_clean)


def test_time_consistent_projection_target_modes_compute_expected_residuals():
    pred_xz = torch.tensor([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]])
    target_xz = torch.tensor([[20.0, 0.0], [22.0, 0.0], [24.0, 0.0]])

    relative, relative_metrics = _compute_time_consistent_projection_residual(
        pred_xz,
        target_xz,
        projection_target_mode="relative_shape",
        mixed_global_weight=0.3,
        mixed_local_weight=1.0,
    )
    absolute, absolute_metrics = _compute_time_consistent_projection_residual(
        pred_xz,
        target_xz,
        projection_target_mode="absolute",
        mixed_global_weight=0.3,
        mixed_local_weight=1.0,
    )
    mixed, mixed_metrics = _compute_time_consistent_projection_residual(
        pred_xz,
        target_xz,
        projection_target_mode="mixed",
        mixed_global_weight=0.3,
        mixed_local_weight=1.0,
    )

    assert torch.allclose(relative, torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    assert torch.allclose(absolute, torch.tensor([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]]))
    assert torch.allclose(mixed, torch.tensor([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]))
    assert relative_metrics["global_offset_norm"] == 10.0
    assert absolute_metrics["local_shape_error"] == 1.0
    assert mixed_metrics["global_offset_norm"] == 10.0
    assert mixed_metrics["local_shape_error"] == 1.0


def test_time_consistent_guidance_callback_bypasses_when_disabled_or_zero():
    assert build_time_consistent_root_guidance_callback(
        vae=_FakeVae(),
        config=TimeConsistentRootGuidanceConfig(enabled=False, alpha=0.2),
    ) is None
    assert build_time_consistent_root_guidance_callback(
        vae=_FakeVae(),
        config=TimeConsistentRootGuidanceConfig(enabled=True, alpha=0.0),
    ) is None
    assert build_time_consistent_root_guidance_callback(
        vae=_FakeVae(),
        config=TimeConsistentRootGuidanceConfig(
            enabled=True,
            alpha=0.2,
            guidance_strength=0.0,
        ),
    ) is None


def test_time_consistent_guidance_applies_once_and_restores_vae_cache():
    model = _FakeModel()
    vae = _FakeVae()
    x_before = torch.tensor([[[1.0]], [[2.0]]])
    predicted_vel = torch.tensor([[[0.5]], [[-0.5]]])
    beta_before = torch.tensor(0.2)
    beta_after = torch.tensor(0.1)
    z_hat = x_before + beta_before * predicted_vel
    # FakeVAE.encode returns 7.0, so delta_clean is known and easy to assert.
    expected_delta = torch.full_like(z_hat, 7.0) - z_hat
    expected_update = (1.0 - beta_after) * expected_delta
    original_decode_cache = (
        list(vae.model._conv_idx),
        [item.clone() for item in vae.model._feat_map],
    )
    original_encode_cache = (
        list(vae.model._enc_conv_idx),
        [item.clone() for item in vae.model._enc_feat_map],
    )
    model.generated[0, :, 0, 0, 0] = torch.tensor([10.0, 20.0])
    step_input = {
        "traj_cond_7d_frame": _target_traj(8).unsqueeze(0),
        "traj_cond_frame_mask": torch.ones(1, 8),
        "traj_start_token": 0,
    }
    cfg = TimeConsistentRootGuidanceConfig(
        enabled=True,
        projection_step=8,
        alpha=1.0,
        guidance_strength=1.0,
        max_delta_per_frame=1.0,
        max_delta_per_chunk=4.0,
        max_latent_delta=100.0,
        frames_per_token=4,
    )

    skipped = apply_time_consistent_root_guidance(
        model=model,
        vae=vae,
        step_input=step_input,
        first_chunk=True,
        commit_index=0,
        current_step=7,
        start_index=0,
        end_index=1,
        x_beta_before_update=x_before,
        predicted_vel=predicted_vel,
        beta_before=torch.tensor(0.3),
        beta_after=torch.tensor(0.2),
        x_after_velocity_update=model.generated[0, :, 0:1, ...],
        config=cfg,
    )
    assert skipped.applied is False
    assert skipped.reason == "not_projection_step"

    result = apply_time_consistent_root_guidance(
        model=model,
        vae=vae,
        step_input=step_input,
        first_chunk=True,
        commit_index=0,
        current_step=8,
        start_index=0,
        end_index=1,
        x_beta_before_update=x_before,
        predicted_vel=predicted_vel,
        beta_before=beta_before,
        beta_after=beta_after,
        x_after_velocity_update=model.generated[0, :, 0:1, ...],
        config=cfg,
    )

    assert result.applied is True
    assert result.reason == "applied"
    assert torch.allclose(
        model.generated[0, :, 0, 0, 0],
        torch.tensor([10.0, 20.0]) + expected_update[:, 0, 0],
    )
    assert result.latent_delta_norm > 0.0
    assert vae.model._conv_idx == original_decode_cache[0]
    assert torch.allclose(vae.model._feat_map[0], original_decode_cache[1][0])
    assert vae.model._enc_conv_idx == original_encode_cache[0]
    assert torch.allclose(vae.model._enc_feat_map[0], original_encode_cache[1][0])


def test_time_consistent_guidance_accepts_multiple_projection_steps():
    model = _FakeModel()
    vae = _FakeVae()
    step_input = {
        "traj_cond_7d_frame": _target_traj(8).unsqueeze(0),
        "traj_cond_frame_mask": torch.ones(1, 8),
        "traj_start_token": 0,
    }
    cfg = TimeConsistentRootGuidanceConfig(
        enabled=True,
        projection_step=(8, 9, 10),
        alpha=1.0,
        guidance_strength=1.0,
        max_delta_per_frame=1.0,
        max_delta_per_chunk=4.0,
        max_latent_delta=100.0,
        frames_per_token=4,
    )

    for current_step in (8, 9, 10):
        model.generated.zero_()
        result = apply_time_consistent_root_guidance(
            model=model,
            vae=vae,
            step_input=step_input,
            first_chunk=True,
            commit_index=0,
            current_step=current_step,
            start_index=0,
            end_index=1,
            x_beta_before_update=torch.tensor([[[1.0]], [[2.0]]]),
            predicted_vel=torch.tensor([[[0.5]], [[-0.5]]]),
            beta_before=torch.tensor(1.0 - current_step / 10.0),
            beta_after=torch.tensor(max(1.0 - (current_step + 1) / 10.0, 0.0)),
            x_after_velocity_update=model.generated[0, :, 0:1, ...],
            config=cfg,
        )
        assert result.applied is True
        assert result.local_step == current_step

    skipped = apply_time_consistent_root_guidance(
        model=model,
        vae=vae,
        step_input=step_input,
        first_chunk=True,
        commit_index=0,
        current_step=7,
        start_index=0,
        end_index=1,
        x_beta_before_update=torch.tensor([[[1.0]], [[2.0]]]),
        predicted_vel=torch.tensor([[[0.5]], [[-0.5]]]),
        beta_before=torch.tensor(0.3),
        beta_after=torch.tensor(0.2),
        x_after_velocity_update=model.generated[0, :, 0:1, ...],
        config=cfg,
    )
    assert skipped.applied is False
    assert skipped.reason == "not_projection_step"
    assert skipped.local_step == 7


def test_time_consistent_guidance_debug_records_clean_estimate_distance_to_final_latent():
    model = _FakeModel()
    vae = _FakeVae()
    model.generated[0, :, 0, 0, 0] = torch.tensor([1.0, 2.0])
    step_input = {
        "traj_cond_7d_frame": _target_traj(8).unsqueeze(0),
        "traj_cond_frame_mask": torch.ones(1, 8),
        "traj_start_token": 0,
    }
    cfg = TimeConsistentRootGuidanceConfig(
        enabled=True,
        projection_step=8,
        alpha=1.0,
        guidance_strength=1.0,
        max_delta_per_frame=1.0,
        max_delta_per_chunk=4.0,
        max_latent_delta=100.0,
        frames_per_token=4,
        debug=True,
    )
    callback = build_time_consistent_root_guidance_callback(vae=vae, config=cfg)
    assert callback is not None
    assert callback.debug_records is vae._time_consistent_guidance_debug_records

    callback(
        model=model,
        step_input=step_input,
        first_chunk=True,
        commit_index=0,
        current_step=8,
        start_index=0,
        end_index=1,
        x_beta_before_update=torch.tensor([[[1.0]], [[2.0]]]),
        predicted_vel=torch.tensor([[[0.5]], [[-0.5]]]),
        beta_before=torch.tensor(0.2),
        beta_after=torch.tensor(0.1),
        x_after_velocity_update=model.generated[0, :, 0:1, ...],
    )
    callback.finalize_commit_debug(model=model, commit_index=0)

    assert callback.debug_records
    record = callback.debug_records[-1]
    assert record["applied"] is True
    assert record["commit_index"] == 0
    assert record["local_step"] == 8
    assert record["z_hat_l2_to_z_final"] > 0.0
    assert -1.0 <= record["z_hat_cos_to_z_final"] <= 1.0
    for key in [
        "one_minus_beta_after",
        "normal_update_norm",
        "delta_clean_raw_norm",
        "delta_clean_after_clamp_norm",
        "clamp_ratio",
        "guidance_update_norm",
        "guidance_to_denoise_ratio",
        "root_abs_error_before_projection",
        "root_abs_error_after_projection",
        "root_relative_error_before_projection",
        "root_relative_error_after_projection",
        "global_offset_norm",
        "local_shape_error",
        "projection_target_mode",
        "latent_cos_delta_clean_predicted_vel",
        "z_hat_after_l2_to_z_final",
        "z_hat_after_cos_to_z_final",
        "decode_z_hat_root_xz_l2_to_z_final",
        "decode_z_hat_motion_l2_to_z_final",
    ]:
        assert key in record
    assert record["normal_update_norm"] > 0.0
    assert record["delta_clean_raw_norm"] >= record["delta_clean_after_clamp_norm"]


def test_late_projection_skips_before_start_noise_threshold():
    model = _FakeModel()
    vae = _FakeVae()
    step_input = {
        "traj_cond_7d_frame": _target_traj(8).unsqueeze(0),
        "traj_cond_frame_mask": torch.ones(1, 8),
        "traj_start_token": 0,
    }
    cfg = LateDenoiseRootProjectionConfig(
        enabled=True,
        alpha=1.0,
        projection_start_step=6,
        frames_per_token=4,
    )

    result = apply_late_denoise_root_projection(
        model=model,
        vae=vae,
        step_input=step_input,
        first_chunk=True,
        commit_index=0,
        current_step=3,
        start_index=0,
        end_index=2,
        noise_level_full=torch.tensor([[0.8, 1.0]]),
        config=cfg,
    )

    assert result.applied is False
    assert result.reason == "before_projection_ramp"
    assert vae.encoded_motion is None
    assert torch.allclose(model.generated, torch.zeros_like(model.generated))


def test_runtime_step_helper_only_passes_projection_callback_when_enabled():
    from eval.runtime.runners import _stream_generate_step_with_projection

    class _RuntimeModel:
        def __init__(self):
            self.callbacks = []

        def stream_generate_step(
            self,
            step_payload,
            *,
            first_chunk,
            condition,
            projection_callback=None,
        ):
            del step_payload, first_chunk, condition
            self.callbacks.append(projection_callback)
            return {"generated": torch.zeros(1, 1, 2)}

    model = _RuntimeModel()
    vae = _FakeVae()
    _stream_generate_step_with_projection(
        model,
        vae,
        {"text": "walk"},
        first_chunk=True,
        condition_provider=lambda **kwargs: None,
        root_projection_config=None,
    )
    enabled_cfg = LateDenoiseRootProjectionConfig(enabled=True, alpha=0.5)
    _stream_generate_step_with_projection(
        model,
        vae,
        {"text": "walk"},
        first_chunk=False,
        condition_provider=lambda **kwargs: None,
        root_projection_config=enabled_cfg,
    )

    assert model.callbacks[0] is None
    assert callable(model.callbacks[1])
    result = model.callbacks[1](
        model=_FakeModel(),
        step_input={
            "traj_cond_7d_frame": _target_traj(8).unsqueeze(0),
            "traj_cond_frame_mask": torch.ones(1, 8),
            "traj_start_token": 0,
        },
        first_chunk=True,
        commit_index=0,
        current_step=9,
        current_time=0.9,
        start_index=0,
        end_index=5,
        time_steps=torch.tensor([0.9]),
        noise_level_full=torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8]]),
        condition=None,
    )
    assert result.applied is True


def test_runtime_step_helper_uses_original_call_shape_when_projection_disabled():
    from eval.runtime.runners import _stream_generate_step_with_projection

    class _LegacyShapeModel:
        def __init__(self):
            self.called = False

        def stream_generate_step(self, step_payload, *, first_chunk, condition):
            del step_payload, first_chunk, condition
            self.called = True
            return {"generated": torch.zeros(1, 1, 2)}

    model = _LegacyShapeModel()
    _stream_generate_step_with_projection(
        model,
        _FakeVae(),
        {"text": "walk"},
        first_chunk=True,
        condition_provider=lambda **kwargs: None,
        root_projection_config=None,
    )

    assert model.called is True


def test_runtime_benchmark_builds_projection_config_from_args():
    from eval.runtime.benchmark import build_runtime_root_projection_config_from_args

    disabled = build_runtime_root_projection_config_from_args(
        SimpleNamespace(
            late_root_projection=False,
            late_root_projection_alpha=0.7,
            late_root_projection_start_step=5,
            late_root_projection_max_delta_per_frame=0.02,
            late_root_projection_max_delta_per_chunk=0.09,
            late_root_projection_latent_blend_gain=0.4,
            late_root_projection_final_step_gain=0.8,
            late_root_projection_no_frame_ramp=False,
        )
    )
    enabled = build_runtime_root_projection_config_from_args(
        SimpleNamespace(
            late_root_projection=True,
            late_root_projection_alpha=0.7,
            late_root_projection_start_step=5,
            late_root_projection_max_delta_per_frame=0.02,
            late_root_projection_max_delta_per_chunk=0.09,
            late_root_projection_latent_blend_gain=0.4,
            late_root_projection_final_step_gain=0.8,
            late_root_projection_no_frame_ramp=True,
        )
    )

    assert disabled is None
    assert enabled is not None
    assert enabled.enabled is True
    assert enabled.alpha == 0.7
    assert enabled.projection_start_step == 5
    assert enabled.max_delta_per_frame == 0.02
    assert enabled.max_delta_per_chunk == 0.09
    assert enabled.latent_blend_gain == 0.4
    assert enabled.final_step_gain == 0.8
    assert enabled.frame_ramp is False


def test_runtime_benchmark_applies_cfg_scale_overrides():
    from eval.runtime.benchmark import apply_runtime_cfg_overrides

    cfg = OmegaConf.create(
        {
            "model": {
                "params": {
                    "cfg_scale_text": 2.0,
                    "cfg_scale_traj": 2.0,
                }
            }
        }
    )

    apply_runtime_cfg_overrides(
        cfg,
        SimpleNamespace(cfg_text=1.25, cfg_traj=3.0),
    )

    assert cfg.model.params.cfg_scale_text == 1.25
    assert cfg.model.params.cfg_scale_traj == 3.0


def test_runtime_turn_case_accepts_projection_config_keyword():
    from eval.runtime.runners import run_turn_case

    signature = inspect.signature(run_turn_case)

    assert "root_projection_config" in signature.parameters
