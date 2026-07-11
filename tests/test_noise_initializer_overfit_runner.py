from __future__ import annotations

import torch

from models.tools.traj_encoder import TrajectoryEncoder

from utils.training.noise_initializer.shadow_rollout import clip_delta_to_base_norm
from utils.training.noise_initializer.text_encoder import (
    resolve_noise_initializer_text_encoder,
)
from utils.training.noise_initializer.config_validate import (
    validate_noise_initializer_overfit_config,
)
from utils.training.noise_initializer.context_builder import NoiseInitializerContext
from utils.training.noise_initializer.overfit_runner import (
    _build_initializer_context_for_commit,
    _decode_latents_to_root_xz,
    append_terminal_hold,
    apply_initializer_to_stream_state,
    advance_token_update_count,
    advance_model_token_update_count,
    build_noise_initializer_from_ldf,
    affected_history_frames,
    encode_initializer_text_embedding,
    format_train_progress_log,
    resolve_train_progress_log_step,
    resolve_training_commit_indices,
    should_apply_initializer,
    should_train_commit,
    should_log_train_progress,
    slice_future_traj_frames,
    slice_initializer_traj_payload,
)


class _FakeTextModel:
    def encode_text_with_cache(self, texts, device):
        assert texts == ["walk"]
        return [torch.arange(12, dtype=torch.float32, device=device).view(3, 4)]


class _FakeTextRollout:
    def get_text_for_commit_index(self, commit_index: int) -> str:
        del commit_index
        return "walk"


class _FakeStreamModel(_FakeTextModel):
    def __init__(self):
        self.generated = torch.arange(1 * 2 * 12, dtype=torch.float32).view(1, 2, 12, 1, 1)
        self.commit_index = 3
        self.current_step = 10
        self.dt = 0.1
        self.chunk_size = 5
        self.num_denoise_steps = 10


def test_encode_initializer_text_embedding_mean_pools_sequence_context():
    embedding = encode_initializer_text_embedding(
        _FakeTextModel(),
        "walk",
        torch.device("cpu"),
    )

    assert embedding.shape == (1, 4)
    assert torch.equal(embedding[0], torch.tensor([4.0, 5.0, 6.0, 7.0]))


def test_encode_initializer_text_embedding_prefers_precomputed_text_encoder(tmp_path):
    cache_path = tmp_path / "text_embeddings.pt"
    torch.save(
        {
            "embeddings": {
                "walk": torch.arange(12, dtype=torch.float32).view(3, 4),
            },
            "text_dim": 4,
        },
        cache_path,
    )
    text_encoder = resolve_noise_initializer_text_encoder(
        {
            "text_encoder": {
                "type": "precomputed_t5_pool",
                "precomputed_text_emb_path": str(cache_path),
                "pooling": "first",
            }
        },
        text_emb_dim=4,
    )

    embedding = encode_initializer_text_embedding(
        model=None,
        text="walk",
        device=torch.device("cpu"),
        text_encoder=text_encoder,
    )

    assert embedding.shape == (1, 4)
    assert torch.equal(embedding[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_should_train_commit_respects_optimize_every_tokens():
    assert should_train_commit(0, optimize_every_tokens=5)
    assert not should_train_commit(4, optimize_every_tokens=5)
    assert should_train_commit(5, optimize_every_tokens=5)
    assert should_train_commit(3, optimize_every_tokens=1)


def test_resolve_training_commit_indices_fixed_snapshot_selects_one_commit():
    commits = resolve_training_commit_indices(
        {
            "training_mode": "fixed_snapshot_overfit",
            "fixed_commit_index": 10,
            "train_steps_per_commit": 1000,
        },
        target_tokens=46,
    )

    assert commits == [10]


def test_resolve_training_commit_indices_online_uses_decision_interval():
    commits = resolve_training_commit_indices(
        {
            "training_mode": "online_multi_commit",
            "max_commits": 12,
            "optimize_every_tokens": 5,
        },
        target_tokens=46,
    )

    assert commits == [0, 5, 10]


def test_should_apply_initializer_matches_fixed_snapshot_training_policy():
    cfg = {
        "training_mode": "fixed_snapshot_overfit",
        "fixed_commit_index": 10,
        "apply_initializer_every_tokens": 5,
    }

    assert not should_apply_initializer(cfg, 0)
    assert not should_apply_initializer(cfg, 5)
    assert should_apply_initializer(cfg, 10)
    assert not should_apply_initializer(cfg, 15)


def test_should_apply_initializer_matches_online_training_decisions():
    cfg = {
        "training_mode": "online_multi_commit",
        "optimize_every_tokens": 5,
    }

    assert [i for i in range(16) if should_apply_initializer(cfg, i)] == [0, 5, 10, 15]


def test_advance_token_update_count_tracks_actual_triangular_update_ranges():
    counts = torch.zeros(8, dtype=torch.long)

    advance_token_update_count(
        counts,
        start_step=0,
        end_step=2,
        dt=0.1,
        chunk_size=5,
    )

    # step 0 updates [0,1), step 1 also updates [0,1)
    assert counts.tolist() == [2, 0, 0, 0, 0, 0, 0, 0]


def test_advance_model_token_update_count_rolls_with_generated_buffer():
    class FakeModel:
        seq_len = 2
        chunk_size = 1
        num_denoise_steps = 10
        dt = 0.1
        token_update_count = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)

    model = FakeModel()
    advance_model_token_update_count(model, start_step=30, start_commit=3)

    assert model.token_update_count.tolist() == [12, 23, 14, 0, 0]


def test_slice_initializer_traj_payload_aligns_memory_to_absolute_commit():
    frame_count = 4 * 40 - 3
    frames = torch.arange(frame_count, dtype=torch.float32).view(1, frame_count, 1)
    frames = frames.expand(-1, -1, 7).clone()
    payload = {
        "traj_cond_7d_frame": frames,
        "traj_cond_frame_mask": torch.ones(1, frame_count),
        "traj_start_token": 0,
        "traj_abs_start_token": 0,
        "traj_num_tokens": 40,
    }

    window, mask, offsets = slice_initializer_traj_payload(
        payload,
        absolute_commit_index=15,
        traj_tokens=20,
        frames_per_token=4,
    )

    assert window.shape == (1, 80, 7)
    assert window[0, 0, 0].item() == 57.0
    assert window[0, -1, 0].item() == 136.0
    assert torch.equal(mask, torch.ones(1, 80))
    assert offsets.tolist() == list(range(20))


def test_should_log_train_progress_respects_positive_interval():
    assert not should_log_train_progress(0, log_every_train_steps=50)
    assert not should_log_train_progress(49, log_every_train_steps=50)
    assert should_log_train_progress(50, log_every_train_steps=50)
    assert not should_log_train_progress(51, log_every_train_steps=50)
    assert should_log_train_progress(100, log_every_train_steps=50)
    assert not should_log_train_progress(50, log_every_train_steps=0)


def test_resolve_train_progress_log_step_emits_snapshot_baseline_and_intervals():
    assert resolve_train_progress_log_step(
        completed_steps=1,
        inner_step=0,
        log_every_train_steps=50,
    ) == 0
    assert resolve_train_progress_log_step(
        completed_steps=50,
        inner_step=49,
        log_every_train_steps=50,
    ) == 50
    assert resolve_train_progress_log_step(
        completed_steps=51,
        inner_step=50,
        log_every_train_steps=50,
    ) is None
    assert resolve_train_progress_log_step(
        completed_steps=101,
        inner_step=0,
        log_every_train_steps=50,
    ) == 100


def test_format_train_progress_log_preserves_training_diagnostics():
    row = {
        "commit_index": 5,
        "inner_step": 49,
        "loss": 0.125,
        "grad_norm_sum": 1.5,
        "history_frames": 16,
        "traj_loss": 0.1,
        "vel_loss": 0.2,
        "delta_reg": 0.3,
        "applied_delta_norm": 0.45,
        "applied_delta_ratio": 0.09,
        "delta_ratio_reg": 0.0081,
        "optimized_frames": 24,
        "raw_delta_norm": 1.0,
        "clipped_delta_norm": 0.5,
        "base_zT_norm": 5.0,
        "clipped_to_base_ratio": 0.1,
        "delta_scale_mean": 0.5,
        "clip_saturation_ratio": 1.0,
    }

    payload = format_train_progress_log(train_step=50, row=row)

    assert payload == {
        "event": "noise_initializer_train_progress",
        "train_step": 50,
        "commit_index": 5,
        "inner_step": 49,
        "loss": 0.125,
        "grad_norm_sum": 1.5,
        "history_frames": 16,
        "traj_loss": 0.1,
        "vel_loss": 0.2,
        "delta_reg": 0.3,
        "applied_delta_norm": 0.45,
        "applied_delta_ratio": 0.09,
        "delta_ratio_reg": 0.0081,
        "optimized_frames": 24,
        "raw_delta_norm": 1.0,
        "clipped_delta_norm": 0.5,
        "base_zT_norm": 5.0,
        "clipped_to_base_ratio": 0.1,
        "delta_scale_mean": 0.5,
        "clip_saturation_ratio": 1.0,
    }


def test_slice_future_traj_frames_uses_causal_token_frame_mapping_and_pads():
    traj = torch.arange(1 * 6 * 7, dtype=torch.float32).view(1, 6, 7)

    sliced = slice_future_traj_frames(
        traj,
        commit_index=1,
        traj_horizon_tokens=2,
        frames_per_token=4,
    )

    assert sliced.shape == (1, 8, 7)
    assert torch.equal(sliced[:, :5], traj[:, 1:6])
    assert torch.equal(sliced[:, 5:], torch.zeros(1, 3, 7))


def test_decode_latents_to_root_xz_decodes_full_prefix_and_slices_shadow_window(monkeypatch):
    class FakeVae:
        def decode(self, latents):
            values = torch.cat(
                [
                    latents[:, 0:1, 0],
                    latents[:, 1:2, 0].expand(-1, 4),
                    latents[:, 2:3, 0].expand(-1, 4),
                ],
                dim=1,
            )
            return values.unsqueeze(-1).expand(-1, -1, 263)

    monkeypatch.setattr(
        "eval.ldf.latent_initializer.optimize_noise._root_xz_from_feature",
        lambda feature: torch.stack([feature[:, 0], feature[:, 0] + 100.0], dim=-1),
    )
    prefix = torch.tensor([[[1.0], [2.0]]])
    shadow = torch.tensor([[[3.0]]], requires_grad=True)

    pred_xz = _decode_latents_to_root_xz(
        FakeVae(),
        shadow,
        committed_prefix_latents=prefix,
        shadow_start_token=2,
        frames_per_token=4,
    )

    assert pred_xz.shape == (1, 4, 2)
    assert torch.equal(pred_xz[0, :, 0], torch.full((4,), 3.0))
    pred_xz.sum().backward()
    assert shadow.grad is not None
    assert shadow.grad.abs().sum() > 0


def test_context_for_commit_prefers_runtime_traj_payload_over_sample_gt():
    model = _FakeStreamModel()
    sample_batch = {
        "traj_cond_7d": torch.zeros(1, 12, 7),
        "traj_cond_mask": torch.ones(1, 12),
    }
    runtime_payload = {
        "traj_cond_7d_frame": torch.full((1, 9, 7), 7.0),
        "traj_cond_frame_mask": torch.ones(1, 9),
        "traj_start_token": 3,
        "traj_abs_start_token": 3,
        "traj_num_tokens": 2,
    }

    context = _build_initializer_context_for_commit(
        model=model,
        sample_batch=sample_batch,
        text_rollout=_FakeTextRollout(),
        commit_index=3,
        device=torch.device("cpu"),
        text_encoder=None,
        history_tokens=2,
        frontier_tokens=2,
        traj_horizon_tokens=2,
        frames_per_token=4,
        beta_threshold=0.999,
        traj_payload=runtime_payload,
    )

    assert torch.equal(context.traj_token_frames, torch.full((1, 2, 4, 7), 7.0))


def test_apply_initializer_to_stream_state_updates_frontier_before_real_rollin():
    model = _FakeStreamModel()
    context = NoiseInitializerContext(
        history_latents=torch.zeros(1, 2, 2),
        active_latents=torch.zeros(1, 1, 2),
        active_beta=torch.tensor([0.5]),
        active_offsets=torch.tensor([0]),
        text_embedding=torch.zeros(1, 4),
        traj_token_frames=torch.zeros(1, 2, 4, 7),
        traj_frame_mask=torch.ones(1, 2, 4),
        frontier_offsets=torch.tensor([2, 3]),
        frontier_base_zT=(
            model.generated[:, :, torch.tensor([5, 6]), 0, 0]
            .permute(0, 2, 1)
            .contiguous()
        ),
        frontier_ids=torch.tensor([5, 6]),
    )
    original = model.generated.clone()

    class Initializer(torch.nn.Module):
        def forward(self, **kwargs):
            return torch.ones_like(kwargs["frontier_base_zT"]) * 0.5

    diagnostics = apply_initializer_to_stream_state(
        model=model,
        initializer=Initializer(),
        context=context,
        alpha=2.0,
    )

    assert torch.equal(model.generated[:, :, :5], original[:, :, :5])
    assert torch.equal(model.generated[:, :, 7:], original[:, :, 7:])
    assert torch.equal(
        model.generated[:, :, 5:7, 0, 0].permute(0, 2, 1),
        context.frontier_base_zT + 1.0,
    )
    assert diagnostics["applied"] is True
    assert diagnostics["delta_norm"] > 0.0


def test_apply_initializer_to_stream_state_can_clip_delta_to_base_norm_ratio():
    model = _FakeStreamModel()
    context = NoiseInitializerContext(
        history_latents=torch.zeros(1, 2, 2),
        active_latents=torch.zeros(1, 1, 2),
        active_beta=torch.tensor([0.5]),
        active_offsets=torch.tensor([0]),
        text_embedding=torch.zeros(1, 4),
        traj_token_frames=torch.zeros(1, 2, 4, 7),
        traj_frame_mask=torch.ones(1, 2, 4),
        frontier_offsets=torch.tensor([2, 3]),
        frontier_base_zT=torch.ones(1, 2, 2),
        frontier_ids=torch.tensor([5, 6]),
    )

    class Initializer(torch.nn.Module):
        def forward(self, **kwargs):
            return torch.ones_like(kwargs["history_latents"][:, :2]) * 10.0

    diagnostics = apply_initializer_to_stream_state(
        model=model,
        initializer=Initializer(),
        context=context,
        alpha=1.0,
        max_delta_norm_ratio=0.1,
    )

    assert diagnostics["raw_delta_norm"] > diagnostics["delta_norm"]
    assert diagnostics["delta_norm"] <= diagnostics["base_zT_norm"] * 0.1001
    assert diagnostics["delta_scale_mean"] < 1.0


def test_clip_delta_to_base_norm_limits_each_sample_residual():
    delta = torch.ones(2, 2, 2) * 10.0
    base = torch.ones(2, 2, 2)

    clipped, scale = clip_delta_to_base_norm(
        delta,
        base,
        max_delta_norm_ratio=0.1,
    )

    assert clipped[0].norm() <= base[0].norm() * 0.1001
    assert clipped[1].norm() <= base[1].norm() * 0.1001
    assert torch.all(scale < 1.0)


def test_affected_history_frames_starts_loss_when_frontier_can_affect_motion():
    context = NoiseInitializerContext(
        history_latents=torch.zeros(1, 1, 2),
        active_latents=torch.zeros(1, 1, 2),
        active_beta=torch.tensor([0.5]),
        active_offsets=torch.tensor([0]),
        text_embedding=torch.zeros(1, 4),
        traj_token_frames=torch.zeros(1, 2, 4, 7),
        traj_frame_mask=None,
        frontier_offsets=torch.tensor([3, 4]),
        frontier_base_zT=torch.zeros(1, 2, 2),
        frontier_ids=torch.tensor([8, 9]),
        local_commit_index=5,
    )

    assert affected_history_frames(context, frames_per_token=4) == 12


def test_affected_history_frames_handles_causal_token_zero_width():
    context = NoiseInitializerContext(
        history_latents=torch.zeros(1, 0, 2),
        active_latents=torch.zeros(1, 1, 2),
        active_beta=torch.tensor([0.5]),
        active_offsets=torch.tensor([0]),
        text_embedding=torch.zeros(1, 4),
        traj_token_frames=torch.zeros(1, 2, 4, 7),
        traj_frame_mask=None,
        frontier_offsets=torch.tensor([1, 2]),
        frontier_base_zT=torch.zeros(1, 2, 2),
        frontier_ids=torch.tensor([1, 2]),
        local_commit_index=0,
    )

    assert affected_history_frames(context, frames_per_token=4) == 1


def test_context_for_commit_can_require_zero_update_count_frontier():
    model = _FakeStreamModel()
    model.token_update_count = torch.zeros(12, dtype=torch.long)
    model.token_update_count[5] = 1
    sample_batch = {
        "traj_cond_7d": torch.zeros(1, 12, 7),
        "traj_cond_mask": torch.ones(1, 12),
    }

    context = _build_initializer_context_for_commit(
        model=model,
        sample_batch=sample_batch,
        text_rollout=_FakeTextRollout(),
        commit_index=3,
        device=torch.device("cpu"),
        text_encoder=None,
        history_tokens=2,
        frontier_tokens=2,
        traj_horizon_tokens=2,
        frames_per_token=4,
        beta_threshold=0.999,
        traj_payload=None,
        token_update_count=model.token_update_count,
        require_zero_update_count=True,
    )

    assert context.frontier_ids.tolist() == [6, 7]


def test_append_terminal_hold_extends_target_with_zero_velocity_segment():
    target_xz = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    mask = torch.ones(2)

    out_xz, out_mask = append_terminal_hold(target_xz, mask, hold_frames=3)

    assert torch.equal(
        out_xz,
        torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        ),
    )
    assert torch.equal(out_mask, torch.ones(5))


def test_validate_noise_initializer_overfit_config_requires_positive_frontier_tokens():
    cfg = {
        "ckpt": "model.ckpt",
        "meta_path": "meta.txt",
        "model": {"params": {"latent_dim": 4, "text_dim": 4096, "frontier_tokens": 0}},
    }

    try:
        validate_noise_initializer_overfit_config(cfg)
    except ValueError as exc:
        assert "frontier_tokens" in str(exc)
    else:
        raise AssertionError("expected frontier_tokens validation error")


def test_validate_noise_initializer_overfit_config_requires_precomputed_text_path():
    cfg = {
        "ckpt": "model.ckpt",
        "meta_path": "meta.txt",
        "model": {"params": {"latent_dim": 4, "text_dim": 4096, "frontier_tokens": 5}},
        "text_encoder": {"type": "precomputed_t5_pool"},
    }

    try:
        validate_noise_initializer_overfit_config(cfg)
    except ValueError as exc:
        assert "precomputed_text_emb_path" in str(exc)
    else:
        raise AssertionError("expected precomputed text path validation error")


def test_validate_fixed_snapshot_mode_rejects_initializer_rollin():
    cfg = {
        "ckpt": "model.ckpt",
        "meta_path": "meta.txt",
        "training_mode": "fixed_snapshot_overfit",
        "fixed_commit_index": 10,
        "apply_initializer_rollin": True,
        "model": {"params": {"latent_dim": 4, "text_dim": 4096, "frontier_tokens": 5}},
    }

    try:
        validate_noise_initializer_overfit_config(cfg)
    except ValueError as exc:
        assert "apply_initializer_rollin" in str(exc)
    else:
        raise AssertionError("expected fixed snapshot rollin validation error")


def test_build_noise_initializer_from_ldf_copies_and_freezes_traj_encoder():
    class FakeLdf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.traj_encoder = TrajectoryEncoder(out_dim=8)

    ldf = FakeLdf()
    cfg = {
        "model": {
            "reuse_ldf_traj_encoder": True,
            "params": {
                "latent_dim": 4,
                "text_dim": 6,
                "frontier_tokens": 3,
                "hidden_dim": 32,
                "traj_emb_dim": 8,
                "freeze_traj_encoder": True,
            },
        }
    }

    initializer = build_noise_initializer_from_ldf(cfg, ldf)

    source_param = next(ldf.traj_encoder.parameters())
    copied_param = next(initializer.traj_encoder.parameters())
    assert torch.equal(source_param, copied_param)
    assert source_param.data_ptr() != copied_param.data_ptr()
    assert all(not parameter.requires_grad for parameter in initializer.traj_encoder.parameters())
