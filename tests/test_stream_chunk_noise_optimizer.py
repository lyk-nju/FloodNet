from __future__ import annotations

import torch

from eval.ldf.latent_initializer.optimize_stream_chunk_noise import (
    _active_token_slice,
    _anchored_absolute_xz_loss,
    _active_state_method_metadata,
    _baseline_optimized_xz_diff,
    _build_window_diagnostics,
    _extract_window_noise,
    _extract_window_tokens,
    _inject_window_noise,
    _relative_xz_loss,
)
from eval.ldf.latent_initializer.true_future_zT_local_oracle import (
    _local_true_zT_method_metadata,
    _select_future_zT_window,
    _should_optimize_commit,
)


def test_active_token_slice_clamps_to_target_tokens():
    assert _active_token_slice(10, horizon_tokens=5, target_tokens=20) == (10, 15)
    assert _active_token_slice(18, horizon_tokens=5, target_tokens=20) == (18, 20)
    assert _active_token_slice(19, horizon_tokens=0, target_tokens=20) == (19, 20)


def test_window_noise_roundtrip_replaces_only_requested_tokens():
    generated_tokens = torch.arange(1 * 4 * 6, dtype=torch.float32).view(1, 4, 6, 1, 1)
    window = _extract_window_noise(generated_tokens, start=2, end=5)

    assert window.shape == (1, 3, 4)
    assert torch.equal(window[0, 0], generated_tokens[0, :, 2, 0, 0])

    replacement = torch.full_like(window, -7.0).requires_grad_(True)
    injected = _inject_window_noise(generated_tokens, replacement, start=2)

    assert torch.equal(injected[:, :, :2], generated_tokens[:, :, :2])
    assert torch.equal(injected[:, :, 5:], generated_tokens[:, :, 5:])
    assert torch.equal(injected[0, :, 2:5, 0, 0], replacement[0].T)
    assert injected.requires_grad


def test_extract_window_tokens_detaches_history_when_requested():
    generated_tokens = torch.arange(1 * 4 * 6, dtype=torch.float32).view(1, 4, 6, 1, 1)
    generated_tokens.requires_grad_(True)

    detached = _extract_window_tokens(generated_tokens, start=1, end=4, detach=True)
    attached = _extract_window_tokens(generated_tokens, start=1, end=4, detach=False)

    assert detached.shape == (1, 3, 4)
    assert attached.shape == (1, 3, 4)
    assert detached.requires_grad is False
    assert attached.requires_grad is True


def test_relative_xz_loss_ignores_absolute_translation_and_penalizes_shape():
    target = torch.tensor([[10.0, 2.0], [11.0, 2.0], [12.0, 2.0]])
    translated_match = target + torch.tensor([100.0, -50.0])
    bent = torch.tensor([[100.0, -50.0], [101.0, -49.0], [102.0, -50.0]])
    mask = torch.ones(3)

    match_loss, match_parts = _relative_xz_loss(
        translated_match,
        target,
        mask,
        lambda_vel=0.1,
    )
    bent_loss, bent_parts = _relative_xz_loss(
        bent,
        target,
        mask,
        lambda_vel=0.1,
    )

    assert float(match_loss.item()) == 0.0
    assert match_parts["traj_loss"] == 0.0
    assert float(bent_loss.item()) > 0.0
    assert bent_parts["vel_loss"] > 0.0


def test_target_anchor_absolute_xz_loss_penalizes_translation_and_masks_history():
    # Two history frames, then three optimized frames.  The prediction has the
    # right local shape but wrong absolute location after anchoring.
    pred_xz = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        requires_grad=True,
    )
    target_xz = torch.tensor(
        [[10.0, 2.0], [11.0, 2.0], [20.0, 2.0], [21.0, 2.0], [22.0, 2.0]]
    )
    mask = torch.ones(5)

    loss, parts = _anchored_absolute_xz_loss(
        pred_xz,
        target_xz,
        mask,
        history_frames=2,
        lambda_vel=0.0,
        anchor_mode="target_anchor_abs",
    )
    loss.backward()

    assert parts["history_frames"] == 2
    assert parts["optimized_frames"] == 3
    assert float(loss.item()) > 0.0
    assert pred_xz.grad is not None
    assert torch.equal(pred_xz.grad[:2], torch.zeros_like(pred_xz.grad[:2]))
    assert bool((pred_xz.grad[2:].abs().sum(dim=-1) > 0).all())


def test_generated_anchor_absolute_xz_loss_uses_generated_anchor():
    pred_xz = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        requires_grad=True,
    )
    target_xz = torch.tensor([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]])
    generated_anchor = torch.tensor([100.0, 0.0])
    mask = torch.ones(4)

    target_anchor_loss, target_parts = _anchored_absolute_xz_loss(
        pred_xz,
        target_xz,
        mask,
        history_frames=2,
        lambda_vel=0.0,
        anchor_mode="target_anchor_abs",
    )
    generated_anchor_loss, generated_parts = _anchored_absolute_xz_loss(
        pred_xz,
        target_xz,
        mask,
        history_frames=2,
        lambda_vel=0.0,
        anchor_mode="generated_anchor_abs",
        generated_anchor_xz=generated_anchor,
    )

    assert float(target_anchor_loss.item()) == 0.0
    assert float(generated_anchor_loss.item()) > 0.0
    assert target_parts["anchor_mode"] == "target_anchor_abs"
    assert generated_parts["anchor_mode"] == "generated_anchor_abs"


def test_active_state_metadata_explicitly_excludes_initial_zT():
    metadata = _active_state_method_metadata()

    assert metadata["method"] == "active_noisy_state_opt"
    assert metadata["optimized_variable"] == "model.generated[start:end] current x_beta"
    assert metadata["not_optimized_variable"] == "initial Gaussian z_T"


def test_window_diagnostics_records_distribution_schedule_and_history_diff():
    generated = torch.zeros(1, 2, 6, 1, 1)
    base_window = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    optimized_window = base_window + 0.5
    before = generated.clone()
    after = generated.clone()
    after[:, :, 2:4, 0, 0] = optimized_window[0].T

    diagnostics = _build_window_diagnostics(
        state_generated=before,
        optimized_generated=after,
        base_window=base_window,
        optimized_window=optimized_window,
        start=2,
        end=4,
        current_step=6,
        commit_index=2,
        dt=0.1,
        chunk_size=5,
    )

    assert diagnostics["base_window_mean"] == float(base_window.mean().item())
    assert diagnostics["optimized_window_std"] == float(optimized_window.std(unbiased=False).item())
    assert diagnostics["optimized_minus_base_l2"] > 0.0
    assert diagnostics["current_step"] == 6
    assert diagnostics["commit_index"] == 2
    assert diagnostics["token_indices"] == [2, 3]
    assert len(diagnostics["token_beta"]) == 2
    assert diagnostics["history_before_after_diff"] == 0.0


def test_baseline_optimized_xz_diff_reports_zero_for_identical_paths():
    xz = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    same = _baseline_optimized_xz_diff(xz, xz)
    shifted = _baseline_optimized_xz_diff(xz, xz + 1.0)

    assert same["mean_l2"] == 0.0
    assert same["max_l2"] == 0.0
    assert shifted["mean_l2"] > 0.0
    assert shifted["max_l2"] > 0.0


def test_local_true_zT_metadata_excludes_active_x_beta():
    metadata = _local_true_zT_method_metadata()

    assert metadata["method"] == "local_true_future_zT_oracle"
    assert metadata["optimized_variable"] == "future model.generated tokens with beta ~= 1"
    assert metadata["not_optimized_variable"] == "active x_beta or committed z0"


def test_select_future_zT_window_skips_active_noisy_tokens():
    # At current_time=1.0 and chunk_size=5, token 4 has beta=0.8, while
    # token 5 and beyond are still pure initial-noise z_T.
    window = _select_future_zT_window(
        commit_index=1,
        current_step=10,
        dt=0.1,
        chunk_size=5,
        generated_tokens=20,
        target_tokens=20,
        zT_horizon_tokens=5,
        beta_threshold=0.999,
    )

    assert window["start"] == 5
    assert window["end"] == 10
    assert window["token_indices"] == [5, 6, 7, 8, 9]
    assert all(beta >= 0.999 for beta in window["token_beta"])


def test_should_optimize_commit_supports_interval_schedule():
    assert _should_optimize_commit(0, optimize_every_tokens=5)
    assert not _should_optimize_commit(1, optimize_every_tokens=5)
    assert not _should_optimize_commit(4, optimize_every_tokens=5)
    assert _should_optimize_commit(5, optimize_every_tokens=5)
    assert _should_optimize_commit(6, optimize_every_tokens=1)
    assert _should_optimize_commit(6, optimize_every_tokens=0)
