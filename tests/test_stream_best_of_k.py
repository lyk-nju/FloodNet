from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from eval.ldf.stream_state import (
    StepOutput,
    capture_runtime_snapshot,
    restore_runtime_snapshot,
)
from eval.ldf.stream_scoring import (
    ConservativeGateConfig,
    CandidateScore,
    choose_candidate,
    compute_chunk_xz_score,
    compute_continuity_score,
    decoded_chunk_root_xz,
    previous_root_xz,
    target_xz_slice,
)


def test_conservative_gate_keeps_candidate0_when_improvement_is_small():
    cfg = ConservativeGateConfig(
        fde_weight=1.0,
        vel_weight=0.5,
        rel_margin=0.10,
        abs_margin=0.03,
        cont_tol=0.03,
        force_candidate0=False,
        switch_cooldown_steps=0,
    )
    scores = [
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
        CandidateScore(index=1, xz_ade=0.19, xz_fde=0.19, pos_cont=0.01, vel_cont=0.01),
    ]
    decision = choose_candidate(scores, cfg, steps_since_switch=100)
    assert decision.selected_index == 0
    assert decision.reason == "keep_candidate0"
    assert decision.candidate0_track == pytest.approx(0.40)


def test_conservative_gate_switches_for_clear_tracking_gain_with_good_continuity():
    cfg = ConservativeGateConfig(
        fde_weight=1.0,
        vel_weight=0.5,
        rel_margin=0.10,
        abs_margin=0.03,
        cont_tol=0.03,
        force_candidate0=False,
        switch_cooldown_steps=0,
    )
    scores = [
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
        CandidateScore(index=1, xz_ade=0.10, xz_fde=0.10, pos_cont=0.02, vel_cont=0.01),
    ]
    decision = choose_candidate(scores, cfg, steps_since_switch=100)
    assert decision.selected_index == 1
    assert decision.reason == "switch_improved_continuous"
    assert decision.selected_track == pytest.approx(0.20)


def test_conservative_gate_uses_candidate0_by_index_when_scores_are_unordered():
    cfg = ConservativeGateConfig(
        fde_weight=1.0,
        vel_weight=0.5,
        rel_margin=0.10,
        abs_margin=0.03,
        cont_tol=0.03,
        force_candidate0=False,
        switch_cooldown_steps=0,
    )
    scores = [
        CandidateScore(index=1, xz_ade=0.10, xz_fde=0.10, pos_cont=0.02, vel_cont=0.01),
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
    ]
    decision = choose_candidate(scores, cfg, steps_since_switch=100)
    assert decision.selected_index == 1
    assert decision.reason == "switch_improved_continuous"
    assert decision.candidate0_track == pytest.approx(0.40)
    assert decision.selected_track == pytest.approx(0.20)
    assert decision.candidate0_continuity == pytest.approx(0.015)


def test_conservative_gate_requires_candidate0_score():
    cfg = ConservativeGateConfig()
    scores = [
        CandidateScore(index=1, xz_ade=0.10, xz_fde=0.10, pos_cont=0.01, vel_cont=0.01),
    ]
    with pytest.raises(ValueError, match="index 0"):
        choose_candidate(scores, cfg, steps_since_switch=100)


def test_conservative_gate_respects_switch_cooldown():
    cfg = ConservativeGateConfig(switch_cooldown_steps=3)
    scores = [
        CandidateScore(index=1, xz_ade=0.01, xz_fde=0.01, pos_cont=0.01, vel_cont=0.01),
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
    ]
    decision = choose_candidate(scores, cfg, steps_since_switch=1)
    assert decision.selected_index == 0
    assert decision.reason == "cooldown"


def test_conservative_gate_rejects_bad_continuity():
    cfg = ConservativeGateConfig(
        fde_weight=1.0,
        vel_weight=0.5,
        rel_margin=0.10,
        abs_margin=0.03,
        cont_tol=0.03,
        force_candidate0=False,
        switch_cooldown_steps=0,
    )
    scores = [
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
        CandidateScore(index=1, xz_ade=0.05, xz_fde=0.05, pos_cont=0.20, vel_cont=0.20),
    ]
    decision = choose_candidate(scores, cfg, steps_since_switch=100)
    assert decision.selected_index == 0
    assert decision.reason == "keep_candidate0"


def test_force_candidate0_overrides_better_candidates():
    cfg = ConservativeGateConfig(
        fde_weight=1.0,
        vel_weight=0.5,
        rel_margin=0.10,
        abs_margin=0.03,
        cont_tol=0.03,
        force_candidate0=True,
        switch_cooldown_steps=0,
    )
    scores = [
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
        CandidateScore(index=1, xz_ade=0.01, xz_fde=0.01, pos_cont=0.01, vel_cont=0.01),
    ]
    decision = choose_candidate(scores, cfg, steps_since_switch=100)
    assert decision.selected_index == 0
    assert decision.reason == "force_candidate0"


def test_chunk_xz_score_uses_ade_and_fde():
    pred = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0], [1.5, 0.0], [1.0, 0.0]], dtype=torch.float32)
    ade, fde = compute_chunk_xz_score(pred, target)
    assert torch.isclose(torch.tensor(ade), torch.tensor(0.5))
    assert torch.isclose(torch.tensor(fde), torch.tensor(1.0))


def test_continuity_score_uses_position_and_velocity_jump():
    prev = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    candidate = torch.tensor([[1.2, 0.0], [2.5, 0.0]], dtype=torch.float32)
    pos_jump, vel_jump = compute_continuity_score(prev, candidate)
    assert torch.isclose(torch.tensor(pos_jump), torch.tensor(0.2), atol=1e-6)
    assert torch.isclose(torch.tensor(vel_jump), torch.tensor(0.3), atol=1e-6)


def test_target_xz_slice_matches_current_7d_eval_contract():
    traj7 = torch.zeros(1, 3, 7, dtype=torch.float32)
    traj7[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    traj7[0, :, 2] = torch.tensor([4.0, 5.0, 6.0])
    traj7[0, :, 3] = 1.0
    out = target_xz_slice({"traj_cond_7d": traj7}, (1, 3))
    assert torch.allclose(out, torch.tensor([[2.0, 5.0], [3.0, 6.0]]))


def test_root_xz_helpers_return_finite_xz_shapes():
    previous = [torch.zeros(2, 263, dtype=torch.float32)]
    decoded = torch.zeros(3, 263, dtype=torch.float32)

    decoded_xz = decoded_chunk_root_xz(decoded, previous)
    previous_xz = previous_root_xz(previous)

    assert decoded_xz.shape == (3, 2)
    assert previous_xz.shape == (2, 2)
    assert torch.isfinite(decoded_xz).all()
    assert torch.isfinite(previous_xz).all()


class _StatefulVAEModel:
    def __init__(self):
        self._conv_num = 1
        self._conv_idx = [3]
        self._feat_map = [torch.tensor([1.0, 2.0])]


class _StatefulVAE:
    def __init__(self):
        self.model = _StatefulVAEModel()


class _StatefulConditioner:
    def __init__(self):
        self.timeline = {"head": torch.tensor([1.0])}
        self.root_plan = {"anchor": torch.tensor([2.0])}


class _StatefulModel:
    def __init__(self):
        self.generated = torch.ones(1, 4, 5, 1, 1)
        self.commit_index = 2
        self.current_step = 7
        self.batch_size = 1
        self.seq_len = 4
        self.num_denoise_steps = 10
        self.dt = 0.1
        self.text_condition_list = [[torch.tensor([3.0])]]
        self._traj_buf = {"value": torch.tensor([4.0])}


def test_runtime_snapshot_deep_clones_mutable_state():
    model = _StatefulModel()
    vae = _StatefulVAE()
    conditioner = _StatefulConditioner()

    snapshot = capture_runtime_snapshot(
        model=model,
        vae=vae,
        stream_conditioner=conditioner,
        first_chunk=False,
        generated_frames=12,
        chunk_frame_ends=[4, 8, 12],
    )
    model.generated.zero_()
    model.text_condition_list[0][0].fill_(9.0)
    vae.model._feat_map[0].fill_(8.0)
    conditioner.timeline["head"].fill_(7.0)

    restore_runtime_snapshot(model=model, vae=vae, stream_conditioner=conditioner, snapshot=snapshot)

    assert torch.allclose(model.generated, torch.ones(1, 4, 5, 1, 1))
    assert torch.allclose(model.text_condition_list[0][0], torch.tensor([3.0]))
    assert torch.allclose(vae.model._feat_map[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(conditioner.timeline["head"], torch.tensor([1.0]))
    assert snapshot.first_chunk is False
    assert snapshot.generated_frames == 12
    assert snapshot.chunk_frame_ends == [4, 8, 12]
    assert model.generated.data_ptr() != snapshot.model_state["generated"].data_ptr()


def test_runtime_snapshot_restores_rng_state():
    model = _StatefulModel()
    vae = _StatefulVAE()
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    snapshot = capture_runtime_snapshot(
        model=model,
        vae=vae,
        stream_conditioner=None,
        first_chunk=True,
        generated_frames=0,
        chunk_frame_ends=[],
    )
    expected_py = random.random()
    expected_np = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_runtime_snapshot(model=model, vae=vae, stream_conditioner=None, snapshot=snapshot)

    assert random.random() == expected_py
    assert float(np.random.rand()) == expected_np
    assert float(torch.rand(1).item()) == expected_torch


def test_step_output_records_explicit_ranges():
    output = StepOutput(
        clean_committed_latent=torch.zeros(1, 4),
        decoded_chunk=torch.zeros(4, 263),
        commit_token_range=(2, 3),
        commit_frame_range=(8, 12),
        decoded_chunk_frame_range=(8, 12),
        target_xz_frame_range=(8, 12),
        debug={"ready_to_commit_token": 2},
    )

    assert output.commit_token_range == (2, 3)
    assert output.commit_frame_range == (8, 12)
    assert output.debug["ready_to_commit_token"] == 2


def test_best_of_k_disabled_uses_original_single_step_path(monkeypatch):
    import eval.ldf.stream_generation as stream_generation

    called = {"selector": 0}

    def _selector_should_not_run(**kwargs):
        called["selector"] += 1
        raise AssertionError("best_of_k <= 1 must bypass selector")

    monkeypatch.setattr(stream_generation, "_stream_generate_step_best_of_k", _selector_should_not_run)

    assert stream_generation._should_use_best_of_k(1) is False
    assert stream_generation._should_use_best_of_k(0) is False
    assert stream_generation._should_use_best_of_k(2) is True
    assert called["selector"] == 0


class _BypassModel:
    input_dim = 4
    noise_steps = 1
    chunk_size = 1
    seq_len = 2
    use_text_cond = True
    param_dtype = torch.float32

    def __init__(self):
        self.batch_size = 1
        self.commit_index = 0
        self.text_condition_list = [[]]

    def init_generated(self, history_length, batch_size, num_denoise_steps, traj_buffer=None):
        self.batch_size = int(batch_size)
        self.commit_index = 0
        self.text_condition_list = [[] for _ in range(batch_size)]

    def encode_text_with_cache(self, text_list, device):
        return [torch.zeros(1, 1, device=device) for _ in text_list]

    def stream_generate_step(self, step_payload, first_chunk=True, condition=None):
        self.commit_index += 1
        latent = torch.full((1, 1, self.input_dim), float(self.commit_index))
        return {"generated": latent}


class _BypassVAE:
    def clear_cache(self):
        pass

    def stream_decode(self, latent, first_chunk=True):
        return torch.zeros(1, 4, 263, dtype=torch.float32)


def test_best_of_k_disabled_skips_config_validation_and_selector(monkeypatch):
    import eval.ldf.stream_generation as stream_generation

    def _selector_should_not_run(**kwargs):
        raise AssertionError("best_of_k <= 1 must bypass selector")

    sample_batch = {
        "name": ["sample"],
        "dataset": ["HumanML3D"],
        "text": ["walk"],
        "token_length": torch.tensor([1], dtype=torch.long),
        "feature_length": torch.tensor([4], dtype=torch.long),
    }

    monkeypatch.setattr(stream_generation, "_stream_generate_step_best_of_k", _selector_should_not_run)
    stream_out = stream_generation.run_stream_generate_step_sample(
        model=_BypassModel(),
        vae=_BypassVAE(),
        sample_batch=sample_batch,
        device=torch.device("cpu"),
        history_length=2,
        num_denoise_steps=1,
        best_of_k=1,
        best_of_k_score="bad",
    )

    assert stream_out["stream_best_of_k"]["enabled"] is False
    assert stream_out["stream_best_of_k"]["k"] == 1
    assert stream_out["stream_best_of_k"]["score"] == "bad"
    assert stream_out["stream_best_of_k"]["records"] == []


class _OneStepModel(_StatefulModel):
    input_dim = 4
    chunk_size = 1

    def stream_generate_step(self, step_payload, first_chunk=True, condition=None):
        self.commit_index += 1
        latent = torch.full((1, 1, self.input_dim), float(self.commit_index))
        return {"generated": latent}


class _OneStepVAE:
    def __init__(self):
        self.model = _StatefulVAEModel()
        self.calls = []

    def stream_decode(self, latent, first_chunk=True):
        self.calls.append((latent.detach().clone(), bool(first_chunk)))
        frames = torch.zeros(1, 4, 263, dtype=torch.float32)
        frames[:, :, 0] = latent[:, :, 0].view(1, 1)
        return frames


class _OneStepStream:
    def build_ldf_condition_provider(self, step_payload, first_chunk=True, device=None):
        return lambda **kwargs: None


def test_run_one_stream_step_returns_clean_commit_ranges():
    from eval.ldf.stream_step import run_one_stream_step

    model = _OneStepModel()
    vae = _OneStepVAE()
    output = run_one_stream_step(
        model=model,
        vae=vae,
        stream=_OneStepStream(),
        step_payload={"text": "walk"},
        first_chunk=True,
        device=torch.device("cpu"),
        local_commit_index=0,
        generated_frames=0,
        frames_per_token=4,
    )

    assert output.clean_committed_latent.shape == (1, 4)
    assert output.decoded_chunk.shape == (4, 263)
    assert output.commit_token_range == (0, 1)
    assert output.commit_frame_range == (0, 4)
    assert output.decoded_chunk_frame_range == (0, 4)
    assert output.debug["ready_to_commit_token"] == 0
