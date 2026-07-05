from __future__ import annotations

import pytest
import torch

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
