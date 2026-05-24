"""Unit tests for eval/benchmark_gates.py (Phase D gate 把关)."""

from __future__ import annotations

from eval.benchmark_gates import (
    GATES,
    Gate,
    all_passed,
    check_gate,
    evaluate,
    improvement_pct,
)


def test_lt_gate_pass_and_fail():
    g = Gate("Gate X", "BENCH", "err", "<", 15.0)
    assert check_gate(g, {"err": 10.0}).passed
    assert not check_gate(g, {"err": 20.0}).passed
    assert not check_gate(g, {"err": 15.0}).passed   # strict <


def test_gt_gate():
    g = Gate("Gate X", "BENCH", "score", ">", 0.8)
    assert check_gate(g, {"score": 0.9}).passed
    assert not check_gate(g, {"score": 0.8}).passed


def test_missing_metric_fails_loud():
    g = Gate("Gate X", "BENCH", "err", "<", 1.0)
    r = check_gate(g, {"other": 0.0})
    assert not r.passed and r.value is None and "missing" in r.reason


def test_nonfinite_metric_fails():
    g = Gate("Gate X", "BENCH", "err", "<", 1.0)
    assert not check_gate(g, {"err": float("nan")}).passed
    assert not check_gate(g, {"err": float("inf")}).passed


def test_evaluate_filters_by_benchmark():
    res = evaluate({}, benchmark="T_BENCH_B")
    assert res and all(r.benchmark == "T_BENCH_B" for r in res)
    # all missing → all fail
    assert not all_passed(res)


def test_evaluate_filters_by_gate_id_gate9_has_three_checks():
    res = evaluate({}, gate_id="Gate 9")
    assert {r.metric for r in res} == {
        "rotation_equivariance_error_median_m", "anchor_misuse_check", "turn_success_rate",
    }


def test_gate5_mask_effect_delta_threshold():
    # Gate 5 (T_BENCH_B) + Gate 10 (T_BENCH_G) both gate mask_effect_delta < 1e-4
    g5 = [g for g in GATES if g.gate_id == "Gate 5"][0]
    assert check_gate(g5, {"mask_effect_delta": 5e-5}).passed
    assert not check_gate(g5, {"mask_effect_delta": 2e-4}).passed


def test_full_pass_for_a_benchmark():
    metrics = {
        "rotation_equivariance_error_median_m": 0.3,
        "anchor_misuse_check": 2.5,
        "turn_success_rate": 0.9,
    }
    res = evaluate(metrics, benchmark="T_BENCH_D")
    assert all_passed(res)
    # one bad metric flips it
    metrics["turn_success_rate"] = 0.5
    assert not all_passed(evaluate(metrics, benchmark="T_BENCH_D"))


def test_all_gates_have_valid_ops():
    for g in GATES:
        assert g.op in ("<", "<=", ">", ">=")


def test_improvement_pct():
    assert improvement_pct(0.4, 0.2) == 50.0          # halved drift (lower better)
    assert improvement_pct(0.4, 0.4) == 0.0
    assert improvement_pct(0.0, 0.2) == 0.0            # baseline 0 → 0
    assert improvement_pct(1.0, 2.0, lower_is_better=False) == 100.0
