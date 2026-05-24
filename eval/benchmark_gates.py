"""Phase D layered-benchmark gates ("把关": 未通过不进下一层).

Single source of truth for the absolute pass/fail thresholds from docs/TODO.md
Phase D (design §10). The actual T_BENCH_* runs need trained models + data and
execute on the runtime box; each run produces a `metrics` dict and calls
`evaluate(metrics, benchmark=...)` here to decide gate pass/fail consistently
(instead of eyeballing numbers against the doc).

Comparative gates (e.g. "R3 improves over R2", "C3 drift < 50% of C0") need two
runs and are checked with `improvement_pct` (documented per task), not encoded
as absolute thresholds here.
"""

from __future__ import annotations

from dataclasses import dataclass

_OPS = {
    "<": lambda v, t: v < t,
    "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t,
    ">=": lambda v, t: v >= t,
}


@dataclass(frozen=True)
class Gate:
    gate_id: str        # e.g. "Gate 5"
    benchmark: str      # e.g. "T_BENCH_B"
    metric: str         # key expected in the metrics dict
    op: str             # "<" | "<=" | ">" | ">="
    threshold: float
    note: str = ""


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    benchmark: str
    metric: str
    passed: bool
    value: float | None
    threshold: float
    op: str
    reason: str


# Absolute-threshold gates (degrees for headings, metres for positions).
GATES: tuple[Gate, ...] = (
    Gate("Gate 3", "T_BENCH_A", "turning_segment_heading_error_median_deg", "<", 15.0,
         "Refiner turning heading median < 15°"),
    Gate("Gate 5", "T_BENCH_B", "mask_effect_delta", "<", 1e-4,
         "horizon-mask: mask=0 frames must not affect encoder output"),
    Gate("Gate 6", "T_BENCH_B", "heading_angle_error_median_deg", "<", 15.0,
         "body GT-7D heading median < 15°"),
    Gate("Gate 8", "T_BENCH_C_GTPlan", "closed_loop_root_drift_median_m", "<", 0.3,
         "closed-loop root drift median < 0.3m over 10s"),
    Gate("Gate 8b", "T_BENCH_C_RefinerPlan", "target_current_error_median_m", "<", 0.5,
         "Refiner-plan closed-loop target/current error median < 0.5m"),
    Gate("Gate 9", "T_BENCH_D", "rotation_equivariance_error_median_m", "<", 0.5,
         "same local intent across initial yaws → consistent final pos < 0.5m"),
    Gate("Gate 9", "T_BENCH_D", "anchor_misuse_check", ">", 1.0,
         "diagnostic must be sensitive: using head anchor would mis-place > 1.0"),
    Gate("Gate 9", "T_BENCH_D", "turn_success_rate", ">", 0.8,
         "rot60/90 turn success rate > 80%"),
    Gate("Gate 10", "T_BENCH_G", "mask_effect_delta", "<", 1e-4,
         "clear/no-traj: mask=0 region must not affect body output"),
    Gate("Gate 11", "T_BENCH_F", "replace_boundary_position_jump_median_m", "<", 0.5,
         "mid-session replace boundary position jump median < 0.5m"),
    Gate("Gate 11", "T_BENCH_F", "commit_boundary_position_jump_median_m", "<", 0.1,
         "commit boundary position jump median < 0.1m"),
    Gate("Gate 11", "T_BENCH_F", "stale_result_rate", "<", 0.05,
         "stale Refiner results / total requests < 5%"),
)


def check_gate(gate: Gate, metrics: dict) -> GateResult:
    """Evaluate one gate against a metrics dict. A missing or non-finite metric
    FAILS the gate (never silently pass)."""
    import math

    if gate.metric not in metrics:
        return GateResult(gate.gate_id, gate.benchmark, gate.metric, False, None,
                          gate.threshold, gate.op,
                          f"metric '{gate.metric}' missing from results")
    value = float(metrics[gate.metric])
    if not math.isfinite(value):
        return GateResult(gate.gate_id, gate.benchmark, gate.metric, False, value,
                          gate.threshold, gate.op,
                          f"metric '{gate.metric}' is non-finite ({value})")
    passed = _OPS[gate.op](value, gate.threshold)
    reason = (f"{gate.metric}={value:g} {gate.op} {gate.threshold:g} → "
              f"{'PASS' if passed else 'FAIL'}")
    return GateResult(gate.gate_id, gate.benchmark, gate.metric, passed, value,
                      gate.threshold, gate.op, reason)


def evaluate(metrics: dict, *, benchmark: str | None = None,
             gate_id: str | None = None) -> list[GateResult]:
    """Check all gates (optionally filtered by benchmark / gate_id) against
    `metrics`. Returns one GateResult per applicable gate check."""
    gates = [
        g for g in GATES
        if (benchmark is None or g.benchmark == benchmark)
        and (gate_id is None or g.gate_id == gate_id)
    ]
    return [check_gate(g, metrics) for g in gates]


def all_passed(results: list[GateResult]) -> bool:
    return bool(results) and all(r.passed for r in results)


def improvement_pct(baseline: float, candidate: float, *,
                    lower_is_better: bool = True) -> float:
    """Relative improvement of `candidate` over `baseline`, in percent, for the
    comparative gates (e.g. C3 drift vs C0). lower_is_better=True → improvement
    is reduction; returns 50.0 for a halving. baseline==0 → 0.0."""
    if baseline == 0:
        return 0.0
    delta = (baseline - candidate) if lower_is_better else (candidate - baseline)
    return 100.0 * delta / abs(baseline)


__all__ = ["Gate", "GateResult", "GATES", "check_gate", "evaluate",
           "all_passed", "improvement_pct"]
