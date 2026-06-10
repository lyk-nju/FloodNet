from __future__ import annotations

import json

from eval.ldf.report import build_checkpoint_comparison, main


def _write_summary(
    path,
    *,
    tag: str,
    ade: float,
    fde: float,
    jump: float,
    path_arc: float = 0.10,
    jitter: float = 0.002,
    yaw: float = 0.15,
    skating: float = 0.01,
):
    payload = {
        "probe_tag": tag,
        "ckpt": f"/ckpts/{tag}.ckpt",
        "summary": {
            "traj/ADE_mean": ade,
            "traj/FDE_mean": fde,
            "path/arc_ADE_mean": path_arc,
            "stream_no_traj/root_ADE_mean": ade + 0.10,
            "stream_no_traj/root_FDE_mean": fde + 0.20,
            "control_gain/root_ADE_delta_mean": 0.10,
            "control_gain/root_FDE_delta_mean": 0.20,
            "stream_boundary/root_jump_mean": jump,
            "stream_vs_offline/root_ade_mean": ade / 2.0,
            "traj/jitter_mean": jitter,
            "stream_gt/yaw_error_mean": yaw,
            "control/Control_L2_dist_mean": ade * 10.0,
            "control/Skating_Ratio_mean": skating,
        },
    }
    path.write_text(json.dumps(payload))


def test_build_checkpoint_comparison_aliases_stream_metrics_and_thresholds(tmp_path):
    baseline_path = tmp_path / "baseline_summary.json"
    candidate_path = tmp_path / "candidate_summary.json"
    _write_summary(baseline_path, tag="full_prefix", ade=0.20, fde=0.30, jump=0.01)
    _write_summary(candidate_path, tag="window_local", ade=0.23, fde=0.32, jump=0.012)

    report = build_checkpoint_comparison(
        baseline_path,
        candidate_path,
        max_root_ade_regression=0.05,
        max_root_fde_regression=0.05,
        max_root_jump_regression=0.005,
    )

    assert report["kind"] == "ldf_stream_checkpoint_comparison"
    assert report["baseline"]["tag"] == "full_prefix"
    assert report["candidate"]["tag"] == "window_local"
    assert report["candidate"]["metrics"]["stream_gt/root_ADE"] == 0.23
    assert report["candidate"]["metrics"]["stream_gt/jitter"] == 0.002
    assert report["candidate"]["metrics"]["stream_gt/yaw_error"] == 0.15
    assert report["candidate"]["metrics"]["stream_no_traj/root_ADE"] == 0.33
    assert report["candidate"]["metrics"]["control_gain/root_ADE_delta"] == 0.10
    assert report["baseline"]["metrics"]["stream_gt/chunk_boundary_root_jump"] == 0.01
    assert report["deltas"]["stream_gt/root_ADE"] == 0.03
    assert report["decision"]["passed"] is True
    assert report["decision"]["failures"] == []


def test_build_checkpoint_comparison_flags_regressions(tmp_path):
    baseline_path = tmp_path / "baseline_summary.json"
    candidate_path = tmp_path / "candidate_summary.json"
    _write_summary(baseline_path, tag="full_prefix", ade=0.20, fde=0.30, jump=0.01)
    _write_summary(candidate_path, tag="window_local", ade=0.31, fde=0.33, jump=0.011)

    report = build_checkpoint_comparison(
        baseline_path,
        candidate_path,
        max_root_ade_regression=0.05,
    )

    assert report["decision"]["passed"] is False
    assert report["decision"]["failures"] == [
        {
            "metric": "stream_gt/root_ADE",
            "delta": 0.11,
            "max_allowed_regression": 0.05,
        }
    ]


def test_build_checkpoint_comparison_flags_quality_guardrail_regressions(tmp_path):
    baseline_path = tmp_path / "baseline_summary.json"
    candidate_path = tmp_path / "candidate_summary.json"
    _write_summary(
        baseline_path,
        tag="full_prefix",
        ade=0.20,
        fde=0.30,
        jump=0.01,
        path_arc=0.10,
        jitter=0.01,
        yaw=0.10,
        skating=0.05,
    )
    _write_summary(
        candidate_path,
        tag="window_local",
        ade=0.20,
        fde=0.30,
        jump=0.01,
        path_arc=0.18,
        jitter=0.04,
        yaw=0.25,
        skating=0.09,
    )

    report = build_checkpoint_comparison(
        baseline_path,
        candidate_path,
        max_path_arc_regression=0.05,
        max_yaw_error_regression=0.10,
        max_jitter_regression=0.02,
        max_foot_skating_regression=0.02,
    )

    assert report["decision"]["passed"] is False
    assert report["decision"]["failures"] == [
        {
            "metric": "stream_gt/path_arc_ADE",
            "delta": 0.08,
            "max_allowed_regression": 0.05,
        },
        {
            "metric": "stream_gt/yaw_error",
            "delta": 0.15,
            "max_allowed_regression": 0.10,
        },
        {
            "metric": "stream_gt/jitter",
            "delta": 0.03,
            "max_allowed_regression": 0.02,
        },
        {
            "metric": "stream_gt/foot_skating",
            "delta": 0.04,
            "max_allowed_regression": 0.02,
        },
    ]


def test_ldf_report_cli_writes_comparison_json(tmp_path):
    baseline_path = tmp_path / "baseline_summary.json"
    candidate_path = tmp_path / "candidate_summary.json"
    out_path = tmp_path / "comparison.json"
    _write_summary(baseline_path, tag="full_prefix", ade=0.20, fde=0.30, jump=0.01)
    _write_summary(candidate_path, tag="window_local", ade=0.21, fde=0.31, jump=0.01)

    rc = main([
        "--baseline", str(baseline_path),
        "--candidate", str(candidate_path),
        "--out", str(out_path),
        "--max-root-ade-regression", "0.05",
        "--max-yaw-error-regression", "0.2",
    ])

    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["decision"]["passed"] is True
    assert payload["candidate"]["metrics"]["stream_gt/root_FDE"] == 0.31
    assert payload["thresholds"]["stream_gt/yaw_error"] == 0.2
