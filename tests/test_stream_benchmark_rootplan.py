from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from eval.runtime.benchmark import (
    _build_turn_metric_target,
    _csv_safe_record,
    _run_babel,
    _run_turn,
    _write_runtime_case_visuals,
    aggregate_runtime_records,
    build_eval_root_plan_from_points,
    build_rootplan_stream_step_payload,
    parse_condition_variants,
    resolve_traj_condition_source,
    write_runtime_report,
    write_stream_summary,
)
from utils.inference_glue import InferenceGlueState, InferenceGlueTimeline
from utils.runtime_rootplan import build_rootplan_stream_payload_from_buffer
from utils.token_frame import token_range_to_frame_slice, token_start_frame
from utils.traj_stream_buffer import TrajStreamBuffer


def _timeline(num_commits: int) -> InferenceGlueTimeline:
    timeline = InferenceGlueTimeline(
        InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )
    for commit_idx in range(1, num_commits + 1):
        timeline.append(
            InferenceGlueState(
                commit_idx=commit_idx,
                world_xz=torch.zeros(2),
                world_yaw=torch.tensor(0.0),
                source="test",
            )
        )
    return timeline


def _moving_z_timeline(num_commits: int) -> InferenceGlueTimeline:
    timeline = InferenceGlueTimeline(
        InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )
    for commit_idx in range(1, num_commits + 1):
        timeline.append(
            InferenceGlueState(
                commit_idx=commit_idx,
                world_xz=torch.tensor(
                    [0.0, float(token_start_frame(commit_idx))],
                    dtype=torch.float32,
                ),
                world_yaw=torch.tensor(0.0),
                source="test",
            )
        )
    return timeline


def test_stream_benchmark_builds_7d_rootplan_payload_for_model_step():
    timeline = _timeline(10)
    model = SimpleNamespace(
        commit_index=10,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(80, 3)
    points[:, 2] = torch.arange(80, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=4,
        traj_horizon_tokens=3,
    )

    start_token = 7
    num_tokens = 7
    frame_slice = token_range_to_frame_slice(start_token, num_tokens)
    assert payload["traj_start_token"] == start_token
    assert payload["traj_abs_start_token"] == start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == 7
    assert payload["body_anchor_abs_token"] == 7
    assert "traj_cond_7d_frame" in payload
    assert "traj_cond_frame_mask" in payload
    assert "traj" not in payload
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert float(payload["traj_cond_7d_frame"][0, 0, 2]) == float(
        token_start_frame(start_token)
    )
    assert payload["traj_cond_frame_mask"].all()


def test_rootplan_payload_covers_all_denoise_substep_windows_within_chunk():
    history_length = 30
    horizon_tokens = 20
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=40,
        chunk_size=5,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=128),
    )
    points = torch.zeros(400, 3)
    points[:, 2] = torch.arange(400, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=40,
    )

    start_token = 11  # earliest substep right token 41, history 30 -> left edge 11
    num_tokens = 54  # final right token 45 + horizon 20 - start 11
    frame_slice = token_range_to_frame_slice(start_token, num_tokens)
    assert payload["traj_start_token"] == start_token
    assert payload["traj_abs_start_token"] == start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == start_token
    assert payload["body_anchor_abs_token"] == start_token
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert float(payload["traj_cond_7d_frame"][0, 0, 2]) == float(
        token_start_frame(start_token)
    )


def test_rootplan_payload_canonicalizes_each_substep_with_its_body_anchor():
    history_length = 30
    horizon_tokens = 20
    timeline = _moving_z_timeline(100)
    model = SimpleNamespace(
        commit_index=40,
        chunk_size=5,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=128),
    )
    points = torch.zeros(400, 3)
    points[:, 2] = torch.arange(400, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=40,
    )

    subpayloads = payload["traj_substep_payloads"]
    assert [p["traj_start_token"] for p in subpayloads] == [11, 12, 13, 14, 15]
    assert [p["body_anchor_token"] for p in subpayloads] == [11, 12, 13, 14, 15]
    for subpayload in subpayloads:
        assert abs(float(subpayload["traj_cond_7d_frame"][0, 0, 2])) < 1e-6


def test_rootplan_payload_uses_absolute_commit_for_plan_and_local_commit_for_model():
    history_length = 30
    horizon_tokens = 3
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=30,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(300, 3)
    points[:, 2] = torch.arange(300, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=60,
    )

    body_anchor_token = 1
    body_anchor_abs_token = 31
    local_start_token = body_anchor_token
    absolute_start_token = body_anchor_abs_token
    num_tokens = history_length + horizon_tokens
    frame_slice = token_range_to_frame_slice(absolute_start_token, num_tokens)
    assert payload["traj_start_token"] == local_start_token
    assert payload["traj_abs_start_token"] == absolute_start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == body_anchor_token
    assert payload["body_anchor_abs_token"] == body_anchor_abs_token
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert float(payload["traj_cond_7d_frame"][0, 0, 2]) == float(
        token_start_frame(absolute_start_token)
    )


def test_rootplan_payload_frame_slice_uses_absolute_start_when_local_start_is_zero():
    history_length = 30
    horizon_tokens = 3
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=1,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(300, 3)
    points[:, 2] = torch.arange(300, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=60,
    )

    body_anchor_token = 0
    body_anchor_abs_token = 59
    local_start_token = body_anchor_token
    absolute_start_token = body_anchor_abs_token
    num_tokens = 2 + horizon_tokens
    frame_slice = token_range_to_frame_slice(absolute_start_token, num_tokens)
    assert payload["traj_start_token"] == local_start_token
    assert payload["traj_abs_start_token"] == absolute_start_token
    assert payload["traj_num_tokens"] == num_tokens
    assert payload["body_anchor_token"] == body_anchor_token
    assert payload["body_anchor_abs_token"] == body_anchor_abs_token
    assert payload["traj_cond_7d_frame"].shape == (
        1,
        frame_slice.stop - frame_slice.start,
        7,
    )
    assert float(payload["traj_cond_7d_frame"][0, 0, 2]) == float(
        token_start_frame(absolute_start_token)
    )


def test_rootplan_payload_masks_until_future_plan_anchor_after_model_roll():
    history_length = 30
    horizon_tokens = 10
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=30,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(300, 3)
    points[:, 2] = torch.arange(300, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(65),
        token_dt=0.20,
        frames_per_token=4,
        source="future_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=60,
    )

    local_start_token = 1
    absolute_start_token = 31
    prefix_tokens = 65 - absolute_start_token
    prefix_frames = (
        token_start_frame(absolute_start_token + prefix_tokens)
        - token_start_frame(absolute_start_token)
    )
    mask = payload["traj_cond_frame_mask"][0].bool()
    assert payload["traj_start_token"] == local_start_token
    assert payload["traj_abs_start_token"] == absolute_start_token
    assert prefix_frames > 0
    assert not mask[:prefix_frames].any()
    assert mask[prefix_frames:].all()


def test_rootplan_substep_payloads_partially_mask_until_replan_anchor_inside_window():
    history_length = 30
    horizon_tokens = 20
    timeline = _timeline(100)
    model = SimpleNamespace(
        commit_index=40,
        chunk_size=5,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=128),
    )
    points = torch.zeros(400, 3)
    points[:, 2] = torch.arange(400, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(40),
        token_dt=0.20,
        frames_per_token=4,
        source="mid_window_replan",
    )
    model._traj_buf.set_root_plan(root_plan)

    payload = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
        absolute_commit_index=40,
    )

    assert [p["traj_abs_start_token"] for p in payload["traj_substep_payloads"]] == [
        11,
        12,
        13,
        14,
        15,
    ]
    for subpayload in payload["traj_substep_payloads"]:
        absolute_start = int(subpayload["traj_abs_start_token"])
        prefix_frames = token_start_frame(40) - token_start_frame(absolute_start)
        mask = subpayload["traj_cond_frame_mask"][0].bool()
        assert 0 < prefix_frames < mask.numel()
        assert not mask[:prefix_frames].any()
        assert mask[prefix_frames:].all()


def test_eval_rootplan_payload_wrapper_matches_shared_runtime_helper():
    history_length = 4
    horizon_tokens = 2
    timeline = _timeline(10)
    model = SimpleNamespace(
        commit_index=10,
        chunk_size=1,
        _traj_buf=TrajStreamBuffer(batch_size=1, buf_len=96),
    )
    points = torch.zeros(80, 3)
    points[:, 2] = torch.arange(80, dtype=torch.float32)
    root_plan = build_eval_root_plan_from_points(
        points,
        anchor_state=timeline.at_commit(0),
        token_dt=0.20,
        frames_per_token=4,
        source="test_route",
    )
    model._traj_buf.set_root_plan(root_plan)

    wrapped = build_rootplan_stream_step_payload(
        model,
        timeline,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
    )
    direct = build_rootplan_stream_payload_from_buffer(
        model._traj_buf,
        timeline,
        local_commit_index=model.commit_index,
        absolute_commit_index=model.commit_index,
        chunk_size=model.chunk_size,
        history_length=history_length,
        traj_horizon_tokens=horizon_tokens,
    )

    assert wrapped["traj_start_token"] == direct["traj_start_token"]
    assert wrapped["traj_abs_start_token"] == direct["traj_abs_start_token"]
    assert torch.equal(wrapped["traj_cond_frame_mask"], direct["traj_cond_frame_mask"])
    assert torch.allclose(wrapped["traj_cond_7d_frame"], direct["traj_cond_7d_frame"])


def test_write_stream_summary_sanitizes_nan_for_strict_json(tmp_path):
    path = tmp_path / "summary.json"
    write_stream_summary(
        path,
        {
            "records": [
                {"ADE": float("nan"), "FDE": float("inf")},
                {"ADE": 1.0, "FDE": -float("inf")},
            ]
        },
    )

    text = path.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(text)
    assert payload["records"][0]["ADE"] is None
    assert payload["records"][0]["FDE"] is None
    assert payload["records"][1]["FDE"] is None


def test_write_stream_summary_sanitizes_numpy_and_torch_scalars(tmp_path):
    path = tmp_path / "summary.json"
    write_stream_summary(
        path,
        {
            "np_bad": np.float32(np.inf),
            "np_ok": np.int64(7),
            "torch_bad": torch.tensor(float("nan")),
            "torch_ok": torch.tensor([1.0, 2.0]),
        },
    )

    payload = json.loads(path.read_text())
    assert payload["np_bad"] is None
    assert payload["np_ok"] == 7
    assert payload["torch_bad"] is None
    assert payload["torch_ok"] == [1.0, 2.0]


def test_write_runtime_report_mirrors_standard_eval_layout(tmp_path):
    payload = {
        "run_id": "step_000500",
        "summary": {"ADE_mean": 0.2},
        "aggregate": {"ADE_mean": 0.2},
        "records": [
            {
                "suite": "humanml3d_control",
                "mode": "gt_traj",
                "sample_id": "000021",
                "ADE": 0.2,
                "FDE": 0.3,
            }
        ],
    }

    write_runtime_report(
        output_dir=tmp_path,
        run_id="step_000500",
        suite_tag="smoke",
        payload=payload,
        records=payload["records"],
    )

    legacy_root = tmp_path / "step_000500"
    metrics_dir = tmp_path / "Runtime" / "metrics" / "smoke" / "step_000500"

    assert (legacy_root / "summary.json").is_file()
    assert (legacy_root / "summary.csv").is_file()
    assert (metrics_dir / "summary.json").is_file()
    assert (metrics_dir / "records.csv").is_file()

    standard_payload = json.loads((metrics_dir / "summary.json").read_text())
    assert standard_payload["run_id"] == "step_000500"
    assert standard_payload["summary"]["ADE_mean"] == 0.2


def test_write_runtime_report_exposes_standard_media_dirs(tmp_path):
    payload = {
        "run_id": "step_000500",
        "summary": {"ADE_mean": 0.2},
        "aggregate": {"ADE_mean": 0.2},
        "records": [
            {
                "suite": "real",
                "mode": "real_predroot",
                "case_name": "real_predroot_rot90_001168",
                "condition_variant": "gt_7d_ldf",
                "ADE": 0.2,
                "FDE": 0.3,
            }
        ],
    }

    dirs = write_runtime_report(
        output_dir=tmp_path,
        run_id="step_000500",
        suite_tag="control",
        payload=payload,
        records=payload["records"],
        artifact_kinds=("metrics", "plot", "video"),
    )

    assert dirs["metrics"] == tmp_path / "Runtime/metrics/control/step_000500"
    assert dirs["plot"] == tmp_path / "Runtime/plot/control/step_000500"
    assert dirs["video"] == tmp_path / "Runtime/video/control/step_000500"
    assert (dirs["metrics"] / "records.csv").is_file()


def test_parse_condition_variants_expands_runtime_control_paths():
    variants = parse_condition_variants("gt_7d_ldf,rootrefiner_7d_ldf,no_traj_ldf")

    assert [variant.name for variant in variants] == [
        "gt_7d_ldf",
        "rootrefiner_7d_ldf",
        "no_traj_ldf",
    ]
    assert variants[0].condition_path == "rootplan_7d"
    assert variants[0].use_root_refiner is False
    assert variants[0].force_no_traj is False
    assert variants[1].condition_path == "rootplan_7d"
    assert variants[1].use_root_refiner is True
    assert variants[1].force_no_traj is False
    assert variants[2].condition_path == "rootplan_7d"
    assert variants[2].use_root_refiner is False
    assert variants[2].force_no_traj is True


def test_aggregate_runtime_records_groups_condition_variants():
    summary = aggregate_runtime_records(
        [
            {
                "suite": "real",
                "mode": "real_predroot",
                "condition_variant": "gt_7d_ldf",
                "ADE": 1.0,
                "FDE": 2.0,
            },
            {
                "suite": "real",
                "mode": "real_predroot",
                "condition_variant": "rootrefiner_7d_ldf",
                "ADE": 3.0,
                "FDE": 4.0,
            },
            {
                "suite": "real",
                "mode": "real_no_traj",
                "condition_variant": "no_traj_ldf",
                "ADE": 5.0,
                "FDE": 6.0,
            },
        ]
    )

    assert summary["by_condition_variant"]["gt_7d_ldf"]["ADE_mean"] == 1.0
    assert (
        summary["by_condition_variant"]["rootrefiner_7d_ldf"]["FDE_mean"] == 4.0
    )
    assert summary["by_suite_variant"]["real/no_traj_ldf"]["ADE_mean"] == 5.0


def test_resolve_traj_condition_source_makes_root_refiner_mode_explicit():
    assert resolve_traj_condition_source("rootplan_7d", None) == "route_7d"
    assert (
        resolve_traj_condition_source("rootplan_7d", object())
        == "root_refiner_7d"
    )
    assert resolve_traj_condition_source("legacy_xyz", None) == "legacy_xyz"
    assert (
        resolve_traj_condition_source("rootplan_7d", object(), no_traj=True)
        == "none"
    )


def test_csv_safe_record_serializes_nested_values_as_json():
    row = _csv_safe_record(
        {
            "suite": "turn",
            "rootplan_replan_commits": [0, 15],
            "rootplan_replan_sources": ["bench_old", "bench_new"],
            "bad": float("nan"),
        }
    )

    assert row["suite"] == "turn"
    assert json.loads(row["rootplan_replan_commits"]) == [0, 15]
    assert json.loads(row["rootplan_replan_sources"]) == ["bench_old", "bench_new"]
    assert row["bad"] is None


def test_aggregate_runtime_records_groups_checkpoint_selection_metrics():
    summary = aggregate_runtime_records(
        [
            {
                "suite": "humanml3d_control",
                "mode": "gt_traj",
                "ADE": 1.0,
                "FDE": 2.0,
                "path_arc": float("nan"),
            },
            {
                "suite": "humanml3d_control",
                "mode": "refiner_traj",
                "ADE": 3.0,
                "FDE": 4.0,
                "path_arc": 6.0,
            },
            {
                "suite": "route_edit",
                "mode": "replace_future",
                "ADE": 5.0,
                "FDE": 6.0,
                "path_arc": 8.0,
            },
        ]
    )

    assert summary["num_records"] == 3
    assert summary["ADE_mean"] == 3.0
    assert summary["ADE_count"] == 3
    assert summary["path_arc_mean"] == 7.0
    assert summary["path_arc_count"] == 2
    assert summary["by_suite"]["humanml3d_control"]["ADE_mean"] == 2.0
    assert summary["by_mode"]["gt_traj"]["FDE_mean"] == 2.0
    assert summary["by_suite_mode"]["route_edit/replace_future"]["ADE_mean"] == 5.0


def test_runtime_case_visuals_write_static_path_and_yaw_plots(tmp_path):
    frames = 8
    pred_root = np.zeros((frames, 3), dtype=np.float32)
    target_root = np.zeros((frames, 3), dtype=np.float32)
    pred_root[:, 2] = np.arange(frames, dtype=np.float32) * 0.12
    target_root[:, 2] = np.arange(frames, dtype=np.float32) * 0.10
    motion = np.zeros((frames, 263), dtype=np.float32)
    motion[:, 2] = pred_root[:, 2]

    _write_runtime_case_visuals(
        output_dir=tmp_path,
        case_name="case_a",
        pred_root=pred_root,
        target_root=target_root,
        motion_263=motion,
        split_tok=1,
    )

    assert (tmp_path / "case_a_plot_world_xz.png").is_file()
    assert (tmp_path / "case_a_plot_yaw.png").is_file()


class _FakeStreamModel:
    chunk_size = 1

    def init_generated(self, history_length, batch_size=1, num_denoise_steps=None):
        self.commit_index = 0
        self.generated = torch.zeros(1, 1, 1)
        self._traj_buf = TrajStreamBuffer(batch_size=1, buf_len=96)

    def stream_generate_step(self, step_input, first_chunk=True):
        self.commit_index += 1
        return {"generated": torch.zeros(1, 1, 263)}


class _FakeVae:
    def clear_cache(self):
        pass

    def stream_decode(self, generated, first_chunk=True):
        return torch.zeros(1, 1, 263)


class _RecordingRootRefinerRuntime:
    def __init__(self):
        self.calls = []

    def build_root_plan(
        self,
        *,
        text,
        plan,
        anchor_state,
        token_dt,
        history_motion_world_5d=None,
    ):
        self.calls.append(
            {
                "text": text,
                "start_commit_index": int(plan.start_commit_index),
                "history": history_motion_world_5d,
                "first_point": plan.points_xyz[0].copy(),
                "num_points": len(plan.points_xyz),
            }
        )
        points = torch.zeros(80, 3)
        points[:, 2] = torch.arange(80, dtype=torch.float32)
        return build_eval_root_plan_from_points(
            points,
            anchor_state=anchor_state,
            token_dt=token_dt,
            frames_per_token=4,
            source="fake_refiner",
        )


def test_babel_root_refiner_rebuilds_plan_when_text_segment_changes():
    feature = torch.zeros(9, 263)
    sample = {
        "token_length": 3,
        "feature_length": 9,
        "feature": feature,
        "traj": torch.zeros(9, 3),
        "text": ["walk", "run"],
        "token_text_end": [1, 3],
    }
    runtime = _RecordingRootRefinerRuntime()

    _run_babel(
        _FakeStreamModel(),
        _FakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=1,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="babel_timestamped",
        condition_path="rootplan_7d",
        root_refiner_runtime=runtime,
    )

    assert [(c["text"], c["start_commit_index"]) for c in runtime.calls] == [
        ("walk", 0),
        ("run", 1),
    ]


def test_babel_root_refiner_replan_receives_generated_root_history():
    feature = torch.zeros(9, 263)
    sample = {
        "token_length": 3,
        "feature_length": 9,
        "feature": feature,
        "traj": torch.zeros(9, 3),
        "text": ["walk", "run"],
        "token_text_end": [1, 3],
    }
    runtime = _RecordingRootRefinerRuntime()

    model = _FakeStreamModel()
    replan_events = []
    _run_babel(
        model,
        _FakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=1,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="babel_timestamped",
        condition_path="rootplan_7d",
        root_refiner_runtime=runtime,
        replan_events=replan_events,
    )

    assert runtime.calls[0]["history"] is None
    assert runtime.calls[1]["history"] is not None
    assert runtime.calls[1]["history"].shape[-1] == 5
    assert len(replan_events) == 2
    assert replan_events[1]["text"] == "run"
    assert not hasattr(model, "_stream_eval_replan_events")


def test_turn_rootplan_matches_web_delayed_replace_semantics():
    feature = torch.zeros(81, 263)
    sample = {
        "token_length": 20,
        "feature_length": 81,
        "feature": feature,
        "traj": torch.zeros(81, 3),
        "text": "walk forward",
    }
    model = _FakeStreamModel()
    replan_events = []

    _run_turn(
        model,
        _FakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=2,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="turn_mid_update",
        angle=45.0,
        delay_tokens=0,
        blend_tokens=2,
        condition_path="rootplan_7d",
        root_refiner_runtime=None,
        replan_events=replan_events,
    )

    sources = [event["source"] for event in replan_events]
    assert sources == ["bench_old", "bench_new"]
    assert "bench_blend" not in sources
    assert not hasattr(model, "_stream_eval_replan_events")


def test_turn_force_no_traj_skips_rootplan_setup():
    feature = torch.zeros(81, 263)
    sample = {
        "token_length": 20,
        "feature_length": 81,
        "feature": feature,
        "traj": torch.zeros(81, 3),
        "text": "walk forward",
    }
    model = _FakeStreamModel()
    replan_events = []

    _run_turn(
        model,
        _FakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=2,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="turn_mid_update",
        angle=45.0,
        delay_tokens=0,
        blend_tokens=2,
        condition_path="rootplan_7d",
        root_refiner_runtime=None,
        replan_events=replan_events,
        force_no_traj=True,
    )

    assert replan_events == []
    assert not model._traj_buf.has_active_plan()


def test_turn_root_refiner_does_not_rebuild_every_blend_token():
    feature = torch.zeros(81, 263)
    traj = torch.zeros(81, 3)
    traj[:, 2] = torch.arange(81, dtype=torch.float32) * 0.1
    sample = {
        "token_length": 20,
        "feature_length": 81,
        "feature": feature,
        "traj": traj,
        "text": "walk forward",
    }
    runtime = _RecordingRootRefinerRuntime()
    replan_events = []

    _run_turn(
        _FakeStreamModel(),
        _FakeVae(),
        sample,
        torch.device("cpu"),
        hl=4,
        nds=1,
        hz=2,
        tdt=0.20,
        wpdt=0.05,
        fps=20.0,
        mode="turn_mid_update",
        angle=45.0,
        delay_tokens=0,
        blend_tokens=3,
        condition_path="rootplan_7d",
        root_refiner_runtime=runtime,
        replan_events=replan_events,
    )

    assert [(c["text"], c["start_commit_index"]) for c in runtime.calls] == [
        ("walk forward", 0),
        ("walk forward", 15),
    ]
    assert runtime.calls[1]["num_points"] < runtime.calls[0]["num_points"]
    assert [event["source"] for event in replan_events] == ["bench_old", "bench_new"]


def test_turn_metric_target_follows_old_path_until_runtime_replacement():
    plan_t = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    old_pts = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    new_pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]],
        dtype=np.float32,
    )

    target_t, target_pts = _build_turn_metric_target(
        old_times=plan_t,
        old_points_xyz=old_pts,
        new_times=plan_t,
        new_points_xyz=new_pts,
        target_frames=21,
        motion_fps=10.0,
        edit_commit=2,
        delay_tokens=3,
        blend_tokens=2,
        token_dt=0.1,
    )

    activation_frame = int(round((2 + 3 + 2) * 0.1 * 10.0))
    assert np.allclose(target_pts[activation_frame - 1], [0.0, 0.0, 0.6])
    assert target_pts[activation_frame, 0] > 0.0
    assert np.allclose(target_t, np.arange(21, dtype=np.float32) / 10.0)


def test_turn_metric_target_reanchors_new_route_to_effective_anchor():
    plan_t = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    old_pts = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    new_pts = np.array(
        [[10.0, 0.0, 0.0], [10.0, 0.0, 1.0], [10.0, 0.0, 2.0]],
        dtype=np.float32,
    )

    _, target_pts = _build_turn_metric_target(
        old_times=plan_t,
        old_points_xyz=old_pts,
        new_times=plan_t,
        new_points_xyz=new_pts,
        target_frames=21,
        motion_fps=10.0,
        edit_commit=2,
        delay_tokens=3,
        blend_tokens=2,
        token_dt=0.1,
        new_anchor_xz=np.array([0.0, 0.5], dtype=np.float32),
    )

    activation_frame = int(round((2 + 3 + 2) * 0.1 * 10.0))
    assert np.allclose(target_pts[activation_frame - 1], [0.0, 0.0, 0.6])
    assert np.allclose(target_pts[activation_frame], [0.0, 0.0, 0.7])
