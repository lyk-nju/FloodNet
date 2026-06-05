import csv
import json
from pathlib import Path

import pytest


def test_common_eval_contracts_are_serializable(tmp_path):
    from eval.common import EvalCase, EvalEvent, EvalPathSpec, EvalRunResult

    path = EvalPathSpec(
        name="stream_gt",
        description="stream_generate_step + GT-7D",
        enabled_by_default=True,
        tags=("ldf", "stream"),
    )
    event = EvalEvent(
        event_type="route_replace_future",
        submit_commit=12,
        effective_commit=16,
        route_version=2,
        payload={"points": [[0.0, 0.0], [1.0, 0.0]]},
    )
    case = EvalCase(
        suite_name="humanml3d_control",
        case_id="000001",
        dataset="humanml3d",
        sample_id="000001",
        path_names=("gt_traj", "refiner_traj"),
        events=(event,),
        metadata={"text": "walk forward"},
    )
    result = EvalRunResult(
        suite_name=case.suite_name,
        case_id=case.case_id,
        metrics={"gt_traj/root_ADE": 0.1},
        artifacts={"plot": tmp_path / "plot_world_xz.png"},
        events=(event,),
    )

    assert path.to_metadata() == {
        "name": "stream_gt",
        "description": "stream_generate_step + GT-7D",
        "enabled_by_default": True,
        "tags": ["ldf", "stream"],
        "metadata": {},
    }
    assert event.to_metadata()["effective_commit"] == 16
    assert case.to_metadata()["path_names"] == ["gt_traj", "refiner_traj"]
    assert result.to_json_dict()["artifacts"] == {"plot": str(tmp_path / "plot_world_xz.png")}


def test_common_artifact_writer_outputs_strict_files(tmp_path):
    from eval.common.artifacts import (
        make_sample_artifact_dir,
        write_eval_csv,
        write_eval_json,
    )

    sample_dir = make_sample_artifact_dir(tmp_path, "ldf", "000001")
    json_path = write_eval_json(sample_dir / "metrics.json", {"value": float("nan")})
    csv_path = write_eval_csv(
        sample_dir / "metrics.csv",
        [
            {"case": "a", "metrics": {"ADE": 0.1}},
            {"case": "b", "metrics": {"ADE": 0.2}, "extra": [1, 2]},
        ],
    )

    assert json_path.exists()
    assert json.loads(json_path.read_text()) == {"value": None}
    csv_text = csv_path.read_text()
    assert "case" in csv_text
    rows = list(csv.DictReader(csv_text.splitlines()))
    assert rows[0]["metrics"] == '{"ADE":0.1}'
    assert rows[1]["extra"] == "[1,2]"


def test_root_refiner_eval_contract_names_are_design_aligned():
    from eval.root_refiner.adapters import (
        DURATION_GROUNDTRUTH,
        DURATION_PRED,
        ROOT_REFINER_ARTIFACT_NAMES,
        ROUTE_MODES,
        build_root_refiner_sample_metadata,
        normalize_duration_mode,
    )

    assert DURATION_PRED == "pred_duration"
    assert DURATION_GROUNDTRUTH == "groundtruth_duration"
    assert "dense_gt" in ROUTE_MODES
    assert "offset_sparse" in ROUTE_MODES
    assert ROOT_REFINER_ARTIFACT_NAMES[DURATION_GROUNDTRUTH]["pred_root_7d"].endswith(
        "groundtruth_duration.npy"
    )
    assert normalize_duration_mode("groundtruth_duration") == "groundtruth_duration"
    with pytest.raises(ValueError, match="unknown duration_mode"):
        normalize_duration_mode("oracle_duration")

    metadata = build_root_refiner_sample_metadata(
        sample_id="000001",
        route_mode="offset_sparse",
        duration_mode="groundtruth_duration",
        offset_frame=20,
        offset_token=5,
        anchor_world_xz=(1.0, 2.0),
        anchor_world_yaw=0.3,
        gt_slice_start=20,
        gt_slice_end=80,
    )
    assert metadata["route_mode"] == "offset_sparse"
    assert metadata["duration_mode"] == "groundtruth_duration"
    assert metadata["gt_slice"] == {"start": 20, "end": 80}


def test_ldf_default_paths_and_text_buckets_are_design_aligned():
    from eval.ldf.cases import (
        DEFAULT_LDF_EVAL_PATHS,
        HUMANML3D_ONLY_DATASET,
        OPTIONAL_LDF_EVAL_PATHS,
        TEXT_BUCKET_MAX_SAMPLES,
        build_ldf_path_specs,
        classify_text_bucket,
    )

    assert HUMANML3D_ONLY_DATASET == "humanml3d"
    assert DEFAULT_LDF_EVAL_PATHS == ("stream_gt", "offline_gt", "stream_no_traj")
    assert OPTIONAL_LDF_EVAL_PATHS == ("offline_no_traj",)
    assert [spec.name for spec in build_ldf_path_specs()] == list(DEFAULT_LDF_EVAL_PATHS)
    assert [spec.name for spec in build_ldf_path_specs(include_offline_no_traj=True)] == [
        "stream_gt",
        "offline_gt",
        "stream_no_traj",
        "offline_no_traj",
    ]
    assert classify_text_bucket("a person is running quickly") == "run"
    assert classify_text_bucket("a person slowly walks") == "walk"
    assert classify_text_bucket("a person waves their arms") == "other"
    assert TEXT_BUCKET_MAX_SAMPLES == 5


def test_runtime_event_policy_and_suite_contracts_are_design_aligned():
    from eval.runtime.events import (
        RootRefinerResponse,
        RuntimeEvent,
        should_accept_root_refiner_response,
    )
    from eval.runtime.state_machine import RuntimeCase, RuntimeRunConfig
    from eval.runtime.suites import RUNTIME_SUITE_NAMES
    from eval.runtime.suites.humanml3d_control import build_humanml3d_control_case
    from eval.runtime.suites.route_edit import build_route_replace_case
    from eval.runtime.suites.text_update import build_text_update_case

    assert RUNTIME_SUITE_NAMES == (
        "humanml3d_control",
        "route_edit",
        "text_update",
        "babel_long_session",
    )

    event = RuntimeEvent(
        event_type="route_replace_future",
        submit_commit=10,
        effective_commit=14,
        route_version=2,
    )
    response = RootRefinerResponse(
        request_id="req-1",
        request_route_version=2,
        request_text_version=1,
        request_submit_commit=10,
        request_base_commit=14,
        response_commit=15,
        plan={"valid_frames": 40},
    )
    decision = should_accept_root_refiner_response(
        response,
        active_route_version=2,
        active_text_version=1,
        min_response_commit=14,
    )
    assert decision == {"accepted": True, "reject_reason": None}

    stale_decision = should_accept_root_refiner_response(
        response,
        active_route_version=3,
        active_text_version=1,
        min_response_commit=14,
    )
    assert stale_decision == {"accepted": False, "reject_reason": "stale_route"}

    cfg = RuntimeRunConfig(seed=7, context_tokens=30, horizon_tokens=20)
    assert cfg.to_metadata()["horizon_tokens"] == 20
    case = RuntimeCase(
        suite_name="route_edit",
        sample_id="synthetic",
        initial_text="walk",
        event_schedule=(event,),
    )
    assert case.to_metadata()["event_schedule"][0]["effective_commit"] == 14
    assert build_humanml3d_control_case(
        sample_id="000001",
        text="walk",
        route_points=[(0.0, 0.0), (1.0, 0.0)],
        duration_tokens=20,
    ).path_names == ("gt_traj", "refiner_traj")
    assert build_route_replace_case(
        case_id="replace",
        initial_text="walk",
        initial_route=[(0.0, 0.0)],
        updated_route=[(1.0, 0.0)],
        submit_commit=5,
        delay_tokens=3,
    ).event_schedule[0].effective_commit == 8
    assert build_text_update_case(
        case_id="walk_to_run",
        route_points=[(0.0, 0.0)],
        text_segments=[
            {"text": "walk", "start_commit": 0, "effective_commit": 0},
            {"text": "run", "start_commit": 10, "effective_commit": 12},
        ],
    ).event_schedule[1].text_version == 2


def test_root_refiner_benchmark_accepts_groundtruth_duration_mode():
    from eval.root_refiner.benchmark import (
        resolve_duration_mode,
        resolve_suite_config,
        resolve_suite_duration_mode,
    )

    assert resolve_duration_mode(duration_mode=None, oracle_duration=False) == "pred_duration"
    assert (
        resolve_duration_mode(duration_mode="groundtruth_duration", oracle_duration=False)
        == "groundtruth_duration"
    )
    assert resolve_duration_mode(duration_mode=None, oracle_duration=True) == "groundtruth_duration"
    assert resolve_suite_config("standard_groundtruth").duration_mode == "groundtruth_duration"
    assert resolve_suite_config("standard_oracle").duration_mode == "groundtruth_duration"
    assert resolve_suite_config("standard_oracle").oracle_duration is True
    assert (
        resolve_suite_duration_mode(
            resolve_suite_config("standard"),
            duration_mode=None,
            oracle_duration=True,
        )
        == "groundtruth_duration"
    )
    with pytest.raises(ValueError, match="conflicting duration options"):
        resolve_duration_mode(duration_mode="pred_duration", oracle_duration=True)


def test_runtime_stale_policy_rejects_old_base_commit():
    from eval.runtime.events import RootRefinerResponse, should_accept_root_refiner_response

    response = RootRefinerResponse(
        request_id="req-old-base",
        request_route_version=2,
        request_text_version=1,
        request_submit_commit=5,
        request_base_commit=5,
        response_commit=30,
        plan={"valid_frames": 40},
    )

    assert should_accept_root_refiner_response(
        response,
        active_route_version=2,
        active_text_version=1,
        min_response_commit=20,
        min_base_commit=20,
    ) == {"accepted": False, "reject_reason": "stale_commit"}


def test_runtime_stale_policy_default_does_not_check_base_commit():
    from eval.runtime.events import RootRefinerResponse, should_accept_root_refiner_response

    response = RootRefinerResponse(
        request_id="req-old-base",
        request_route_version=2,
        request_text_version=1,
        request_submit_commit=5,
        request_base_commit=5,
        response_commit=30,
        plan={"valid_frames": 40},
    )

    assert should_accept_root_refiner_response(
        response,
        active_route_version=2,
        active_text_version=1,
        min_response_commit=20,
    ) == {"accepted": True, "reject_reason": None}


def test_runtime_route_edit_records_delay_tokens():
    from eval.runtime.suites.route_edit import build_route_replace_case

    case = build_route_replace_case(
        case_id="replace",
        initial_text="walk",
        initial_route=[(0.0, 0.0)],
        updated_route=[(1.0, 0.0)],
        submit_commit=5,
        delay_tokens=3,
    )

    assert case.event_schedule[0].metadata["delay_tokens"] == 3


def test_babel_runtime_case_validates_segment_metadata_lengths():
    from eval.runtime.suites.babel_long_session import (
        BabelAssemblyMetadata,
        build_babel_runtime_case,
    )

    with pytest.raises(ValueError, match="bad.*segment metadata length mismatch"):
        build_babel_runtime_case(
            assembly=BabelAssemblyMetadata(
                babel_sequence_id="bad",
                segment_ids=("a", "b"),
                segment_texts=("walk", "run"),
                segment_start_commits=(0,),
                segment_end_commits=(10, 20),
            ),
            route_schedule=[],
        )
