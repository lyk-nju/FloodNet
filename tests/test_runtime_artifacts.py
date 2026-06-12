from pathlib import Path
import csv
import json

import numpy as np
import pytest

from eval.runtime.artifacts import (
    RuntimeArtifactLayout,
    infer_ckpt_tag,
    read_experiment_root_plan,
    write_experiment_metrics,
    write_experiment_root_plan,
    write_manifest,
    write_records_csv,
    write_runtime_debug_report,
    write_root_diagnostic_artifacts,
    write_root_diagnostics_summary,
    write_source_json,
    write_summary_json,
)


def test_runtime_layout_builds_root_source_paths(tmp_path: Path):
    layout = RuntimeArtifactLayout(
        output_root=tmp_path,
        ckpt_tag="step_485000",
        run_id="20260611_220231",
    )

    assert layout.run_dir == tmp_path / "runtime" / "step_485000" / "20260611_220231"
    assert layout.root_source_dir("gtroot") == layout.run_dir / "gtroot"
    assert layout.experiment_dir("gtroot", "web_stream") == layout.run_dir / "gtroot" / "web_stream"
    assert (
        layout.experiment_dir("gtroot", "rotation", "rot_090")
        == layout.run_dir / "gtroot" / "rotation" / "rot_090"
    )
    assert (
        layout.experiment_dir("rootrefiner", "turn", "delay", "delay_020")
        == layout.run_dir / "rootrefiner" / "turn" / "delay" / "delay_020"
    )


def test_runtime_layout_builds_root_diagnostic_paths(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")

    assert (
        layout.root_diagnostic_dir("web_stream")
        == layout.run_dir / "root_diagnostics" / "web_stream" / "gtroot_vs_rootrefiner"
    )
    assert (
        layout.root_diagnostic_dir("rotation", "rot_090")
        == layout.run_dir / "root_diagnostics" / "rotation" / "rot_090" / "gtroot_vs_rootrefiner"
    )
    assert (
        layout.root_diagnostic_dir("turn", "delay", "delay_020")
        == layout.run_dir / "root_diagnostics" / "turn" / "delay" / "delay_020" / "gtroot_vs_rootrefiner"
    )


def test_runtime_layout_rejects_empty_path_parts(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")

    with pytest.raises(ValueError, match="empty"):
        layout.experiment_dir("gtroot", "")

    with pytest.raises(ValueError, match="empty"):
        layout.root_diagnostic_dir("rotation", "")


def test_infer_ckpt_tag_from_common_checkpoint_paths():
    assert infer_ckpt_tag("outputs/step_485000.ckpt") == "step_485000"
    assert infer_ckpt_tag("/tmp/refiner_step_127500.ckpt") == "refiner_step_127500"
    assert infer_ckpt_tag("/tmp/model.ckpt") == "model"


def test_runtime_artifact_writers_create_top_level_files(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")

    manifest_path = write_manifest(layout, {"ckpt": "outputs/step_485000.ckpt"})
    summary_path = write_summary_json(layout, {"mean_ADE": 0.1})

    assert manifest_path == layout.run_dir / "manifest.json"
    assert summary_path == layout.run_dir / "summary.json"
    assert json.loads(manifest_path.read_text())["ckpt"] == "outputs/step_485000.ckpt"
    assert json.loads(summary_path.read_text())["mean_ADE"] == 0.1


def test_write_source_json_uses_root_source_directory(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")

    path = write_source_json(layout, "rootrefiner", {"condition_source": "rootrefiner_7d"})

    assert path == layout.run_dir / "rootrefiner" / "source.json"
    assert json.loads(path.read_text())["condition_source"] == "rootrefiner_7d"


def test_write_records_csv_json_encodes_nested_cells(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")
    records = [
        {"case": "a", "ADE": 0.2, "nested": {"root": "gtroot"}},
        {"case": "b", "FDE": 0.4, "values": [1, 2]},
    ]

    path = write_records_csv(layout, records)

    assert path == layout.run_dir / "records.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["case"] == "a"
    assert rows[0]["nested"] == '{"root":"gtroot"}'
    assert rows[1]["values"] == "[1,2]"
    assert "ADE" in rows[0]
    assert "FDE" in rows[0]


def test_write_runtime_debug_report_writes_new_layout(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")

    paths = write_runtime_debug_report(
        layout,
        manifest={"ckpt": "outputs/step_485000.ckpt"},
        summary={"run_id": "20260611_220231"},
        records=[{"case_name": "web_stream", "ADE": 0.1}],
        source_payloads={
            "gtroot": {"condition_source": "gt_motion_7d"},
            "notraj": {"condition_source": "none"},
        },
    )

    assert paths["run_dir"] == layout.run_dir
    assert (layout.run_dir / "manifest.json").exists()
    assert (layout.run_dir / "summary.json").exists()
    assert (layout.run_dir / "records.csv").exists()
    assert json.loads((layout.run_dir / "gtroot" / "source.json").read_text())[
        "condition_source"
    ] == "gt_motion_7d"
    assert json.loads((layout.run_dir / "notraj" / "source.json").read_text())[
        "condition_source"
    ] == "none"


def test_write_experiment_metrics_uses_leaf_experiment_directory(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")

    path = write_experiment_metrics(
        layout,
        root_source="gtroot",
        family="rotation",
        parts=("rot_090",),
        metrics={"ADE": 0.25},
    )

    assert path == layout.run_dir / "gtroot" / "rotation" / "rot_090" / "metrics.json"
    assert json.loads(path.read_text())["ADE"] == 0.25


def test_write_and_read_experiment_root_plan_uses_leaf_directory(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")
    root_7d = np.arange(14, dtype=np.float32).reshape(2, 7)

    path = write_experiment_root_plan(
        layout,
        root_source="gtroot",
        family="rotation",
        parts=("rot_090",),
        root_7d_world=root_7d,
        num_tokens=2,
    )
    loaded = read_experiment_root_plan(
        layout,
        root_source="gtroot",
        family="rotation",
        parts=("rot_090",),
    )

    assert path == layout.run_dir / "gtroot" / "rotation" / "rot_090" / "root_plan_world.npz"
    assert loaded is not None
    loaded_root, loaded_tokens = loaded
    np.testing.assert_allclose(loaded_root, root_7d)
    assert loaded_tokens == 2


def test_write_root_diagnostic_artifacts_writes_metrics_and_npz(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")
    gt = np.zeros((2, 7), dtype=np.float32)
    pred = np.ones((2, 7), dtype=np.float32)

    out_dir = write_root_diagnostic_artifacts(
        layout,
        family="rotation",
        parts=("rot_090",),
        metrics={"xyz_ADE": 1.0},
        gt_root_7d=gt,
        pred_root_7d=pred,
        gt_num_tokens=2,
        pred_num_tokens=3,
    )

    expected = layout.root_diagnostic_dir("rotation", "rot_090")
    assert out_dir == expected
    assert json.loads((expected / "metrics.json").read_text())["xyz_ADE"] == 1.0
    arrays = np.load(expected / "root_plans.npz")
    np.testing.assert_allclose(arrays["gt_root_7d"], gt)
    np.testing.assert_allclose(arrays["pred_root_7d"], pred)
    assert int(arrays["gt_num_tokens"]) == 2
    assert int(arrays["pred_num_tokens"]) == 3


def test_write_root_diagnostics_summary_writes_json_and_csv(tmp_path: Path):
    layout = RuntimeArtifactLayout(tmp_path, "step_485000", "20260611_220231")

    paths = write_root_diagnostics_summary(
        layout,
        summary={"xyz_ADE_mean": 0.5},
        records=[{"family": "web_stream", "xyz_ADE": 0.5}],
    )

    assert paths["summary"] == layout.run_dir / "root_diagnostics" / "summary.json"
    assert paths["records"] == layout.run_dir / "root_diagnostics" / "summary.csv"
    assert json.loads(paths["summary"].read_text())["xyz_ADE_mean"] == 0.5
    with paths["records"].open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["family"] == "web_stream"
    assert rows[0]["xyz_ADE"] == "0.5"
