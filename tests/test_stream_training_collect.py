from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.stream_training_smoke import SmokeRunConfig, build_validation_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _touch(path: Path, *, mtime: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ckpt")
    os.utime(path, (mtime, mtime))


def _validation_manifest(tmp_path: Path) -> tuple[Path, dict]:
    cfg = SmokeRunConfig(
        config="configs/ldf.yaml",
        python="python",
        resume_ckpt=str(tmp_path / "baseline.ckpt"),
        vae_ckpt=str(tmp_path / "vae.ckpt"),
        raw_data_root=str(tmp_path / "raw_data"),
        output_dir=str(tmp_path / "runs"),
        exp_name="stream_validation",
        max_steps=10,
        val_split=str(tmp_path / "val.txt"),
    )
    payload = build_validation_manifest(cfg)
    manifest = tmp_path / "validation_plan.json"
    _write_json(manifest, payload)
    return manifest, payload


def _write_passed_outputs(payload: dict, *, failed_stage: str | None = None) -> None:
    _write_json(
        Path(payload["stream_eval"]["baseline"]["summary"]),
        {"kind": "ldf_stream_eval", "summary": {"stream_gt/root_ADE": 0.1}},
    )
    for idx, stage in enumerate(payload["stages"]):
        ckpt = Path(stage["config"]["output_dir"]) / "checkpoints" / "last.ckpt"
        _touch(ckpt, mtime=10 + idx)
        _write_json(
            Path(stage["post_training_eval"]["candidate_eval"]["summary"]),
            {"kind": "ldf_stream_eval", "summary": {"stream_gt/root_ADE": 0.1}},
        )
        passed = stage["stage"] != failed_stage
        failures = [] if passed else [{"metric": "stream_gt/root_ADE", "delta": 0.2}]
        _write_json(
            Path(stage["post_training_eval"]["comparison"]["summary"]),
            {
                "kind": "ldf_stream_checkpoint_comparison",
                "decision": {"passed": passed, "failures": failures},
            },
        )


def test_collect_validation_status_reports_passed_stages_and_newest_ckpt(tmp_path):
    from scripts.stream_training_collect import collect_validation_status

    manifest, payload = _validation_manifest(tmp_path)
    _write_passed_outputs(payload)
    first_stage = payload["stages"][0]
    old_ckpt = Path(first_stage["config"]["output_dir"]) / "old.ckpt"
    _touch(old_ckpt, mtime=1)

    status = collect_validation_status(manifest)

    assert status["kind"] == "ldf_stream_training_validation_status"
    assert status["overall"]["status"] == "passed"
    assert status["baseline"]["status"] == "present"
    assert [stage["status"] for stage in status["stages"]] == [
        "passed" for _ in payload["stages"]
    ]
    assert status["stages"][0]["candidate_ckpt"].endswith(
        "01_smoke_full_prefix/checkpoints/last.ckpt"
    )
    candidate_argv = status["stages"][0]["commands"]["candidate_eval"]["argv"]
    assert "{candidate_ckpt}" not in candidate_argv
    assert status["stages"][0]["candidate_ckpt"] in candidate_argv
    assert status["stages"][0]["commands"]["comparison"]["argv"][-2:] == [
        "--max-root-ade-regression",
        "0.05",
    ]


def test_collect_validation_status_marks_missing_outputs_pending(tmp_path):
    from scripts.stream_training_collect import collect_validation_status

    manifest, _payload = _validation_manifest(tmp_path)

    status = collect_validation_status(manifest)

    assert status["overall"]["status"] == "pending"
    assert status["baseline"]["status"] == "missing"
    assert status["stages"][0]["status"] == "pending"
    assert "candidate checkpoint not found" in status["stages"][0]["missing"]
    assert "candidate summary missing" in status["stages"][0]["missing"]
    assert "comparison missing" in status["stages"][0]["missing"]


def test_collect_validation_status_marks_failed_comparison_failed(tmp_path):
    from scripts.stream_training_collect import collect_validation_status

    manifest, payload = _validation_manifest(tmp_path)
    _write_passed_outputs(payload, failed_stage="03_overfit_full_prefix")

    status = collect_validation_status(manifest)

    assert status["overall"]["status"] == "failed"
    failed = [
        stage for stage in status["stages"] if stage["stage"] == "03_overfit_full_prefix"
    ][0]
    assert failed["status"] == "failed"
    assert failed["failures"] == [{"metric": "stream_gt/root_ADE", "delta": 0.2}]


def test_stream_training_collect_cli_writes_status_and_uses_exit_codes(tmp_path):
    from scripts.stream_training_collect import main

    manifest, payload = _validation_manifest(tmp_path)
    _write_passed_outputs(payload)
    out = tmp_path / "status.json"

    rc = main(["--manifest", str(manifest), "--out", str(out)])

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["overall"]["status"] == "passed"

    pending_out = tmp_path / "pending.json"
    pending_manifest, _ = _validation_manifest(tmp_path / "pending")
    pending_rc = main(["--manifest", str(pending_manifest), "--out", str(pending_out)])
    assert pending_rc == 2
    assert json.loads(pending_out.read_text())["overall"]["status"] == "pending"
