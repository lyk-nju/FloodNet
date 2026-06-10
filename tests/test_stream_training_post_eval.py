from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from scripts.stream_training_smoke import SmokeRunConfig, build_validation_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _touch(path: Path, *, mtime: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ckpt")
    os.utime(path, (mtime, mtime))


def _write_summary_cmd(summary_path: str, *, extra_arg: str | None = None) -> list[str]:
    code = (
        "import json,sys; "
        "from pathlib import Path; "
        "p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); "
        "payload={'kind':'ldf_stream_eval','summary':{'stream_gt/root_ADE':0.1}}; "
        "payload['extra_arg']=sys.argv[2] if len(sys.argv)>2 else ''; "
        "p.write_text(json.dumps(payload))"
    )
    cmd = [sys.executable, "-c", code, summary_path]
    if extra_arg is not None:
        cmd.append(extra_arg)
    return cmd


def _write_comparison_cmd(path: str, *, passed: bool = True) -> list[str]:
    code = (
        "import json,sys; "
        "from pathlib import Path; "
        "p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); "
        "passed=sys.argv[2]=='true'; "
        "failures=[] if passed else [{'metric':'stream_gt/root_ADE','delta':0.2}]; "
        "p.write_text(json.dumps({'kind':'ldf_stream_checkpoint_comparison',"
        "'decision':{'passed':passed,'failures':failures}})); "
        "raise SystemExit(0 if passed else 1)"
    )
    return [sys.executable, "-c", code, path, "true" if passed else "false"]


def _manifest_with_fake_commands(tmp_path: Path, *, comparison_passed: bool = True):
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
    baseline_summary = payload["stream_eval"]["baseline"]["summary"]
    payload["stream_eval"]["baseline"]["argv"] = _write_summary_cmd(baseline_summary)
    payload["stream_eval"]["baseline"]["command"] = "fake baseline"
    for stage in payload["stages"]:
        post_eval = stage["post_training_eval"]
        candidate_summary = post_eval["candidate_eval"]["summary"]
        comparison_path = post_eval["comparison"]["summary"]
        post_eval["candidate_eval"]["argv"] = _write_summary_cmd(
            candidate_summary,
            extra_arg="{candidate_ckpt}",
        )
        post_eval["candidate_eval"]["command"] = "fake candidate"
        post_eval["comparison"]["argv"] = _write_comparison_cmd(
            comparison_path,
            passed=comparison_passed,
        )
        post_eval["comparison"]["command"] = "fake report"
    manifest = tmp_path / "validation_plan.json"
    _write_json(manifest, payload)
    return manifest, payload


def test_run_post_training_eval_executes_missing_eval_and_report_commands(tmp_path):
    from scripts.stream_training_post_eval import run_post_training_eval

    manifest, payload = _manifest_with_fake_commands(tmp_path)
    first_stage = payload["stages"][0]
    old_ckpt = Path(first_stage["config"]["output_dir"]) / "old.ckpt"
    new_ckpt = Path(first_stage["config"]["output_dir"]) / "checkpoints" / "last.ckpt"
    _touch(old_ckpt, mtime=1)
    _touch(new_ckpt, mtime=10)
    for stage in payload["stages"][1:]:
        _touch(Path(stage["config"]["output_dir"]) / "checkpoints" / "last.ckpt", mtime=10)

    result = run_post_training_eval(manifest)

    assert result["kind"] == "ldf_stream_training_post_eval_run"
    assert result["overall"]["status"] == "passed"
    assert {cmd["status"] for cmd in result["commands"]} == {"ran"}
    candidate_payload = json.loads(
        Path(first_stage["post_training_eval"]["candidate_eval"]["summary"]).read_text()
    )
    assert candidate_payload["extra_arg"] == str(new_ckpt)


def test_run_post_training_eval_skips_existing_outputs(tmp_path):
    from scripts.stream_training_post_eval import run_post_training_eval

    manifest, payload = _manifest_with_fake_commands(tmp_path)
    _write_json(
        Path(payload["stream_eval"]["baseline"]["summary"]),
        {"kind": "ldf_stream_eval", "summary": {"stream_gt/root_ADE": 0.1}},
    )
    for stage in payload["stages"]:
        _touch(Path(stage["config"]["output_dir"]) / "checkpoints" / "last.ckpt")
        _write_json(
            Path(stage["post_training_eval"]["candidate_eval"]["summary"]),
            {"kind": "ldf_stream_eval", "summary": {"stream_gt/root_ADE": 0.1}},
        )
        _write_json(
            Path(stage["post_training_eval"]["comparison"]["summary"]),
            {"kind": "ldf_stream_checkpoint_comparison", "decision": {"passed": True, "failures": []}},
        )

    result = run_post_training_eval(manifest)

    assert result["overall"]["status"] == "passed"
    assert {cmd["status"] for cmd in result["commands"]} == {"skipped"}


def test_run_post_training_eval_reports_pending_stage_without_checkpoint(tmp_path):
    from scripts.stream_training_post_eval import run_post_training_eval

    manifest, _payload = _manifest_with_fake_commands(tmp_path)

    result = run_post_training_eval(manifest)

    assert result["overall"]["status"] == "pending"
    assert "candidate checkpoint not found" in result["commands"][1]["missing"]


def test_run_post_training_eval_cli_writes_status_and_uses_exit_codes(tmp_path):
    from scripts.stream_training_post_eval import main

    manifest, payload = _manifest_with_fake_commands(tmp_path, comparison_passed=False)
    for stage in payload["stages"]:
        _touch(Path(stage["config"]["output_dir"]) / "checkpoints" / "last.ckpt")
    out = tmp_path / "post_eval_status.json"

    rc = main(["--manifest", str(manifest), "--out", str(out)])

    assert rc == 1
    status = json.loads(out.read_text())
    assert status["overall"]["status"] == "failed"
