"""Run missing post-training LDF stream eval/report commands from a manifest.

This script executes only the validation closure recorded by
`scripts/stream_training_smoke.py --validation-plan --manifest ...`: baseline
stream eval, per-stage candidate stream eval, and per-stage comparison reports.
It does not launch training. After command execution it writes the same status
shape as `stream_training_collect.py`, plus a command execution log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from FloodNet.eval.common.json import write_json_strict
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from eval.common.json import write_json_strict

from scripts.stream_training_collect import (
    _load_json,
    _materialized_command,
    _newest_match,
    collect_validation_status,
)


def _path_exists(path: str) -> bool:
    return bool(path) and Path(path).is_file()


def _run_argv(argv: list[str], *, dry_run: bool) -> int | None:
    if dry_run:
        return None
    return int(subprocess.call(argv))


def _command_record(
    *,
    name: str,
    kind: str,
    summary: str,
    command: dict[str, Any],
    status: str,
    returncode: int | None = None,
    missing: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "summary": summary,
        "status": status,
        "returncode": returncode,
        "argv": command.get("argv", []),
        "command": command.get("command", ""),
        "missing": missing or [],
    }


def _execute_entry(
    *,
    name: str,
    kind: str,
    entry: dict[str, Any],
    ckpt: str | None = None,
    dry_run: bool = False,
    rerun: bool = False,
    comparison_command: bool = False,
) -> dict[str, Any]:
    summary = str(entry.get("summary") or "")
    command = _materialized_command(entry, ckpt)
    if not rerun and _path_exists(summary):
        return _command_record(
            name=name,
            kind=kind,
            summary=summary,
            command=command,
            status="skipped",
        )
    if not command["ready"]:
        return _command_record(
            name=name,
            kind=kind,
            summary=summary,
            command=command,
            status="pending",
            missing=["command is not ready"],
        )
    returncode = _run_argv(command["argv"], dry_run=dry_run)
    if dry_run:
        status = "dry_run"
    elif returncode == 0:
        status = "ran"
    elif comparison_command and _path_exists(summary):
        # eval.ldf.report intentionally returns 1 when guardrails fail. If it
        # wrote a comparison JSON, let the collector classify the checkpoint.
        status = "ran"
    else:
        status = "failed"
    return _command_record(
        name=name,
        kind=kind,
        summary=summary,
        command=command,
        status=status,
        returncode=returncode,
    )


def _stage_selected(stage: str, selected: set[str] | None) -> bool:
    return selected is None or stage in selected


def run_post_training_eval(
    manifest_path: str | Path,
    *,
    dry_run: bool = False,
    rerun: bool = False,
    stages: list[str] | None = None,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("kind") != "ldf_stream_training_validation_plan":
        raise ValueError(
            "manifest kind must be 'ldf_stream_training_validation_plan', "
            f"got {manifest.get('kind')!r}"
        )
    selected = set(stages) if stages else None
    commands: list[dict[str, Any]] = []

    baseline_entry = manifest.get("stream_eval", {}).get("baseline", {})
    commands.append(
        _execute_entry(
            name="baseline_full_prefix",
            kind="baseline_eval",
            entry=baseline_entry,
            dry_run=dry_run,
            rerun=rerun,
        )
    )

    for stage in manifest.get("stages", []):
        stage_name = str(stage.get("stage") or "")
        if not _stage_selected(stage_name, selected):
            continue
        ckpt_glob = str(stage.get("expected_candidate_ckpt_glob") or "")
        ckpt = _newest_match(ckpt_glob) if ckpt_glob else None
        post_eval = stage.get("post_training_eval", {})
        candidate_eval = post_eval.get("candidate_eval", {})
        comparison = post_eval.get("comparison", {})
        if ckpt is None:
            commands.append(
                _command_record(
                    name=stage_name,
                    kind="candidate_eval",
                    summary=str(candidate_eval.get("summary") or ""),
                    command=_materialized_command(candidate_eval, None),
                    status="pending",
                    missing=["candidate checkpoint not found"],
                )
            )
            commands.append(
                _command_record(
                    name=stage_name,
                    kind="comparison",
                    summary=str(comparison.get("summary") or ""),
                    command=_materialized_command(comparison, None),
                    status="pending",
                    missing=["candidate checkpoint not found"],
                )
            )
            continue

        candidate_record = _execute_entry(
            name=stage_name,
            kind="candidate_eval",
            entry=candidate_eval,
            ckpt=ckpt,
            dry_run=dry_run,
            rerun=rerun,
        )
        commands.append(candidate_record)
        candidate_summary = str(candidate_eval.get("summary") or "")
        if not dry_run and not _path_exists(candidate_summary):
            commands.append(
                _command_record(
                    name=stage_name,
                    kind="comparison",
                    summary=str(comparison.get("summary") or ""),
                    command=_materialized_command(comparison, ckpt),
                    status="pending",
                    missing=["candidate summary missing"],
                )
            )
            continue
        commands.append(
            _execute_entry(
                name=stage_name,
                kind="comparison",
                entry=comparison,
                ckpt=ckpt,
                dry_run=dry_run,
                rerun=rerun,
                comparison_command=True,
            )
        )

    validation_status = collect_validation_status(manifest_path)
    execution_failed = [
        cmd for cmd in commands if cmd["status"] == "failed"
    ]
    if execution_failed:
        overall_status = "failed"
    elif dry_run:
        overall_status = "pending"
    else:
        overall_status = validation_status["overall"]["status"]

    return {
        "kind": "ldf_stream_training_post_eval_run",
        "manifest": str(manifest_path),
        "commands": commands,
        "validation_status": validation_status,
        "overall": {
            "status": overall_status,
            "failed_commands": execution_failed,
            "validation": validation_status["overall"],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run commands even when their recorded summary JSON already exists.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help="Optional stage name to run. Repeat for multiple stages.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_post_training_eval(
        args.manifest,
        dry_run=bool(args.dry_run),
        rerun=bool(args.rerun),
        stages=args.stage,
    )
    write_json_strict(args.out, result)
    status = result["overall"]["status"]
    if status == "passed":
        return 0
    if status == "failed":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
