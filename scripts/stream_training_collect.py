"""Collect LDF stream-training validation results from a manifest.

The smoke helper writes a manifest with train/eval/report commands. This
collector is the read-only counterpart: after those commands have run on a data
machine, it checks which stage outputs exist and whether each comparison report
passed the configured guardrails.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
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


CANDIDATE_CKPT_PLACEHOLDER = "{candidate_ckpt}"


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be a dict: {path}")
    return payload


def _newest_match(pattern: str) -> str | None:
    matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
    files = [path for path in matches if path.is_file()]
    if not files:
        return None
    newest = max(files, key=lambda path: (path.stat().st_mtime, str(path)))
    return str(newest)


def _baseline_status(manifest: dict[str, Any]) -> dict[str, Any]:
    baseline = manifest.get("stream_eval", {}).get("baseline", {})
    summary = str(baseline.get("summary") or "")
    present = bool(summary) and Path(summary).is_file()
    return {
        "summary": summary,
        "status": "present" if present else "missing",
        "missing": [] if present else ["baseline summary missing"],
    }


def _comparison_decision(path: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "failed", [{"reason": f"invalid comparison JSON: {exc}"}]
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        return "failed", [{"reason": "comparison decision missing"}]
    failures = decision.get("failures", [])
    if failures is None:
        failures = []
    if not isinstance(failures, list):
        failures = [{"reason": "comparison failures field is not a list"}]
    return ("passed" if bool(decision.get("passed")) and not failures else "failed", failures)


def _quote_cmd(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _materialized_command(entry: dict[str, Any], ckpt: str | None) -> dict[str, Any]:
    argv = [str(part) for part in entry.get("argv", [])]
    if ckpt is not None:
        argv = [
            ckpt if part == CANDIDATE_CKPT_PLACEHOLDER else part
            for part in argv
        ]
    ready = bool(argv) and CANDIDATE_CKPT_PLACEHOLDER not in argv
    return {
        "ready": ready,
        "argv": argv,
        "command": _quote_cmd(argv) if argv else "",
    }


def _stage_status(stage: dict[str, Any]) -> dict[str, Any]:
    post_eval = stage.get("post_training_eval", {})
    candidate_eval = post_eval.get("candidate_eval", {})
    comparison = post_eval.get("comparison", {})

    ckpt_glob = str(stage.get("expected_candidate_ckpt_glob") or "")
    ckpt = _newest_match(ckpt_glob) if ckpt_glob else None
    candidate_summary = str(candidate_eval.get("summary") or "")
    comparison_summary = str(comparison.get("summary") or "")

    missing: list[str] = []
    if ckpt is None:
        missing.append("candidate checkpoint not found")
    if not candidate_summary or not Path(candidate_summary).is_file():
        missing.append("candidate summary missing")
    if not comparison_summary or not Path(comparison_summary).is_file():
        missing.append("comparison missing")

    status = "pending"
    failures: list[dict[str, Any]] = []
    if not missing:
        status, failures = _comparison_decision(comparison_summary)

    return {
        "stage": stage.get("stage"),
        "description": stage.get("description"),
        "status": status,
        "candidate_ckpt": ckpt,
        "candidate_ckpt_glob": ckpt_glob,
        "candidate_summary": candidate_summary,
        "comparison": comparison_summary,
        "commands": {
            "candidate_eval": _materialized_command(candidate_eval, ckpt),
            "comparison": _materialized_command(comparison, ckpt),
        },
        "missing": missing,
        "failures": failures,
    }


def _overall_status(
    baseline: dict[str, Any], stages: list[dict[str, Any]]
) -> dict[str, Any]:
    failed = [stage["stage"] for stage in stages if stage["status"] == "failed"]
    pending = [stage["stage"] for stage in stages if stage["status"] == "pending"]
    if baseline["status"] != "present":
        pending.insert(0, "baseline_full_prefix")
    if failed:
        status = "failed"
    elif not pending and stages:
        status = "passed"
    else:
        status = "pending"
    return {
        "status": status,
        "failed_stages": failed,
        "pending_stages": pending,
    }


def collect_validation_status(manifest_path: str | Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("kind") != "ldf_stream_training_validation_plan":
        raise ValueError(
            "manifest kind must be 'ldf_stream_training_validation_plan', "
            f"got {manifest.get('kind')!r}"
        )
    baseline = _baseline_status(manifest)
    stages = [_stage_status(stage) for stage in manifest.get("stages", [])]
    return {
        "kind": "ldf_stream_training_validation_status",
        "manifest": str(manifest_path),
        "baseline": baseline,
        "stages": stages,
        "overall": _overall_status(baseline, stages),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    status = collect_validation_status(args.manifest)
    write_json_strict(args.out, status)
    overall = status["overall"]["status"]
    if overall == "passed":
        return 0
    if overall == "failed":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
