"""Shared helpers for LDF stream-training validation manifests."""

from __future__ import annotations

import glob
import json
import shlex
from pathlib import Path
from typing import Any


CANDIDATE_CKPT_PLACEHOLDER = "{candidate_ckpt}"


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from `path`."""
    with Path(path).open() as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be a dict: {path}")
    return payload


def newest_match(pattern: str) -> str | None:
    """Return the newest file matching a glob pattern, or None."""
    matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
    files = [path for path in matches if path.is_file()]
    if not files:
        return None
    newest = max(files, key=lambda path: (path.stat().st_mtime, str(path)))
    return str(newest)


def quote_command(argv: list[str]) -> str:
    """Shell-quote an argv list for manifest display."""
    return " ".join(shlex.quote(str(part)) for part in argv)


def materialize_command(entry: dict[str, Any], ckpt: str | None) -> dict[str, Any]:
    """Replace the candidate-checkpoint placeholder in one manifest command."""
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
        "command": quote_command(argv) if argv else "",
    }
