"""Shared artifact helpers for eval packages."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from eval.common.json import json_sanitize, write_json_strict


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_sample_artifact_dir(
    output_root: str | Path,
    suite_name: str,
    sample_id: str,
) -> Path:
    return ensure_dir(Path(output_root) / "cases" / str(suite_name) / str(sample_id))


def write_eval_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    write_json_strict(out, payload)
    return out


def _csv_cell(value: Any) -> Any:
    clean = json_sanitize(value)
    if isinstance(clean, (dict, list)):
        return json.dumps(clean, separators=(",", ":"), allow_nan=False)
    return clean


def write_eval_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    materialized = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})
    return out


__all__ = [
    "ensure_dir",
    "make_sample_artifact_dir",
    "write_eval_csv",
    "write_eval_json",
]
