"""Runtime debug evaluation artifact layout helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from eval.common.json import json_sanitize, write_json_strict


def _path_part(value: str, *, label: str) -> str:
    part = str(value).strip()
    if not part:
        raise ValueError(f"{label} contains an empty path part")
    if "/" in part or "\\" in part:
        raise ValueError(f"{label} must be a single path part, got {value!r}")
    return part


def infer_ckpt_tag(ckpt_path: str | Path) -> str:
    """Return a filesystem-safe checkpoint tag such as ``step_485000``."""

    stem = Path(ckpt_path).name
    if stem.endswith(".ckpt"):
        stem = stem[:-5]
    return _path_part(stem, label="ckpt_tag")


@dataclass(frozen=True)
class RuntimeArtifactLayout:
    """Directory contract for runtime debug evaluation outputs."""

    output_root: str | Path
    ckpt_tag: str
    run_id: str

    @property
    def run_dir(self) -> Path:
        return (
            Path(self.output_root)
            / "runtime"
            / _path_part(self.ckpt_tag, label="ckpt_tag")
            / _path_part(self.run_id, label="run_id")
        )

    def root_source_dir(self, root_source: str) -> Path:
        return self.run_dir / _path_part(root_source, label="root_source")

    def experiment_dir(self, root_source: str, family: str, *parts: str) -> Path:
        out = self.root_source_dir(root_source) / _path_part(family, label="family")
        for idx, part in enumerate(parts):
            out = out / _path_part(part, label=f"parts[{idx}]")
        return out

    def root_diagnostic_dir(self, family: str, *parts: str) -> Path:
        out = self.run_dir / "root_diagnostics" / _path_part(family, label="family")
        for idx, part in enumerate(parts):
            out = out / _path_part(part, label=f"parts[{idx}]")
        return out / "gtroot_vs_rootrefiner"


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(layout: RuntimeArtifactLayout, payload: Mapping[str, Any]) -> Path:
    path = _ensure_parent(layout.run_dir / "manifest.json")
    write_json_strict(path, dict(payload))
    return path


def write_summary_json(layout: RuntimeArtifactLayout, payload: Mapping[str, Any]) -> Path:
    path = _ensure_parent(layout.run_dir / "summary.json")
    write_json_strict(path, dict(payload))
    return path


def write_source_json(
    layout: RuntimeArtifactLayout,
    root_source: str,
    payload: Mapping[str, Any],
) -> Path:
    path = _ensure_parent(layout.root_source_dir(root_source) / "source.json")
    write_json_strict(path, dict(payload))
    return path


def write_experiment_metrics(
    layout: RuntimeArtifactLayout,
    *,
    root_source: str,
    family: str,
    parts: tuple[str, ...] = (),
    metrics: Mapping[str, Any],
) -> Path:
    path = _ensure_parent(
        layout.experiment_dir(root_source, family, *parts) / "metrics.json"
    )
    write_json_strict(path, dict(metrics))
    return path


def write_runtime_debug_report(
    layout: RuntimeArtifactLayout,
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    source_payloads: Mapping[str, Mapping[str, Any]] = (),
) -> dict[str, Path]:
    paths = {
        "run_dir": layout.run_dir,
        "manifest": write_manifest(layout, manifest),
        "summary": write_summary_json(layout, summary),
        "records": write_records_csv(layout, records),
    }
    for root_source, payload in dict(source_payloads).items():
        paths[f"source:{root_source}"] = write_source_json(layout, root_source, payload)
    return paths


def _csv_cell(value: Any) -> Any:
    clean = json_sanitize(value)
    if isinstance(clean, (dict, list)):
        return json.dumps(clean, separators=(",", ":"), allow_nan=False)
    return clean


def write_records_csv(
    layout: RuntimeArtifactLayout,
    records: Iterable[Mapping[str, Any]],
) -> Path:
    path = _ensure_parent(layout.run_dir / "records.csv")
    rows = [dict(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})
    return path


def write_root_diagnostic_artifacts(
    layout: RuntimeArtifactLayout,
    *,
    family: str,
    parts: tuple[str, ...] = (),
    metrics: Mapping[str, Any],
    gt_root_7d: Any,
    pred_root_7d: Any,
    gt_num_tokens: int,
    pred_num_tokens: int,
) -> Path:
    out_dir = layout.root_diagnostic_dir(family, *parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_strict(out_dir / "metrics.json", dict(metrics))
    np.savez(
        out_dir / "root_plans.npz",
        gt_root_7d=np.asarray(gt_root_7d, dtype=np.float32),
        pred_root_7d=np.asarray(pred_root_7d, dtype=np.float32),
        gt_num_tokens=np.asarray(int(gt_num_tokens), dtype=np.int64),
        pred_num_tokens=np.asarray(int(pred_num_tokens), dtype=np.int64),
    )
    return out_dir


def write_root_diagnostics_summary(
    layout: RuntimeArtifactLayout,
    *,
    summary: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Path]:
    out_dir = layout.run_dir / "root_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    records_path = out_dir / "summary.csv"
    write_json_strict(summary_path, dict(summary))
    rows = [dict(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    with records_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})
    return {"summary": summary_path, "records": records_path}


__all__ = [
    "infer_ckpt_tag",
    "RuntimeArtifactLayout",
    "write_experiment_metrics",
    "write_manifest",
    "write_records_csv",
    "write_runtime_debug_report",
    "write_root_diagnostic_artifacts",
    "write_root_diagnostics_summary",
    "write_source_json",
    "write_summary_json",
]
