"""Refiner standalone benchmark (T_A_10, Benchmark A).

Evaluates RootRefiner predictions against RefinerDataset targets WITHOUT the
body model. Computes the metric suite from docs/TODO.md §T_A_10:

    num_token_top1_accuracy / num_token_top3_accuracy / num_token_MAE
    xyz_ADE / xyz_FDE
    heading_error_deg (median)
    fwd_speed_MAE      (per-frame fwd_delta channel; "speed" is the metric label)
    lateral_speed_MAE  (metric label; per-frame lateral displacement MAE,
                        not meters/second)
    yaw_rate_MAE       (per-frame yaw_delta channel)
    smoothness_acc_mean

Outputs a JSON summary + per-sample CSV.

⚠ Done-criteria thresholds (num_token top-1 > 0.5, heading_error_deg < 30°
median) require a TRAINED checkpoint (T_A_09). With random weights the pipeline
runs end-to-end but the numbers are meaningless — the smoke test only checks
that metrics are finite and the report is written.

References:
- docs/TODO.md §T_A_10 lines 1441-1478.
- docs/design.md §10.3 (Benchmark A metric definitions).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common.json import json_sanitize, write_json_strict  # noqa: E402
from eval.root_refiner.adapters import (  # noqa: E402
    DURATION_GROUNDTRUTH,
    DURATION_PRED,
    normalize_duration_mode,
)
from eval.root_refiner.metrics import (  # noqa: E402
    _heading_to_yaw,
    _lateral_component,
    compute_sample_metrics,
)
from utils.refiner.path_feature_stats import compute_sampling_config_hash  # noqa: E402


@dataclass(frozen=True)
class RootRefinerEvalSuite:
    name: str
    path_modes: tuple[str | None, ...]
    default_max_samples: int | None = None
    force_no_path_aug: bool = True
    duration_mode: str = DURATION_PRED

    @property
    def oracle_duration(self) -> bool:
        """Legacy compatibility alias for groundtruth-duration eval."""
        return self.duration_mode == DURATION_GROUNDTRUTH

    def to_json_dict(self) -> dict:
        return {
            "name": self.name,
            "path_modes": [
                mode if mode is not None else "mixed" for mode in self.path_modes
            ],
            "default_max_samples": self.default_max_samples,
            "force_no_path_aug": self.force_no_path_aug,
            "duration_mode": self.duration_mode,
            "oracle_duration": self.oracle_duration,
        }


_ROOT_REFINER_SUITES = {
    "smoke": RootRefinerEvalSuite(
        name="smoke",
        path_modes=(None,),
        default_max_samples=50,
        force_no_path_aug=True,
    ),
    "standard": RootRefinerEvalSuite(
        name="standard",
        path_modes=("dense_path", "sparse_path", "goal_point"),
        default_max_samples=None,
        force_no_path_aug=True,
    ),
    "standard_oracle": RootRefinerEvalSuite(
        name="standard_oracle",
        path_modes=("dense_path", "sparse_path", "goal_point"),
        default_max_samples=None,
        force_no_path_aug=True,
        duration_mode=DURATION_GROUNDTRUTH,
    ),
    "standard_groundtruth": RootRefinerEvalSuite(
        name="standard_groundtruth",
        path_modes=("dense_path", "sparse_path", "goal_point"),
        default_max_samples=None,
        force_no_path_aug=True,
        duration_mode=DURATION_GROUNDTRUTH,
    ),
    "stress": RootRefinerEvalSuite(
        name="stress",
        path_modes=("sparse_path", "goal_point"),
        default_max_samples=None,
        force_no_path_aug=False,
    ),
}


def resolve_suite_config(suite: str) -> RootRefinerEvalSuite:
    key = str(suite)
    if key not in _ROOT_REFINER_SUITES:
        valid = ", ".join(sorted(_ROOT_REFINER_SUITES))
        raise ValueError(f"unknown RootRefiner eval suite {key!r}; expected one of: {valid}")
    return _ROOT_REFINER_SUITES[key]


# ---------------------------------------------------------------------------
# Aggregate benchmark
# ---------------------------------------------------------------------------


def _get_eval_sample(
    dataset,
    idx: int,
    *,
    force_path_mode: str | None,
    force_no_path_aug: bool,
    force_mode: str | None = None,
    force_num_tokens: int | None = None,
    force_anchor_frame: int | None = None,
) -> dict:
    if not hasattr(dataset, "get_sample"):
        if force_path_mode is not None:
            raise ValueError("dataset does not support force_path_mode")
        return dataset[idx]

    kwargs = {
        "force_no_path_aug": bool(force_no_path_aug),
        "force_text_idx": 0,
    }
    if force_path_mode is not None:
        kwargs["force_path_mode"] = force_path_mode
    if force_mode is not None:
        kwargs["force_mode"] = force_mode
    if force_num_tokens is not None:
        kwargs["force_num_tokens"] = int(force_num_tokens)
    if force_anchor_frame is not None:
        kwargs["force_anchor_frame"] = int(force_anchor_frame)
    try:
        return dataset.get_sample(idx, **kwargs)
    except TypeError as exc:
        if force_path_mode is None:
            raise
        raise ValueError(
            "dataset.get_sample must support force_path_mode for RootRefiner "
            f"suite path-mode evaluation; got {force_path_mode!r}"
        ) from exc


def _to_int(value, default: int = 0) -> int:
    if value is None:
        return int(default)
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def build_eval_task_specs(dataset, max_samples: int = -1) -> list[dict]:
    """Freeze underlying RootRefiner tasks before path-mode bucketing.

    Each spec captures the dataset index, full/sliding mode, target duration,
    and anchor frame. Path-mode suites can then rebuild dense/sparse/goal
    conditions over identical tasks instead of comparing different random
    anchors or horizons.
    """
    if hasattr(dataset, "reset_rng"):
        dataset.reset_rng()
    n = len(dataset) if max_samples < 0 else min(int(max_samples), len(dataset))
    specs = []
    for idx in range(n):
        sample = _get_eval_sample(
            dataset,
            idx,
            force_path_mode=None,
            force_no_path_aug=True,
        )
        mode = str(sample.get("mode", "sample"))
        num_tokens = _to_int(sample.get("num_tokens"), default=0)
        anchor_frame = _to_int(sample.get("anchor_frame"), default=0)
        specs.append(
            {
                "idx": idx,
                "mode": mode,
                "num_tokens": num_tokens,
                "anchor_frame": anchor_frame,
                "task_key": f"{idx}:{mode}:{num_tokens}:{anchor_frame}",
            }
        )
    return specs


def _nanmean_from_samples(per_sample: list[dict], key: str) -> float:
    vals = [s[key] for s in per_sample if key in s and not math.isnan(s[key])]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _nanmedian_from_samples(per_sample: list[dict], key: str) -> float:
    vals = sorted(s[key] for s in per_sample if key in s and not math.isnan(s[key]))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2)


def summarize_per_sample(
    per_sample: list[dict],
    *,
    duration_mode: str | None = None,
    oracle_duration: bool = False,
) -> dict:
    resolved_duration_mode = resolve_duration_mode(
        duration_mode=duration_mode,
        oracle_duration=oracle_duration,
    )
    use_groundtruth_duration = resolved_duration_mode == DURATION_GROUNDTRUTH
    n = len(per_sample)
    unique_task_keys = {
        str(s.get("task_key", s.get("idx", i)))
        for i, s in enumerate(per_sample)
    }
    n_unique = len(unique_task_keys)
    return {
        "n_samples": n_unique,
        "n_records": n,
        "n_unique_tasks": n_unique,
        "duration_mode": resolved_duration_mode,
        "oracle_duration": use_groundtruth_duration,
        "num_token_top1_accuracy": (
            sum(int(s.get("num_token_top1_hit", 0)) for s in per_sample) / n
            if n else float("nan")
        ),
        "num_token_top3_accuracy": (
            sum(int(s.get("num_token_top3_hit", 0)) for s in per_sample) / n
            if n else float("nan")
        ),
        "num_token_MAE": (
            sum(
                abs(int(s["pred_num_tokens"]) - int(s["gt_num_tokens"]))
                for s in per_sample
            ) / n
            if n else float("nan")
        ),
        "xyz_ADE": _nanmean_from_samples(per_sample, "xyz_ADE"),
        "xyz_FDE": _nanmean_from_samples(per_sample, "xyz_FDE"),
        "heading_error_deg": _nanmedian_from_samples(per_sample, "heading_error_deg"),
        "fwd_speed_MAE": _nanmean_from_samples(per_sample, "fwd_speed_MAE"),
        "lateral_speed_MAE": _nanmean_from_samples(per_sample, "lateral_speed_MAE"),
        "yaw_rate_MAE": _nanmean_from_samples(per_sample, "yaw_rate_MAE"),
        "smoothness_acc_mean": _nanmean_from_samples(per_sample, "smoothness"),
    }


def resolve_duration_mode(
    *,
    duration_mode: str | None = None,
    oracle_duration: bool = False,
) -> str:
    if duration_mode is None:
        return DURATION_GROUNDTRUTH if oracle_duration else DURATION_PRED
    resolved = normalize_duration_mode(duration_mode)
    if oracle_duration and resolved != DURATION_GROUNDTRUTH:
        raise ValueError(
            "conflicting duration options: --oracle_duration is compatible only "
            f"with duration_mode={DURATION_GROUNDTRUTH!r}"
        )
    return resolved


def resolve_suite_duration_mode(
    suite_cfg: RootRefinerEvalSuite,
    *,
    duration_mode: str | None = None,
    oracle_duration: bool = False,
) -> str:
    if duration_mode is not None or oracle_duration:
        return resolve_duration_mode(
            duration_mode=duration_mode,
            oracle_duration=oracle_duration,
        )
    return normalize_duration_mode(suite_cfg.duration_mode)


_CKPT_CONFIG_CONTRACT_KEYS = (
    ("model", "params", "n_hist"),
    ("model", "params", "n_path"),
    ("model", "params", "max_tokens"),
    ("model", "params", "min_tokens"),
    ("model", "params", "frames_per_token"),
)


def _nested_get(mapping, path: tuple[str, ...]):
    cur = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def validate_ckpt_eval_config_compatible(ckpt_cfg: dict, eval_cfg: dict) -> None:
    """Fail fast when benchmark --config disagrees with checkpoint hparams."""
    mismatches = []
    for path in _CKPT_CONFIG_CONTRACT_KEYS:
        ckpt_value = _nested_get(ckpt_cfg, path)
        eval_value = _nested_get(eval_cfg, path)
        if ckpt_value is None or eval_value is None:
            continue
        if ckpt_value != eval_value:
            dotted = ".".join(path)
            mismatches.append(f"{dotted}: ckpt={ckpt_value!r} eval={eval_value!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            "RootRefiner benchmark config mismatch between checkpoint hparams "
            f"and --config: {joined}. Use the training config for this checkpoint."
        )


@torch.no_grad()
def run_benchmark(
    model,
    dataset,
    text_encoder,
    device="cpu",
    max_samples: int = -1,
    oracle_duration: bool = False,
    duration_mode: str | None = None,
    force_path_mode: str | None = None,
    force_no_path_aug: bool = True,
    task_specs: list[dict] | None = None,
) -> dict:
    """Run inference over `dataset` and aggregate the metric suite.

    `model` is a RootRefiner (eval mode). `text_encoder` must have
    `.encode(list[str], device=...) -> [B, text_emb_dim]`.

    `duration_mode`: `pred_duration` uses model-predicted duration;
    `groundtruth_duration` feeds GT num_tokens so trajectory metrics
    (xyz_ADE/FDE, heading, ...) measure the WAYPOINT DECODER ALONE under the
    correct horizon, isolating it from num_token-head prediction error.
    `oracle_duration` is a deprecated alias for `groundtruth_duration`.
    Default = real inference (model picks its expected-round duration, so
    trajectory metrics use the common GT/predicted prefix).
    top-k duration metrics still use classification logits; num_token_MAE uses
    the actual predicted duration that drives inference.

    Returns dict with `summary` (aggregate metrics) and `per_sample` (list).
    """
    duration_mode_resolved = resolve_duration_mode(
        duration_mode=duration_mode,
        oracle_duration=oracle_duration,
    )
    use_groundtruth_duration = duration_mode_resolved == DURATION_GROUNDTRUTH
    model = model.to(device).eval()
    min_tokens = model.min_tokens

    # Reproducibility: when no frozen task list is provided, get_sample advances
    # the dataset RNG every call, so reset it to the base seed for comparable
    # single-pass/oracle runs.
    if task_specs is None and hasattr(dataset, "reset_rng"):
        dataset.reset_rng()

    if task_specs is None:
        n = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))
        task_specs = [{"idx": idx} for idx in range(n)]
    else:
        n = len(task_specs)

    per_sample = []

    for spec in task_specs:
        idx = int(spec["idx"])
        # force_no_path_aug controls whether the benchmark measures clean path
        # conditions or intentionally stresses offset/sparse path augmentation.
        # force_text_idx=0 pins the canonical first caption (training randomizes
        # over all captions for augmentation; eval must stay comparable).
        sample = _get_eval_sample(
            dataset,
            idx,
            force_path_mode=force_path_mode,
            force_no_path_aug=force_no_path_aug,
            force_mode=spec.get("mode"),
            force_num_tokens=spec.get("num_tokens"),
            force_anchor_frame=spec.get("anchor_frame"),
        )
        text_emb = text_encoder.encode([sample["text"]], device=device)
        # groundtruth_duration: teacher-force the GT horizon so trajectory
        # metrics isolate the waypoint decoder. The model honors num_tokens when
        # given, regardless of eval mode; see RootRefiner.forward.
        oracle_nt = (
            sample["num_tokens"].reshape(1).to(device)
            if use_groundtruth_duration
            else None
        )
        out = model(
            text_emb=text_emb,
            path=sample["path"].unsqueeze(0).to(device),
            path_valid_mask=sample["path_valid_mask"].unsqueeze(0).to(device),
            path_control_mask=sample["path_control_mask"].unsqueeze(0).to(device),
            path_mode=[sample.get("path_mode", "dense_path")],
            path_features=sample["path_features"].unsqueeze(0).to(device),
            history_motion=sample["history_motion"].unsqueeze(0).to(device),
            history_mask=sample["history_mask"].unsqueeze(0).to(device),
            offset_start_frames=sample.get(
                "offset_start_frames",
                torch.tensor(0, dtype=torch.long),
            ).reshape(1).to(device),
            num_tokens=oracle_nt,
        )
        logits = out["num_token_logits"][0]                  # [K]
        # Model emits NORMALIZED 5D. Assemble physical 7D at the boundary
        # (unnormalize xyz → unit heading → append fwd_delta / yaw_delta) so
        # metrics are computed in physical space. GT `target_waypoints` is
        # PHYSICAL-then-z-scored, so unnormalize its xyz the same way and
        # re-derive its deltas (do NOT trust target_wp[..., 5:7] — those are
        # z-scored physical deltas that would mismatch the freshly-derived
        # physical pred deltas in scale and offset).
        from utils.motion_process import build_physical_7d_from_normalized_5d
        wp_mean = getattr(dataset, "_wp_mean", None)
        wp_std = getattr(dataset, "_wp_std", None)
        wp_norm_idx = getattr(dataset, "_wp_norm_idx", None)
        pred_wp = build_physical_7d_from_normalized_5d(
            out["waypoints"][0].cpu(), wp_mean, wp_std, wp_norm_idx,
        )                                                    # [max_frames, 7] physical
        gt_source = sample.get("target_waypoints", sample.get("waypoints"))
        gt5_norm = gt_source[..., :5]
        gt_wp = build_physical_7d_from_normalized_5d(
            gt5_norm, wp_mean, wp_std, wp_norm_idx,
        )                                                    # [max_frames, 7] physical
        mask = sample.get("target_mask", sample.get("waypoints_mask"))
        if not use_groundtruth_duration:
            used_tokens = int(out["used_num_tokens"][0].detach().cpu().item())
            valid_eff = (
                model.frames_per_token * used_tokens
                - (model.frames_per_token - 1)
            )
            common = torch.arange(mask.shape[0]) < int(valid_eff)
            mask = mask.bool() & common

        # num_token metrics.
        gt_class = int(sample["num_tokens"].item()) - min_tokens
        gt_class = max(0, min(gt_class, logits.shape[-1] - 1))
        pred_class = int(logits.argmax().item())
        pred_num_tokens = int(out["pred_num_tokens"][0].detach().cpu().item())
        top3 = torch.topk(logits, k=min(3, logits.shape[-1])).indices.tolist()
        top1_hit = int(pred_class == gt_class)
        top3_hit = int(gt_class in top3)

        m = compute_sample_metrics(pred_wp, gt_wp, mask)
        m["idx"] = idx
        m["task_key"] = spec.get("task_key", f"{idx}")
        m["mode"] = sample.get("mode", spec.get("mode"))
        m["anchor_frame"] = _to_int(sample.get("anchor_frame"), default=0)
        m["path_mode"] = sample.get("path_mode", force_path_mode or "mixed")
        m["num_token_top1_hit"] = top1_hit
        m["num_token_top3_hit"] = top3_hit
        m["pred_num_tokens"] = pred_num_tokens
        m["argmax_num_tokens"] = pred_class + min_tokens
        m["gt_num_tokens"] = gt_class + min_tokens
        per_sample.append(m)

    summary = summarize_per_sample(
        per_sample,
        duration_mode=duration_mode_resolved,
    )
    return {"summary": summary, "per_sample": per_sample}


def _suite_sample_limit(
    suite_cfg: RootRefinerEvalSuite,
    max_samples: int,
) -> int:
    if max_samples is not None and int(max_samples) > 0:
        return int(max_samples)
    if suite_cfg.default_max_samples is not None:
        return int(suite_cfg.default_max_samples)
    return -1


def build_refiner_dataset_from_clips(
    cfg: dict,
    clips,
    *,
    dataset_cls,
    seed: int = 0,
):
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg["model"]["params"]
    sampling_cfg = cfg.get("sampling", {}) or {}
    path_condition_cfg = sampling_cfg.get("path_condition", {}) or {}
    offset_cfg = path_condition_cfg.get("offset_start", {}) or {}
    sparse_cfg = path_condition_cfg.get("sparse_path", {}) or {}
    normalize = bool(data_cfg.get("normalize", False))
    path_feature_stats_dir = data_cfg.get("path_feature_stats_dir") if normalize else None
    return dataset_cls(
        clips,
        n_hist=model_cfg["n_hist"],
        n_path=model_cfg["n_path"],
        max_tokens=model_cfg["max_tokens"],
        min_tokens=model_cfg["min_tokens"],
        frames_per_token=model_cfg["frames_per_token"],
        full_plan_ratio=sampling_cfg.get("full_plan_ratio", 0.5),
        horizon_policy=sampling_cfg.get("horizon_policy", "random"),
        path_condition_policy=path_condition_cfg.get("policy", "dense_path"),
        path_condition_ratios=path_condition_cfg.get("ratios"),
        offset_start_enabled=bool(offset_cfg.get("enabled", False)),
        offset_start_prob=float(offset_cfg.get("prob", 0.0)),
        offset_start_max_frames=int(offset_cfg.get("max_frames", 40)),
        offset_start_apply_to=tuple(
            offset_cfg.get("apply_to", ("dense_path", "sparse_path"))
        ),
        sparse_path_point_range=tuple(sparse_cfg.get("point_range", (3, 8))),
        normalize=normalize,
        stats_dir=data_cfg.get("stats_dir") if normalize else None,
        path_feature_stats_dir=path_feature_stats_dir,
        sampling_config_hash=(
            compute_sampling_config_hash(cfg)
            if path_feature_stats_dir is not None
            else None
        ),
        seed=seed,
    )


def _path_mode_label(path_mode: str | None) -> str:
    return path_mode if path_mode is not None else "mixed"


def _add_prefixed_summary(
    target: dict,
    *,
    prefix: str,
    summary: dict,
) -> None:
    for key, value in summary.items():
        target[f"{prefix}/{key}"] = value


def build_eval_payload(
    result: dict,
    *,
    suite: str = "single",
    suite_config: RootRefinerEvalSuite | None = None,
    runs: list[dict] | None = None,
) -> dict:
    if result.get("schema_version") == "root_refiner_eval.v1":
        return result
    return {
        "schema_version": "root_refiner_eval.v1",
        "evaluator": "root_refiner",
        "suite": str(suite),
        "suite_config": (
            suite_config.to_json_dict() if suite_config is not None else None
        ),
        "summary": result["summary"],
        "runs": runs or [],
        "per_sample": result["per_sample"],
    }


def _json_sanitize(value):
    return json_sanitize(value)


def run_suite_benchmark(
    model,
    dataset,
    text_encoder,
    *,
    suite: str,
    device="cpu",
    max_samples: int = -1,
    duration_mode: str | None = None,
    oracle_duration: bool = False,
) -> dict:
    suite_cfg = resolve_suite_config(suite)
    duration_mode_resolved = resolve_suite_duration_mode(
        suite_cfg,
        duration_mode=duration_mode,
        oracle_duration=oracle_duration,
    )
    sample_limit = _suite_sample_limit(suite_cfg, max_samples)
    task_specs = build_eval_task_specs(dataset, max_samples=sample_limit)
    runs: list[dict] = []
    per_sample: list[dict] = []

    for path_mode in suite_cfg.path_modes:
        label = _path_mode_label(path_mode)
        run = run_benchmark(
            model,
            dataset,
            text_encoder,
            device=device,
            max_samples=sample_limit,
            duration_mode=duration_mode_resolved,
            force_path_mode=path_mode,
            force_no_path_aug=suite_cfg.force_no_path_aug,
            task_specs=task_specs,
        )
        run_samples = []
        for sample in run["per_sample"]:
            sample = dict(sample)
            sample["suite"] = suite_cfg.name
            sample["run_name"] = label
            sample["path_mode"] = label
            run_samples.append(sample)
        runs.append(
            {
                "name": label,
                "path_mode": label,
                "summary": run["summary"],
            }
        )
        per_sample.extend(run_samples)

    summary = summarize_per_sample(
        per_sample,
        duration_mode=duration_mode_resolved,
    )
    summary["n_samples"] = len(task_specs)
    summary["n_records"] = len(per_sample)
    summary["n_unique_tasks"] = len(task_specs)
    summary["num_runs"] = len(runs)
    for run in runs:
        _add_prefixed_summary(
            summary,
            prefix=f"path_mode/{run['path_mode']}",
            summary=run["summary"],
        )

    return build_eval_payload(
        {"summary": summary, "per_sample": per_sample},
        suite=suite_cfg.name,
        suite_config=RootRefinerEvalSuite(
            name=suite_cfg.name,
            path_modes=suite_cfg.path_modes,
            default_max_samples=suite_cfg.default_max_samples,
            force_no_path_aug=suite_cfg.force_no_path_aug,
            duration_mode=duration_mode_resolved,
        ),
        runs=runs,
    )


def write_report(result: dict, output_dir: str | Path) -> None:
    """Write metrics.json plus legacy summary.json + per_sample.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = _json_sanitize(build_eval_payload(result))
    write_json_strict(out / "metrics.json", payload)
    write_json_strict(out / "summary.json", payload["summary"])
    per_sample = payload["per_sample"]
    if per_sample:
        keys = sorted({key for sample in per_sample for key in sample.keys()})
        with (out / "per_sample.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_sample)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_model_from_ckpt(ckpt_path: str, device: str):
    """Load RootRefiner weights from a Lightning checkpoint or raw state_dict."""
    from train_refiner import RefinerLightningModule

    ckpt = torch.load(ckpt_path, map_location=device)
    if "hyper_parameters" in ckpt and "cfg" in ckpt["hyper_parameters"]:
        cfg = ckpt["hyper_parameters"]["cfg"]
        module = RefinerLightningModule(cfg)
        module.load_state_dict(ckpt["state_dict"])
        return module.refiner, module.text_encoder, cfg
    raise ValueError(
        f"checkpoint {ckpt_path} missing hyper_parameters.cfg; "
        "pass a Lightning checkpoint saved by train_refiner.py"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/root_refiner.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/refiner_bench")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--split_file", type=str, default=None,
                         help="Eval split; defaults to data.val_split_file or the dataset default.")
    parser.add_argument(
        "--oracle_duration",
        action="store_true",
        default=False,
        help=(
            "Deprecated alias for --duration_mode groundtruth_duration: "
            "teacher-force GT num_tokens for trajectory metrics."
        ),
    )
    parser.add_argument(
        "--duration_mode",
        type=str,
        choices=(DURATION_PRED, DURATION_GROUNDTRUTH),
        default=None,
        help=(
            "Duration mode for RootRefiner trajectory evaluation. "
            "Defaults to the selected suite's mode, or pred_duration in "
            "legacy single-pass mode."
        ),
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=sorted(_ROOT_REFINER_SUITES),
        default=None,
        help="Optional layered eval suite. Omit to preserve legacy single-pass behavior.",
    )
    args = parser.parse_args(argv)

    from datasets.humanml3d_refiner import HumanML3DRefinerDataset as RefinerDataset
    from scripts.compute_5d_stats import load_clips_from_dir

    from train_refiner import resolve_cfg_interpolations

    from train_refiner import _load_cfg
    cfg = _load_cfg(args.config)
    # A-P0-1: resolve ${data.raw_data_dir} etc. (the model cfg comes from the
    # ckpt's saved hparams, which train_refiner already resolved).
    cfg = resolve_cfg_interpolations(cfg)

    model, text_encoder, ckpt_cfg = _load_model_from_ckpt(args.ckpt, args.device)
    validate_ckpt_eval_config_compatible(ckpt_cfg, cfg)

    data_cfg = cfg.get("data", {})
    split_file = args.split_file or data_cfg.get("val_split_file")
    clips = load_clips_from_dir(
        data_cfg["raw_data_dir"],
        dataset=data_cfg.get("dataset", "humanml3d"),
        split_file=split_file,
        feature_path=data_cfg.get("feature_path"),
        text_path=data_cfg.get("text_path"),
    )
    dataset = build_refiner_dataset_from_clips(
        cfg,
        clips,
        dataset_cls=RefinerDataset,
        seed=0,
    )

    if args.suite:
        result = run_suite_benchmark(
            model,
            dataset,
            text_encoder,
            suite=args.suite,
            device=args.device,
            max_samples=args.max_samples,
            duration_mode=args.duration_mode,
            oracle_duration=args.oracle_duration,
        )
    else:
        result = run_benchmark(model, dataset, text_encoder, device=args.device,
                                max_samples=args.max_samples,
                                oracle_duration=args.oracle_duration,
                                duration_mode=args.duration_mode)
    write_report(result, args.output_dir)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
