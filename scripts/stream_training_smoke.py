"""Launch or print a reproducible LDF stream-training smoke command.

This helper does not fake data. It performs a path preflight, then builds the
`train_ldf.py --override ...` command needed to validate the flag-gated
window-local stream training path on a data machine.

`--raw-data-root` follows `configs/ldf.yaml`: it is the parent `raw_data`
directory containing `HumanML3D/`, not the `HumanML3D/` directory itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path


CANDIDATE_CKPT_PLACEHOLDER = "{candidate_ckpt}"


@dataclass
class SmokeRunConfig:
    config: str = "configs/ldf.yaml"
    python: str = sys.executable
    resume_ckpt: str = ""
    vae_ckpt: str = ""
    raw_data_root: str = ""
    output_dir: str = "outputs/stream_training_smoke"
    exp_name: str = "ldf_stream_training_smoke"
    max_steps: int = 0
    devices: str = "1"
    accelerator: str = "cpu"
    sample_policy: str = "variable_history"
    motion_aux_loss: str = "latent_only"
    context_tokens: int = 30
    min_history_tokens: int = 8
    horizon_tokens: int = 20
    history_tokens_min: int = 0
    history_tokens_max: str = "auto"
    horizon_tokens_min: int = 5
    horizon_tokens_max: int = 25
    z_stats_dir: str = ""
    train_split: str = ""
    val_split: str = ""
    debug: bool = True
    dry_run: bool = False
    print_only: bool = False
    validation_plan: bool = False
    manifest: str = ""
    eval_out_dir: str = ""
    eval_max_samples: int = 5
    eval_num_runs: int = 1
    eval_stream_mode: str = "stream_generate_step"
    max_root_ade_regression: float | None = 0.05
    max_root_fde_regression: float | None = None
    max_path_arc_regression: float | None = None
    max_yaw_error_regression: float | None = None
    max_jitter_regression: float | None = None
    max_foot_skating_regression: float | None = None
    max_root_jump_regression: float | None = None


@dataclass(frozen=True)
class ValidationPlanEntry:
    stage: str
    description: str
    config: SmokeRunConfig


def _bool_str(value: bool) -> str:
    return "true" if bool(value) else "false"


def _quote_cmd(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in cmd)


def _split_override(value: str | None) -> list[str]:
    return [value] if value else []


def _threshold_str(value: float) -> str:
    return f"{float(value):g}"


def _stream_eval_out_dir(cfg: SmokeRunConfig) -> str:
    if cfg.eval_out_dir:
        return cfg.eval_out_dir
    return str(Path(cfg.output_dir) / "stream_eval")


def build_validation_plan(base: SmokeRunConfig) -> list[ValidationPlanEntry]:
    """Expand one base config into the standard stream-training validation plan."""
    stages = [
        (
            "01_smoke_latent",
            "one-step or short smoke with active-left window sampling and latent-only loss",
            "latent_only",
        ),
        (
            "02_overfit_latent",
            "small overfit with active-left window sampling and latent-only loss",
            "latent_only",
        ),
        (
            "03_overfit_full_prefix",
            "small overfit with active-left window sampling and full-prefix motion auxiliary loss",
            "full_prefix",
        ),
    ]
    out = []
    for stage, description, motion_aux_loss in stages:
        out.append(
            ValidationPlanEntry(
                stage=stage,
                description=description,
                config=replace(
                    base,
                    motion_aux_loss=motion_aux_loss,
                    exp_name=f"{base.exp_name}_{stage}",
                    output_dir=str(Path(base.output_dir) / stage),
                    validation_plan=False,
                ),
            )
        )
    return out


def build_train_command(cfg: SmokeRunConfig) -> list[str]:
    if cfg.motion_aux_loss not in {"latent_only", "full_prefix", "disabled"}:
        raise ValueError(
            "motion_aux_loss must be 'latent_only', 'full_prefix', or "
            f"'disabled', got {cfg.motion_aux_loss!r}"
        )
    if int(cfg.max_steps) <= 0:
        raise ValueError("max_steps must be an absolute Lightning target step > 0")

    overrides = [
        "train=true",
        f"debug={_bool_str(cfg.debug)}",
        f"exp_name={cfg.exp_name}",
        f"save_dir={cfg.output_dir}",
        f"resume_ckpt={cfg.resume_ckpt}",
        f"test_ckpt={cfg.resume_ckpt}",
        f"test_vae_ckpt={cfg.vae_ckpt}",
        f"dirs.raw_data={cfg.raw_data_root}",
        f"trainer.max_steps={int(cfg.max_steps)}",
        f"trainer.accelerator={cfg.accelerator}",
        f"trainer.devices={cfg.devices}",
        "trainer.log_every_n_steps=1",
        "validation.validation_steps=1",
        "validation.test_steps=1",
        "validation.save_every_n_steps=1",
        "validation.eval_generation_metrics=false",
        "t2m_metric=false",
        "data.num_workers=0",
        "data.train_bs=1",
        "data.val_bs=1",
        "model.params.traj_encoder_in_dim=7",
        "data.traj_feat_dim=7",
        "model.params.self_forcing_enabled=true",
        "stream_training.enabled=true",
        f"stream_training.context_tokens={int(cfg.context_tokens)}",
        "stream_training.window_sampling.enabled=true",
        f"stream_training.window_sampling.history_tokens_min={int(cfg.history_tokens_min)}",
        f"stream_training.window_sampling.history_tokens_max={cfg.history_tokens_max}",
        f"stream_training.window_sampling.horizon_tokens_min={int(cfg.horizon_tokens_min)}",
        f"stream_training.window_sampling.horizon_tokens_max={int(cfg.horizon_tokens_max)}",
        "stream_training.anchor_move_in_rollout=false",
        "stream_training.latent_source=precomputed_slice",
        f"stream_training.motion_aux_loss={cfg.motion_aux_loss}",
        "history_corruption.enabled=true",
        "anchor_canonicalize.enabled=true",
        "body_aux_loss.enabled=true",
    ]
    overrides.extend(_split_override(f"history_corruption.z_stats_dir={cfg.z_stats_dir}" if cfg.z_stats_dir else None))
    overrides.extend(_split_override(f"data.train_meta_paths.0={cfg.train_split}" if cfg.train_split else None))
    overrides.extend(_split_override(f"data.val_meta_paths.0={cfg.val_split}" if cfg.val_split else None))
    overrides.extend(_split_override(f"data.test_probe_meta_paths.test.0={cfg.val_split}" if cfg.val_split else None))

    return [cfg.python, "train_ldf.py", "--config", cfg.config, "--override", *overrides]


def build_stream_eval_command(
    cfg: SmokeRunConfig,
    *,
    ckpt: str,
    run_name: str,
    probe_tag: str,
) -> list[str]:
    """Build a deterministic LDF stream-eval command for a trained checkpoint."""
    if cfg.eval_stream_mode not in {"stream_generate", "stream_generate_step"}:
        raise ValueError(
            "eval_stream_mode must be 'stream_generate' or "
            f"'stream_generate_step', got {cfg.eval_stream_mode!r}"
        )
    cmd = [
        cfg.python,
        "-m",
        "eval.ldf.stream_metrics",
        "--config",
        cfg.config,
        "--ckpt",
        ckpt,
        "--vae_ckpt",
        cfg.vae_ckpt,
        "--stream_mode",
        cfg.eval_stream_mode,
        "--out_dir",
        _stream_eval_out_dir(cfg),
        "--run_name",
        run_name,
        "--probe_tag",
        probe_tag,
        "--max_samples",
        str(int(cfg.eval_max_samples)),
        "--num_runs",
        str(int(cfg.eval_num_runs)),
    ]
    if cfg.val_split:
        cmd.extend(["--meta_paths", cfg.val_split])
    return cmd


def build_report_command(
    cfg: SmokeRunConfig,
    *,
    baseline_summary: str,
    candidate_summary: str,
    out_path: str,
) -> list[str]:
    cmd = [
        cfg.python,
        "-m",
        "eval.ldf.report",
        "--baseline",
        baseline_summary,
        "--candidate",
        candidate_summary,
        "--out",
        out_path,
    ]
    if cfg.max_root_ade_regression is not None:
        cmd.extend(
            [
                "--max-root-ade-regression",
                _threshold_str(float(cfg.max_root_ade_regression)),
            ]
        )
    if cfg.max_root_fde_regression is not None:
        cmd.extend(
            [
                "--max-root-fde-regression",
                _threshold_str(float(cfg.max_root_fde_regression)),
            ]
        )
    if cfg.max_path_arc_regression is not None:
        cmd.extend(
            [
                "--max-path-arc-regression",
                _threshold_str(float(cfg.max_path_arc_regression)),
            ]
        )
    if cfg.max_yaw_error_regression is not None:
        cmd.extend(
            [
                "--max-yaw-error-regression",
                _threshold_str(float(cfg.max_yaw_error_regression)),
            ]
        )
    if cfg.max_jitter_regression is not None:
        cmd.extend(
            [
                "--max-jitter-regression",
                _threshold_str(float(cfg.max_jitter_regression)),
            ]
        )
    if cfg.max_foot_skating_regression is not None:
        cmd.extend(
            [
                "--max-foot-skating-regression",
                _threshold_str(float(cfg.max_foot_skating_regression)),
            ]
        )
    if cfg.max_root_jump_regression is not None:
        cmd.extend(
            [
                "--max-root-jump-regression",
                _threshold_str(float(cfg.max_root_jump_regression)),
            ]
        )
    return cmd


def _stream_eval_entry(cfg: SmokeRunConfig, *, ckpt: str, run_name: str, probe_tag: str) -> dict:
    cmd = build_stream_eval_command(
        cfg,
        ckpt=ckpt,
        run_name=run_name,
        probe_tag=probe_tag,
    )
    summary = str(Path(_stream_eval_out_dir(cfg)) / run_name / "summary.json")
    return {
        "run_name": run_name,
        "probe_tag": probe_tag,
        "summary": summary,
        "command": _quote_cmd(cmd),
        "argv": cmd,
    }


def _comparison_entry(
    cfg: SmokeRunConfig,
    *,
    stage: str,
    baseline_summary: str,
    candidate_summary: str,
) -> dict:
    out_path = str(Path(_stream_eval_out_dir(cfg)) / f"{stage}_comparison.json")
    cmd = build_report_command(
        cfg,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        out_path=out_path,
    )
    return {
        "summary": out_path,
        "command": _quote_cmd(cmd),
        "argv": cmd,
    }


def build_validation_manifest(base: SmokeRunConfig) -> dict:
    plan = build_validation_plan(base)
    baseline_eval = _stream_eval_entry(
        base,
        ckpt=base.resume_ckpt,
        run_name="baseline_full_prefix",
        probe_tag="baseline_full_prefix",
    )
    stages = []
    for entry in plan:
        cmd = build_train_command(entry.config)
        candidate_eval = _stream_eval_entry(
            base,
            ckpt=CANDIDATE_CKPT_PLACEHOLDER,
            run_name=entry.stage,
            probe_tag=entry.stage,
        )
        comparison = _comparison_entry(
            base,
            stage=entry.stage,
            baseline_summary=baseline_eval["summary"],
            candidate_summary=candidate_eval["summary"],
        )
        stages.append(
            {
                "stage": entry.stage,
                "description": entry.description,
                "config": asdict(entry.config),
                "command": _quote_cmd(cmd),
                "argv": cmd,
                "missing_paths": find_missing_paths(entry.config),
                "expected_candidate_ckpt_glob": str(
                    Path(entry.config.output_dir) / "**" / "*.ckpt"
                ),
                "post_training_eval": {
                    "candidate_ckpt_placeholder": CANDIDATE_CKPT_PLACEHOLDER,
                    "candidate_eval": candidate_eval,
                    "comparison": comparison,
                },
            }
        )
    return {
        "kind": "ldf_stream_training_validation_plan",
        "base": asdict(base),
        "stream_eval": {
            "out_dir": _stream_eval_out_dir(base),
            "baseline": baseline_eval,
        },
        "stages": stages,
    }


def write_manifest(path: str, payload: dict) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def find_missing_paths(cfg: SmokeRunConfig) -> list[str]:
    required = [
        cfg.config,
        cfg.resume_ckpt,
        cfg.vae_ckpt,
        cfg.raw_data_root,
        cfg.train_split,
        cfg.val_split,
        cfg.z_stats_dir,
    ]
    if cfg.raw_data_root:
        required.append(
            str(Path(cfg.raw_data_root) / "HumanML3D" / "t5_text_embeddings.pt")
        )
    return [path for path in required if path and not Path(path).exists()]


def parse_args(argv: list[str] | None = None) -> SmokeRunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ldf.yaml")
    parser.add_argument("--python", default=os.environ.get("PY", sys.executable))
    parser.add_argument("--resume-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument(
        "--raw-data-root",
        "--raw-data-dir",
        dest="raw_data_root",
        required=True,
        help=(
            "Path to the raw_data root. This is the parent directory containing "
            "HumanML3D/, not the HumanML3D directory itself."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/stream_training_smoke")
    parser.add_argument("--exp-name", default="ldf_stream_training_smoke")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--devices", default="1")
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument(
        "--sample-policy",
        choices=("variable_history", "fixed_window"),
        default="variable_history",
        help="Deprecated legacy option; stream-training v2 always uses active-left window sampling.",
    )
    parser.add_argument(
        "--motion-aux-loss",
        choices=("latent_only", "full_prefix", "disabled"),
        default="latent_only",
    )
    parser.add_argument("--context-tokens", type=int, default=30)
    parser.add_argument("--min-history-tokens", type=int, default=8)
    parser.add_argument("--horizon-tokens", type=int, default=20)
    parser.add_argument("--history-tokens-min", type=int, default=0)
    parser.add_argument("--history-tokens-max", default="auto")
    parser.add_argument("--horizon-tokens-min", type=int, default=5)
    parser.add_argument("--horizon-tokens-max", type=int, default=25)
    parser.add_argument("--z-stats-dir", default="")
    parser.add_argument("--train-split", default="")
    parser.add_argument("--val-split", default="")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument(
        "--validation-plan",
        action="store_true",
        help=(
            "Print or run the standard validation matrix: latent smoke, "
            "latent overfit, and full-prefix overfit."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional JSON manifest path recording commands, configs, and preflight results.",
    )
    parser.add_argument(
        "--eval-out-dir",
        default="",
        help=(
            "Optional output root for post-training stream eval commands in the "
            "manifest. Defaults to <output-dir>/stream_eval."
        ),
    )
    parser.add_argument("--eval-max-samples", type=int, default=5)
    parser.add_argument("--eval-num-runs", type=int, default=1)
    parser.add_argument(
        "--eval-stream-mode",
        choices=("stream_generate", "stream_generate_step"),
        default="stream_generate_step",
    )
    parser.add_argument("--max-root-ade-regression", type=float, default=0.05)
    parser.add_argument("--max-root-fde-regression", type=float, default=None)
    parser.add_argument("--max-path-arc-regression", type=float, default=None)
    parser.add_argument("--max-yaw-error-regression", type=float, default=None)
    parser.add_argument("--max-jitter-regression", type=float, default=None)
    parser.add_argument("--max-foot-skating-regression", type=float, default=None)
    parser.add_argument("--max-root-jump-regression", type=float, default=None)
    ns = parser.parse_args(argv)
    return SmokeRunConfig(**vars(ns))


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    if cfg.validation_plan:
        plan = build_validation_plan(cfg)
        for entry in plan:
            print(f"# {entry.stage}: {entry.description}")
            print(_quote_cmd(build_train_command(entry.config)))
        if cfg.manifest:
            write_manifest(cfg.manifest, build_validation_manifest(cfg))
        if cfg.print_only:
            return 0
        missing: list[str] = []
        seen: set[str] = set()
        for entry in plan:
            for path in find_missing_paths(entry.config):
                if path not in seen:
                    seen.add(path)
                    missing.append(path)
        if missing:
            print("Missing required paths:", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            return 2
        if cfg.dry_run:
            return 0
        for entry in plan:
            rc = subprocess.call(build_train_command(entry.config))
            if rc:
                return rc
        return 0

    cmd = build_train_command(cfg)
    print(_quote_cmd(cmd))
    if cfg.manifest:
        write_manifest(
            cfg.manifest,
            {
                "kind": "ldf_stream_training_smoke",
                "base": asdict(cfg),
                "command": _quote_cmd(cmd),
                "argv": cmd,
                "missing_paths": find_missing_paths(cfg),
            },
        )
    if cfg.print_only:
        return 0
    missing = find_missing_paths(cfg)
    if missing:
        print("Missing required paths:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2
    if cfg.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
