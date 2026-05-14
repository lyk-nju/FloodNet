"""Async eval watcher: polls for evaluation requests and dispatches them to the
appropriate runner outside the training loop, with GPU isolation.

Supports two modes:
  --mode inline      Polls for request JSON files emitted by emit_eval_request.
                     Launches run_eval.py for each pending request.
  --mode generation  Polls for checkpoint files directly.
                     Launches eval_generation_metrics.py for each pending checkpoint.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


THREAD_LIMIT_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _subprocess_env(cuda_visible_devices: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_LIMIT_ENV)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def _discover_config(run_dir: Path) -> Path:
    sanity_dir = run_dir / "sanity_check"
    yaml_files = sorted(sanity_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No yaml found under {sanity_dir}")
    return yaml_files[0]


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"completed": [], "failed": {}}
    try:
        with open(state_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("completed"), list):
            if not isinstance(data.get("failed"), dict):
                data["failed"] = {}
            return data
    except Exception:
        pass
    return {"completed": [], "failed": {}}


def _save_state(state_path: Path, state: dict[str, Any]):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


# ------------------------------------------------------------------
# Inline mode
# ------------------------------------------------------------------


def _iter_inline_requests(
    request_dir: Path,
    state: dict[str, Any],
    min_request_age_sec: float,
    max_failures: int,
):
    completed = set(state.get("completed", []))
    failed = state.get("failed", {})
    now = time.time()
    requests = []
    for request_path in sorted(request_dir.glob("step_*.json")):
        request_key = str(request_path.resolve())
        if request_key in completed:
            continue
        if max_failures > 0 and int(failed.get(request_key, 0)) >= max_failures:
            continue
        if now - request_path.stat().st_mtime < min_request_age_sec:
            continue
        try:
            with open(request_path, "r") as f:
                payload = json.load(f)
            step = int(payload["step"])
            ckpt_path = Path(payload["ckpt_path"])
        except Exception:
            continue
        if not ckpt_path.exists():
            continue
        requests.append((step, request_path, payload))
    requests.sort(key=lambda x: x[0])
    return requests


def _build_inline_eval_command(
    runner_script: Path,
    config_path: Path,
    request_payload: dict[str, Any],
    artifact_root: Path,
    default_root_dir: Path,
    args,
) -> list[str]:
    cmd = [
        sys.executable,
        str(runner_script),
        "--config",
        str(config_path),
        "--ckpt",
        str(request_payload["ckpt_path"]),
        "--artifact_root",
        str(artifact_root),
        "--default_root_dir",
        str(default_root_dir),
        "--devices",
        str(args.devices),
    ]
    if args.accelerator is not None:
        cmd.extend(["--accelerator", args.accelerator])
    return cmd


def _run_inline_mode(args, run_dir: Path, config_path: Path, project_root: Path):
    runner_script = (project_root / "eval" / "run_eval.py").resolve()
    async_root = run_dir / "async_eval"
    request_dir = async_root / "requests"
    state_path = async_root / "watcher_state_inline.json"
    state = _load_state(state_path)

    print(f"[eval-watcher|inline] run_dir={run_dir}")
    print(f"[eval-watcher|inline] config={config_path}")
    print(f"[eval-watcher|inline] request_dir={request_dir}")

    training_done_marker = async_root / "training_done"
    last_activity = time.time()

    while True:
        pending = _iter_inline_requests(
            request_dir, state, args.min_request_age_sec, args.max_failures
        )
        if not pending:
            idle_sec = time.time() - last_activity
            # Drain mode: training_done exists but requests too young;
            # wait and retry only if there are unfinished requests.
            if training_done_marker.exists():
                completed = set(state.get("completed", []))
                failed = state.get("failed", {})
                unfinished = []
                for request_path in request_dir.glob("step_*.json"):
                    request_key = str(request_path.resolve())
                    if request_key in completed:
                        continue
                    if (
                        args.max_failures > 0
                        and int(failed.get(request_key, 0)) >= args.max_failures
                    ):
                        continue
                    unfinished.append(request_path)
                if not unfinished:
                    print("[eval-watcher|inline] no pending requests; exit.")
                    return
                time.sleep(args.poll_interval_sec)
                continue
            if args.once:
                print("[eval-watcher|inline] no pending requests; exit.")
                return
            if args.idle_timeout_min > 0 and idle_sec > args.idle_timeout_min * 60:
                print(
                    f"[eval-watcher|inline] idle for {idle_sec:.0f}s "
                    f"(timeout={args.idle_timeout_min}min); exit."
                )
                return
            time.sleep(args.poll_interval_sec)
            continue
        last_activity = time.time()

        for step, request_path, payload in pending:
            request_key = str(request_path.resolve())
            artifact_root = Path(
                payload.get("artifact_root") or payload.get("run_dir") or run_dir
            ).resolve()
            config_path_for_request = (
                Path(payload["config_path"]).resolve()
                if payload.get("config_path")
                else config_path
            )
            step_tag = str(payload.get("step_tag") or f"step_{step:06d}")
            probe_tags = [str(tag) for tag in payload.get("probe_tags", [])]

            cmd = _build_inline_eval_command(
                runner_script=runner_script,
                config_path=config_path_for_request,
                request_payload=payload,
                artifact_root=artifact_root,
                default_root_dir=async_root,
                args=args,
            )
            print(
                f"[eval-watcher|inline] evaluating step={step} "
                f"ckpt={Path(payload['ckpt_path']).name}"
            )
            print(f"[eval-watcher|inline] cmd={' '.join(cmd)}")
            env = _subprocess_env(args.cuda_visible_devices)
            result = subprocess.run(cmd, cwd=project_root, env=env)
            if result.returncode != 0:
                failed = state.setdefault("failed", {})
                failed[request_key] = int(failed.get(request_key, 0)) + 1
                _save_state(state_path, state)
                print(
                    f"[eval-watcher|inline] evaluation failed for step={step} "
                    f"with code={result.returncode} "
                    f"(failures={failed[request_key]}/{args.max_failures})"
                )
                if args.once:
                    raise SystemExit(result.returncode)
                break

            if _expected_inline_summaries_exist(artifact_root, step_tag, probe_tags):
                state.setdefault("completed", []).append(request_key)
                state.setdefault("failed", {}).pop(request_key, None)
                _save_state(state_path, state)
                print(f"[eval-watcher|inline] finished step={step}")
            else:
                print(
                    f"[eval-watcher|inline] step={step} completed without summary; "
                    f"keep pending."
                )
                break

        if args.once:
            return
        time.sleep(args.poll_interval_sec)


def _expected_inline_summaries_exist(
    artifact_root: Path, step_tag: str, probe_tags: list[str]
) -> bool:
    """Return True once inline eval has written all expected summary files."""
    expected_probe_tags = probe_tags or [
        path.parent.name
        for path in artifact_root.glob(f"*/metrics/*/{step_tag}/summary.json")
    ]
    if not expected_probe_tags:
        return False

    for probe_tag in expected_probe_tags:
        matches = list(
            artifact_root.glob(f"*/metrics/{probe_tag}/{step_tag}/summary.json")
        )
        if not matches:
            return False
    return True


# ------------------------------------------------------------------
# Generation mode
# ------------------------------------------------------------------

def _parse_step_from_ckpt(ckpt_path: Path) -> int | None:
    stem = ckpt_path.stem
    if "step=" in stem:
        try:
            return int(stem.split("step=")[-1])
        except ValueError:
            return None
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None


def _metrics_exist(step_dir: Path) -> bool:
    return any(step_dir.glob("**/metrics.json"))


def _resolve_eval_defaults(config_path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    validation = cfg.get("validation", {})
    return {
        "num_runs": int(validation.get("eval_num_runs", 1)),
        "seg_size": int(validation.get("eval_seg_size", 20)),
        "forward_control_loss": bool(
            validation.get("eval_forward_control_loss", True)
        ),
        "seed": int(cfg.get("seed", 1234)),
    }


def _resolve_probe_specs(config_path: Path) -> list[tuple[str, list[str]]]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    probe_cfg = cfg.get("data", {}).get("test_probe_meta_paths", None)
    if probe_cfg:
        return [
            (str(probe_tag), list(meta_paths))
            for probe_tag, meta_paths in probe_cfg.items()
        ]
    return [("test", list(cfg.get("data", {}).get("test_meta_paths", [])))]


def _build_generation_eval_command(
    eval_script: Path,
    config_path: Path,
    ckpt_path: Path,
    probe_tag: str,
    meta_paths: list[str],
    step_dir: Path,
    args,
    defaults: dict[str, Any],
) -> list[str]:
    cmd = [
        sys.executable,
        str(eval_script),
        "--config",
        str(config_path),
        "--ckpt",
        str(ckpt_path),
        "--out_dir",
        str(step_dir),
        "--probe_tag",
        probe_tag,
        "--num_runs",
        str(args.num_runs if args.num_runs is not None else defaults["num_runs"]),
        "--seg_size",
        str(args.seg_size if args.seg_size is not None else defaults["seg_size"]),
        "--seed",
        str(args.seed if args.seed is not None else defaults["seed"]),
    ]
    if meta_paths:
        cmd.extend(["--meta_paths", *meta_paths])
    if args.max_batches > 0:
        cmd.extend(["--max_batches", str(args.max_batches)])
    if args.no_ema:
        cmd.append("--no_ema")
    forward_ctrl_loss = (
        args.forward_control_loss
        if args.forward_control_loss is not None
        else defaults["forward_control_loss"]
    )
    if forward_ctrl_loss:
        cmd.append("--forward_control_loss")
    if args.t2m_metric:
        cmd.append("--t2m_metric")
    return cmd


def _iter_generation_checkpoints(
    run_dir: Path, state: dict[str, Any], min_ckpt_age_sec: float
):
    completed = set(state.get("completed", []))
    now = time.time()
    ckpt_paths = []
    for ckpt_path in sorted(run_dir.glob("step_*.ckpt")):
        ckpt_key = str(ckpt_path.resolve())
        if ckpt_key in completed:
            continue
        if now - ckpt_path.stat().st_mtime < min_ckpt_age_sec:
            continue
        step = _parse_step_from_ckpt(ckpt_path)
        if step is None:
            continue
        ckpt_paths.append((step, ckpt_path))
    ckpt_paths.sort(key=lambda x: x[0])
    return ckpt_paths


def _run_generation_mode(args, run_dir: Path, config_path: Path, project_root: Path):
    defaults = _resolve_eval_defaults(config_path)
    probe_specs = _resolve_probe_specs(config_path)
    eval_script = (project_root / "eval" / "eval_generation_metrics.py").resolve()
    async_root = run_dir / "async_eval"
    state_path = async_root / "watcher_state.json"
    state = _load_state(state_path)

    training_done_marker = async_root / "training_done"
    last_activity = time.time()

    print(f"[eval-watcher|generation] run_dir={run_dir}")
    print(f"[eval-watcher|generation] config={config_path}")
    print(f"[eval-watcher|generation] probes={probe_specs}")

    while True:
        pending = _iter_generation_checkpoints(
            run_dir, state, args.min_ckpt_age_sec
        )
        if not pending:
            idle_sec = time.time() - last_activity
            # Drain: training_done but young checkpoints remain; wait.
            if training_done_marker.exists() and list(run_dir.glob("step_*.ckpt")):
                time.sleep(args.poll_interval_sec)
                continue
            if args.once or training_done_marker.exists():
                print("[eval-watcher|generation] no pending checkpoints; exit.")
                return
            if args.idle_timeout_min > 0 and idle_sec > args.idle_timeout_min * 60:
                print(
                    f"[eval-watcher|generation] idle for {idle_sec:.0f}s "
                    f"(timeout={args.idle_timeout_min}min); exit."
                )
                return
            time.sleep(args.poll_interval_sec)
            continue
        last_activity = time.time()

        for step, ckpt_path in pending:
            ckpt_key = str(ckpt_path.resolve())
            all_probe_done = True
            for probe_tag, meta_paths in probe_specs:
                step_dir = async_root / f"step_{step:06d}" / probe_tag
                if _metrics_exist(step_dir):
                    print(
                        f"[eval-watcher|generation] metrics already exist for "
                        f"step={step} probe={probe_tag}; skip."
                    )
                    continue

                all_probe_done = False
                step_dir.mkdir(parents=True, exist_ok=True)
                cmd = _build_generation_eval_command(
                    eval_script=eval_script,
                    config_path=config_path,
                    ckpt_path=ckpt_path,
                    probe_tag=probe_tag,
                    meta_paths=meta_paths,
                    step_dir=step_dir,
                    args=args,
                    defaults=defaults,
                )
                print(
                    f"[eval-watcher|generation] evaluating step={step} "
                    f"probe={probe_tag} ckpt={ckpt_path.name}"
                )
                print(f"[eval-watcher|generation] cmd={' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=project_root, env=_subprocess_env())
                if result.returncode != 0:
                    print(
                        f"[eval-watcher|generation] evaluation failed for "
                        f"step={step} probe={probe_tag} "
                        f"with code={result.returncode}"
                    )
                    if args.once:
                        raise SystemExit(result.returncode)
                    all_probe_done = False
                    break

            done = all_probe_done or all(
                _metrics_exist(
                    async_root / f"step_{step:06d}" / probe_tag
                )
                for probe_tag, _ in probe_specs
            )
            if done:
                state.setdefault("completed", []).append(ckpt_key)
                _save_state(state_path, state)
                print(f"[eval-watcher|generation] finished step={step}")
            else:
                break

        if args.once:
            return
        time.sleep(args.poll_interval_sec)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Async eval watcher for FloodNet."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inline", "generation"],
        default="inline",
        help="Evaluation mode: 'inline' (request JSONs) or 'generation' (ckpt files).",
    )
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Training run directory."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config yaml. Defaults to run_dir/sanity_check/*.yaml.",
    )
    parser.add_argument(
        "--poll_interval_sec", type=float, default=30.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--min_request_age_sec", type=float, default=30.0,
        help="Only process requests/checkpoints older than this age.",
    )
    parser.add_argument("--min_ckpt_age_sec", type=float, default=30.0)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--idle_timeout_min",
        type=float,
        default=0.0,
        help="Exit if idle for this many minutes. 0 = never timeout (rely on training_done marker).",
    )
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument(
        "--max_failures",
        type=int,
        default=3,
        help="Skip an inline request after this many failed eval attempts. 0 = retry forever.",
    )
    # Generation-mode options
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--seg_size", type=int, default=None)
    parser.add_argument("--forward_control_loss", action="store_true", default=None)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--t2m_metric", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    # Clean up pidfile when watcher exits.
    pidfile = run_dir / "async_eval" / "watcher.pid"
    import atexit
    atexit.register(lambda: pidfile.unlink(missing_ok=True))

    config_path = (
        Path(args.config).resolve() if args.config else _discover_config(run_dir)
    )
    project_root = Path(__file__).parent.parent.resolve()

    if args.mode == "inline":
        _run_inline_mode(args, run_dir, config_path, project_root)
    else:
        _run_generation_mode(args, run_dir, config_path, project_root)


if __name__ == "__main__":
    main()

