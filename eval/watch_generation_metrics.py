import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch a training run directory and evaluate new checkpoints asynchronously."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Training run directory.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config yaml to use for evaluation. Defaults to run_dir/sanity_check/*.yaml.",
    )
    parser.add_argument(
        "--poll_interval_sec",
        type=float,
        default=30.0,
        help="Watcher polling interval in seconds.",
    )
    parser.add_argument(
        "--min_ckpt_age_sec",
        type=float,
        default=30.0,
        help="Only evaluate checkpoints older than this age to avoid half-written files.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=None,
        help="Override validation.eval_num_runs.",
    )
    parser.add_argument(
        "--seg_size",
        type=int,
        default=None,
        help="Override validation.eval_seg_size.",
    )
    parser.add_argument(
        "--forward_control_loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override validation.eval_forward_control_loss.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Pass through to eval_generation_metrics.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override evaluation seed. Defaults to config seed.",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Pass through to eval_generation_metrics.py.",
    )
    parser.add_argument(
        "--t2m_metric",
        action="store_true",
        help="Pass through to eval_generation_metrics.py.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Evaluate current pending checkpoints once, then exit.",
    )
    return parser.parse_args()


def _discover_config(run_dir: Path) -> Path:
    sanity_dir = run_dir / "sanity_check"
    yaml_files = sorted(sanity_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No yaml found under {sanity_dir}")
    return yaml_files[0]


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


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"completed": []}
    try:
        with open(state_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("completed"), list):
            return data
    except Exception:
        pass
    return {"completed": []}


def _save_state(state_path: Path, state: dict[str, Any]):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def _metrics_exist(step_dir: Path) -> bool:
    return any(step_dir.glob("**/metrics.json"))


def _resolve_eval_defaults(config_path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    validation = cfg.get("validation", {})
    return {
        "num_runs": int(validation.get("eval_num_runs", 1)),
        "seg_size": int(validation.get("eval_seg_size", 20)),
        "forward_control_loss": bool(validation.get("eval_forward_control_loss", True)),
        "seed": int(cfg.get("seed", 1234)),
    }


def _resolve_probe_specs(config_path: Path) -> list[tuple[str, list[str]]]:
    cfg = OmegaConf.load(config_path)
    probe_cfg = cfg.get("data", {}).get("test_probe_meta_paths", None)
    if probe_cfg:
        return [(str(probe_tag), list(meta_paths)) for probe_tag, meta_paths in probe_cfg.items()]
    return [("test", list(cfg.get("data", {}).get("test_meta_paths", [])))]


def _build_eval_command(
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


def _iter_pending_checkpoints(run_dir: Path, state: dict[str, Any], min_ckpt_age_sec: float):
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


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    config_path = Path(args.config).resolve() if args.config else _discover_config(run_dir)
    defaults = _resolve_eval_defaults(config_path)
    probe_specs = _resolve_probe_specs(config_path)
    eval_script = Path(__file__).with_name("eval_generation_metrics.py").resolve()
    project_root = eval_script.parent.parent
    async_root = run_dir / "async_eval"
    state_path = async_root / "watcher_state.json"
    state = _load_state(state_path)

    print(f"[async-eval] run_dir={run_dir}")
    print(f"[async-eval] config={config_path}")
    print(f"[async-eval] probes={probe_specs}")
    print(f"[async-eval] poll_interval_sec={args.poll_interval_sec}")
    print(f"[async-eval] min_ckpt_age_sec={args.min_ckpt_age_sec}")

    while True:
        pending = _iter_pending_checkpoints(run_dir, state, args.min_ckpt_age_sec)
        if not pending:
            if args.once:
                print("[async-eval] no pending checkpoints; exit.")
                return
            time.sleep(args.poll_interval_sec)
            continue

        for step, ckpt_path in pending:
            ckpt_key = str(ckpt_path.resolve())
            all_probe_done = True
            for probe_tag, meta_paths in probe_specs:
                step_dir = async_root / f"step_{step:06d}" / probe_tag
                if _metrics_exist(step_dir):
                    print(f"[async-eval] metrics already exist for step={step} probe={probe_tag}; skip.")
                    continue

                all_probe_done = False
                step_dir.mkdir(parents=True, exist_ok=True)
                cmd = _build_eval_command(
                    eval_script=eval_script,
                    config_path=config_path,
                    ckpt_path=ckpt_path,
                    probe_tag=probe_tag,
                    meta_paths=meta_paths,
                    step_dir=step_dir,
                    args=args,
                    defaults=defaults,
                )
                print(f"[async-eval] evaluating step={step} probe={probe_tag} ckpt={ckpt_path.name}")
                print(f"[async-eval] cmd={' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=project_root)
                if result.returncode != 0:
                    print(
                        f"[async-eval] evaluation failed for step={step} probe={probe_tag} "
                        f"with code={result.returncode}"
                    )
                    if args.once:
                        raise SystemExit(result.returncode)
                    all_probe_done = False
                    break

            if all_probe_done or all(_metrics_exist(async_root / f"step_{step:06d}" / probe_tag) for probe_tag, _ in probe_specs):
                state.setdefault("completed", []).append(ckpt_key)
                _save_state(state_path, state)
                print(f"[async-eval] finished step={step}")
            else:
                break

        if args.once:
            return

        time.sleep(args.poll_interval_sec)


if __name__ == "__main__":
    main()
