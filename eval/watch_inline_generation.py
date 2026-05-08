import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch async inline-eval requests and evaluate them outside the training loop."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--poll_interval_sec", type=float, default=30.0)
    parser.add_argument("--min_request_age_sec", type=float, default=30.0)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    return parser.parse_args()


def _discover_config(run_dir: Path) -> Path:
    sanity_dir = run_dir / "sanity_check"
    yaml_files = sorted(sanity_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No yaml found under {sanity_dir}")
    return yaml_files[0]


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


def _expected_inline_summaries_exist(
    artifact_root: Path, step_tag: str, probe_tags: list[str]
) -> bool:
    if probe_tags:
        for probe_tag in probe_tags:
            pattern = f"*/metrics/{probe_tag}/{step_tag}/summary.json"
            if not any(artifact_root.glob(pattern)):
                return False
        return True
    return any(artifact_root.glob(f"*/metrics/*/{step_tag}/summary.json"))


def _iter_pending_requests(request_dir: Path, state: dict[str, Any], min_request_age_sec: float):
    completed = set(state.get("completed", []))
    now = time.time()
    requests = []
    for request_path in sorted(request_dir.glob("step_*.json")):
        request_key = str(request_path.resolve())
        if request_key in completed:
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


def _build_eval_command(
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


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    config_path = Path(args.config).resolve() if args.config else _discover_config(run_dir)
    project_root = Path(__file__).parent.parent.resolve()
    runner_script = Path(__file__).with_name("run_inline_generation_eval.py").resolve()
    async_root = run_dir / "async_eval"
    request_dir = async_root / "requests"
    state_path = async_root / "watcher_state_inline.json"
    state = _load_state(state_path)

    print(f"[async-inline-eval] run_dir={run_dir}")
    print(f"[async-inline-eval] config={config_path}")
    print(f"[async-inline-eval] request_dir={request_dir}")
    print(f"[async-inline-eval] poll_interval_sec={args.poll_interval_sec}")
    print(f"[async-inline-eval] min_request_age_sec={args.min_request_age_sec}")

    while True:
        pending = _iter_pending_requests(request_dir, state, args.min_request_age_sec)
        if not pending:
            if args.once:
                print("[async-inline-eval] no pending requests; exit.")
                return
            time.sleep(args.poll_interval_sec)
            continue

        for step, request_path, payload in pending:
            request_key = str(request_path.resolve())
            artifact_root = Path(payload.get("artifact_root") or payload.get("run_dir") or run_dir).resolve()
            config_path_for_request = (
                Path(payload["config_path"]).resolve()
                if payload.get("config_path")
                else config_path
            )
            step_tag = str(payload.get("step_tag") or f"step_{step:06d}")
            probe_tags = [str(tag) for tag in payload.get("probe_tags", [])]
            if _expected_inline_summaries_exist(artifact_root, step_tag, probe_tags):
                state.setdefault("completed", []).append(request_key)
                _save_state(state_path, state)
                print(f"[async-inline-eval] results already exist for step={step}; mark completed.")
                continue

            cmd = _build_eval_command(
                runner_script=runner_script,
                config_path=config_path_for_request,
                request_payload=payload,
                artifact_root=artifact_root,
                default_root_dir=async_root,
                args=args,
            )
            print(f"[async-inline-eval] evaluating step={step} ckpt={Path(payload['ckpt_path']).name}")
            print(f"[async-inline-eval] cmd={' '.join(cmd)}")
            env = os.environ.copy()
            if args.cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            result = subprocess.run(cmd, cwd=project_root, env=env)
            if result.returncode != 0:
                print(
                    f"[async-inline-eval] evaluation failed for step={step} "
                    f"with code={result.returncode}"
                )
                if args.once:
                    raise SystemExit(result.returncode)
                break

            if _expected_inline_summaries_exist(artifact_root, step_tag, probe_tags):
                state.setdefault("completed", []).append(request_key)
                _save_state(state_path, state)
                print(f"[async-inline-eval] finished step={step}")
            else:
                print(f"[async-inline-eval] step={step} completed without summary; keep pending.")
                break

        if args.once:
            return
        time.sleep(args.poll_interval_sec)


if __name__ == "__main__":
    main()
