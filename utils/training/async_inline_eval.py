from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from lightning.pytorch.utilities import rank_zero_info

from .inline_eval_runtime import get_test_probe_tags
from .module_step import compute_checkpoint_step_info, compute_step_semantics
from .test_probes import build_test_probe_tags


def async_test_mode_enabled(cfg) -> bool:
    return str(cfg.get("validation", {}).get("test_mode", "inline")) == "async"


def get_async_eval_root(save_dir: str | Path) -> Path:
    return Path(save_dir) / "async_eval"


def get_run_config_path(save_dir: str | Path, exp_name: str) -> Path:
    return Path(save_dir) / "sanity_check" / f"{exp_name}.yaml"


def get_async_eval_ckpt_path(save_dir: str | Path, step: int) -> Path:
    return get_async_eval_root(save_dir) / "ckpts" / f"step_{step:06d}.ckpt"


def get_async_eval_request_path(save_dir: str | Path, step: int) -> Path:
    return get_async_eval_root(save_dir) / "requests" / f"step_{step:06d}.json"


def emit_async_test_request(module):
    trainer = getattr(module, "trainer", None)
    if trainer is None:
        return
    if trainer.sanity_checking or not async_test_mode_enabled(module.cfg):
        return
    if module.global_step <= 0:
        return
    test_steps = int(module.cfg.validation.test_steps)
    if test_steps <= 0 or module.global_step % test_steps != 0:
        return

    step_semantics = compute_step_semantics(module)
    step_info = compute_checkpoint_step_info(module)
    step = int(step_semantics.absolute_step)
    ckpt_path = get_async_eval_ckpt_path(module.cfg.save_dir, step)
    request_path = get_async_eval_request_path(module.cfg.save_dir, step)
    if ckpt_path.exists() and request_path.exists():
        return

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.parent.mkdir(parents=True, exist_ok=True)

    trainer.strategy.barrier()
    trainer.save_checkpoint(str(ckpt_path), weights_only=False)
    trainer.strategy.barrier()

    if trainer.is_global_zero:
        run_dir = Path(module.cfg.save_dir).resolve()
        config_path = get_run_config_path(module.cfg.save_dir, module.cfg.exp_name).resolve()
        payload = {
            "step": step,
            "step_tag": step_info.step_tag,
            "run_dir": str(run_dir),
            "artifact_root": str(run_dir),
            "config_path": str(config_path),
            "ckpt_path": str(ckpt_path),
            "probe_tags": get_test_probe_tags(module),
            "created_at": time.time(),
            "test_mode": "async",
        }
        tmp_path = request_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, request_path)
        rank_zero_info(
            f"[async-test] emitted request step={step} ckpt={ckpt_path.name}"
        )


def maybe_launch_async_eval_watcher(cfg, save_dir: str):
    if not async_test_mode_enabled(cfg):
        return None
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    watcher_script = Path(__file__).parents[2] / "eval" / "eval_watcher.py"
    eval_device = str(cfg.get("async_eval_device", "0"))
    log_path = Path(save_dir) / "async_eval" / "watcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = eval_device
    cmd = [
        sys.executable,
        str(watcher_script),
        "--run_dir",
        str(save_dir),
        "--poll_interval_sec",
        "30",
    ]
    log_file = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    rank_zero_info(
        f"[async-eval] watcher launched pid={proc.pid} device={eval_device} log={log_path}"
    )
    return proc


def emit_resume_ckpt_eval_request(cfg, save_dir: str, resume_ckpt: str):
    """Emit an initial async eval request for the resume checkpoint at startup.

    Reads the checkpoint's global_step to set the correct step tag, then
    writes a request JSON so the watcher evaluates the resume model as a
    baseline before any training checkpoints are emitted.
    """
    if not resume_ckpt:
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return
    try:
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        global_step = int(ckpt.get("global_step", 0))
    except Exception as exc:
        rank_zero_info(
            f"[async-eval] could not read global_step from resume ckpt ({exc!r}); "
            f"skipping initial eval request"
        )
        return

    request_dir = get_async_eval_root(save_dir) / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"step_{global_step:06d}.json"

    if request_path.exists():
        return

    run_dir = Path(save_dir).resolve()
    config_path = get_run_config_path(save_dir, cfg.exp_name).resolve()
    probe_tags = build_test_probe_tags(cfg)

    payload = {
        "step": global_step,
        "step_tag": f"step_{global_step:06d}",
        "run_dir": str(run_dir),
        "artifact_root": str(run_dir),
        "config_path": str(config_path),
        "ckpt_path": str(Path(resume_ckpt).resolve()),
        "probe_tags": probe_tags,
        "created_at": time.time(),
        "test_mode": "async",
    }
    tmp_path = request_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, request_path)
    rank_zero_info(
        f"[async-eval] emitted initial eval request for resume ckpt "
        f"step={global_step} ckpt={resume_ckpt}"
    )
