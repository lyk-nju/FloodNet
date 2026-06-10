from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from eval.eval_watcher import (
    _build_stream_eval_cmd,
    _mark_completed_without_summary,
    _should_mark_completed_without_summary,
    _stream_summary_exists,
)
from utils.training.async_inline_eval import build_stream_eval_request
from utils.training.async_inline_eval import emit_resume_eval


def test_mark_completed_without_summary_is_terminal_and_deduped():
    state = {"completed": [], "failed": {"request.json": 2}}

    _mark_completed_without_summary(state, "request.json")
    _mark_completed_without_summary(state, "request.json")

    assert state["completed"] == ["request.json"]
    assert state["completed_without_summary"] == ["request.json"]
    assert state["failed"] == {}


def test_stream_eval_request_requires_summary_file():
    assert _should_mark_completed_without_summary({}) is True
    assert _should_mark_completed_without_summary({"stream_eval": {"enabled": False}}) is True
    assert _should_mark_completed_without_summary({"stream_eval": {"enabled": True}}) is False


def test_build_stream_eval_request_uses_validation_config(tmp_path):
    cfg = {
        "test_vae_ckpt": "/ckpt/vae.ckpt",
        "validation": {
            "stream_eval": {
                "enabled": True,
                "stream_mode": "stream_generate_step",
                "max_samples": 7,
                "num_runs": 2,
                "max_batches": 3,
                "compute_offline_baseline": True,
                "compute_no_traj_baseline": False,
            }
        },
    }

    request = build_stream_eval_request(cfg, tmp_path, "step_000500")

    assert request == {
        "enabled": True,
        "stream_mode": "stream_generate_step",
        "out_dir": str(tmp_path / "async_eval" / "stream_eval"),
        "run_name": "step_000500",
        "probe_tag": "step_000500",
        "max_samples": 7,
        "num_runs": 2,
        "max_batches": 3,
        "vae_ckpt": "/ckpt/vae.ckpt",
        "compute_offline_baseline": True,
        "compute_no_traj_baseline": False,
    }


def test_emit_resume_eval_writes_stream_eval_request(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    ckpt_path = tmp_path / "resume.ckpt"
    torch.save({"global_step": 500}, ckpt_path)
    cfg = OmegaConf.create(
        {
            "exp_name": "ldf_run",
            "test_vae_ckpt": "/ckpt/vae.ckpt",
            "data": {"test_probe_meta_paths": {"test": ["val.txt"]}},
            "validation": {
                "stream_eval": {
                    "enabled": True,
                    "max_samples": 7,
                    "num_runs": 2,
                }
            },
        }
    )

    emit_resume_eval(cfg, tmp_path, str(ckpt_path))

    request_path = tmp_path / "async_eval" / "requests" / "step_000500.json"
    payload = json.loads(request_path.read_text())
    assert payload["ckpt_path"] == str(ckpt_path.resolve())
    assert payload["stream_eval"]["enabled"] is True
    assert payload["stream_eval"]["run_name"] == "step_000500"
    assert payload["stream_eval"]["max_samples"] == 7
    assert payload["stream_eval"]["num_runs"] == 2


def test_eval_watcher_builds_stream_eval_command_and_summary_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ok")
    payload = {
        "ckpt_path": str(tmp_path / "step_000500.ckpt"),
        "stream_eval": {
            "enabled": True,
            "stream_mode": "stream_generate_step",
            "out_dir": str(tmp_path / "stream_eval"),
            "run_name": "step_000500",
            "probe_tag": "step_000500",
            "max_samples": 7,
            "num_runs": 2,
            "max_batches": 3,
            "vae_ckpt": "/ckpt/vae.ckpt",
            "compute_offline_baseline": True,
            "compute_no_traj_baseline": False,
        },
    }

    cmd = _build_stream_eval_cmd(config_path, payload)

    assert cmd == [
        "python",
        "-m",
        "eval.ldf.stream_metrics",
        "--config",
        str(config_path),
        "--ckpt",
        str(tmp_path / "step_000500.ckpt"),
        "--vae_ckpt",
        "/ckpt/vae.ckpt",
        "--stream_mode",
        "stream_generate_step",
        "--out_dir",
        str(tmp_path / "stream_eval"),
        "--run_name",
        "step_000500",
        "--probe_tag",
        "step_000500",
        "--max_samples",
        "7",
        "--num_runs",
        "2",
        "--max_batches",
        "3",
        "--compute_offline_baseline",
        "--no_compute_no_traj_baseline",
    ]
    assert _stream_summary_exists(payload) is False
    summary = Path(payload["stream_eval"]["out_dir"]) / "step_000500" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}")
    assert _stream_summary_exists(payload) is True
