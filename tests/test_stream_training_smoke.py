from __future__ import annotations

import json
from pathlib import Path

from scripts.stream_training_smoke import (
    SmokeRunConfig,
    build_validation_manifest,
    build_validation_plan,
    build_train_command,
    find_missing_paths,
    main,
)


def test_build_train_command_enables_stream_training_window_sampling_full_prefix(tmp_path):
    cfg = SmokeRunConfig(
        config="configs/ldf.yaml",
        python="python",
        resume_ckpt="/ckpt/body.ckpt",
        vae_ckpt="/ckpt/vae.ckpt",
        raw_data_root="/data/raw_data",
        output_dir=str(tmp_path),
        exp_name="stream_smoke",
        max_steps=318000,
        devices="1",
        accelerator="cpu",
        motion_aux_loss="full_prefix",
        context_tokens=30,
        history_tokens_min=0,
        history_tokens_max="auto",
        horizon_tokens_min=5,
        horizon_tokens_max=25,
        z_stats_dir="/stats/body",
        train_split="/data/raw_data/HumanML3D/train_min.txt",
        val_split="/data/raw_data/HumanML3D/val_min.txt",
    )

    cmd = build_train_command(cfg)

    assert cmd[:4] == ["python", "train_ldf.py", "--config", "configs/ldf.yaml"]
    overrides = set(cmd[5:])
    assert "stream_training.enabled=true" in overrides
    assert "stream_training.motion_aux_loss=full_prefix" in overrides
    assert "stream_training.context_tokens=30" in overrides
    assert "stream_training.window_sampling.enabled=true" in overrides
    assert "stream_training.window_sampling.history_tokens_min=0" in overrides
    assert "stream_training.window_sampling.history_tokens_max=auto" in overrides
    assert "stream_training.window_sampling.horizon_tokens_min=5" in overrides
    assert "stream_training.window_sampling.horizon_tokens_max=25" in overrides
    assert not any(item.startswith("horizon_sim.") for item in overrides)
    assert "trainer.accelerator=cpu" in overrides
    assert "trainer.devices=1" in overrides
    assert "trainer.max_steps=318000" in overrides
    assert "dirs.raw_data=/data/raw_data" in overrides
    assert "validation.validation_steps=1" in overrides
    assert "validation.save_every_n_steps=1" in overrides
    assert "history_corruption.z_stats_dir=/stats/body" in overrides
    assert "data.train_meta_paths.0=/data/raw_data/HumanML3D/train_min.txt" in overrides
    assert "data.val_meta_paths.0=/data/raw_data/HumanML3D/val_min.txt" in overrides
    assert "data.test_probe_meta_paths.test.0=/data/raw_data/HumanML3D/val_min.txt" in overrides


def test_find_missing_paths_reports_required_inputs(tmp_path):
    existing = tmp_path / "exists"
    existing.write_text("ok")
    cfg = SmokeRunConfig(
        config=str(existing),
        resume_ckpt=str(tmp_path / "missing_resume.ckpt"),
        vae_ckpt=str(tmp_path / "missing_vae.ckpt"),
        raw_data_root=str(tmp_path / "missing_raw"),
        train_split=str(existing),
        val_split=str(existing),
        z_stats_dir=str(tmp_path / "missing_stats"),
    )

    missing = find_missing_paths(cfg)

    assert str(tmp_path / "missing_resume.ckpt") in missing
    assert str(tmp_path / "missing_vae.ckpt") in missing
    assert str(tmp_path / "missing_raw") in missing
    assert str(tmp_path / "missing_raw" / "HumanML3D" / "t5_text_embeddings.pt") in missing
    assert str(tmp_path / "missing_stats") in missing
    assert str(existing) not in missing


def test_build_validation_plan_expands_required_stream_training_stages(tmp_path):
    base = SmokeRunConfig(
        config="configs/ldf.yaml",
        python="python",
        resume_ckpt="/ckpt/body.ckpt",
        vae_ckpt="/ckpt/vae.ckpt",
        raw_data_root="/data/raw_data",
        output_dir=str(tmp_path / "stream_validation"),
        exp_name="stream_validation",
        max_steps=318000,
        devices="1",
        accelerator="cpu",
        motion_aux_loss="full_prefix",
        context_tokens=30,
    )

    plan = build_validation_plan(base)

    assert [entry.stage for entry in plan] == [
        "01_smoke_latent",
        "02_overfit_latent",
        "03_overfit_full_prefix",
    ]
    assert [
        entry.config.motion_aux_loss
        for entry in plan
    ] == [
        "latent_only",
        "latent_only",
        "full_prefix",
    ]
    assert plan[2].config.exp_name == "stream_validation_03_overfit_full_prefix"
    assert plan[2].config.output_dir == str(
        tmp_path / "stream_validation" / "03_overfit_full_prefix"
    )

    cmd = build_train_command(plan[2].config)
    overrides = set(cmd[5:])
    assert "stream_training.motion_aux_loss=full_prefix" in overrides
    assert "stream_training.window_sampling.enabled=true" in overrides


def test_validation_plan_print_only_outputs_all_stages_without_preflight(capsys):
    rc = main([
        "--resume-ckpt", "/missing/body.ckpt",
        "--vae-ckpt", "/missing/vae.ckpt",
        "--raw-data-root", "/missing/raw_data",
        "--max-steps", "318000",
        "--validation-plan",
        "--print-only",
    ])

    out = capsys.readouterr().out
    assert rc == 0
    assert "# 01_smoke_latent:" in out
    assert "# 02_overfit_latent:" in out
    assert "# 03_overfit_full_prefix:" in out
    assert "stream_training.motion_aux_loss=full_prefix" in out
    assert "stream_training.window_sampling.enabled=true" in out


def test_validation_plan_writes_manifest_for_dry_run(tmp_path):
    manifest = tmp_path / "validation_plan.json"
    existing = tmp_path / "exists"
    existing.write_text("ok")
    raw_root = tmp_path / "raw_data"
    raw_h3d = raw_root / "HumanML3D"
    raw_h3d.mkdir(parents=True)
    (raw_h3d / "t5_text_embeddings.pt").write_text("ok")

    rc = main([
        "--config", str(existing),
        "--resume-ckpt", str(existing),
        "--vae-ckpt", str(existing),
        "--raw-data-root", str(raw_root),
        "--train-split", str(existing),
        "--val-split", str(existing),
        "--z-stats-dir", str(existing),
        "--max-steps", "318000",
        "--output-dir", str(tmp_path / "runs"),
        "--exp-name", "stream_validation",
        "--validation-plan",
        "--dry-run",
        "--manifest", str(manifest),
    ])

    assert rc == 0
    payload = json.loads(manifest.read_text())
    assert payload["kind"] == "ldf_stream_training_validation_plan"
    assert payload["base"]["max_steps"] == 318000
    assert payload["base"]["raw_data_root"] == str(raw_root)
    assert [stage["stage"] for stage in payload["stages"]] == [
        "01_smoke_latent",
        "02_overfit_latent",
        "03_overfit_full_prefix",
    ]
    assert payload["stages"][2]["config"]["motion_aux_loss"] == "full_prefix"
    assert payload["stages"][2]["missing_paths"] == []
    assert "stream_training.motion_aux_loss=full_prefix" in payload["stages"][2]["command"]


def test_validation_manifest_includes_stream_eval_and_report_commands(tmp_path):
    base = SmokeRunConfig(
        config="configs/ldf.yaml",
        python="python",
        resume_ckpt="/ckpt/full_prefix.ckpt",
        vae_ckpt="/ckpt/vae.ckpt",
        raw_data_root="/data/raw_data",
        output_dir=str(tmp_path / "runs"),
        exp_name="stream_validation",
        max_steps=318000,
        val_split="/data/raw_data/HumanML3D/val_min.txt",
        eval_max_samples=5,
        eval_num_runs=1,
        max_root_ade_regression=0.05,
        max_path_arc_regression=0.08,
        max_yaw_error_regression=0.2,
        max_jitter_regression=0.01,
        max_foot_skating_regression=0.03,
    )

    payload = build_validation_manifest(base)

    baseline = payload["stream_eval"]["baseline"]
    assert baseline["summary"] == str(
        tmp_path / "runs" / "stream_eval" / "baseline_full_prefix" / "summary.json"
    )
    assert baseline["argv"][:5] == [
        "python",
        "-m",
        "eval.ldf.stream_metrics",
        "--config",
        "configs/ldf.yaml",
    ]
    assert "--run_name" in baseline["argv"]
    assert "baseline_full_prefix" in baseline["argv"]
    assert "--meta_paths" in baseline["argv"]
    assert "/data/raw_data/HumanML3D/val_min.txt" in baseline["argv"]

    stage_eval = payload["stages"][2]["post_training_eval"]
    assert stage_eval["candidate_ckpt_placeholder"] == "{candidate_ckpt}"
    assert stage_eval["candidate_eval"]["summary"] == str(
        tmp_path
        / "runs"
        / "stream_eval"
        / "03_overfit_full_prefix"
        / "summary.json"
    )
    assert "{candidate_ckpt}" in stage_eval["candidate_eval"]["argv"]
    assert stage_eval["comparison"]["summary"] == str(
        tmp_path / "runs" / "stream_eval" / "03_overfit_full_prefix_comparison.json"
    )
    assert stage_eval["comparison"]["argv"] == [
        "python",
        "-m",
        "eval.ldf.report",
        "--baseline",
        baseline["summary"],
        "--candidate",
        stage_eval["candidate_eval"]["summary"],
        "--out",
        stage_eval["comparison"]["summary"],
        "--max-root-ade-regression",
        "0.05",
        "--max-path-arc-regression",
        "0.08",
        "--max-yaw-error-regression",
        "0.2",
        "--max-jitter-regression",
        "0.01",
        "--max-foot-skating-regression",
        "0.03",
    ]
