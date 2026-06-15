from __future__ import annotations

import torch

from utils.training.ldf.self_forcing import resolve_sf_runtime


def _write_ckpt(path, *, global_step: int):
    torch.save({"global_step": int(global_step)}, path)


def test_preserving_optimizer_does_not_rewrite_scheduler_horizon(tmp_path):
    ckpt = tmp_path / "resume.ckpt"
    _write_ckpt(ckpt, global_step=485000)

    resume_step, phase_steps, scheduler_steps = resolve_sf_runtime(
        absolute_target_step=700000,
        resume_ckpt=str(ckpt),
        model_self_forcing_enabled=True,
        configured_num_training_steps=700000,
        reset_optimizer_on_resume=False,
    )

    assert resume_step == 485000
    assert phase_steps == 215000
    assert scheduler_steps == 700000


def test_resetting_optimizer_rewrites_scheduler_horizon_to_resume_phase(tmp_path):
    ckpt = tmp_path / "resume.ckpt"
    _write_ckpt(ckpt, global_step=485000)

    resume_step, phase_steps, scheduler_steps = resolve_sf_runtime(
        absolute_target_step=700000,
        resume_ckpt=str(ckpt),
        model_self_forcing_enabled=True,
        configured_num_training_steps=700000,
        reset_optimizer_on_resume=True,
    )

    assert resume_step == 485000
    assert phase_steps == 215000
    assert scheduler_steps == 215000
