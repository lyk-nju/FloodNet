from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StepSemantics:
    resume_step_offset: int
    phase_step: int
    absolute_step: int
    phase_total_steps: int
    absolute_target_step: int
    progress: float


@dataclass(frozen=True)
class CheckpointStepInfo:
    metric_value: float
    step_tag: str
    filename_step: int


def load_resume_step_offset(ckpt_path: str | None) -> int:
    if not ckpt_path:
        return 0
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "global_step" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain `global_step`")
    return int(ckpt["global_step"])


def resolve_runtime_max_steps(
    absolute_target_step: int,
    resume_step_offset: int = 0,
    *,
    self_forcing_enabled: bool,
) -> int:
    if not self_forcing_enabled:
        return int(absolute_target_step)
    runtime_max_steps = int(absolute_target_step) - int(resume_step_offset)
    if runtime_max_steps <= 0:
        raise ValueError(
            "trainer.max_steps must be greater than resume checkpoint global_step "
            f"for self-forcing resume: max_steps={absolute_target_step}, "
            f"resume_step_offset={resume_step_offset}"
        )
    return runtime_max_steps


def resolve_runtime_scheduler_steps(
    configured_num_training_steps: int,
    *,
    absolute_target_step: int,
    runtime_max_steps: int,
) -> int:
    """Map lr-scheduler horizon to the runtime phase when config follows trainer.max_steps.

    Most configs set ``lr_scheduler.params.num_training_steps`` to
    ``${trainer.max_steps}``. After self-forcing resume rewrites the Trainer's
    runtime ``max_steps`` to a shorter phase length, the scheduler should follow
    that shorter horizon too. If the scheduler was configured independently, we
    preserve the explicit override.
    """
    configured = int(configured_num_training_steps)
    if configured == int(absolute_target_step):
        return int(runtime_max_steps)
    return configured


def build_step_semantics(
    *,
    phase_step: int,
    trainer_max_steps: int | None,
    resume_step_offset: int = 0,
    self_forcing_enabled: bool,
) -> StepSemantics:
    phase_total_steps = (
        int(trainer_max_steps)
        if trainer_max_steps is not None and int(trainer_max_steps) > 0
        else 1
    )
    # `phase_step` is always phase-relative (= lightning global_step minus the
    # resume offset). The absolute step (used for checkpoint naming / logging)
    # should match Lightning's global_step in *both* SF and non-SF resume
    # cases, otherwise checkpoints saved after a non-SF resume collapse to the
    # same filename (e.g. 5000/10000/15000 instead of 245000/250000/255000).
    absolute_step = int(resume_step_offset) + int(phase_step)
    if self_forcing_enabled:
        # SF mode: trainer.max_steps was rewritten to a phase-relative length,
        # so the absolute target adds the resume offset back.
        absolute_target_step = int(resume_step_offset) + phase_total_steps
    else:
        # Non-SF mode: trainer.max_steps is already an absolute target.
        absolute_target_step = phase_total_steps
    progress = min(1.0, float(phase_step) / float(max(phase_total_steps, 1)))
    return StepSemantics(
        resume_step_offset=int(resume_step_offset),
        phase_step=int(phase_step),
        absolute_step=int(absolute_step),
        phase_total_steps=int(phase_total_steps),
        absolute_target_step=int(absolute_target_step),
        progress=float(progress),
    )


def build_checkpoint_step_info(
    semantics: StepSemantics,
    *,
    include_next_step: bool,
) -> CheckpointStepInfo:
    step_value = semantics.absolute_step + (1 if include_next_step else 0)
    return CheckpointStepInfo(
        metric_value=float(step_value),
        step_tag=f"step_{step_value:06d}",
        filename_step=int(step_value),
    )
