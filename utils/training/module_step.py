from __future__ import annotations

from .step_semantics import _make_step_info, build_step_semantics


def _get_module_phase_step(module) -> int:
    resume_step_offset = int(getattr(module, "_resume_step_offset", 0))
    return max(0, int(module.global_step) - resume_step_offset)


def compute_step_semantics(module, phase_step: int | None = None):
    trainer = getattr(module, "trainer", None)
    trainer_max_steps = (
        getattr(trainer, "max_steps", None) if trainer is not None else None
    )
    resume_step_offset = int(getattr(module, "_resume_step_offset", 0))
    sf_enabled = bool(getattr(module.model, "self_forcing_enabled", False))
    if sf_enabled and trainer_max_steps is not None and int(trainer_max_steps) > 0:
        phase_total = int(trainer_max_steps) - resume_step_offset
        trainer_max_steps = max(1, phase_total)
    return build_step_semantics(
        phase_step=(
            _get_module_phase_step(module)
            if phase_step is None
            else int(phase_step)
        ),
        trainer_max_steps=trainer_max_steps,
        resume_step_offset=resume_step_offset,
        self_forcing_enabled=sf_enabled,
    )


def ckpt_step_info(module, *, include_next_step: bool = False):
    return _make_step_info(
        compute_step_semantics(module),
        include_next_step=include_next_step,
    )
