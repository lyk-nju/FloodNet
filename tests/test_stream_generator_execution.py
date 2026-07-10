from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from utils.inference.stream_execution import RootFeedbackConfig
from utils.inference.stream_generator import StreamGenerator


class _FakeLdf(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))


def _generator():
    return StreamGenerator(ldf_model=_FakeLdf(), device="cpu")


def test_execute_step_requires_authoritative_runtime_session():
    generator = _generator()

    with pytest.raises(RuntimeError, match="attached StreamRuntimeSession"):
        generator.execute_step()


def test_attach_rejects_session_owned_by_another_kernel():
    generator = _generator()
    other = _generator()

    with pytest.raises(ValueError, match="kernel"):
        generator.attach_runtime_session(SimpleNamespace(kernel=other))


def test_execute_step_delegates_to_attached_session():
    generator = _generator()
    event = object()
    session = SimpleNamespace(kernel=generator, step=lambda: event)
    generator.attach_runtime_session(session)

    assert generator.execute_step() is event


def test_execute_step_rejects_hidden_per_step_state_changes():
    generator = _generator()
    session = SimpleNamespace(kernel=generator, step=lambda: object())
    generator.attach_runtime_session(session)

    with pytest.raises(ValueError, match="runtime commands"):
        generator.execute_step(text="walk")
    with pytest.raises(ValueError, match="runtime commands"):
        generator.execute_step(traj_input={})
    with pytest.raises(ValueError, match="runtime commands"):
        generator.execute_step(num_denoise_steps=10)


def test_configure_execution_updates_root_feedback_policy_during_migration():
    generator = _generator()

    generator.configure_execution(
        root_feedback=RootFeedbackConfig(enabled=True, xz_blend_alpha=0.25)
    )

    assert generator.root_feedback_config.enabled is True
    assert generator.root_feedback_config.xz_blend_alpha == 0.25
