"""Formal random-state snapshots for transactional stream retries."""

from __future__ import annotations

import random

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RNGStreamState:
    python_state: object
    numpy_state: tuple
    torch_cpu_state: torch.Tensor
    torch_cuda_states: tuple[tuple[int, torch.Tensor], ...]


def _device_index(device: int | str | torch.device) -> int:
    parsed = torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
    if parsed.type != "cuda":
        raise ValueError(f"RNG CUDA device must be CUDA, got {parsed}")
    if parsed.index is None:
        return int(torch.cuda.current_device())
    return int(parsed.index)


def snapshot_rng_state(devices=()) -> RNGStreamState:
    """Capture Python, NumPy, Torch CPU, and selected CUDA RNG states."""
    indices = tuple(dict.fromkeys(_device_index(device) for device in devices))
    if indices and not torch.cuda.is_available():
        raise RuntimeError("CUDA RNG snapshot requested but CUDA is unavailable")
    cuda_states = tuple(
        (index, torch.cuda.get_rng_state(index).clone()) for index in indices
    )
    return RNGStreamState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_rng_state(state: RNGStreamState) -> None:
    """Restore all random generators captured by :func:`snapshot_rng_state`."""
    if not isinstance(state, RNGStreamState):
        raise TypeError("state must be RNGStreamState")
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_cpu_state.clone())
    for index, cuda_state in state.torch_cuda_states:
        torch.cuda.set_rng_state(cuda_state.clone(), int(index))


__all__ = ["RNGStreamState", "restore_rng_state", "snapshot_rng_state"]
