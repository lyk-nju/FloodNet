"""Runtime state snapshots for serial stream best-of-K selection."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class StepOutput:
    clean_committed_latent: Any
    decoded_chunk: Any
    commit_token_range: Tuple[int, int]
    commit_frame_range: Tuple[int, int]
    decoded_chunk_frame_range: Tuple[int, int]
    target_xz_frame_range: Tuple[int, int]
    debug: Dict[str, Any]


@dataclass(frozen=True)
class StreamRuntimeSnapshot:
    model_state: Dict[str, Any]
    vae_state: Dict[str, Any]
    conditioner_state: Dict[str, Any]
    rng_state: Dict[str, Any]
    first_chunk: bool
    generated_frames: int
    chunk_frame_ends: list[int]


@dataclass(frozen=True)
class CandidateState:
    index: int
    step_output: StepOutput
    snapshot: StreamRuntimeSnapshot


_MODEL_ATTRS = (
    "generated",
    "commit_index",
    "current_step",
    "batch_size",
    "seq_len",
    "num_denoise_steps",
    "dt",
    "text_condition_list",
    "_traj_buf",
)
_VAE_MODEL_ATTRS = ("_conv_num", "_conv_idx", "_feat_map")
_CONDITIONER_ATTRS = ("timeline", "root_plan", "_anchor_xz", "_anchor_yaw")


def deep_clone_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [deep_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(deep_clone_state(item) for item in value)
    if isinstance(value, dict):
        return {
            key: deep_clone_state(item)
            for key, item in value.items()
        }
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return deep_clone_state(state)


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(deep_clone_state(state["torch_cpu"]))
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(deep_clone_state(cuda_state))


def _capture_attrs(obj: Any, attrs: tuple[str, ...]) -> Dict[str, Any]:
    if obj is None:
        return {}
    return {
        name: deep_clone_state(getattr(obj, name))
        for name in attrs
        if hasattr(obj, name)
    }


def _restore_attrs(obj: Any, state: Dict[str, Any]) -> None:
    if obj is None:
        return
    for name, value in state.items():
        setattr(obj, name, deep_clone_state(value))


def capture_runtime_snapshot(
    *,
    model: Any,
    vae: Any,
    stream_conditioner: Optional[Any],
    first_chunk: bool,
    generated_frames: int,
    chunk_frame_ends: list[int],
) -> StreamRuntimeSnapshot:
    vae_model = getattr(vae, "model", None)
    return StreamRuntimeSnapshot(
        model_state=_capture_attrs(model, _MODEL_ATTRS),
        vae_state=_capture_attrs(vae_model, _VAE_MODEL_ATTRS),
        conditioner_state=_capture_attrs(stream_conditioner, _CONDITIONER_ATTRS),
        rng_state=_capture_rng_state(),
        first_chunk=bool(first_chunk),
        generated_frames=int(generated_frames),
        chunk_frame_ends=[int(frame) for frame in chunk_frame_ends],
    )


def restore_runtime_snapshot(
    *,
    model: Any,
    vae: Any,
    stream_conditioner: Optional[Any],
    snapshot: StreamRuntimeSnapshot,
) -> None:
    _restore_attrs(model, snapshot.model_state)
    _restore_attrs(getattr(vae, "model", None), snapshot.vae_state)
    _restore_attrs(stream_conditioner, snapshot.conditioner_state)
    _restore_rng_state(snapshot.rng_state)
