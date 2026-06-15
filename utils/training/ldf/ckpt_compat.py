"""Checkpoint compatibility helpers for trajectory-conditioning rewrites."""

from __future__ import annotations

import torch

# Key prefixes owned by the trajectory-conditioning subsystem.
_LEGACY_TRAJ_PREFIXES = (
    "local_traj_encoder.",
    "traj_encoder.",
    "controlnet.traj_in_proj.",
    "model.traj_in_proj.",
    "controlnet.traj_type_embed",
)


def _key_matches_legacy_traj(key: str) -> bool:
    for p in _LEGACY_TRAJ_PREFIXES:
        # Match either "<module>.<key>" or "<prefix>.<module>.<key>".
        if key.startswith(p) or f".{p}" in key:
            return True
    return False


def strip_legacy_traj_encoder_weights(state_dict: dict, own_state: dict) -> int:
    """Strip incompatible trajectory-conditioning weights in place.

    The trajectory encoder and projection are treated as one logical unit. If
    any tracked key is missing or shape-mismatched against the live model, all
    tracked keys are removed. Fresh checkpoints round-trip unchanged.
    """
    legacy_keys = [k for k in state_dict.keys() if _key_matches_legacy_traj(k)]
    if not legacy_keys:
        return 0

    any_mismatch = False
    for key in legacy_keys:
        v = state_dict[key]
        if not torch.is_tensor(v):
            continue
        own = own_state.get(key)
        if own is None or tuple(v.shape) != tuple(own.shape):
            any_mismatch = True
            break
    if not any_mismatch:
        return 0

    n = 0
    for key in legacy_keys:
        if key in state_dict:
            del state_dict[key]
            n += 1
    return n


# Kept for older callers that imported the former expansion helper.
def expand_traj_input_4d_to_7d(state_dict: dict, target_in_dim: int) -> int:
    """No-op compatibility shim; use strip_legacy_traj_encoder_weights instead."""
    del state_dict, target_in_dim
    return 0


__all__ = [
    "strip_legacy_traj_encoder_weights",
    "expand_traj_input_4d_to_7d",
]
