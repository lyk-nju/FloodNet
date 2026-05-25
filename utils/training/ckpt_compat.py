"""Checkpoint compatibility shims for the 7D traj-encoder rewrite.

The old 4D-era encoder (Conv1d 4→32→4 + per-token MLP 4→64→64) and the new 7D
encoder (Conv1d 7→64→128 + LayerNorm + MLP 128→128→128, traj_out_dim=128) have
incompatible shapes throughout. We do **not** try to migrate weights — the old
traj encoder is dropped entirely and the new one trains from scratch.

`strip_legacy_traj_encoder_weights` removes any incoming key whose shape no
longer matches the model:
  * local_traj_encoder.*
  * traj_encoder.*
  * controlnet.traj_in_proj.* (input dim changed 64 → 128)

The backbone, ControlNet attention layers, time/text embeddings, and the
T_B_02 optional fields (mask_emb / z_mean / z_std) all keep loading as before —
only the trajectory-conditioning weights are dropped on a legacy ckpt.
"""

from __future__ import annotations

import torch

# Key prefixes whose owning submodules were rewritten and have new shapes /
# layouts. Any key whose suffix-tail matches one of these is a candidate for
# stripping (we additionally check shape mismatch before dropping, so a fresh
# 7D ckpt round-trips cleanly).
_LEGACY_TRAJ_PREFIXES = (
    "local_traj_encoder.",
    "traj_encoder.",
    "controlnet.traj_in_proj.",
    "model.traj_in_proj.",
)


def _key_matches_legacy_traj(key: str) -> bool:
    for p in _LEGACY_TRAJ_PREFIXES:
        # Match either "<module>.<key>" or "<prefix>.<module>.<key>".
        if key.startswith(p) or f".{p}" in key:
            return True
    return False


def strip_legacy_traj_encoder_weights(state_dict: dict, own_state: dict) -> int:
    """Drop incoming traj-encoder/traj_in_proj keys that don't match the rewritten
    7D model. Returns the number of keys stripped.

    A legacy ckpt's traj submodules were entirely rewritten — even keys whose
    raw shape happens to match (e.g. ``traj_in_proj.bias`` is still ``(dim,)``)
    carry stale 4D-era statistics, so the safe rule for the traj prefixes is
    "strip unconditionally": any incoming traj-prefix key whose tensor or shape
    doesn't match the live model's same-key tensor is removed, AND if any key
    under a given traj-prefix is being stripped, all sibling keys for that
    prefix are stripped too — otherwise a surviving bias would silently
    re-introduce non-zero init into the supposedly zero-init projection.

    Modifies `state_dict` in place. Caller is expected to load with strict=False.
    """
    legacy_keys = [k for k in state_dict.keys() if _key_matches_legacy_traj(k)]
    if not legacy_keys:
        return 0

    # Phase 1: shape-mismatch check, grouped by prefix so a partial-mismatch
    # group strips its siblings too.
    by_prefix: dict[str, list[str]] = {}
    for key in legacy_keys:
        for p in _LEGACY_TRAJ_PREFIXES:
            if key.startswith(p) or f".{p}" in key:
                # Use the full prefix path (including any LightningModule "model." outer).
                idx = key.find(p)
                full_prefix = key[: idx + len(p)]
                by_prefix.setdefault(full_prefix, []).append(key)
                break

    n = 0
    for full_prefix, keys in by_prefix.items():
        # Mismatched if ANY sibling has no own-state counterpart or shape mismatch.
        mismatched = False
        for key in keys:
            v = state_dict[key]
            if not torch.is_tensor(v):
                continue
            own = own_state.get(key)
            if own is None or tuple(v.shape) != tuple(own.shape):
                mismatched = True
                break
        if mismatched:
            for key in keys:
                if key in state_dict:
                    del state_dict[key]
                    n += 1
    return n


# --- Legacy 4D→7D expansion is gone. Keep a stub so existing callers (train_ldf,
# generate_ldf, tests) keep importing without crash; it now strips legacy weights
# instead of reshaping them. ---
def expand_traj_input_4d_to_7d(state_dict: dict, target_in_dim: int) -> int:
    """Backward-compat shim: legacy 4D ckpts no longer reshape into the new 7D
    encoder (architecture changed too much). This function is now a no-op and
    returns 0 — call `strip_legacy_traj_encoder_weights` against the live
    model's state dict instead.
    """
    del state_dict, target_in_dim
    return 0


__all__ = [
    "strip_legacy_traj_encoder_weights",
    "expand_traj_input_4d_to_7d",
]
