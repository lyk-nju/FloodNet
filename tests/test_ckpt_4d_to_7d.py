"""Tests for the 7D traj-encoder rewrite ckpt-compat shim.

The legacy 4D encoder is dropped entirely — `strip_legacy_traj_encoder_weights`
removes any incoming key whose shape no longer matches the new 7D model so a
legacy ckpt loads with `strict=False` and the new traj encoder trains from
scratch. The kept-stub `expand_traj_input_4d_to_7d` is a no-op that returns 0.
"""

from __future__ import annotations

import torch

from models.tools.traj_encoder import LocalTrajEncoder, TrajEncoder
from utils.training.ckpt_compat import (
    expand_traj_input_4d_to_7d,
    strip_legacy_traj_encoder_weights,
)

_LOCAL_PREFIX = "local_traj_encoder."
_TRAJ_PREFIX = "traj_encoder."
_CN_TRAJ_IN_PROJ = "controlnet.traj_in_proj."


def _own_state_for_new_model() -> dict:
    """The (post-rewrite) submodule shapes the live model exposes."""
    le = LocalTrajEncoder()                       # 7 → 64 → 128
    te = TrajEncoder()                            # 128 → 128 → 128
    own = {}
    for k, v in le.state_dict().items():
        own[_LOCAL_PREFIX + k] = v
    for k, v in te.state_dict().items():
        own[_TRAJ_PREFIX + k] = v
    own[_CN_TRAJ_IN_PROJ + "weight"] = torch.zeros(1024, 128)
    own[_CN_TRAJ_IN_PROJ + "bias"] = torch.zeros(1024)
    return own


def _legacy_4d_state() -> dict:
    """Synthesize a legacy 4D ckpt's traj-encoder slice with the old shapes."""
    sd = {}
    # Old LocalTrajEncoder: Conv1d 4→32→4 (kernel_size=3 → weight [out, in, 3])
    sd[_LOCAL_PREFIX + "net.0.weight"] = torch.randn(32, 4, 3)
    sd[_LOCAL_PREFIX + "net.0.bias"] = torch.randn(32)
    sd[_LOCAL_PREFIX + "net.2.weight"] = torch.randn(4, 32, 1)
    sd[_LOCAL_PREFIX + "net.2.bias"] = torch.randn(4)
    # Old TrajEncoder: 4 → 64 → 64
    sd[_TRAJ_PREFIX + "mlp.0.weight"] = torch.randn(64, 4)
    sd[_TRAJ_PREFIX + "mlp.0.bias"] = torch.randn(64)
    sd[_TRAJ_PREFIX + "mlp.2.weight"] = torch.randn(64, 64)
    sd[_TRAJ_PREFIX + "mlp.2.bias"] = torch.randn(64)
    # Old ControlNet traj_in_proj: 64 → hidden (e.g. 1024)
    sd[_CN_TRAJ_IN_PROJ + "weight"] = torch.randn(1024, 64)
    sd[_CN_TRAJ_IN_PROJ + "bias"] = torch.randn(1024)
    # Some unrelated backbone weight to ensure stripping is targeted.
    sd["model.blocks.0.self_attn.q.weight"] = torch.randn(1024, 1024)
    return sd


def test_strip_drops_all_legacy_traj_keys():
    sd = _legacy_4d_state()
    n_traj_keys = sum(
        1 for k in sd
        if k.startswith(_LOCAL_PREFIX)
        or k.startswith(_TRAJ_PREFIX)
        or k.startswith(_CN_TRAJ_IN_PROJ)
    )
    assert n_traj_keys == 10
    n = strip_legacy_traj_encoder_weights(sd, _own_state_for_new_model())
    assert n == 10
    # Backbone key untouched.
    assert "model.blocks.0.self_attn.q.weight" in sd
    # All traj keys stripped.
    for k in list(sd.keys()):
        assert not k.startswith(_LOCAL_PREFIX)
        assert not k.startswith(_TRAJ_PREFIX)
        assert not k.startswith(_CN_TRAJ_IN_PROJ)


def test_strip_keeps_already_7d_keys():
    """A fresh 7D ckpt round-trips: shapes match → nothing gets stripped."""
    own = _own_state_for_new_model()
    sd = {k: v.clone() for k, v in own.items()}
    n = strip_legacy_traj_encoder_weights(sd, own)
    assert n == 0
    assert set(sd.keys()) == set(own.keys())


def test_strip_handles_lightning_module_prefix():
    """LightningModule ckpts often nest the model under a prefix (e.g. "model.")."""
    own = _own_state_for_new_model()
    own_prefixed = {f"foo.{k}": v for k, v in own.items()}
    legacy = _legacy_4d_state()
    legacy_prefixed = {f"foo.{k}": v for k, v in legacy.items()}
    n = strip_legacy_traj_encoder_weights(legacy_prefixed, own_prefixed)
    # Same 10 traj keys stripped; backbone weight (foo.model.blocks...) keeps.
    assert n == 10
    assert "foo.model.blocks.0.self_attn.q.weight" in legacy_prefixed


def test_expand_is_noop_stub():
    """The legacy expander is preserved as a no-op import-compat stub."""
    sd = _legacy_4d_state()
    before = {k: v.clone() for k, v in sd.items()}
    n = expand_traj_input_4d_to_7d(sd, target_in_dim=7)
    assert n == 0
    assert set(sd.keys()) == set(before.keys())
    for k in sd:
        assert torch.equal(sd[k], before[k])


def test_legacy_ckpt_loads_into_new_encoders_with_strict_false():
    """End-to-end: strip + load(strict=False) leaves new encoders at init."""
    own = _own_state_for_new_model()
    sd = _legacy_4d_state()
    strip_legacy_traj_encoder_weights(sd, own)
    le = LocalTrajEncoder()
    te = TrajEncoder()
    le_sd = {k[len(_LOCAL_PREFIX):]: v for k, v in sd.items() if k.startswith(_LOCAL_PREFIX)}
    te_sd = {k[len(_TRAJ_PREFIX):]: v for k, v in sd.items()
             if k.startswith(_TRAJ_PREFIX) and not k.startswith(_LOCAL_PREFIX)}
    # Both subdicts should be empty post-strip (every legacy key was the wrong shape).
    assert le_sd == {} and te_sd == {}
    res_le = le.load_state_dict(le_sd, strict=False)
    res_te = te.load_state_dict(te_sd, strict=False)
    # All weights still expected (none loaded) → missing keys cover the entire encoder.
    assert len(res_le.missing_keys) > 0 and res_le.unexpected_keys == []
    assert len(res_te.missing_keys) > 0 and res_te.unexpected_keys == []
