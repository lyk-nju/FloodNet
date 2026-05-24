"""Unit tests for T_B_08: 4D→7D traj-encoder ckpt expansion hook.

Covers docs/TODO.md §T_B_08 Done-criteria 1-4. The full DiffForcingWanModel needs
t5 deps, so the expansion is verified at the traj-encoder level (where the
shapes actually change) + against the real step_460000.ckpt when present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.tools.traj_encoder import LocalTrajEncoder, TrajEncoder
from utils.training.ckpt_compat import expand_traj_input_4d_to_7d

_CKPT = Path(__file__).resolve().parent.parent / "step_460000.ckpt"
_LOCAL_PREFIX = "local_traj_encoder."
_TRAJ_PREFIX = "traj_encoder."


def _encoders(in_dim, hidden_local=32, hidden_traj=64, out_dim=64, seed=0):
    torch.manual_seed(seed)
    le = LocalTrajEncoder(in_dim=in_dim, hidden_dim=hidden_local).eval()
    torch.manual_seed(seed + 100)
    te = TrajEncoder(in_dim=in_dim, hidden_dim=hidden_traj, out_dim=out_dim).eval()
    return le, te


def _combined_sd(le, te):
    sd = {}
    for k, v in le.state_dict().items():
        sd[_LOCAL_PREFIX + k] = v.clone()
    for k, v in te.state_dict().items():
        sd[_TRAJ_PREFIX + k] = v.clone()
    return sd


def _load_into(le7, te7, sd):
    le_sd = {k[len(_LOCAL_PREFIX):]: v for k, v in sd.items() if k.startswith(_LOCAL_PREFIX)}
    te_sd = {k[len(_TRAJ_PREFIX):]: v for k, v in sd.items()
             if k.startswith(_TRAJ_PREFIX) and not k.startswith(_LOCAL_PREFIX)}
    le7.load_state_dict(le_sd, strict=True)
    te7.load_state_dict(te_sd, strict=True)


# ---------------------------------------------------------------------------
# expansion shapes / no-op
# ---------------------------------------------------------------------------


def test_expand_shapes_and_count():
    le4, te4 = _encoders(4)
    sd = _combined_sd(le4, te4)
    n = expand_traj_input_4d_to_7d(sd, target_in_dim=7)
    assert n == 4   # net.0.w (in), net.2.w (out), net.2.b (out), mlp.0.w (in)
    assert sd[_LOCAL_PREFIX + "net.0.weight"].shape == (32, 7, 3)
    assert sd[_LOCAL_PREFIX + "net.2.weight"].shape == (7, 32, 1)
    assert sd[_LOCAL_PREFIX + "net.2.bias"].shape == (7,)
    assert sd[_TRAJ_PREFIX + "mlp.0.weight"].shape == (64, 7)


def test_noop_when_target_is_4d():
    le4, te4 = _encoders(4)
    sd = _combined_sd(le4, te4)
    before = {k: v.clone() for k, v in sd.items()}
    n = expand_traj_input_4d_to_7d(sd, target_in_dim=4)
    assert n == 0
    for k in sd:
        assert torch.equal(sd[k], before[k])


def test_raw_input_axis_uses_safe_xz_map():
    """net.0 in-channels: old x(0)→new0, old z(1)→new2, legacy(2,3) dropped, rest 0."""
    le4, te4 = _encoders(4)
    sd = _combined_sd(le4, te4)
    old = sd[_LOCAL_PREFIX + "net.0.weight"].clone()   # [32,4,3]
    expand_traj_input_4d_to_7d(sd, 7)
    new = sd[_LOCAL_PREFIX + "net.0.weight"]            # [32,7,3]
    assert torch.equal(new[:, 0, :], old[:, 0, :])     # x
    assert torch.equal(new[:, 2, :], old[:, 1, :])     # z
    for c in (1, 3, 4, 5, 6):                          # y, cos, sin, fwd, yaw
        assert torch.count_nonzero(new[:, c, :]) == 0
    # legacy heading (old 2,3) must NOT appear anywhere in the new weight
    assert not torch.equal(new[:, 1, :], old[:, 2, :])


def test_internal_axes_in_place_zero_pad():
    """net.2 out + mlp.0 in keep old 4 channels in place, append 3 zeros."""
    le4, te4 = _encoders(4)
    sd = _combined_sd(le4, te4)
    old_net2 = sd[_LOCAL_PREFIX + "net.2.weight"].clone()
    old_mlp0 = sd[_TRAJ_PREFIX + "mlp.0.weight"].clone()
    expand_traj_input_4d_to_7d(sd, 7)
    assert torch.equal(sd[_LOCAL_PREFIX + "net.2.weight"][:4], old_net2)
    assert torch.count_nonzero(sd[_LOCAL_PREFIX + "net.2.weight"][4:]) == 0
    assert torch.equal(sd[_TRAJ_PREFIX + "mlp.0.weight"][:, :4], old_mlp0)
    assert torch.count_nonzero(sd[_TRAJ_PREFIX + "mlp.0.weight"][:, 4:]) == 0


# ---------------------------------------------------------------------------
# Done #1: 4D ckpt loads into 7D encoders with no shape error
# ---------------------------------------------------------------------------


def test_expanded_4d_loads_into_7d_strict():
    le4, te4 = _encoders(4)
    sd = _combined_sd(le4, te4)
    expand_traj_input_4d_to_7d(sd, 7)
    le7 = LocalTrajEncoder(in_dim=7, hidden_dim=32)
    te7 = TrajEncoder(in_dim=7, hidden_dim=64, out_dim=64)
    _load_into(le7, te7, sd)   # strict=True inside → raises on any mismatch


# ---------------------------------------------------------------------------
# Done #3: 7D input [x,0,z,0,0,0,0] reproduces old 4D forward on [x,z,0,0]
# Done #2: differs from old 4D forward with NONZERO legacy heading
# ---------------------------------------------------------------------------


def _build_pair():
    le4, te4 = _encoders(4)
    sd = _combined_sd(le4, te4)
    expand_traj_input_4d_to_7d(sd, 7)
    le7 = LocalTrajEncoder(in_dim=7, hidden_dim=32).eval()
    te7 = TrajEncoder(in_dim=7, hidden_dim=64, out_dim=64).eval()
    _load_into(le7, te7, sd)
    return le4, te4, le7, te7


def test_equivalence_xz_only_input():
    le4, te4, le7, te7 = _build_pair()
    B, T = 2, 5
    xz = torch.randn(B, T, 4, 2)
    x4 = torch.zeros(B, T, 4, 4)
    x4[..., 0] = xz[..., 0]   # x
    x4[..., 1] = xz[..., 1]   # z
    x7 = torch.zeros(B, T, 4, 7)
    x7[..., 0] = xz[..., 0]   # x
    x7[..., 2] = xz[..., 1]   # z
    with torch.no_grad():
        o4 = te4(le4(x4))
        o7 = te7(le7(x7))
    assert torch.allclose(o4, o7, atol=1e-5)


def test_legacy_heading_dropped_changes_output():
    le4, te4, le7, te7 = _build_pair()
    B, T = 2, 5
    xz = torch.randn(B, T, 4, 2)
    x4_h = torch.zeros(B, T, 4, 4)
    x4_h[..., 0] = xz[..., 0]
    x4_h[..., 1] = xz[..., 1]
    x4_h[..., 2] = torch.randn(B, T, 4)   # nonzero legacy heading channels
    x4_h[..., 3] = torch.randn(B, T, 4)
    x7 = torch.zeros(B, T, 4, 7)
    x7[..., 0] = xz[..., 0]
    x7[..., 2] = xz[..., 1]
    with torch.no_grad():
        o4_h = te4(le4(x4_h))   # uses legacy heading
        o7 = te7(le7(x7))       # legacy heading dropped
    assert not torch.allclose(o4_h, o7, atol=1e-4)


# ---------------------------------------------------------------------------
# Done #4: gradient flows into the new zero-init channels (fine-tune可推进)
# ---------------------------------------------------------------------------


def test_gradient_flows_into_new_channels():
    le4, te4 = _encoders(4)
    sd = _combined_sd(le4, te4)
    expand_traj_input_4d_to_7d(sd, 7)
    le7 = LocalTrajEncoder(in_dim=7, hidden_dim=32)
    te7 = TrajEncoder(in_dim=7, hidden_dim=64, out_dim=64)
    _load_into(le7, te7, sd)
    x7 = torch.randn(2, 5, 4, 7)   # nonzero on ALL 7 channels incl. new ones
    out = te7(le7(x7)).sum()
    out.backward()
    g = le7.net[0].weight.grad      # [32,7,3]
    assert g is not None
    # new raw channels y(1), cos(3), sin(4), fwd(5), yaw(6) receive gradient
    for c in (1, 3, 4, 5, 6):
        assert g[:, c, :].abs().sum() > 0


# ---------------------------------------------------------------------------
# Done #2 (real): step_460000.ckpt traj weights expand + load + forward
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CKPT.is_file(), reason="step_460000.ckpt not present")
def test_real_ckpt_traj_expand_load_forward():
    full = torch.load(_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sub = {k: v for k, v in full.items()
           if k.startswith(_LOCAL_PREFIX) or k.startswith(_TRAJ_PREFIX)}
    assert sub[_TRAJ_PREFIX + "mlp.0.weight"].shape[1] == 4   # old ckpt is 4D
    n = expand_traj_input_4d_to_7d(sub, 7)
    assert n == 4
    le7 = LocalTrajEncoder(in_dim=7, hidden_dim=32)
    te7 = TrajEncoder(in_dim=7, hidden_dim=64, out_dim=64)
    _load_into(le7, te7, sub)
    with torch.no_grad():
        out = te7(le7(torch.randn(1, 3, 4, 7)))
    assert out.shape == (1, 3, 64)
