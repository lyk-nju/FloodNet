"""Unit tests for T_B_07: traj_encoder 4D→7D (flag-gated) + frame→token mask.

Covers docs/TODO.md §T_B_07 T01-T08 + a default-4D regression guard.
The 4D path stays the default; in_dim=7 activates the 7D path.
"""

from __future__ import annotations

import pytest
import torch

from models.tools.traj_encoder import LocalTrajEncoder, TrajEncoder
from utils.token_frame import (
    frames_to_token_mask,
    num_frames_for_tokens,
    token_start_frame,
)
from utils.traj_batch import encode_traj_batch


def _encoders(in_dim, out_dim=16, seed=0):
    torch.manual_seed(seed)
    le = LocalTrajEncoder(in_dim=in_dim).eval()
    torch.manual_seed(seed + 100)
    te = TrajEncoder(in_dim=in_dim, out_dim=out_dim).eval()
    return le, te


# ---------------------------------------------------------------------------
# T01 — 7D forward shapes
# ---------------------------------------------------------------------------


def test_local_encoder_7d_forward_shape():
    enc = LocalTrajEncoder(in_dim=7)
    out = enc(torch.randn(2, 5, 4, 7))   # (B, T_token, 4 frames, 7)
    assert out.shape == (2, 5, 7)


def test_traj_encoder_7d_forward_shape():
    enc = TrajEncoder(in_dim=7, out_dim=64)
    out = enc(torch.randn(2, 5, 7))
    assert out.shape == (2, 5, 64)


# ---------------------------------------------------------------------------
# T02 — 4D legacy input into a 7D-configured encoder is rejected
# ---------------------------------------------------------------------------


def test_traj_encoder_rejects_4d_when_configured_7d():
    enc = TrajEncoder(in_dim=7)
    with pytest.raises(ValueError):
        enc(torch.randn(2, 5, 4))


def test_local_encoder_rejects_4d_when_configured_7d():
    enc = LocalTrajEncoder(in_dim=7)
    with pytest.raises(ValueError):
        enc(torch.randn(2, 5, 4, 4))


def test_default_4d_path_unchanged():
    """Flag-gated: default in_dim=4 keeps the legacy shapes (no regression)."""
    le = LocalTrajEncoder()           # in_dim=4
    assert le(torch.randn(2, 3, 4, 4)).shape == (2, 3, 4)
    te = TrajEncoder()                # in_dim=4
    assert te(torch.randn(2, 3, 4)).shape == (2, 3, 64)


# ---------------------------------------------------------------------------
# T03 — mask effectiveness (the hard requirement): inputs differing ONLY on
# masked frames produce the same encoder output.
# ---------------------------------------------------------------------------


def test_mask_effectiveness_through_encode_traj_batch():
    B, seq_len, D = 1, 6, 7
    T_frame = num_frames_for_tokens(seq_len)
    le, te = _encoders(D)

    base = torch.randn(B, T_frame, D)
    mask = torch.ones(B, T_frame)
    mask[:, 5:9] = 0                          # masked frames

    x1 = {"traj_features": base.clone(), "traj_cond_mask": mask.clone()}
    x2_feats = base.clone()
    x2_feats[:, 5:9] = torch.randn(B, 4, D)   # differ ONLY on masked frames
    x2 = {"traj_features": x2_feats, "traj_cond_mask": mask.clone()}

    with torch.no_grad():
        o1 = encode_traj_batch(x1, seq_len, "cpu", le, te)
        o2 = encode_traj_batch(x2, seq_len, "cpu", le, te)
    assert torch.allclose(o1, o2, atol=1e-5)


# ---------------------------------------------------------------------------
# T04 — partial token: 1 valid + 3 invalid frames
# ---------------------------------------------------------------------------


def test_partial_token_invalid_frames_dont_affect_output():
    """(a) Within a token, changing the masked (invalid) frames leaves the
    encoder output unchanged (zero-out happens before frames_to_tokens)."""
    B, seq_len, D = 1, 4, 7
    T_frame = num_frames_for_tokens(seq_len)
    le, te = _encoders(D)

    base = torch.randn(B, T_frame, D)
    mask = torch.ones(B, T_frame)
    # token 1 spans frames [1,4]; keep frame 1 valid, mask frames 2,3,4.
    mask[:, 2:5] = 0
    x1 = {"traj_features": base.clone(), "traj_cond_mask": mask.clone()}
    x2_feats = base.clone()
    x2_feats[:, 2:5] = 99.0                   # change only invalid frames
    x2 = {"traj_features": x2_feats, "traj_cond_mask": mask.clone()}
    with torch.no_grad():
        o1 = encode_traj_batch(x1, seq_len, "cpu", le, te)
        o2 = encode_traj_batch(x2, seq_len, "cpu", le, te)
    assert torch.allclose(o1, o2, atol=1e-5)


def test_partial_token_mask_is_one_by_or_aggregation():
    """(b) A token with >=1 valid frame aggregates to token_mask = 1."""
    mask = torch.zeros(1, num_frames_for_tokens(4))
    mask[0, 1] = 1.0                          # frame 1 (in token 1's span) valid
    tok = frames_to_token_mask(mask, num_tokens=4)
    assert tok[0, 1] == 1.0                   # token 1 valid (any-aggregation)
    assert tok[0, 0] == 0.0                   # token 0 (frame 0) invalid


# ---------------------------------------------------------------------------
# T05 — all-zero frame range → token_mask 0 there
# ---------------------------------------------------------------------------


def test_all_zero_frames_give_zero_token_mask():
    mask = torch.ones(1, num_frames_for_tokens(5))
    # zero out token 3's frames [9,12]
    mask[0, token_start_frame(3):token_start_frame(3) + 4] = 0.0
    tok = frames_to_token_mask(mask, num_tokens=5)
    assert tok[0, 3] == 0.0
    assert tok[0, 2] == 1.0 and tok[0, 4] == 1.0


# ---------------------------------------------------------------------------
# T06 — token 0 (1 frame) vs token k≥1 (4 frames) boundary
# ---------------------------------------------------------------------------


def test_token0_one_frame_vs_tokenk_four_frames():
    T_frame = num_frames_for_tokens(3)        # tokens 0,1,2
    # only frame 0 valid → token 0 valid, tokens 1,2 invalid
    m0 = torch.zeros(1, T_frame)
    m0[0, 0] = 1.0
    t0 = frames_to_token_mask(m0, 3)
    assert t0[0, 0] == 1.0 and t0[0, 1] == 0.0 and t0[0, 2] == 0.0
    # only frame 4 valid (last frame of token 1) → token 1 valid only
    m1 = torch.zeros(1, T_frame)
    m1[0, 4] = 1.0
    t1 = frames_to_token_mask(m1, 3)
    assert t1[0, 1] == 1.0 and t1[0, 0] == 0.0 and t1[0, 2] == 0.0


# ---------------------------------------------------------------------------
# T07 / T08 — external interface: frame-level only
# ---------------------------------------------------------------------------


def test_encode_traj_batch_accepts_frame_level_7d():
    B, seq_len, D = 2, 6, 7
    T_frame = num_frames_for_tokens(seq_len)
    le, te = _encoders(D)
    x = {
        "traj_features": torch.randn(B, T_frame, D),
        "traj_cond_mask": torch.ones(B, T_frame),
    }
    with torch.no_grad():
        out = encode_traj_batch(x, seq_len, "cpu", le, te)
    assert out.shape == (B, seq_len, 16)


def test_all_zero_mask_returns_none_no_control():
    """B-P0-1: a fully-masked traj batch → encode_traj_batch returns None so the
    model runs its no-control path (the encoder bias would otherwise emit a
    nonzero embedding that ControlNet treats as constant control)."""
    import torch.nn as nn

    B, seq_len, D = 2, 6, 7
    T_frame = num_frames_for_tokens(seq_len)
    x = {
        "traj_features": torch.randn(B, T_frame, D),
        "traj_cond_mask": torch.zeros(B, T_frame),   # everything masked
    }
    out = encode_traj_batch(x, seq_len, "cpu", nn.Identity(), nn.Identity())
    assert out is None


def test_horizon_fully_truncated_returns_none():
    """horizon_tokens=0 zeros the whole mask → no valid traj → None."""
    le, te = _encoders(7)
    seq_len = 6
    T_frame = num_frames_for_tokens(seq_len)
    x = {"traj_features": torch.randn(1, T_frame, 7)}   # no base mask → all-ones
    out = encode_traj_batch(x, seq_len, "cpu", le, te, horizon_tokens=0)
    assert out is None


def test_partial_mask_still_returns_embedding():
    """A mask with SOME valid frames is still encoded (not None)."""
    le, te = _encoders(7)
    B, seq_len, D = 1, 6, 7
    T_frame = num_frames_for_tokens(seq_len)
    mask = torch.zeros(B, T_frame)
    mask[:, :10] = 1.0
    x = {"traj_features": torch.randn(B, T_frame, D), "traj_cond_mask": mask}
    out = encode_traj_batch(x, seq_len, "cpu", le, te)
    assert out is not None and out.shape == (B, seq_len, 16)


def test_encode_traj_batch_rejects_token_level_input():
    """Token-level parallel path is disabled: shape[1]==seq_len (seq_len>1)
    means pre-tokenized data was mis-fed → raise."""
    B, seq_len, D = 2, 6, 7
    le, te = _encoders(D)
    x = {"traj_features": torch.randn(B, seq_len, D)}   # token-level shaped
    with pytest.raises(ValueError):
        encode_traj_batch(x, seq_len, "cpu", le, te)
