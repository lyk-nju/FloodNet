"""Unit tests for the 7D-only traj encoder rewrite.

Contract:
  LocalTrajEncoder: (B, T, 4, 7) → (B, T, 128)        (Conv1d 7→64→128 + masked-mean)
  TrajEncoder    : (B, T, 128)   → (B, T, out_dim)    (LayerNorm + 2-layer MLP)

The 4D legacy path is gone; legacy ckpts have their traj weights stripped at
load time via utils.training.ckpt_compat.strip_legacy_traj_encoder_weights.
"""

from __future__ import annotations

import pytest
import torch

from models.tools.traj_encoder import (
    LOCAL_OUT_DIM,
    TRAJ_OUT_DIM,
    LocalTrajEncoder,
    TrajEncoder,
)
from utils.token_frame import (
    frames_to_token_mask,
    num_frames_for_tokens,
    token_start_frame,
)
from utils.traj_batch import encode_traj_batch


def _encoders(out_dim=16, seed=0):
    """Build a (LocalTrajEncoder, TrajEncoder) pair matching the 7D contract."""
    torch.manual_seed(seed)
    le = LocalTrajEncoder().eval()                                    # 7 → 128
    torch.manual_seed(seed + 100)
    te = TrajEncoder(in_dim=LOCAL_OUT_DIM, out_dim=out_dim).eval()    # 128 → out_dim
    return le, te


# ---------------------------------------------------------------------------
# Forward shapes / 7D-only contract
# ---------------------------------------------------------------------------


def test_local_encoder_7d_forward_shape():
    enc = LocalTrajEncoder()
    out = enc(torch.randn(2, 5, 4, 7))
    assert out.shape == (2, 5, LOCAL_OUT_DIM)


def test_traj_encoder_default_forward_shape():
    enc = TrajEncoder()                                  # 128 → 128
    out = enc(torch.randn(2, 5, LOCAL_OUT_DIM))
    assert out.shape == (2, 5, TRAJ_OUT_DIM)


def test_local_encoder_rejects_non_7d_input():
    enc = LocalTrajEncoder()
    with pytest.raises(ValueError):
        enc(torch.randn(2, 5, 4, 4))                     # last dim != 7


def test_local_encoder_rejects_non_7_in_dim_param():
    with pytest.raises(ValueError):
        LocalTrajEncoder(in_dim=4)                       # 7D-only constructor


def test_traj_encoder_rejects_dim_mismatch():
    enc = TrajEncoder()                                  # configured for 128
    with pytest.raises(ValueError):
        enc(torch.randn(2, 5, 7))                        # last dim != 128


# ---------------------------------------------------------------------------
# Mask effectiveness through the full encode_traj_batch pipeline
# ---------------------------------------------------------------------------


def test_mask_effectiveness_through_encode_traj_batch():
    """Inputs differing ONLY on masked frames produce identical output."""
    B, seq_len, D = 1, 6, 7
    T_frame = num_frames_for_tokens(seq_len)
    le, te = _encoders()

    base = torch.randn(B, T_frame, D)
    mask = torch.ones(B, T_frame)
    mask[:, 5:9] = 0

    x1 = {"traj_features": base.clone(), "traj_cond_mask": mask.clone()}
    x2_feats = base.clone()
    x2_feats[:, 5:9] = torch.randn(B, 4, D)
    x2 = {"traj_features": x2_feats, "traj_cond_mask": mask.clone()}

    with torch.no_grad():
        o1 = encode_traj_batch(x1, seq_len, "cpu", le, te)
        o2 = encode_traj_batch(x2, seq_len, "cpu", le, te)
    assert torch.allclose(o1, o2, atol=1e-5)


def test_partial_token_invalid_frames_dont_affect_output():
    """Within a token, changing only masked frames leaves output unchanged."""
    B, seq_len, D = 1, 4, 7
    T_frame = num_frames_for_tokens(seq_len)
    le, te = _encoders()

    base = torch.randn(B, T_frame, D)
    mask = torch.ones(B, T_frame)
    mask[:, 2:5] = 0
    x1 = {"traj_features": base.clone(), "traj_cond_mask": mask.clone()}
    x2_feats = base.clone()
    x2_feats[:, 2:5] = 99.0
    x2 = {"traj_features": x2_feats, "traj_cond_mask": mask.clone()}
    with torch.no_grad():
        o1 = encode_traj_batch(x1, seq_len, "cpu", le, te)
        o2 = encode_traj_batch(x2, seq_len, "cpu", le, te)
    assert torch.allclose(o1, o2, atol=1e-5)


def test_partial_token_mask_is_one_by_or_aggregation():
    """A token with >=1 valid frame aggregates to token_mask = 1."""
    mask = torch.zeros(1, num_frames_for_tokens(4))
    mask[0, 1] = 1.0
    tok = frames_to_token_mask(mask, num_tokens=4)
    assert tok[0, 1] == 1.0
    assert tok[0, 0] == 0.0


def test_all_zero_frames_give_zero_token_mask():
    mask = torch.ones(1, num_frames_for_tokens(5))
    mask[0, token_start_frame(3):token_start_frame(3) + 4] = 0.0
    tok = frames_to_token_mask(mask, num_tokens=5)
    assert tok[0, 3] == 0.0
    assert tok[0, 2] == 1.0 and tok[0, 4] == 1.0


def test_token0_one_frame_vs_tokenk_four_frames():
    T_frame = num_frames_for_tokens(3)
    m0 = torch.zeros(1, T_frame)
    m0[0, 0] = 1.0
    t0 = frames_to_token_mask(m0, 3)
    assert t0[0, 0] == 1.0 and t0[0, 1] == 0.0 and t0[0, 2] == 0.0
    m1 = torch.zeros(1, T_frame)
    m1[0, 4] = 1.0
    t1 = frames_to_token_mask(m1, 3)
    assert t1[0, 1] == 1.0 and t1[0, 0] == 0.0 and t1[0, 2] == 0.0


# ---------------------------------------------------------------------------
# encode_traj_batch external contract
# ---------------------------------------------------------------------------


def test_encode_traj_batch_accepts_frame_level_7d():
    B, seq_len, D = 2, 6, 7
    T_frame = num_frames_for_tokens(seq_len)
    le, te = _encoders(out_dim=16)
    x = {
        "traj_features": torch.randn(B, T_frame, D),
        "traj_cond_mask": torch.ones(B, T_frame),
    }
    with torch.no_grad():
        out = encode_traj_batch(x, seq_len, "cpu", le, te)
    assert out.shape == (B, seq_len, 16)


def test_all_zero_mask_returns_none_no_control():
    """Fully-masked traj batch → encode_traj_batch returns None (no-control path)."""
    import torch.nn as nn

    B, seq_len, D = 2, 6, 7
    T_frame = num_frames_for_tokens(seq_len)
    x = {
        "traj_features": torch.randn(B, T_frame, D),
        "traj_cond_mask": torch.zeros(B, T_frame),
    }
    out = encode_traj_batch(x, seq_len, "cpu", nn.Identity(), nn.Identity())
    assert out is None


def test_horizon_fully_truncated_returns_none():
    le, te = _encoders()
    seq_len = 6
    T_frame = num_frames_for_tokens(seq_len)
    x = {"traj_features": torch.randn(1, T_frame, 7)}
    out = encode_traj_batch(x, seq_len, "cpu", le, te, horizon_tokens=0)
    assert out is None


def test_partial_mask_still_returns_embedding():
    le, te = _encoders(out_dim=16)
    B, seq_len, D = 1, 6, 7
    T_frame = num_frames_for_tokens(seq_len)
    mask = torch.zeros(B, T_frame)
    mask[:, :10] = 1.0
    x = {"traj_features": torch.randn(B, T_frame, D), "traj_cond_mask": mask}
    out = encode_traj_batch(x, seq_len, "cpu", le, te)
    assert out is not None and out.shape == (B, seq_len, 16)


def test_invalid_token_embeddings_zeroed_post_encoder():
    """Tokens whose covered frames are ALL masked get a zero embedding."""
    le, te = _encoders(out_dim=16)
    B, seq_len, D = 1, 8, 7
    T_frame = num_frames_for_tokens(seq_len)
    mask = torch.zeros(B, T_frame)
    mask[:, :17] = 1.0
    x = {"traj_features": torch.randn(B, T_frame, D), "traj_cond_mask": mask}
    out = encode_traj_batch(x, seq_len, "cpu", le, te)
    tmask = frames_to_token_mask(mask, seq_len)[0]
    invalid = tmask == 0
    assert invalid.any()
    assert torch.count_nonzero(out[0][invalid]) == 0
    assert torch.count_nonzero(out[0][~invalid]) > 0


def test_encode_traj_batch_rejects_token_level_input():
    B, seq_len, D = 2, 6, 7
    le, te = _encoders()
    x = {"traj_features": torch.randn(B, seq_len, D)}
    with pytest.raises(ValueError):
        encode_traj_batch(x, seq_len, "cpu", le, te)


def test_encode_traj_batch_returns_token_mask_when_requested():
    """return_token_mask=True returns (emb, token_mask) — the token_mask is
    the OR of the frame-mask aggregation, used by ControlNet to gate
    traj_in_proj output post-projection."""
    B, seq_len, D = 1, 6, 7
    T_frame = num_frames_for_tokens(seq_len)
    le, te = _encoders(out_dim=16)
    mask = torch.zeros(B, T_frame)
    mask[:, :10] = 1.0
    x = {"traj_features": torch.randn(B, T_frame, D), "traj_cond_mask": mask}
    out, tmask = encode_traj_batch(
        x, seq_len, "cpu", le, te, return_token_mask=True,
    )
    assert out.shape == (B, seq_len, 16)
    assert tmask is not None and tmask.shape == (B, seq_len)
    # any-aggregation: tokens with any valid frame are 1
    expected = frames_to_token_mask(mask, seq_len)
    assert torch.allclose(tmask, expected)
