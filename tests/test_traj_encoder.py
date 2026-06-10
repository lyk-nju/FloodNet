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
from models.diffusion_forcing_wan import DiffForcingWanModel
from utils.token_frame import (
    frames_to_token_mask,
    num_frames_for_tokens,
    token_range_to_frame_slice,
    token_start_frame,
)
from utils.traj_batch import (
    build_traj_token_mask,
    encode_traj_batch,
    frames_to_token_mask_range,
    frames_to_tokens,
    frames_to_tokens_range,
)


def _encoders(out_dim=16, seed=0):
    """Build a (LocalTrajEncoder, TrajEncoder) pair matching the 7D contract."""
    torch.manual_seed(seed)
    le = LocalTrajEncoder().eval()                                    # 7 → 128
    torch.manual_seed(seed + 100)
    te = TrajEncoder(in_dim=LOCAL_OUT_DIM, out_dim=out_dim).eval()    # 128 → out_dim
    return le, te


class _KeepFourFrames(torch.nn.Module):
    """Test-only local encoder: expose the four grouped frame values."""

    def forward(self, x, frame_mask=None):
        return x.squeeze(-1)


class _KeepFirstChannelFourFrames(torch.nn.Module):
    """Test-only local encoder for 7D stream payloads."""

    def forward(self, x, frame_mask=None):
        return x[..., 0]


def _dummy_stream_model():
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model.batch_size = 1
    model.local_traj_encoder = _KeepFirstChannelFourFrames()
    model.traj_encoder = torch.nn.Identity()
    return model


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


def test_local_encoder_zeros_invalid_frames_before_conv():
    """LocalTrajEncoder must zero invalid frames internally — values on
    masked frames must not bleed into neighbors via the kernel-size-3 conv.
    """
    enc = LocalTrajEncoder().eval()
    base = torch.randn(1, 2, 4, 7)
    mask = torch.ones(1, 2, 4)
    mask[:, :, 1] = 0.0   # frame 1 invalid in every token

    perturbed = base.clone()
    perturbed[:, :, 1] = 1e3   # huge value on the masked frame

    with torch.no_grad():
        o_clean = enc(base, frame_mask=mask)
        o_perturbed = enc(perturbed, frame_mask=mask)
    assert torch.allclose(o_clean, o_perturbed, atol=1e-5)


# ---------------------------------------------------------------------------
# Mask effectiveness through the full encode_traj_batch pipeline
# ---------------------------------------------------------------------------


def test_frames_to_tokens_range_prefix_matches_legacy_helper():
    B, seq_len, D = 1, 4, 1
    feats = torch.arange(num_frames_for_tokens(seq_len), dtype=torch.float32).view(B, -1, D)
    legacy = frames_to_tokens(feats, seq_len)
    ranged = frames_to_tokens_range(feats, 0, seq_len)
    assert torch.allclose(ranged, legacy)


def test_frames_to_tokens_range_non_prefix_uses_four_frames_per_token():
    B, seq_len, D = 1, 3, 1
    frame_slice = token_range_to_frame_slice(5, seq_len)
    assert frame_slice.stop - frame_slice.start == 12
    feats = torch.arange(12, dtype=torch.float32).view(B, -1, D)

    grouped = frames_to_tokens_range(feats, 5, seq_len).squeeze(0).squeeze(-1)

    expected = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ]
    )
    assert torch.allclose(grouped, expected)


def test_frames_to_token_mask_range_non_prefix_uses_local_window_origin():
    mask = torch.zeros(1, 8)
    mask[:, 6] = 1.0
    tok = frames_to_token_mask_range(mask, 2, start_token_idx=10)
    assert torch.allclose(tok, torch.tensor([[0.0, 1.0]]))


def test_encode_traj_batch_supports_per_sample_start_token():
    seq_len = 2
    feats = torch.zeros(2, 8, 1)
    feats[0, :, 0] = torch.arange(8, dtype=torch.float32)
    feats[1, :, 0] = 100.0 + torch.arange(8, dtype=torch.float32)
    x = {
        "traj_features": feats,
        "traj_start_token": torch.tensor([0, 5]),
    }

    out = encode_traj_batch(
        x,
        seq_len,
        "cpu",
        _KeepFourFrames(),
        torch.nn.Identity(),
    )

    expected = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]],
            [[100.0, 101.0, 102.0, 103.0], [104.0, 105.0, 106.0, 107.0]],
        ]
    )
    assert torch.allclose(out, expected)


def test_build_traj_token_mask_supports_per_sample_start_token():
    seq_len = 2
    feats = torch.zeros(2, 8, 7)
    mask = torch.zeros(2, 8)
    mask[0, 0] = 1.0
    mask[1, 6] = 1.0
    x = {
        "traj_features": feats,
        "traj_cond_mask": mask,
        "traj_start_token": torch.tensor([0, 5]),
    }

    token_mask = build_traj_token_mask(x, seq_len, "cpu")

    assert torch.allclose(token_mask, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_encode_traj_batch_threads_traj_start_token_into_grouping():
    seq_len = 3
    feats = torch.arange(12, dtype=torch.float32).view(1, 12, 1)
    x = {
        "traj_features": feats,
        "traj_cond_mask": torch.ones(1, 12),
        "traj_start_token": 5,
    }

    out = encode_traj_batch(
        x,
        seq_len,
        "cpu",
        _KeepFourFrames(),
        torch.nn.Identity(),
    )

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ]
    )
    assert torch.allclose(out, expected)


def test_stream_direct_7d_payload_uses_window_start_token():
    model = _dummy_stream_model()
    feats = torch.zeros(1, 12, 7)
    feats[..., 0] = torch.arange(12, dtype=torch.float32).view(1, 12)
    x = {
        "traj_cond_7d_frame": feats,
        "traj_cond_frame_mask": torch.ones(1, 12),
        "traj_start_token": 5,
    }

    emb, lens, token_mask = model._build_stream_direct_traj_condition(
        x, model_sl=3, window_start_token=5, device="cpu",
    )

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ]
    )
    assert torch.allclose(emb, expected)
    assert torch.equal(lens, torch.tensor([3]))
    assert torch.allclose(token_mask, torch.ones(1, 3))


def test_stream_direct_7d_payload_can_encode_future_horizon_tokens():
    model = _dummy_stream_model()
    feats = torch.zeros(1, 20, 7)
    feats[..., 0] = torch.arange(20, dtype=torch.float32).view(1, 20)
    x = {
        "traj_cond_7d_frame": feats,
        "traj_cond_frame_mask": torch.ones(1, 20),
        "traj_start_token": 5,
    }

    emb, lens, token_mask = model._build_stream_direct_traj_condition(
        x, model_sl=3, window_start_token=5, device="cpu", traj_sl=5,
    )

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
            ]
        ]
    )
    assert torch.allclose(emb, expected)
    assert torch.equal(lens, torch.tensor([5]))
    assert torch.allclose(token_mask, torch.ones(1, 5))


def test_stream_direct_7d_payload_groups_frames_by_absolute_start_token():
    model = _dummy_stream_model()
    frame_slice = token_range_to_frame_slice(5, 2)
    feats = torch.zeros(1, frame_slice.stop - frame_slice.start, 7)
    feats[..., 0] = torch.arange(feats.shape[1], dtype=torch.float32).view(1, -1)
    x = {
        "traj_cond_7d_frame": feats,
        "traj_cond_frame_mask": torch.ones(1, feats.shape[1]),
        "traj_start_token": 0,
        "traj_abs_start_token": 5,
        "traj_num_tokens": 2,
    }

    emb, lens, token_mask = model._build_stream_direct_traj_condition(
        x, model_sl=2, window_start_token=0, device="cpu",
    )

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
            ]
        ]
    )
    assert torch.allclose(emb, expected)
    assert torch.equal(lens, torch.tensor([2]))
    assert torch.allclose(token_mask, torch.ones(1, 2))


def test_stream_direct_7d_payload_rejects_payload_start_after_latent_window_start():
    model = _dummy_stream_model()
    feats = torch.zeros(1, 12, 7)
    feats[..., 0] = torch.arange(12, dtype=torch.float32).view(1, 12)
    x = {
        "traj_cond_7d_frame": feats,
        "traj_cond_frame_mask": torch.ones(1, 12),
        "traj_start_token": 6,
        "traj_abs_start_token": 16,
        "traj_num_tokens": 3,
    }

    with pytest.raises(ValueError, match="starts after current latent window start"):
        model._build_stream_direct_traj_condition(
            x, model_sl=3, window_start_token=5, device="cpu",
        )


def test_stream_direct_7d_payload_accepts_runtime_history_plus_horizon_shape():
    model = _dummy_stream_model()
    model_sl = 30
    horizon_tokens = 3
    traj_start_token = 1
    traj_abs_start_token = 31
    traj_num_tokens = model_sl + horizon_tokens
    frame_slice = token_range_to_frame_slice(traj_abs_start_token, traj_num_tokens)
    feats = torch.zeros(1, frame_slice.stop - frame_slice.start, 7)
    feats[..., 0] = torch.arange(feats.shape[1], dtype=torch.float32).view(1, -1)
    x = {
        "traj_cond_7d_frame": feats,
        "traj_cond_frame_mask": torch.ones(1, feats.shape[1]),
        "traj_start_token": traj_start_token,
        "traj_abs_start_token": traj_abs_start_token,
        "traj_num_tokens": traj_num_tokens,
    }

    emb, lens, token_mask = model._build_stream_direct_traj_condition(
        x,
        model_sl=model_sl,
        window_start_token=traj_start_token,
        device="cpu",
    )

    assert emb.shape == (1, traj_num_tokens, 4)
    assert torch.equal(lens, torch.tensor([traj_num_tokens]))
    assert torch.allclose(token_mask, torch.ones(1, traj_num_tokens))
    assert torch.allclose(emb[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.allclose(emb[0, -1], torch.tensor([128.0, 129.0, 130.0, 131.0]))


def test_stream_direct_7d_payload_selects_matching_substep_payload():
    model = _dummy_stream_model()
    top_feats = torch.full((1, 12, 7), 100.0)
    sub_feats = torch.zeros(1, 8, 7)
    sub_feats[..., 0] = torch.arange(8, dtype=torch.float32).view(1, 8)
    x = {
        "traj_cond_7d_frame": top_feats,
        "traj_cond_frame_mask": torch.ones(1, 12),
        "traj_start_token": 5,
        "traj_abs_start_token": 15,
        "traj_num_tokens": 3,
        "traj_substep_payloads": [
            {
                "traj_cond_7d_frame": torch.full((1, 12, 7), 200.0),
                "traj_cond_frame_mask": torch.ones(1, 12),
                "traj_start_token": 5,
                "traj_abs_start_token": 15,
                "traj_num_tokens": 3,
            },
            {
                "traj_cond_7d_frame": sub_feats,
                "traj_cond_frame_mask": torch.ones(1, 8),
                "traj_start_token": 6,
                "traj_abs_start_token": 16,
                "traj_num_tokens": 2,
            },
        ],
    }

    emb, lens, token_mask = model._build_stream_direct_traj_condition(
        x,
        model_sl=2,
        window_start_token=6,
        device="cpu",
    )

    expected = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
    assert torch.allclose(emb, expected)
    assert torch.equal(lens, torch.tensor([2]))
    assert torch.allclose(token_mask, torch.ones(1, 2))


def test_stream_direct_7d_payload_slices_from_earlier_payload_start():
    model = _dummy_stream_model()
    feats = torch.zeros(1, 12, 7)
    feats[..., 0] = torch.arange(12, dtype=torch.float32).view(1, 12)
    x = {
        "traj_cond_7d_frame": feats,
        "traj_cond_frame_mask": torch.ones(1, 12),
        "traj_start_token": 5,
    }

    emb, lens, token_mask = model._build_stream_direct_traj_condition(
        x, model_sl=2, window_start_token=6, device="cpu",
    )

    expected = torch.tensor(
        [
            [
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ]
    )
    assert torch.allclose(emb, expected)
    assert torch.equal(lens, torch.tensor([2]))
    assert torch.allclose(token_mask, torch.ones(1, 2))


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
