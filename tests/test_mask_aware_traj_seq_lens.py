"""B-P0-1 mask-aware traj_seq_lens: build_traj_token_mask + _get_traj_seq_lens
truncation + _prepare_traj_condition wiring.

The build_traj_token_mask logic is pure (no model). The _get_traj_seq_lens /
_prepare_traj_condition tests build a tiny DiffForcingWanModel via a precomputed
text fixture (no T5 ckpt). The ControlNet-residual value-invariance test (tail
mask=0 change → residual unchanged; sparse middle-hole non-regression) needs real
VAE-latent shapes and is run on the runtime box — see _get_traj_seq_lens note.
"""

from __future__ import annotations

import pytest
import torch

from utils.token_frame import num_frames_for_tokens
from utils.traj_batch import build_traj_token_mask

_TEXT_DIM = 4096


# ---------------------------------------------------------------------------
# build_traj_token_mask (single source, pure)
# ---------------------------------------------------------------------------


def test_token_mask_horizon_pure_suffix():
    seq_len = 6
    T_frame = num_frames_for_tokens(seq_len)
    x = {"traj_features": torch.randn(1, T_frame, 7)}   # no base mask → all-ones
    # horizon=4, active_end=0 → cutoff token_start_frame(4)=13 → tokens 0..3 valid
    tm = build_traj_token_mask(x, seq_len, "cpu", horizon_tokens=4, horizon_active_end_token=0)
    assert tm.shape == (1, seq_len)
    assert tm[0].tolist() == [1, 1, 1, 1, 0, 0]


def test_token_mask_middle_hole():
    seq_len = 6
    T_frame = num_frames_for_tokens(seq_len)
    mask = torch.ones(1, T_frame)
    mask[:, 5:9] = 0   # token 2 frames [5,8] → hole; tokens 3,4,5 valid
    x = {"traj_features": torch.randn(1, T_frame, 7), "traj_cond_mask": mask}
    tm = build_traj_token_mask(x, seq_len, "cpu")
    assert tm[0, 2] == 0
    assert tm[0, 3] == 1 and tm[0, 5] == 1   # hole, not a suffix


def test_token_mask_none_when_no_mask_no_horizon():
    x = {"traj_features": torch.randn(1, 21, 7)}
    assert build_traj_token_mask(x, 6, "cpu") is None


# ---------------------------------------------------------------------------
# model-level _get_traj_seq_lens (tiny model)
# ---------------------------------------------------------------------------


@pytest.fixture
def model(tmp_path):
    from models.diffusion_forcing_wan import DiffForcingWanModel

    p = tmp_path / "t5.pt"
    torch.save({"embeddings": {"": torch.zeros(2, _TEXT_DIM)}, "text_dim": _TEXT_DIM}, p)
    return DiffForcingWanModel(
        input_dim=4, hidden_dim=64, ffn_dim=128, freq_dim=64,
        num_heads=2, num_layers=1, text_len=8, traj_encoder_in_dim=7,
        use_precomputed_text_emb=True, precomputed_text_emb_path=str(p),
    )


def _x(seq_len, *, mask=None):
    T_frame = num_frames_for_tokens(seq_len)
    x = {"traj_features": torch.randn(1, T_frame, 7),
         "feature_length": torch.tensor([seq_len])}
    if mask is not None:
        x["traj_cond_mask"] = mask
    return x


def test_seq_lens_horizon_truncates_pure_suffix(model):
    seq_len = 6
    sl = model._get_traj_seq_lens(_x(seq_len), seq_len, "cpu",
                                  horizon_tokens=4, horizon_active_end=0)
    assert sl.tolist() == [4]   # base 6 truncated to the valid prefix 4


def test_seq_lens_middle_hole_not_truncated(model):
    seq_len = 6
    T_frame = num_frames_for_tokens(seq_len)
    mask = torch.ones(1, T_frame)
    mask[:, 5:9] = 0   # hole at token 2; tokens 3..5 valid
    sl = model._get_traj_seq_lens(_x(seq_len, mask=mask), seq_len, "cpu")
    assert sl.tolist() == [6]   # middle hole must NOT shorten attention


def test_seq_lens_all_zero_mask_is_zero(model):
    seq_len = 6
    T_frame = num_frames_for_tokens(seq_len)
    mask = torch.zeros(1, T_frame)
    sl = model._get_traj_seq_lens(_x(seq_len, mask=mask), seq_len, "cpu")
    assert sl.tolist() == [0]


def test_seq_lens_no_horizon_no_mask_is_base(model):
    seq_len = 6
    sl = model._get_traj_seq_lens(_x(seq_len), seq_len, "cpu")
    assert sl.tolist() == [6]   # full length when nothing masks


def test_prepare_traj_condition_returns_truncated_seq_lens(model):
    """Wiring: _prepare_traj_condition threads horizon into _get_traj_seq_lens."""
    seq_len = 6
    _, traj_seq_lens, dropped, _tmask = model._prepare_traj_condition(
        _x(seq_len), seq_len, "cpu", traj_dropped_override=False,
        horizon_tokens=4, horizon_active_end=0,
    )
    assert not dropped
    assert traj_seq_lens.tolist() == [4]   # truncated via the wired horizon
