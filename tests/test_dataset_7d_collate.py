"""Unit tests for T_B_09: dataset 7D traj collate (flag-gated).

Real HumanML3D/BABEL data isn't on this host, so the dataset __getitem__ path is
exercised via its building block `extract_root_traj_feats_7d_263` (which the
flag-gated branch calls). Plus prepare_model_input preference + collate padding.
"""

from __future__ import annotations

import numpy as np
import torch

from datasets.multi import collate_fn
from utils.motion_process import (
    extract_root_traj_feats_7d_263,
    recover_root_rot_pos,
    root_to_traj_feats_7d,
)
from utils.training.model_batch import prepare_model_input


# ---------------------------------------------------------------------------
# 1. dataset emission building block: 263D → [T, 7]
# ---------------------------------------------------------------------------


def test_extract_root_traj_feats_7d_shape_and_first_frame_zero():
    T = 20
    feat263 = np.random.randn(T, 263).astype(np.float32)
    out = extract_root_traj_feats_7d_263(feat263)
    assert out.shape == (T, 7)
    # 7D = [x, y, z, cos, sin, fwd_delta, yaw_delta]; first-frame deltas are 0.
    assert abs(float(out[0, 5])) < 1e-6   # fwd_delta[0]
    assert abs(float(out[0, 6])) < 1e-6   # yaw_delta[0]
    # cos^2 + sin^2 ~= 1 (unit heading)
    assert np.allclose(out[:, 3] ** 2 + out[:, 4] ** 2, 1.0, atol=1e-4)


def test_extract_7d_matches_recover_plus_root_to_traj_feats_7d():
    feat263 = np.random.randn(15, 263).astype(np.float32)
    out = extract_root_traj_feats_7d_263(feat263)
    vec = torch.from_numpy(feat263).float().unsqueeze(0)
    quat, pos = recover_root_rot_pos(vec)
    ref = root_to_traj_feats_7d(quat, pos).squeeze(0).numpy()
    assert np.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. prepare_model_input prefers traj_cond_7d over the 4D path
# ---------------------------------------------------------------------------


def test_prepare_model_input_prefers_traj_cond_7d():
    traj7 = np.random.randn(10, 7).astype(np.float32)
    batch = {
        "token": torch.zeros(4, 8),
        "token_length": 4,
        "traj_cond_7d": traj7,
        "traj_cond": np.random.randn(10, 3).astype(np.float32),
        "traj": np.random.randn(10, 3).astype(np.float32),
        "traj_length": 10,
        "traj_cond_mask": np.ones(10, dtype=np.float32),
    }
    mb = prepare_model_input(batch)
    # 7D routed as the traj feature; raw xyz kept for control-loss GT.
    assert mb["traj_features"] is traj7
    assert mb["traj"].shape[-1] == 3


def test_prepare_model_input_falls_back_to_4d_without_7d():
    batch = {
        "token": torch.zeros(4, 8),
        "token_length": 4,
        "traj_cond": np.random.randn(10, 3).astype(np.float32),
        "traj": np.random.randn(10, 3).astype(np.float32),
        "traj_length": 10,
        "traj_cond_mask": np.ones(10, dtype=np.float32),
        "traj_features": np.random.randn(10, 4).astype(np.float32),
    }
    mb = prepare_model_input(batch)
    # 4D path: traj_features dropped so encode_traj_batch uses traj_cond xyz.
    assert "traj_features" not in mb
    assert mb["traj"].shape[-1] == 3


# ---------------------------------------------------------------------------
# 2. collate pads traj_cond_7d to [B, T_max, 7]
# ---------------------------------------------------------------------------


def test_collate_pads_traj_cond_7d():
    b = [
        {"traj_cond_7d": np.random.randn(5, 7).astype(np.float32)},
        {"traj_cond_7d": np.random.randn(8, 7).astype(np.float32)},
    ]
    out = collate_fn(b)
    assert out["traj_cond_7d"].shape == (2, 8, 7)
    # first sample's tail is zero-padded
    assert torch.allclose(out["traj_cond_7d"][0, 5:], torch.zeros(3, 7))
