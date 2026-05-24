"""7D-path tests that need real data / VAE — RUN ON THE DATA MACHINE.

These are skipped automatically when the HumanML3D data dir is absent (e.g. on a
dev box). On a machine with data they verify the parts of the flag-gated 4D→7D
migration that cannot be unit-tested without data:

  - the dataset actually emits `traj_cond_7d` [T,7] in 7D mode (T_B_09 __getitem__).

Run with:  ./scripts/run_pytest.sh tests/test_7d_datamachine.py -v

Manual runtime checks NOT auto-tested here (verify by launching a short 7D
fine-tune; see docs/TODO.md T_B_11):
  - step_460000.ckpt loads into the 7D model via the T_B_08 hook (no shape error);
  - one SF step: total loss + body_aux/* (root_xz/root_y/heading/fwd_delta/
    yaw_delta) are finite and decreasing;
  - anchor_canonicalize/valid_frac ~ 1.0 on normal-length clips;
  - history_corruption/applied and horizon_sim/horizon_tokens log sane values;
  - traj_encoder_in_dim=7 with body_aux_loss.enabled=false raises at startup.
"""

from __future__ import annotations

import os

import pytest

_CFG_PATH = "configs/ldf.yaml"


def _resolve_raw_data_dir() -> str | None:
    """B-P1-1: locate the HumanML3D data root. Priority:
      1. $FLOODNET_RAW_DATA_DIR (override on any machine);
      2. the config's resolved dirs.raw_data.
    Returns None if neither yields an existing HumanML3D/train.txt.
    """
    candidates = []
    env = os.environ.get("FLOODNET_RAW_DATA_DIR")
    if env:
        candidates.append(env)
    try:
        from utils.initialize import Config

        candidates.append(str(Config(_CFG_PATH).config.dirs.raw_data))
    except Exception:
        pass
    for root in candidates:
        if os.path.exists(os.path.join(root, "HumanML3D", "train.txt")):
            return root
    return None


_RAW_DIR = _resolve_raw_data_dir()

pytestmark = pytest.mark.skipif(
    _RAW_DIR is None,
    reason="HumanML3D data absent (set FLOODNET_RAW_DATA_DIR; run on data machine)",
)


def _make_cfg(traj_feat_dim: int):
    """Build a resolved ldf cfg pinned to <raw_dir>/HumanML3D/{train,val}.txt
    (NOT the config's train_difficult.txt, which may not exist on this host)."""
    from omegaconf import OmegaConf

    from utils.initialize import Config

    cfg = OmegaConf.create(OmegaConf.to_container(Config(_CFG_PATH).config, resolve=True))
    hml = os.path.join(_RAW_DIR, "HumanML3D")
    OmegaConf.update(cfg, "dirs.raw_data", _RAW_DIR)
    OmegaConf.update(cfg, "data.train_meta_paths", [os.path.join(hml, "train.txt")])
    OmegaConf.update(cfg, "data.val_meta_paths", [os.path.join(hml, "val.txt")])
    OmegaConf.update(cfg, "data.traj_feat_dim", traj_feat_dim)
    return cfg


def test_7d_dataset_emits_traj_cond_7d():
    from datasets.humanml3d import HumanML3DDataset

    ds = HumanML3DDataset(_make_cfg(7), split="train")
    assert len(ds) > 0
    item = ds[0]
    assert "traj_cond_7d" in item, "7D dataset must emit traj_cond_7d"
    arr = item["traj_cond_7d"]
    assert arr.shape[-1] == 7
    # first-frame fwd_delta / yaw_delta are zero (root_to_traj_feats_7d v1 rule)
    assert abs(float(arr[0, 5])) < 1e-4
    assert abs(float(arr[0, 6])) < 1e-4


def test_4d_default_dataset_has_no_traj_cond_7d():
    from datasets.humanml3d import HumanML3DDataset

    ds = HumanML3DDataset(_make_cfg(4), split="train")
    item = ds[0]
    assert "traj_cond_7d" not in item            # 4D path unchanged
    assert "traj_features" in item               # legacy 4D feature still emitted
