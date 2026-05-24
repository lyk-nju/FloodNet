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

import pytest

_CFG_PATH = "configs/ldf.yaml"


def _load_cfg_and_data_root():
    """Return (cfg, first_train_meta_path) or (None, None) if unavailable."""
    try:
        from utils.initialize import Config

        cfg = Config(_CFG_PATH).config
        meta = cfg.data.train_meta_paths[0]   # resolves ${dirs.raw_data}
        return cfg, str(meta)
    except Exception:
        return None, None


_CFG, _META = _load_cfg_and_data_root()


def _data_present() -> bool:
    import os
    return _META is not None and os.path.exists(_META)


pytestmark = pytest.mark.skipif(
    not _data_present(), reason="HumanML3D data dir absent (run on the data machine)"
)


def test_7d_dataset_emits_traj_cond_7d():
    from omegaconf import OmegaConf

    from datasets.humanml3d import HumanML3DDataset

    cfg = OmegaConf.create(OmegaConf.to_container(_CFG, resolve=True))
    OmegaConf.update(cfg, "data.traj_feat_dim", 7)
    ds = HumanML3DDataset(cfg, split="train")
    assert len(ds) > 0
    item = ds[0]
    assert "traj_cond_7d" in item, "7D dataset must emit traj_cond_7d"
    arr = item["traj_cond_7d"]
    assert arr.shape[-1] == 7
    # first-frame fwd_delta / yaw_delta are zero (root_to_traj_feats_7d v1 rule)
    assert abs(float(arr[0, 5])) < 1e-4
    assert abs(float(arr[0, 6])) < 1e-4


def test_4d_default_dataset_has_no_traj_cond_7d():
    from omegaconf import OmegaConf

    from datasets.humanml3d import HumanML3DDataset

    cfg = OmegaConf.create(OmegaConf.to_container(_CFG, resolve=True))
    OmegaConf.update(cfg, "data.traj_feat_dim", 4)   # legacy default
    ds = HumanML3DDataset(cfg, split="train")
    item = ds[0]
    assert "traj_cond_7d" not in item            # 4D path unchanged
    assert "traj_features" in item               # legacy 4D feature still emitted
