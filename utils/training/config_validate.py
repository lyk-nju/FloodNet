"""Config consistency validation for the flag-gated 4D→7D traj migration (T_B_10).

The 7D path is controlled by two flags that MUST agree:
  - data.traj_feat_dim                 (dataset emits 4D or 7D traj_cond)
  - model.params.traj_encoder_in_dim   (encoder consumes 4D or 7D)
A mismatch silently feeds 7D data into a 4D encoder (or vice-versa). This module
fails fast at startup instead.
"""

from __future__ import annotations

from omegaconf import OmegaConf


def validate_traj_dim_consistency(cfg) -> int:
    """Check the two traj-dim flags agree and are 4 or 7. Returns the dim.

    Raises ValueError on mismatch or an unsupported value.
    """
    data_dim = int(OmegaConf.select(cfg, "data.traj_feat_dim", default=4))
    model_dim = int(OmegaConf.select(cfg, "model.params.traj_encoder_in_dim", default=4))
    if data_dim != model_dim:
        raise ValueError(
            f"traj dim mismatch: data.traj_feat_dim={data_dim} != "
            f"model.params.traj_encoder_in_dim={model_dim}. They must agree "
            "(both 4 = legacy, both 7 = 7D fine-tune). See T_B_10."
        )
    if data_dim not in (4, 7):
        raise ValueError(f"traj dim must be 4 or 7, got {data_dim}.")
    return data_dim


__all__ = ["validate_traj_dim_consistency"]
