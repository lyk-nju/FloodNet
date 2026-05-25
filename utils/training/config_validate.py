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


def validate_7d_requires_self_forcing(cfg) -> None:
    """7D path requires self-forcing — fail fast otherwise. No-op for 4D.

    The two correctness guarantees of the 7D traj path live ONLY inside the
    self-forcing trainer:
      - body_aux_loss, which is the SOLE supervision of the new 7D heading
        channels (computed in SelfForcingTrainer; the regular _step path never
        computes it), and
      - the body-window world->local canonicalize (apply_body_window_canonicalize,
        called only from the SF path) that matches the streaming-inference
        distribution.
    So a 7D config with self_forcing_enabled=false would silently train the new
    heading channels unsupervised on uncanonicalized world-frame traj cond. The
    in-SF "traj_encoder_in_dim=7 requires body_aux_loss" guard never runs when SF
    is off, so enforce 7D => self_forcing here (at module construction).
    """
    dim = int(OmegaConf.select(cfg, "model.params.traj_encoder_in_dim", default=4))
    if dim != 7:
        return
    sf = bool(OmegaConf.select(cfg, "model.params.self_forcing_enabled", default=False))
    if not sf:
        raise ValueError(
            "traj_encoder_in_dim=7 requires model.params.self_forcing_enabled=true: "
            "7D heading supervision (body_aux_loss) and the body-window "
            "canonicalize are self-forcing-only, so a non-SF 7D run trains the new "
            "heading channels unsupervised on world-frame traj cond. Enable "
            "self_forcing or use the 4D path. See T_B_10."
        )


__all__ = ["validate_traj_dim_consistency", "validate_7d_requires_self_forcing"]
