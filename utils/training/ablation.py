"""Body 7D fine-tune ablation config deltas (T_B_12).

Each ablation is a set of `--override key=value` config deltas applied on top of
configs/ldf.yaml to produce one training/benchmark variant. The six variants
isolate each gap-closure subitem:

    all_on               5 subitems on, 7D
    no_corruption        history corruption off (T_B_03 / §2.1)
    no_horizon_sim       horizon simulation off (T_B_04 / §2.2)
    no_anchor_canonical  body-window canonicalize off → absolute world xyz (T_B_05 / §2.3)
    no_heading_loss      heading term zeroed in body aux loss (T_B_06 / §2.4)

(The former `no_7d` 4D-legacy baseline was removed: the traj encoder is now 7D
only — LocalTrajEncoder rejects in_dim=4 — so a 4D ablation can't be built.)

Bools are emitted lowercase so utils/initialize._convert_value parses them.

CLI: `python -m utils.training.ablation <name>` prints the space-joined override
args for that ablation (consumed by scripts/bench_body_7d_ablation.sh).
"""

from __future__ import annotations

_BASE_7D = {
    "model.params.traj_encoder_in_dim": 7,
    "data.traj_feat_dim": 7,
}


def body_ablation_overrides() -> dict[str, dict]:
    """Return {ablation_name: {config_dotpath: value}}."""
    base = dict(_BASE_7D)
    return {
        "all_on": dict(base),
        "no_corruption": {**base, "history_corruption.enabled": False},
        "no_horizon_sim": {**base, "horizon_sim.enabled": False},
        "no_anchor_canonical": {**base, "anchor_canonicalize.enabled": False},
        # body_aux_loss can't be disabled in 7D (the heading channels need
        # supervision — _check_preconditions raises); ablate by zeroing the
        # heading weight while keeping the other four terms.
        "no_heading_loss": {**base, "body_aux_loss.weights.heading": 0.0},
    }


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def overrides_to_cli(overrides: dict) -> list[str]:
    """Format an override dict as ['key=value', ...] for --override."""
    return [f"{k}={_fmt(v)}" for k, v in overrides.items()]


def apply_overrides(cfg, overrides: dict):
    """Apply an override dict to an OmegaConf cfg (returns a new cfg)."""
    from omegaconf import OmegaConf

    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in overrides.items():
        OmegaConf.update(out, key, value, force_add=True)
    return out


__all__ = [
    "body_ablation_overrides",
    "overrides_to_cli",
    "apply_overrides",
]


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else ""
    table = body_ablation_overrides()
    if name not in table:
        sys.stderr.write(
            f"unknown ablation '{name}'. choices: {', '.join(table)}\n"
        )
        sys.exit(2)
    print(" ".join(overrides_to_cli(table[name])))
