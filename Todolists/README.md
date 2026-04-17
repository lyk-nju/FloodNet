## Todolists — FloodNet ControlNet Trajectory Control

This folder contains the design specification, implementation guide, and experiment records for FloodNet's ControlNet-based trajectory control.

---

## Documents

### Design Specs

- **[`target.md`](target.md)**: Overall spec, global consistency rules, cross-task dependency order, code map, implementation status table, bug fix log, and diagnosed experiment findings.
- **[`Task1-input.md`](Task1-input.md)**: Trajectory condition fields, exact batch tensor shapes, token/frame alignment (VAE factor 4), mask semantics, stream buffer specification.
- **[`Task2-controlnet.md`](Task2-controlnet.md)**: `WanControlNet` class — constructor signature, forward I/O shapes, zero-init residual heads, `init_from_backbone`, freeze strategy.
- **[`Task3-backbone-inject.md`](Task3-backbone-inject.md)**: `WanModel.forward` residual injection — exact code, tensor shapes, compatibility with FlexTraj, verification steps.
- **[`Task4-inference-stream.md`](Task4-inference-stream.md)**: `generate` / `stream_generate` / `stream_generate_step` — required call sequence, CFG handling (2-batch and else paths), bug fixes (CFG null branch + stream LocalTrajEncoder).
- **[`Task5-loss.md`](Task5-loss.md)**: Motion-space control loss — full-sequence VAE decode, supervision **only** on the active-window frame slice, axis convention `[x, z]` = indices `[0, 2]`.
- **[`Task6-config.md`](Task6-config.md)**: Complete YAML key reference table, freeze strategy code, checkpoint compatibility, mutual exclusion rules.
- **[`Task7-scheduled-sampling.md`](Task7-scheduled-sampling.md)**: Scheduled Sampling training strategy — reduces teacher-forcing exposure bias; `scheduled_sampling_prob` config key; recommended curriculum.
- **[`Task8-smooth-root.md`](Task8-smooth-root.md)**: Smooth Root trajectory conditioning — borrowed from Kimodo (NVIDIA); reduces error accumulation for long sequences.

### Analysis & Experiments

- **[`Experiment-TODO-forward-generate-gap.md`](Experiment-TODO-forward-generate-gap.md)**: Root cause analysis of the forward/generate gap. Experiments 1–3 completed; findings: error accumulation is primary cause, TF bias secondary (~29%).
- **[`Note-control-loss-origin.md`](Note-control-loss-origin.md)**: Background note on why full-sequence VAE decode is required for correct active-window control loss.
- **[`01_controlnet_trajectory_on_flooddiffusion.md`](01_controlnet_trajectory_on_flooddiffusion.md)**: Original design brief for the ControlNet trajectory task.

---

## Current Status Summary

| Task | Status | Key File |
|------|--------|---------|
| §1 Input fields | Done | `datasets/humanml3d.py`, `utils/traj_batch.py` |
| §2 WanControlNet | Done | `models/tools/wan_controlnet.py` |
| §3 Backbone inject | Done | `models/tools/wan_model.py` |
| §4 Inference paths | Done + Bug-fixed | `models/diffusion_forcing_wan.py` |
| §5 Control loss | Done | `train_ldf.py` |
| §6 Config | Done | `configs/ldf.yaml` |
| §7 Scheduled Sampling | Done (code), pending validation | `models/diffusion_forcing_wan.py` |
| §8 Smooth Root | In Progress | `datasets/humanml3d.py` (planned) |

---

## Important Rules

**When changing any design decision** (e.g., residual injection point, loss time range, mask semantics, CFG behavior), update `target.md` global rules **and** the relevant Task document. Inconsistencies between training and inference are the most common source of subtle bugs.

**Key bugs fixed (2026-04)**:
1. CFG null branch reused cond ControlNet residuals in the `else` path → fixed in all three inference functions
2. `_stream_build_traj_emb` traj_buffer path bypassed `LocalTrajEncoder` → fixed with same 4-frame grouping as training path

See [`target.md`](target.md) §Diagnosed Issues and §Global Consistency Rules for full details.
