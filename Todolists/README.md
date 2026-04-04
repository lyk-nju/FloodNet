## Todolists — FloodNet ControlNet Trajectory Control

This folder contains the design specification and implementation guide for adding ControlNet-based trajectory control to FloodDiffusion.

- **[`target.md`](target.md)**: Overall spec, global consistency rules, cross-task dependency order, code map, implementation status table.
- **[`Task1-input.md`](Task1-input.md)**: Trajectory condition fields, exact batch tensor shapes, token/frame alignment (VAE factor 4), mask semantics, stream buffer specification.
- **[`Task2-controlnet.md`](Task2-controlnet.md)**: `WanControlNet` class — constructor signature, forward I/O shapes, zero-init residual heads, `init_from_backbone`, freeze strategy.
- **[`Task3-backbone-inject.md`](Task3-backbone-inject.md)**: `WanModel.forward` residual injection — exact code, tensor shapes, compatibility with FlexTraj, verification steps.
- **[`Task4-inference-stream.md`](Task4-inference-stream.md)**: `generate` / `stream_generate` / `stream_generate_step` — required call sequence, CFG handling for ControlNet, traj cache consistency.
- **[`Task5-loss.md`](Task5-loss.md)**: Motion-space control loss — exact formula, VAE decode steps, active window alignment, axis convention `[x, z]` = indices `[0, 2]`.
- **[`Task6-config.md`](Task6-config.md)**: Complete YAML key reference table, freeze strategy code, checkpoint compatibility, mutual exclusion rules.

**When changing any design decision** (e.g., residual injection point, loss time range, mask semantics), update `target.md` global rules **and** the relevant Task document. Inconsistencies between training and inference are the most common source of subtle bugs.
