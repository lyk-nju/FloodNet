## FloodNet ControlNet Trajectory Control: Design Spec and Code Map

This document decomposes "adding a ControlNet branch to FloodDiffusion / Diffusion Forcing (Wan backbone) for trajectory control" into implementable module tasks, and defines **global consistency rules** that must be updated everywhere when any one rule changes.

---

## Implementation Status

| Task | File | Status | Summary |
|------|------|--------|---------|
| §1 | [`Task1-input.md`](Task1-input.md) | **Done** | `traj_features`/mask/length fields, token/frame alignment, stream buffer |
| §2 | [`Task2-controlnet.md`](Task2-controlnet.md) | **Done** | `WanControlNet` structure, zero-init residuals, initialization/freezing |
| §3 | [`Task3-backbone-inject.md`](Task3-backbone-inject.md) | **Done** | `WanModel.forward` residual injection interface |
| §4 | [`Task4-inference-stream.md`](Task4-inference-stream.md) | **Done** | ControlNet in all inference paths; CFG reuses cond residuals for null |
| §5 | [`Task5-loss.md`](Task5-loss.md) | **Done** | `L_control_xz` after VAE decode, active window alignment |
| §6 | [`Task6-config.md`](Task6-config.md) | **Done** | Config keys, freeze strategy, checkpoint compatibility |

---

## Task Dependency Order

```
Task6 (config keys) ──► Task2 (WanControlNet init)
                              │
Task1 (input fields) ─────────┼──► Task4 (inference paths)
                              │         │
                        Task3 (inject)  │
                              │         │
                         Task5 (loss) ◄─┘
```

- Task6 must be settled first (config keys flow into Task2 constructor and Task1 dataset fields).
- Task1 and Task2 can proceed in parallel once Task6 is settled.
- Task3 requires Task2 (needs `WanControlNet` to exist) and `WanModel` (already present).
- Task4 requires Task1 (traj fields) + Task2 (controlnet instance) + Task3 (injection point).
- Task5 requires Task4 (access to `control_aux["pred_x0_latent_list"]`).

---

## Global Consistency Rules (must be kept in sync everywhere)

1. **ControlNet scope**: ControlNet outputs **per-layer residuals added to the backbone's hidden states**; applies only to **latent tokens**. Under standard ControlNet training, **the backbone receives no traj tokens / traj embeddings** — trajectory conditions enter only through the ControlNet branch, then propagate via residuals. This keeps backbone behavior identical to the original FloodDiffusion.

2. **Residual tensor shape**: each residual is `(B, seq_len, dim)` — **only latent token positions**. Not `(B, 2*seq_len, dim)` (that would corrupt traj token positions when FlexTraj is also enabled).

3. **Zero-init rule**: all ControlNet residual output heads (`zero_out[i]`) are initialized to exact zero weights and zero bias. This ensures the model with ControlNet added has identical behavior to the backbone alone at initialization, which stabilizes Stage-1 training.

4. **Active window rule (FloodDiffusion)**:
   - The diffusion forcing MSE is computed only on the **last `chunk_size` token positions** (the active window).
   - `L_control_xz` must be aligned to the **same active window**: same token indices → same frame indices after VAE decode.
   - **Implementation detail** (`train_ldf.py::_compute_control_loss_xz`): VAE **decodes the full predicted latent sequence** (so `recover_root_rot_pos` integrates from the real clip start), then root-xz supervision uses only the **frame slice** corresponding to the active window (`chunk_size` tokens × temporal factor 4). This avoids the “window-only decode ⇒ first frame forced to (0,0)” mismatch; see [`Task5-loss.md`](Task5-loss.md).

5. **Coordinate convention**:
   - Root trajectory `traj` has shape `(T, 3)`. **Index 0 = x (ground plane), index 1 = y (vertical height), index 2 = z (ground plane)**.
   - Control loss supervises **only x and z** (indices `[0, 2]`). Height y is not supervised by the control loss (it is determined by pose and the main MSE loss).

6. **Training stage convention**:
   - **Stage-1**: Freeze backbone `WanModel` + text encoder + VAE. Train only `WanControlNet` + `TrajEncoder` (+ optionally `WanModel.traj_in_proj` / `traj_type_embed` if FlexTraj mode is also enabled on ControlNet).
   - **Stage-2** (optional): Unfreeze a small number of backbone layers with a much lower learning rate.

7. **CFG convention**: When CFG is active (`cfg_scale != 1.0`), the ControlNet branch uses the **same traj_emb** and the **same conditional `controlnet_residuals`** for both passes; only the backbone **text context** differs (cond vs null). Traj is not dropped in the null pass.

---

## Minimum config switches

Two keys are sufficient to enable ControlNet mode:

```yaml
use_controlnet_traj: true          # enables WanControlNet branch
freeze_backbone_for_controlnet: true  # Stage-1 freezing
```

`use_traj_cond` (the FlexTraj / backbone traj path) is **mutually exclusive** with `use_controlnet_traj` — do not enable both. The model constructor prints a warning and disables `use_traj_cond` if both are set.

---

## Reusable components (no rewrite needed)

| Component | Location | Description |
|-----------|----------|-------------|
| Triangular noise schedule | `diffusion_forcing_wan.py::_get_noise_levels` | Already correct |
| Active window MSE | `diffusion_forcing_wan.py::forward` | Already correct |
| ControlNet auxiliary output | `control_aux["pred_x0_latent_list"]` | Already set in `forward()` |
| Traj emb cache for streaming | `_traj_stream_version`, `_traj_emb_cache` in `diffusion_forcing_wan.py` | Shared by backbone and ControlNet |
| `TrajEncoder` | `models/tools/traj_encoder.py` | MLP from 4D → `traj_out_dim` |
| `build_traj_emb_from_batch` | `utils/traj_batch.py` | Constructs traj_emb from batch dict |
