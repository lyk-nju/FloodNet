## FloodNet ControlNet Trajectory Control: Design Spec and Code Map

This document decomposes "adding a ControlNet branch to FloodDiffusion / Diffusion Forcing (Wan backbone) for trajectory control" into implementable module tasks, and defines **global consistency rules** that must be updated everywhere when any one rule changes.

---

## Implementation Status

| Task | File | Status | Summary |
|------|------|--------|---------|
| §1 | [`Task1-input.md`](Task1-input.md) | **Done** | `traj_features`/mask/length fields, token/frame alignment, stream buffer |
| §2 | [`Task2-controlnet.md`](Task2-controlnet.md) | **Done** | `WanControlNet` structure, zero-init residuals, initialization/freezing |
| §3 | [`Task3-backbone-inject.md`](Task3-backbone-inject.md) | **Done** | `WanModel.forward` residual injection interface |
| §4 | [`Task4-inference-stream.md`](Task4-inference-stream.md) | **Done (Bug-fixed 2026-04)** | ControlNet in all inference paths; CFG null branch now uses separately-computed residuals; stream `_stream_build_traj_emb` LocalTrajEncoder bug fixed |
| §5 | [`Task5-loss.md`](Task5-loss.md) | **Done** | `L_control_xz` after VAE decode, active window alignment |
| §6 | [`Task6-config.md`](Task6-config.md) | **Done** | Config keys, freeze strategy, checkpoint compatibility |
| §7 | [`Task7-scheduled-sampling.md`](Task7-scheduled-sampling.md) | **Done** | Scheduled Sampling training strategy; `scheduled_sampling_prob` config key |
| §8 | [`Task8-smooth-root.md`](Task8-smooth-root.md) | **In Progress** | Smooth root trajectory conditioning borrowed from Kimodo (NVIDIA) |

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

Task7 (scheduled sampling) ──► forward() in diffusion_forcing_wan.py
Task8 (smooth root)        ──► traj_batch.py / datasets/humanml3d.py
```

- Task6 must be settled first (config keys flow into Task2 constructor and Task1 dataset fields).
- Task1 and Task2 can proceed in parallel once Task6 is settled.
- Task3 requires Task2 (needs `WanControlNet` to exist) and `WanModel` (already present).
- Task4 requires Task1 (traj fields) + Task2 (controlnet instance) + Task3 (injection point).
- Task5 requires Task4 (access to `control_aux["pred_x0_latent_list"]`).
- Task7 is self-contained (only touches `forward()` and config).
- Task8 touches trajectory preprocessing pipeline; requires re-tokenizing traj features if applied to precomputed data.

---

## Global Consistency Rules (must be kept in sync everywhere)

1. **ControlNet scope**: ControlNet outputs **per-layer residuals added to the backbone's hidden states**; applies only to **latent tokens**. Under standard ControlNet training, **the backbone receives no traj tokens / traj embeddings** — trajectory conditions enter only through the ControlNet branch, then propagate via residuals. This keeps backbone behavior identical to the original FloodDiffusion.

2. **Residual tensor shape**: each residual is `(B, seq_len, dim)` — **only latent token positions**. Not `(B, 2*seq_len, dim)` (that would corrupt traj token positions when FlexTraj is also enabled).

3. **Zero-init rule**: all ControlNet residual output heads (`zero_out[i]`) are initialized to exact zero weights and zero bias. This ensures the model with ControlNet added has identical behavior to the backbone alone at initialization, which stabilizes Stage-1 training.

4. **Active window rule (FloodDiffusion)**:
   - The diffusion forcing MSE is computed only on the **last `chunk_size` token positions** (the active window).
   - `L_control_xz` must be aligned to the **same active window**: same token indices → same frame indices after VAE decode.
   - **Implementation detail** (`train_ldf.py::_compute_control_loss_xz`): VAE **decodes the full predicted latent sequence** (so `recover_root_rot_pos` integrates from the real clip start), then root-xz supervision uses only the **frame slice** corresponding to the active window (`chunk_size` tokens × temporal factor 4). This avoids the "window-only decode ⇒ first frame forced to (0,0)" mismatch; see [`Task5-loss.md`](Task5-loss.md).

5. **Coordinate convention**:
   - Root trajectory `traj` has shape `(T, 3)`. **Index 0 = x (ground plane), index 1 = y (vertical height), index 2 = z (ground plane)**.
   - Control loss supervises **only x and z** (indices `[0, 2]`). Height y is not supervised by the control loss (it is determined by pose and the main MSE loss).

6. **Training stage convention**:
   - **Stage-1**: Freeze backbone `WanModel` + text encoder + VAE. Train only `WanControlNet` + `TrajEncoder` (+ optionally `WanModel.traj_in_proj` / `traj_type_embed` if FlexTraj mode is also enabled on ControlNet).
   - **Stage-2** (optional): Unfreeze a small number of backbone layers with a much lower learning rate.

7. **CFG convention (corrected 2026-04)**: When `cfg_scale != 1.0` and the text context fits the 2-batch layout (`n == batch_size`), a single 2-batch forward pass is used — both conditional and unconditional branches run together, with ControlNet receiving context-appropriate text for each half. When the `else` path is taken (e.g., frame-aligned multi-segment text that cannot be batched), `_maybe_controlnet_residuals` is called **separately** with `text_null_context` for the null branch. Both paths are now correct. See [`Task4-inference-stream.md`](Task4-inference-stream.md) §Bug Fixes.

   > **Previous (wrong) Rule 7**: "null pass reuses cond residuals unchanged." This was a design error that has since been fixed.

8. **Scheduled Sampling (2026-04)**: During training, with probability `scheduled_sampling_prob`, context tokens (prefix before the active window, noise_level ≈ 0) are replaced with the model's own x0 prediction from a `no_grad` forward pass. This reduces teacher-forcing / generate exposure bias. See [`Task7-scheduled-sampling.md`](Task7-scheduled-sampling.md).

9. **LocalTrajEncoder consistency rule**: Frame-level traj features `(B, T_frames, 4)` must **always** pass through `local_traj_encoder` before reaching `traj_encoder`. The token-level shortcut in `_build_traj_emb` (when `feats_frame.shape[1] == seq_len`) is only triggered when features are pre-aggregated at the token level — training data from `datasets/humanml3d.py` always provides frame-level features. The stream path `_stream_build_traj_emb` using `traj_buffer` was previously bypassing `local_traj_encoder`; this is now fixed.

---

## Diagnosed Issues and Experimental Findings (2026-04)

See [`Experiment-TODO-forward-generate-gap.md`](Experiment-TODO-forward-generate-gap.md) for full experiment protocol and results.

### Root Causes of forward/generate Gap

| Cause | Contribution | Evidence |
|-------|-------------|---------|
| Error accumulation (position = ∫velocity) | **Primary** | Exp 1: 000021 segment MSE monotonically 0.17→10.70 over 180 frames |
| Teacher-forcing exposure bias | Secondary (~29% of gap by proxy) | Exp 3: TF=0.14, SF-proxy=0.40, Generate=1.02 |
| Training active-window sampling bias | Minor | Exp 2: B_uniform≈0, A_random≈0.16; bias present but small |

### Bug Fixes Applied (2026-04)

| Bug | Severity | Location | Fix |
|-----|---------|----------|-----|
| CFG null branch reused cond ControlNet residuals (`else` path) | Medium | `generate()`, `stream_generate()`, `stream_generate_step()` | Separate `_maybe_controlnet_residuals` call with `text_null_context` for null branch |
| Stream `_stream_build_traj_emb` bypassed `local_traj_encoder` | Medium | `diffusion_forcing_wan.py::_stream_build_traj_emb` traj_buffer path | Apply same 4-frame grouping + `local_traj_encoder` as non-stream path |
| Exp 2 frame-level tensor truncated to token count | Low (tooling) | `tools/run_gap_experiments.py` | Explicit `frame_level_keys` set for correct truncation |

> **Note**: The CFG bug in the `else` branch has minimal practical impact for standard HumanML3D evaluation because `_try_cfg_double_batch_text_context` successfully builds `ctx_2b` when `n == batch_size` (the 2-batch path is taken). The `else` path is only triggered for frame-aligned multi-segment text. The stream LocalTrajEncoder bug affects only streaming inference (`generate_ldf.py` demo path), not the offline `generate()` used for validation.

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
| `LocalTrajEncoder` | `models/tools/traj_encoder.py` | Conv1d aggregation of 4 frames per token |
