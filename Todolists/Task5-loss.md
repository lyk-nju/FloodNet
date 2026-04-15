## Task5 — Explicit Trajectory Loss (motion space)

Corresponds to [`target.md`](target.md) §5.

**Status: Done.** The control loss is implemented in `train_ldf.py::_compute_control_loss_xz`. This document specifies the exact formula and alignment requirements for future changes.

---

## Goal

In addition to the diffusion forcing MSE (latent space), introduce an explicit supervision loss in motion space:

1. Take the per-sample predicted clean latent sequence `pred_x0_latent` from `control_aux` (`pred_x0_latent_list`).
2. **VAE-decode the full latent sequence** `(T_token, z_dim)` → motion features `(T_frame, 263)` (frozen VAE).
3. Extract the root trajectory (x, z) from the **full** decoded motion (global integration from clip start).
4. **Supervise only inside the active window**: slice both pred and GT root xz to the frame range that corresponds to the last `chunk_size` tokens (`token_to_frame = 4`), then apply `traj_mask` on that slice (masked mean; same spatial convention as [`target.md`](target.md) rule 4).
5. Weight the loss by `control_loss_weight` and add to the total loss.

**Fallback**: If `chunk_size_tokens` is `None` or `T_token ≤ chunk_size_tokens`, the frame slice is the full sequence (no tail-only restriction).

---

## Loss Formula

```
L_control_xz = mean_over_masked_frames(
    sum_over_xz_dims( (pred_root_xz - gt_root_xz)^2 )
)
```

More precisely (per sample `i`; see `train_ldf.py::_compute_control_loss_xz`):
- Decode **full** `pred_latent_full` → `decoded` → `pred_traj_full = extract_root_trajectory_263_torch(decoded.unsqueeze(0))`  — shape `(1, T_frame, 3)`.
- If active window applies: `start_f = (T_token - chunk_size_tokens) * 4`, `end_f = T_token * 4`; take `pred_traj = pred_traj_full[:, pred_sl, :]` (same `pred_sl` / `gt_sl` logic as code) and the same frame slice on `traj` / `traj_mask`. Else use the full-overlap slices for `pred_sl` / `gt_sl` and length `L`.
- `pred_root_xz = pred_traj[..., [0, 2]]`, `gt_root_xz = gt_traj[..., [0, 2]]` on that slice.
- `mask = traj_mask` on the same frame indices; per-frame `err = ((pred_root_xz - gt_root_xz) ** 2).sum(dim=-1)`.
- Accumulate `(err * mask).sum()` over samples and divide by total `mask.sum()` (global masked mean for the batch step).

**Important axis convention**: `traj` is `(T, 3)` with layout `[x, y_height, z]`. Only `[0, 2]` are used. Index 1 is vertical height and is **not** supervised by the control loss.

---

## Code Entry Points

### Model side: `diffusion_forcing_wan.py::forward`

The model populates `control_aux` when `use_traj_cond or use_controlnet_traj` is True:

```python
pred_x0_latent_list = []
for i, (pred_i, noise_i) in enumerate(zip(pred_x0, noisy_input_list)):
    # pred_x0 here is the predicted clean latent token sequence
    pred_x0_latent_list.append(pred_i)

loss_dict["control_aux"] = {"pred_x0_latent_list": pred_x0_latent_list}
```

Each element of `pred_x0_latent_list` is a `Tensor` of shape `(T_token, 4)` — one per sample in the batch.

### Training side: `train_ldf.py::CustomLightningModule._step`

```python
if "control_aux" in out:
    pred_list = out["control_aux"]["pred_x0_latent_list"]   # list of (T_token, 4) tensors
    traj = batch["traj"]
    traj_mask = batch["traj_mask"]
    traj_length = batch["traj_length"]
    chunk_size_tokens = getattr(self.model, "chunk_size", None)  # e.g. DiffForcingWanModel.chunk_size default 5

    loss_control = self._compute_control_loss_xz(
        pred_list,
        traj,
        traj_mask,
        traj_length,
        self.vae,
        self.device,
        chunk_size_tokens=chunk_size_tokens,
    )
    total_loss = total_loss + control_loss_weight * loss_control
```

### `_compute_control_loss_xz` function signature

```python
def _compute_control_loss_xz(
    pred_list,
    traj,
    traj_mask,
    traj_length,
    vae,
    device,
    chunk_size_tokens: int | None = None,
    token_to_frame: int = 4,
) -> torch.Tensor | None:
    ...
```

---

## Active Window Alignment

**What the current implementation does** (`train_ldf.py::_compute_control_loss_xz`):

1. **Always** `vae.decode(pred_latent_full)` on the **entire** predicted latent sequence for each sample (correct global root integration from clip start).
2. When `chunk_size_tokens` is set **and** `T_token > chunk_size_tokens`, compute frame indices  
   `start_f = (T_token - chunk_size_tokens) * token_to_frame`, `end_f = T_token * token_to_frame`,  
   then take the **same** `[start_f:end_f)` slice on pred traj, GT `traj`, and `traj_mask` (clamped to valid lengths). **Supervision = masked MSE only on this slice** — aligned with the diffusion forcing MSE active window (last `chunk_size` tokens).
3. When `chunk_size_tokens` is `None` or `T_token ≤ chunk_size_tokens`, the slice is the full valid overlap (equivalent to supervising all decoded frames in range).

**Not done (and usually undesirable)**: decoding **only** the last `chunk_size` tokens without full-sequence decode — that would re-anchor `recover_root_rot_pos` at the window start and reintroduce the “GT window start ≠ (0,0)” mismatch discussed in [`Note-control-loss-origin.md`](Note-control-loss-origin.md).

---

## VAE Decode Step

```python
# pred_latent_full: (T_token, 4) — full-sequence predicted clean latent for one sample
decoded = vae.decode(pred_latent_full.unsqueeze(0))[0].float()   # (T_frame, 263)
pred_traj_full = extract_root_trajectory_263_torch(decoded.unsqueeze(0))  # (1, T_frame, 3)

# Then slice pred_traj_full, traj, traj_mask to [start_f:end_f) when active window applies
# pred_root_xz = pred_traj[..., [0, 2]]
```

The VAE decode upsamples tokens by factor 4: `T_frame ≈ T_token * 4` (boundary ±1 possible depending on VAE padding).

---

## Length Alignment

After choosing `pred_sl` / `gt_sl` (either full overlap or active-window slice), use  
`L = min(pred_sl.stop - pred_sl.start, gt_sl.stop - gt_sl.start)` so pred/GT/mask stay the same length within the supervised region; skip the sample if `L <= 0`.

---

## Numerical Stability Notes

- Compute loss in `float32`, not `bf16`. The VAE output is already cast via `.float()`.
- Add a small epsilon (`1e-8`) when dividing by `mask.sum()` to avoid division by zero when no mask positions are active for a given sample (possible with low `mask_ratio`).
- When `traj_dropout` is active during training, some samples will have no `control_aux`. The `_step` function should skip the control loss when `control_aux` is absent.

---

## Verification Checklist

1. **Loss axis**: Confirm `batch["traj"][:, :, 1]` (height) does NOT appear in the loss computation. Search for `[0, 2]` in `train_ldf.py::_compute_control_loss_xz`.

2. **Loss decreases**: After ~1000 training steps with `use_controlnet_traj=True`, `control` loss in wandb should decrease from initial value. If it remains flat or NaN, check gradient flow (see item 4).

3. **Mask coverage / window**: When `T_token > chunk_size_tokens`, confirm supervised frames lie in `[start_f, end_f)` (active window). `traj_mask.mean()` should reflect `mask_ratio` (e.g., ~1.0 for full trajectory conditioning) **within** that slice.

4. **Gradient reaches TrajEncoder**: With `freeze_backbone_for_controlnet=True`, after one step, assert `model.traj_encoder.mlp[0].weight.grad is not None`. The gradient path is: `L_control → pred_x0 → ControlNet → TrajEncoder`.

5. **No gradient in VAE**: The VAE is frozen. Verify `vae.encoder.parameters()` all have `requires_grad=False`.
