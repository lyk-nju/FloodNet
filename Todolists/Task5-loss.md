## Task5 — Explicit Trajectory Loss (motion space)

Corresponds to [`target.md`](target.md) §5.

**Status: Done.** The control loss is implemented in `train_ldf.py::_compute_control_loss_xz`. This document specifies the exact formula and alignment requirements for future changes.

---

## Goal

In addition to the diffusion forcing MSE (latent space), introduce an explicit supervision loss in motion space:

1. Take `pred_x0_latent` (the predicted clean latent at each active window token) from `control_aux`.
2. Decode each sample's predicted latent through the VAE to get 263D motion features.
3. Extract the root trajectory (x, z) from the decoded motion.
4. Compare with GT root trajectory (x, z) at frames covered by `traj_mask`.
5. Weight the loss by `control_loss_weight` and add to the total loss.

---

## Loss Formula

```
L_control_xz = mean_over_masked_frames(
    sum_over_xz_dims( (pred_root_xz - gt_root_xz)^2 )
)
```

More precisely:
- `pred_root_xz = extract_root_trajectory_263_torch(decoded_pred)[..., [0, 2]]`  — shape `(T_pred, 2)`
- `gt_root_xz = batch["traj"][i, :L, [0, 2]]` — shape `(L, 2)`, where L = min length of pred and GT
- `mask = batch["traj_mask"][i, :L]` — shape `(L,)`, values 0 or 1
- Per-frame error: `err = ((pred_root_xz - gt_root_xz) ** 2).sum(dim=-1)` — shape `(L,)`
- Masked mean: `loss_i = (err * mask).sum() / (mask.sum() + eps)`

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
    traj_mask = batch["traj_mask"]                          # (B, T_frame_max), float32

    loss_control = self._compute_control_loss_xz(
        pred_list,
        batch,
        traj_mask,
        self.vae,
        self.device,
    )
    total_loss = total_loss + control_loss_weight * loss_control
```

### `_compute_control_loss_xz` function signature

```python
def _compute_control_loss_xz(
    pred_x0_latent_list: list[torch.Tensor],  # each: (T_token_i, 4) — predicted clean latent
    batch: dict,                              # contains "traj" (B, T_max, 3) and "traj_length" (B,)
    traj_mask: torch.Tensor,                  # (B, T_frame_max)
    vae: nn.Module,                           # VAE decoder
    device: torch.device,
) -> torch.Tensor:                            # scalar loss
```

---

## Active Window Alignment

**What the current implementation does**: The control loss is computed on all valid frames for each sample (up to `min(T_pred, T_gt, traj_length)`). It is **not** restricted to the last `chunk_size` token positions.

**Alignment with diffusion MSE active window**:
- The diffusion MSE is only computed on the last `chunk_size` token positions (active window).
- The control loss is computed on all frames where `traj_mask = 1`.
- These are consistent by design: `traj_mask` is set to 1 at randomly sampled token positions, and the VAE decode produces a full-length motion sequence from the predicted tokens.

**If you want to restrict control loss to active window only** (for more focused supervision):
1. Take only the last `chunk_size` elements of `pred_x0_latent_list[i]`.
2. Decode only those tokens: `vae.decode(pred_last_chunk.unsqueeze(0))`.
3. Compare against the corresponding GT frame range.

This is not currently done but is a valid option for experimentation.

---

## VAE Decode Step

```python
# pred_latent: (T_token, 4) — single sample predicted clean latent
with torch.no_grad():
    decoded = vae.decode(pred_latent.unsqueeze(0))[0].float()
    # decoded shape: (T_frame, 263)

# Extract root trajectory
pred_traj_3d = extract_root_trajectory_263_torch(decoded)
# pred_traj_3d shape: (T_frame, 3) with [x, y_height, z]

pred_root_xz = pred_traj_3d[:, [0, 2]]   # (T_frame, 2)
```

The VAE decode upsamples tokens by factor 4: `T_frame = T_token * 4` (approximately; exact boundary may differ by ±1 depending on VAE padding).

---

## Length Alignment

```python
T_pred = pred_root_xz.shape[0]             # from VAE decode
T_gt = batch["traj_length"][i].item()      # from batch
L = min(T_pred, T_gt)
gt_xz = batch["traj"][i, :L, [0, 2]]
mask = traj_mask[i, :L]
pred_xz = pred_root_xz[:L]
```

This `min` ensures the comparison is over the valid overlap region, handling cases where the predicted sequence is longer or shorter than the GT.

---

## Numerical Stability Notes

- Compute loss in `float32`, not `bf16`. The VAE output is already cast via `.float()`.
- Add a small epsilon (`1e-8`) when dividing by `mask.sum()` to avoid division by zero when no mask positions are active for a given sample (possible with low `mask_ratio`).
- When `traj_dropout` is active during training, some samples will have no `control_aux`. The `_step` function should skip the control loss when `control_aux` is absent.

---

## Verification Checklist

1. **Loss axis**: Confirm `batch["traj"][:, :, 1]` (height) does NOT appear in the loss computation. Search for `[0, 2]` in `train_ldf.py::_compute_control_loss_xz`.

2. **Loss decreases**: After ~1000 training steps with `use_controlnet_traj=True`, `control` loss in wandb should decrease from initial value. If it remains flat or NaN, check gradient flow (see item 4).

3. **Mask coverage**: `traj_mask.mean()` should match `mask_ratio` configuration (e.g., ~1.0 for full trajectory conditioning). Log it in the first epoch.

4. **Gradient reaches TrajEncoder**: With `freeze_backbone_for_controlnet=True`, after one step, assert `model.traj_encoder.mlp[0].weight.grad is not None`. The gradient path is: `L_control → pred_x0 → ControlNet → TrajEncoder`.

5. **No gradient in VAE**: The VAE is frozen. Verify `vae.encoder.parameters()` all have `requires_grad=False`.
