## Task6 — Configuration and Freeze Strategy (YAML / checkpoint compatibility)

Corresponds to [`target.md`](target.md) §6.

**Status: Done.** All config keys are implemented and read by `DiffForcingWanModel.__init__`. This document serves as the definitive config key reference.

---

## Goal

All ControlNet switches, loss weights, and freeze strategies are centrally managed in YAML and map 1:1 to constructor parameters, with:

- No effect on runs that do not use ControlNet.
- Old checkpoints loadable with `strict=False` (new parameters have safe defaults).

---

## Complete Config Key Reference

All keys below live under `model.params` in `configs/ldf.yaml` (or `ldf_tiny.yaml`).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_controlnet_traj` | bool | `false` | Enable the `WanControlNet` branch. Mutually exclusive with `use_traj_cond`. |
| `freeze_backbone_for_controlnet` | bool | `false` | Stage-1: freeze `WanModel` + text encoder + VAE. Only ControlNet + TrajEncoder are trainable. |
| `controlnet_init_from_backbone` | bool | `true` | After creating ControlNet, copy weights from backbone. Requires `use_controlnet_traj: true`. |
| `control_loss_weight` | float | `1.0` | Multiplier for `L_control_xz` added to total training loss. |
| `use_traj_cond` | bool | `false` | Enable FlexTraj (traj tokens in backbone self-attention). **Do not set true with `use_controlnet_traj: true`.** |
| `freeze_backbone_for_traj` | bool | `false` | Legacy freeze flag for FlexTraj. Ignored when `use_controlnet_traj: true`. |
| `traj_encoder_in_dim` | int | `4` | Input dimension of `TrajEncoder`. Must be 4 for `[x, z, cos(ψ), sin(ψ)]`. |
| `traj_out_dim` | int | `64` | Output dimension of `TrajEncoder` (= `traj_enc_dim` passed to ControlNet). |
| `traj_dropout` | float | `0.1` | Probability of dropping trajectory conditioning during training (for robustness). |
| `use_traj_emb_cache` | bool | `true` | Cache `traj_emb` in streaming mode to avoid recomputing when trajectory is unchanged. |
| `noise_steps` | int | `10` | Number of diffusion denoising steps. |
| `chunk_size` | int | `5` | Active window size (number of token positions in the last chunk). |
| `prediction_type` | str | `"vel"` | Diffusion prediction target. `"vel"` (velocity) or `"x0"` (clean latent). |
| `cfg_scale` | float | `4.5` | Classifier-free guidance scale at inference. 1.0 = no guidance. |
| `dropout` | float | `0.1` | General text dropout probability during training (for CFG). |
| `input_dim` | int | `4` | Input dimension of the noisy latent (VAE z-dim). |
| `causal` | bool | `false` | Use causal (unidirectional) self-attention. False = bidirectional. |

---

## Example Config (ControlNet Stage-1)

```yaml
model:
    target: models.diffusion_forcing_wan.DiffForcingWanModel
    ema_decay: 0.99
    params:
        input_dim: 4
        noise_steps: 10
        use_controlnet_traj: true
        freeze_backbone_for_controlnet: true
        controlnet_init_from_backbone: true
        traj_encoder_in_dim: 4
        traj_out_dim: 64
        traj_dropout: 0.1
        use_traj_emb_cache: true
        control_loss_weight: 1.0
```

---

## Freeze Strategy Details

When `freeze_backbone_for_controlnet: true`, `DiffForcingWanModel.__init__` calls:

```python
# Freeze everything first
for p in self.model.parameters():
    p.requires_grad_(False)
for p in self.text_encoder.parameters():
    p.requires_grad_(False)
for p in self.vae.parameters():
    p.requires_grad_(False)

# Unfreeze ControlNet branch
if self.controlnet is not None:
    for p in self.controlnet.parameters():
        p.requires_grad_(True)

# Unfreeze TrajEncoder
for p in self.traj_encoder.parameters():
    p.requires_grad_(True)
```

The backbone `WanModel.traj_in_proj` and `traj_type_embed` remain frozen in ControlNet mode (they are part of `self.model`). If FlexTraj is also enabled on the backbone (not recommended), they would need to be unfrozen separately.

---

## Checkpoint Compatibility

### Loading old FloodDiffusion checkpoints into ControlNet model

Old checkpoints have no `controlnet.*` keys. Use `strict=False`:

```python
# In train_ldf.py on_load_checkpoint or resume logic:
missing_keys, unexpected_keys = model.load_state_dict(ckpt["state_dict"], strict=False)
# Expected missing: all "controlnet.*" keys
# Expected unexpected: none
```

The `controlnet.*` parameters will be initialized by the constructor (via `init_from_backbone` or random init) — this is correct behavior.

### Saving ControlNet checkpoints

The standard Lightning checkpoint saves all module parameters, including `controlnet.*`. When resuming from a ControlNet checkpoint, `strict=True` is fine (all keys present).

### Loading ControlNet checkpoints into non-ControlNet model

If `use_controlnet_traj: false` but the checkpoint has `controlnet.*` keys, use `strict=False`. The ControlNet parameters will be ignored.

---

## Config Validation

`DiffForcingWanModel.__init__` enforces these mutual exclusion rules:

1. `use_controlnet_traj=True` and `use_traj_cond=True` → warning printed, `use_traj_cond` is silently disabled.
2. `use_controlnet_traj=True` and `freeze_backbone_for_traj=True` → warning printed, `freeze_backbone_for_traj` is silently disabled (use `freeze_backbone_for_controlnet` instead).

---

## Verification Checklist

1. **Trainable parameter count**: With `freeze_backbone_for_controlnet: true`, `sum(p.numel() for p in model.parameters() if p.requires_grad)` should equal `controlnet_params + traj_encoder_params` (approximately half the total for a mirrored architecture).

2. **Config round-trip**: Save a checkpoint, reload with the same config. Assert loss is same as end of previous run (no accidental re-initialization).3. **Old checkpoint loads**: Load a FloodDiffusion checkpoint (no `controlnet.*`) with `strict=False`. Assert no unexpected keys in `unexpected_keys`. Assert `missing_keys` contains only `controlnet.*` entries.4. **`control_loss_weight=0.0`**: Setting this to 0 should disable the control loss contribution without disabling the ControlNet forward pass. Verify in `train_ldf.py::_step`.5. **`use_traj_emb_cache`**: With `True`, verify the cache hit/miss logic in `_stream_compute_traj_emb`. The `traj_emb` should be recomputed only when `_traj_stream_version` changes.