## Task3 — Backbone Injection Point (WanModel receives and applies residuals)

Corresponds to [`target.md`](target.md) §3.

**Status: Done.** `WanModel.forward` already accepts and applies `controlnet_residuals`. This document specifies the exact injection semantics and how to verify correctness.

---

## Goal

With **minimal changes** to `WanModel.forward`, accept ControlNet residuals and add them to the backbone's hidden states after each transformer block, affecting **only latent token positions**.

---

## Exact Implementation (already in place)

**File**: `FloodNet/models/tools/wan_model.py` — `WanModel.forward`

### Added parameter

```python
def forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    traj_emb=None,
    traj_seq_lens=None,
    controlnet_residuals=None,    # List[Tensor] | None
):
```

### Validation at entry (before block loop)

```python
if controlnet_residuals is not None and len(controlnet_residuals) != len(self.blocks):
    raise ValueError(
        f"controlnet_residuals length {len(controlnet_residuals)} "
        f"!= num_layers {len(self.blocks)}"
    )
```

### Injection inside block loop

```python
for i, block in enumerate(self.blocks):
    x = block(x, **kwargs)
    if controlnet_residuals is not None:
        r = controlnet_residuals[i].to(dtype=x.dtype, device=x.device)
        if r.dim() != 3 or r.size(0) != x.size(0) or r.size(1) != seq_len:
            raise ValueError(
                "controlnet_residuals[i] must have shape (B, seq_len, dim); "
                f"got {tuple(r.shape)} with seq_len={seq_len}"
            )
        # Only apply residual to latent tokens (first seq_len positions).
        x[:, :seq_len, :] = x[:, :seq_len, :] + r
```

### Why `x[:, :seq_len, :]` and not `x`

When FlexTraj is enabled (`traj_emb is not None`), `x` has shape `(B, 2*seq_len, dim)` — the first `seq_len` tokens are latent and the second `seq_len` tokens are trajectory. The residual must **only** be applied to the latent half. With FlexTraj disabled, `x.shape[1] == seq_len`, so `x[:, :seq_len, :]` is equivalent to `x` in that case.

---

## Tensor Shape Reference

When ControlNet mode is active (`use_controlnet_traj=True`, `use_traj_cond=False`):

| Variable | Shape | Notes |
|----------|-------|-------|
| `x` before blocks | `(B, seq_len, dim)` | Padded noisy latent (no FlexTraj) |
| `controlnet_residuals[i]` | `(B, seq_len, dim)` | From `WanControlNet.zero_out[i]` |
| `x[:, :seq_len, :]` | `(B, seq_len, dim)` | Target slice for addition |
| `x` after blocks | `(B, seq_len, dim)` | With residuals applied |

When ControlNet + FlexTraj are both enabled (not recommended, but supported):

| Variable | Shape | Notes |
|----------|-------|-------|
| `x` before blocks | `(B, 2*seq_len, dim)` | Latent + traj tokens concatenated |
| `controlnet_residuals[i]` | `(B, seq_len, dim)` | Only latent size |
| `x[:, :seq_len, :]` | `(B, seq_len, dim)` | Only latent half gets the residual |

---

## Compatibility Requirements

- When `controlnet_residuals=None` (default), the entire injection logic is skipped — no performance overhead, no behavior change.
- `generate()`, `stream_generate()`, and `stream_generate_step()` do not call `WanModel` directly; they call `_maybe_controlnet_residuals` first, then pass the result as `controlnet_residuals` to `WanModel`. No changes are needed inside `WanModel` for inference.
- The injection happens **before** the head (`self.head`), which only sees the latent half. The residuals are already applied to the correct slice before the head is called.

---

## Code Path in `DiffForcingWanModel.forward`

The full forward pass in training:

```
1. Build traj_emb from batch fields (build_traj_emb_from_batch)
2. traj_emb_backbone = None  (ControlNet mode: backbone is traj-unconditional)
3. controlnet_residuals = _maybe_controlnet_residuals(
       noisy_input, t, context, seq_len, traj_emb, traj_seq_lens)
4. pred = self.model(
       x=noisy_input, t=t, context=context, seq_len=seq_len,
       traj_emb=traj_emb_backbone,      # None
       controlnet_residuals=controlnet_residuals
   )
5. Compute diffusion MSE loss (active window only)
6. Collect pred_x0_latent_list → control_aux for Task5 control loss
```

---

## Verification Checklist

1. **Residual dimension validation**: Call `model.model.forward(..., controlnet_residuals=residuals)` where one residual has `shape=(B, seq_len+1, dim)`. It must raise `ValueError`.

2. **Residual count validation**: Pass `controlnet_residuals` with length ≠ `num_layers`. It must raise `ValueError`.

3. **No gradient in frozen backbone**: With `freeze_backbone_for_controlnet=True`, verify `model.model.blocks[0].norm1.weight.grad is None` after `.backward()`.

4. **Correct slice**: Create synthetic residuals with known values (e.g., all-ones). After forward, verify that `x[:, :seq_len, :]` has been modified but `x[:, seq_len:, :]` (traj half, if FlexTraj) is unchanged.

5. **Backward through residual to ControlNet**: With ControlNet trainable, after `loss.backward()`, `model.controlnet.zero_out[0].weight.grad` must be non-zero (the residual head receives gradient).
