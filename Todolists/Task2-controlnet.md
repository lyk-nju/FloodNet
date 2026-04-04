## Task2 — ControlNet Branch (WanControlNet)

Corresponds to [`target.md`](target.md) §2.

**Status: Done.** `WanControlNet` is implemented in `models/tools/wan_controlnet.py`. This document specifies the design contract and verification requirements.

---

## Goal

A parallel branch that mirrors `WanModel` but outputs **per-layer residuals** to be added to the backbone's hidden states:

- Receives the same inputs as the backbone (noisy latent + time + text + traj_emb).
- Outputs a `list[Tensor]` of residuals, one per transformer block.
- Residual output heads are **zero-initialized**: model with ControlNet added ≈ model without ControlNet at initialization.
- Stage-1 training: backbone frozen, only ControlNet + TrajEncoder are trainable.

---

## Module Location

```
FloodNet/models/tools/wan_controlnet.py
    class WanControlNet(nn.Module)
```

---

## Constructor Signature

```python
WanControlNet(
    *,
    model_type: str = "t2v",        # "t2v" or "i2v"
    patch_size: tuple = (1, 1, 1),  # must match backbone patch_size
    text_len: int = 512,            # max text token length
    in_dim: int = 256,              # patch input channels (match backbone)
    dim: int = 1024,                # transformer hidden dim (match backbone)
    ffn_dim: int = 2048,            # FFN hidden dim (match backbone)
    freq_dim: int = 256,            # sinusoidal time embedding dim (match backbone)
    text_dim: int = 4096,           # T5 text embedding dim (match backbone)
    out_dim: int = 256,             # unused; kept for interface parity with WanModel
    num_heads: int = 8,             # attention heads (match backbone)
    num_layers: int = 8,            # number of blocks (must match backbone exactly)
    window_size: tuple = (-1, -1),  # match backbone
    qk_norm: bool = True,           # match backbone
    cross_attn_norm: bool = True,   # match backbone
    eps: float = 1e-6,              # match backbone
    causal: bool = False,           # match backbone (False for bidirectional)
    traj_enc_dim: int = 0,          # traj embedding dim; 0 = no traj branch
)
```

**Critical**: `num_layers` must equal the backbone's `num_layers`. Mismatch causes a `ValueError` in `WanModel.forward` (it checks `len(controlnet_residuals) == len(self.blocks)`).

---

## Forward Signature

```python
def forward(
    self,
    x: List[torch.Tensor],          # List of [C_in, 1, H, W] noisy latent tensors (one per sample)
    t: torch.Tensor,                # (B, seq_len) diffusion time steps (noise levels)
    context: List[torch.Tensor],    # List of [L_text, 4096] text embeddings
    seq_len: int,                   # maximum sequence length (pad target)
    y: Optional[List[torch.Tensor]] = None,   # i2v conditioning (unused for t2v)
    traj_emb: Optional[torch.Tensor] = None,  # (B, T, traj_enc_dim) from TrajEncoder
    traj_seq_lens: Optional[torch.Tensor] = None,  # (B,) valid lengths in traj_emb
) -> List[torch.Tensor]:
    # Returns: list of length num_layers, each tensor (B, seq_len, dim)
```

---

## Initialization Strategy

### Zero-init residual heads

In `__init__`, after creating blocks:
```python
self.zero_out = nn.ModuleList([_zero_linear(dim) for _ in range(num_layers)])
```
Where `_zero_linear(dim)` creates `nn.Linear(dim, dim)` with both weight and bias set to zero.

This ensures: at init, ControlNet output = zero → backbone unaffected → training starts stable.

### Copy weights from backbone (`init_from_backbone`)

```python
def init_from_backbone(self, backbone: WanModel) -> None:
    self.load_state_dict(backbone.state_dict(), strict=False)
    # Re-zero the residual heads (they may have been overwritten by the copy).
    for m in self.zero_out:
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
```

This is called in `DiffForcingWanModel.__init__` when `controlnet_init_from_backbone=True`:
```python
if self.controlnet_init_from_backbone:
    self.controlnet.init_from_backbone(self.model)
```

`strict=False` is required because `WanControlNet` has `zero_out` layers that `WanModel` does not have. Any keys present in the backbone but not in the ControlNet, or vice versa, are silently skipped.

---

## How Residuals Are Produced

Inside `WanControlNet.forward`, after the `i`-th block:
```
h_i = block(h, **kwargs)          # same as WanModel block
residual_i = zero_out[i](h_i)    # zero-init linear → initially outputs zeros
residuals.append(residual_i[:, :seq_len, :])  # only latent token positions
```

Shape of each residual: `(B, seq_len, dim)`.

---

## Trainable vs Frozen Parameters (Stage-1)

When `freeze_backbone_for_controlnet=True`, `DiffForcingWanModel.__init__` sets:

**Frozen** (`.requires_grad_(False)`):
- `self.model.*` — all backbone `WanModel` parameters
- `self.text_encoder.*` — T5 or precomputed embedding table
- `self.vae.*` — VAE encoder/decoder

**Trainable** (`.requires_grad_(True)`):
- `self.controlnet.*` — all ControlNet parameters
- `self.traj_encoder.*` — TrajEncoder MLP
- (Optional) `self.model.traj_in_proj`, `self.model.traj_type_embed` if FlexTraj is also enabled on the backbone

---

## How ControlNet Is Called in `DiffForcingWanModel`

The helper method `_maybe_controlnet_residuals` in `diffusion_forcing_wan.py`:

```python
def _maybe_controlnet_residuals(self, noisy_input, t, context, seq_len, traj_emb, traj_seq_lens):
    if self.controlnet is None:
        return None
    return self.controlnet(
        x=noisy_input, t=t, context=context, seq_len=seq_len,
        traj_emb=traj_emb, traj_seq_lens=traj_seq_lens
    )
```

Then the backbone is called with residuals:
```python
controlnet_residuals = self._maybe_controlnet_residuals(...)
output = self.model(
    x=noisy_input, t=t, context=context, seq_len=seq_len,
    traj_emb=traj_emb_backbone,           # None in ControlNet mode (backbone gets no traj)
    controlnet_residuals=controlnet_residuals,
)
```

---

## Verification Checklist

1. **Zero-init equivalence**: Before any training step, with `use_controlnet_traj=True`, the output of `DiffForcingWanModel.forward` on the same input should be numerically equal to the output with `use_controlnet_traj=False` (difference < 1e-5). This confirms the zero-init is working.

2. **Parameter count**: `sum(p.numel() for p in model.controlnet.parameters())` should be approximately equal to `sum(p.numel() for p in model.model.parameters())` (ControlNet mirrors backbone architecture).

3. **Frozen backbone during training**: After one training step with `freeze_backbone_for_controlnet=True`, verify `model.model.blocks[0].norm1.weight.grad is None`.

4. **Gradient flow to TrajEncoder**: After one training step, verify `model.traj_encoder.mlp[0].weight.grad is not None`.

5. **`init_from_backbone` correct**: After calling `init_from_backbone`, `controlnet.patch_embedding.weight` should equal `backbone.patch_embedding.weight`. But `controlnet.zero_out[0].weight` should equal `torch.zeros(dim, dim)`.

6. **Residuals shape**: `len(residuals) == num_layers` and each `residuals[i].shape == (B, seq_len, dim)`.
