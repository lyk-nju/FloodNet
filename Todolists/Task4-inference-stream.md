## Task4 — Inference and Streaming (generate / stream_generate / stream_generate_step)

Corresponds to [`target.md`](target.md) §4.

**Status: Done (Bug-fixed 2026-04).** ControlNet is wired into `generate()`, `stream_generate()`, and `stream_generate_step()`. Two bugs were discovered and fixed in April 2026; see §Bug Fixes below.

---

## Goal

ControlNet must be active in all three inference paths, and all three must handle the trajectory condition consistently:

- `generate` — offline full-sequence generation (used by validation, eval_control_loss)
- `stream_generate` — streaming generator wrapper (used by `generate_ldf.py`)
- `stream_generate_step` — single streaming step (used by web demo and online generation)

---

## Required Call Sequence (all three paths)

Every call to the backbone `WanModel` must follow this order:

```
1. Compute / retrieve traj_emb:
   - Non-stream: _build_traj_emb(x, seq_len, device, training_dropout=False)
   - Stream:     _stream_build_traj_emb(x, end_index, device)   [uses traj buffer + cache]

2. Compute ControlNet residuals (if controlnet is enabled):
   controlnet_residuals = _maybe_controlnet_residuals(
       noisy_input, t, context, seq_len,
       traj_emb=traj_emb,
       traj_seq_lens=traj_seq_lens
   )

3. Call backbone with residuals:
   pred = self.model(
       x=noisy_input, t=t, context=context, seq_len=seq_len,
       traj_emb=None,            # ControlNet mode: backbone gets no traj
       traj_seq_lens=None,
       controlnet_residuals=controlnet_residuals,
   )
```

The ControlNet always receives `traj_emb`; the backbone always receives `None` for `traj_emb` in standard ControlNet mode.

---

## CFG (Classifier-Free Guidance) Handling

When `cfg_scale != 1.0`, inference uses one of two paths depending on the text context layout:

### Path A: 2-Batch (normal case — almost always taken)

`_try_cfg_double_batch_text_context` succeeds when `len(text_context) == batch_size` (standard single-caption inputs). A single 2×batch forward pass handles both cond and null:

```python
ctx_2b = [cond_texts..., null_texts...]       # cond first, null second
traj_cn_2b = cat([traj_emb, traj_emb], dim=0) # both halves see traj

controlnet_residuals = _maybe_controlnet_residuals(
    noisy_2b, t_2b, ctx_2b, seq_len, traj_cn_2b, ...
)
# ControlNet internally sees cond_text for first B samples, null_text for second B
# → cond_residuals and null_residuals computed in one shot

pred_2b = self.model(noisy_2b, t_2b, ctx_2b, ..., controlnet_residuals=controlnet_residuals)
predicted_result = [cfg_scale * pred_2b[i] - (cfg_scale-1) * pred_2b[i + B] for i in range(B)]
```

This path is **always correct** and is used for standard HumanML3D evaluation.

### Path B: Separate passes (`else` branch — frame-aligned multi-segment text)

When text context is frame-aligned (e.g., multi-segment text with token-length encoding), `_try_cfg_double_batch_text_context` returns `None` and two separate forward passes are made:

```python
# Cond pass
controlnet_residuals_cond = _maybe_controlnet_residuals(
    noisy_input, t, all_text_context, seq_len, traj_emb, traj_seq_lens
)
pred_cond = self.model(..., context=all_text_context, controlnet_residuals=controlnet_residuals_cond)

# Null pass — Bug fix (2026-04): must recompute residuals with null text
if cfg_scale != 1.0:
    controlnet_residuals_null = _maybe_controlnet_residuals(
        noisy_input, t, text_null_context, seq_len, traj_emb, traj_seq_lens
    )
    pred_null = self.model(..., context=text_null_context, controlnet_residuals=controlnet_residuals_null)
    pred = [cfg_scale * pv - (cfg_scale-1) * pvn for pv, pvn in zip(pred_cond, pred_null)]
```

> **Previous (wrong) behavior**: The null pass reused `controlnet_residuals_cond`, making CFG uncond branch text-conditioned. Fixed in all three inference functions.

---

## Traj Cache Consistency

`DiffForcingWanModel` maintains a version-keyed cache to avoid recomputing `traj_emb` when the trajectory has not changed:

```python
self._traj_stream_version: int      # incremented when traj buffer changes
self._traj_emb_cache: dict          # {(key, start_t, end_t, version): traj_emb_tensor}
```

**Rules**:
1. When a new trajectory chunk is written to `traj_buffer`, increment `_traj_stream_version`.
2. `_stream_build_traj_emb` checks the current version and recomputes only if version changed.
3. ControlNet and backbone share the **same** cached `traj_emb` object — do not call `_stream_build_traj_emb` twice.
4. If no traj is provided for a step (user did not specify trajectory), `traj_emb = None` is passed to both ControlNet and backbone.

---

## Bug Fixes (2026-04)

### Bug 1: CFG null branch reused cond ControlNet residuals (else path)

**Location**: `generate()`, `stream_generate()`, `stream_generate_step()` — the `else` branch of the CFG block.

**Problem**: When the 2-batch path is unavailable (frame-aligned text), the null backbone pass was called with `controlnet_residuals=controlnet_residuals_cond` instead of separately-computed null residuals. Since ControlNet has cross-attention with text, this made the null branch text-conditioned — breaking CFG.

**Fix**: Add a separate `_maybe_controlnet_residuals(noisy_input, t, text_null_context, ...)` call before the null backbone forward, and pass its result as `controlnet_residuals_null`.

**Practical impact**: Minimal for standard HumanML3D evaluation (single-caption case always takes the 2-batch path). Affects frame-aligned multi-segment text inference.

---

### Bug 2: `_stream_build_traj_emb` (traj_buffer path) bypassed `LocalTrajEncoder`

**Location**: `diffusion_forcing_wan.py::_stream_build_traj_emb`, `traj_buffer` branch (line ~1342 before fix).

**Problem**: The stream traj buffer stores xyz root positions `(B, T_frames, 3)`. The code called:
```python
emb = self.traj_encoder(xyz_traj_to_features_4d(traj_slice))
```
This passed frame-level `(B, T_frames, 4)` features directly to `traj_encoder` (token-level MLP), **skipping `local_traj_encoder`** which is responsible for aggregating 4 frames per token via Conv1d.

During training, `_build_traj_emb` (non-stream) always goes through `local_traj_encoder`. This created a train/inference distribution mismatch for the streaming path.

**Fix**: Apply the same 4-frame grouping as `_build_traj_emb`:
1. Convert `traj_slice` (xyz frames) → frame-level 4D features via `xyz_traj_to_features_4d`
2. Group into `(B, ctx_len, 4, 4)` windows following the causal VAE convention
3. Apply `local_traj_encoder` → `(B, ctx_len, 4)`
4. Then apply `traj_encoder` → `(B, ctx_len, traj_out_dim)`

**Practical impact**: Only affects streaming inference (`stream_generate_step` and `stream_generate` when `traj_buffer` is used). Offline `generate()` uses `_build_traj_emb` which was always correct.

---

## Verification Checklist

1. **ControlNet active in all three paths**: With `use_controlnet_traj=True`, verify `_maybe_controlnet_residuals` is called during `generate()`, `stream_generate()`, and `stream_generate_step()`.

2. **Backbone receives no traj**: With `use_controlnet_traj=True`, verify `traj_emb_backbone is None` at the `self.model(...)` call site in all three paths.

3. **Cache invalidation**: In streaming mode, feed two steps with different traj values. Verify `_traj_stream_version` increments between steps and `_traj_emb_cache` is refreshed.

4. **CFG 2-batch path (normal)**: With `cfg_scale=2.0` and single-caption text (standard HumanML3D), `ctx_2b` must be non-None. Both cond and null residuals are computed in one ControlNet forward.

5. **CFG else path (multi-segment)**: With frame-aligned text, `ctx_2b` is None. Verify `controlnet_residuals_null` is computed from `text_null_context`, not reused from cond.

6. **Stream LocalTrajEncoder**: After the Bug 2 fix, verify streaming traj emb shape matches non-stream: both should produce `(B, ctx_len, traj_out_dim)`. A quick smoke test: feed the same trajectory via `generate()` and `stream_generate_step()` and check that `traj_emb` tensors are numerically similar.

7. **`generate` and `stream_generate` agreement**: On the same input (fix random seed), both should show trajectory following with similar quality metrics (not numerically identical due to different chunking).
