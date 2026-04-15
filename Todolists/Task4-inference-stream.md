## Task4 — Inference and Streaming (generate / stream_generate / stream_generate_step)

Corresponds to [`target.md`](target.md) §4.

**Status: Done (CFG).** ControlNet is wired into `generate()`, `stream_generate()`, and `stream_generate_step()`. CFG reuses conditional `controlnet_residuals` for the null backbone pass (text-only CFG); see [`target.md`](target.md) rule 7.

---

## Goal

ControlNet must be active in all three inference paths, and all three must handle the trajectory condition consistently:

- `generate` — offline full-sequence generation
- `stream_generate` — streaming wrapper (calls `stream_generate_step` in a loop)
- `stream_generate_step` — the core step, used by web demo and online generation

---

## Required Call Sequence (all three paths)

Every call to the backbone `WanModel` must follow this order:

```
1. Compute / retrieve traj_emb:
   - Non-stream: build_traj_emb_from_batch(x, seq_len, device, traj_encoder, ...)
   - Stream:     _stream_compute_traj_emb(...)   [uses traj buffer + cache]

2. Compute ControlNet residuals (if controlnet is enabled):
   controlnet_residuals = _maybe_controlnet_residuals(
       noisy_input, t, context, seq_len,
       traj_emb=traj_emb,       # same traj_emb as above
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

When `cfg_scale != 1.0`, inference runs **two forward passes**: one conditional (with text) and one unconditional (null text). The final prediction is:

```
pred = cfg_scale * cond_pred - (cfg_scale - 1) * null_pred
```

**Implemented behavior** (`diffusion_forcing_wan.py`): ControlNet is called **once** per denoising step with the **conditional** text context. The CFG null pass calls the backbone with **null text** and the **same** `controlnet_residuals` tensor (traj injection unchanged). See [`target.md`](target.md) rule 7.

```python
controlnet_residuals = _maybe_controlnet_residuals(
    noisy_input, t, context_cond, seq_len, traj_emb, traj_seq_lens
)
pred_cond = self.model(..., context=context_cond, controlnet_residuals=controlnet_residuals)
pred_null = self.model(..., context=context_null, controlnet_residuals=controlnet_residuals)
pred = cfg_scale * pred_cond - (cfg_scale - 1) * pred_null
```

---

## Traj Cache Consistency

`DiffForcingWanModel` maintains a version-keyed cache to avoid recomputing `traj_emb` when the trajectory has not changed:

```python
self._traj_stream_version: int      # incremented when traj buffer changes
self._traj_emb_cache: dict          # {version: traj_emb_tensor}
```

**Rules**:
1. When a new trajectory chunk is written to `traj_buffer`, increment `_traj_stream_version`.
2. `_stream_compute_traj_emb` checks the current version and recomputes only if version changed.
3. ControlNet and backbone share the **same** cached `traj_emb` object — do not call `_stream_compute_traj_emb` twice.
4. If no traj is provided for a step (user did not specify trajectory), `traj_emb = None` is passed to both ControlNet and backbone.

---

## Step-by-Step: Adding ControlNet to `stream_generate_step`

If `stream_generate_step` does not yet call ControlNet, add the following (file: `diffusion_forcing_wan.py`):

1. After computing `traj_emb` (via `_stream_compute_traj_emb`):
   ```python
   traj_seq_lens = self._get_traj_seq_lens(...)  # existing helper
   controlnet_residuals = self._maybe_controlnet_residuals(
       noisy_feature_input, t, context, seq_len, traj_emb, traj_seq_lens
   )
   ```

2. Pass to backbone:
   ```python
   pred = self.model(
       x=noisy_feature_input, t=t, context=context, seq_len=seq_len,
       traj_emb=None,   # ControlNet mode
       controlnet_residuals=controlnet_residuals,
   )
   ```

3. Do **not** change `traj_emb` passed to backbone — it must remain `None` in ControlNet mode.

---

## Verification Checklist

1. **ControlNet active in all three paths**: With `use_controlnet_traj=True`, set a breakpoint or add a log inside `_maybe_controlnet_residuals`. Verify it is called during `generate()`, `stream_generate()`, and `stream_generate_step()`.

2. **Backbone receives no traj**: With `use_controlnet_traj=True`, verify `traj_emb_backbone is None` at the `self.model(...)` call site in all three paths.

3. **Cache invalidation**: In streaming mode, feed two steps with different traj values. Verify `_traj_stream_version` increments between steps and `_traj_emb_cache` is refreshed.

4. **CFG consistency**: With `cfg_scale=2.0`, null pass must use the **same** `controlnet_residuals` as cond. Verify output changes when text changes but traj is fixed; verify output changes when traj changes but text is fixed.

5. **`generate` and `stream_generate` agreement**: On the same input (fix random seed), `generate` and `stream_generate` should produce outputs with similar quality metrics (they won't be numerically identical due to different chunking, but both should show trajectory following).
