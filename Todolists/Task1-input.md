## Task1 — Input Fields and Alignment (traj condition / mask / stream buffer)

Corresponds to [`target.md`](target.md) §1.

**Status: Done.** The dataset pipeline and stream buffer are implemented. This document serves as a specification for verification and future changes.

---

## Goal

Unify the "trajectory condition" into a consistent set of fields across training and streaming inference:

- Token/frame alignment is unambiguous.
- Mask semantics are consistent (last-frame sampling, not mixed with mean pooling).
- ControlNet and backbone `WanModel` receive the **same** `traj_emb` instance during inference.

---

## Batch Field Specification

All fields below are produced by `HumanML3DDataset._process()` and padded by `collate_fn`.

| Field | Shape | Dtype | Semantics |
|-------|-------|-------|-----------|
| `traj` | `(B, T_frame, 3)` | float32 | Full root xyz for loss/visualization. Index 0=x, 1=y(height), 2=z. |
| `traj_length` | `(B,)` | int64 | Number of valid frames in `traj` for each sample. |
| `traj_features` | `(B, T_frame, 4)` | float32 | Path-heading features `[x, z, cos(ψ), sin(ψ)]`. Same temporal length as `traj`. |
| `traj_mask` | `(B, T_frame)` | float32 | Frame-level mask: 1 = trajectory known at this frame, 0 = unknown. Derived from `token_mask` by 4× repeat then truncated to `traj_length`. |
| `token_mask` | `(B, T_token)` | float32 | Token-level mask: 1 = trajectory condition provided for this token. Sampled from `mask_ratio` (e.g., 1.0 = full trajectory, (0.2, 0.3) = random sparsity). |
| `token` | `(B, T_token, 4)` | float32 | VAE latent tokens (the diffusion model's input). |
| `token_length` | `(B,)` | int64 | Number of valid tokens for each sample. |
| `feature` | `(B, T_frame, 263)` | float32 | Raw HumanML3D motion features. |
| `feature_length` | `(B,)` | int64 | Number of valid feature frames. |

**Key constraint**: `T_token == T_frame // 4` for each sample (VAE temporal downsampling factor = 4). The collate_fn pads sequences so the batch tensors are `(B, T_max, ...)`.

---

## Token ↔ Frame Alignment

VAE downsampling factor = 4. Token index `k` corresponds to frames `[4k, 4k+1, 4k+2, 4k+3]`.

**Last-frame semantics (current implementation)**:
- `traj_features[k]` = features computed from the **last frame** of the token's window, i.e., frame `4k+3`.
- `token_mask[k] = 1` expands to `traj_mask[4k], traj_mask[4k+1], traj_mask[4k+2], traj_mask[4k+3]` all set to 1.
- Implemented in `HumanML3DDataset._process()`:
  ```
  token_mask = self.sample_token_mask(token_length)      # (T_token,) with values 0 or 1
  traj_mask = np.repeat(token_mask, 4)[:traj_length]    # (T_frame,) aligned to frame resolution
  ```

**Where `traj_features` is computed**:
- `datasets/humanml3d.py::_process()` — calls `path_heading_features_from_root_xyz(output["traj"])`.
- `utils/traj_batch.path_heading_features_from_root_xyz(traj_xyz)` — takes `(T, 3)` xyz, returns `(T, 4)` `[x, z, cos(ψ), sin(ψ)]`.
- ψ = xz path heading angle: `ψ_t = atan2(Δz_t, Δx_t)`, where `Δx_t = x_{t+1} - x_t` (forward difference; first frame uses `t+1 - t`, subsequent frames use `t - t-1`).

---

## Crop Alignment (Critical)

When a long motion is randomly cropped to `window_length`, the feature and token must use the **same crop window**:

1. `process_feature(feature)` picks `crop_start` randomly → returns cropped feature of length `feature_length`.
2. `process_token(token, crop_start, feature_length)` aligns token:
   - `token_start = crop_start // 4`
   - `token_len = feature_length // 4`
   - `token = token[token_start : token_start + token_len]`
3. `traj` is derived **after cropping**: `extract_root_trajectory_263(feature)` where `feature` is already cropped.
4. `traj_features` is derived from `traj`, so alignment is automatic.

**Bug to avoid**: Do not compute `traj` from the original (uncropped) feature and then crop it separately — the `crop_start` offsets will not align with VAE token boundaries.

---

## `TrajEncoder` Input Spec

File: `models/tools/traj_encoder.py::TrajEncoder`

```
Input:  (B, T, 4)   — [x, z, cos(ψ), sin(ψ)] at token or frame resolution
Output: (B, T, traj_out_dim)   — trajectory embeddings, default traj_out_dim=64
```

The `TrajEncoder` is a 2-layer MLP: `Linear(4, hidden_dim) → GELU → Linear(hidden_dim, out_dim)`.

**In `build_traj_emb_from_batch`** (`utils/traj_batch.py`):
- Prefers `x["traj_features"]` if present (frame-level, then `F.interpolate` to `seq_len` tokens).
- Falls back to `x["traj"]` xyz → `xyz_traj_to_features_4d` → encoder.
- Both paths end with `traj_encoder(feats)` returning `(B, seq_len, traj_out_dim)`.

---

## Stream Buffer Specification

`DiffForcingWanModel` streaming state (fields on `self` after `init_generated()`):

| Field | Shape | Description |
|-------|-------|-------------|
| `traj_buffer` | `(B, buf_len, 3)` | Frame-level root xyz, updated per step. |
| `traj_features_buffer` | `(B, buf_len_token, 4)` | Token-level precomputed path-heading features (optional). |
| `traj_features_mask_buffer` | `(B, buf_len_token)` | Token-level mask for the features above. |
| `_traj_stream_version` | int | Incremented when trajectory buffer changes. |
| `_traj_emb_cache` | dict | Keyed by `_traj_stream_version`; caches `traj_emb` tensor. |

**Rules**:
1. ControlNet and backbone share the **same** `traj_emb` computed by `_stream_compute_traj_emb(...)` — do not compute separate embeddings.
2. When a new trajectory chunk is written to the buffer, `_traj_stream_version` must be bumped. The next call will recompute and re-cache `traj_emb`.
3. If the external caller provides no traj fields for a step, the buffer must be cleared (or the step must skip trajectory conditioning) to prevent stale trajectory leaking.

---

## Verification Checklist

Run these checks after any change to the input pipeline:

1. **Token-frame alignment**: For a single sample, verify `token_length == feature_length // 4` (allow off-by-one only if VAE has boundary padding — document it explicitly if so).

2. **Traj features shape**: `batch["traj_features"].shape == (B, T_max_frame, 4)` and `batch["traj"].shape == (B, T_max_frame, 3)`.

3. **Mask expansion**: Pick a sample with `token_mask[k] = 1`. Verify `traj_mask[4k], traj_mask[4k+1], traj_mask[4k+2], traj_mask[4k+3]` are all 1.

4. **Stream cache invalidation**: Call `stream_generate_step` twice with different traj values; assert `_traj_stream_version` increments and `traj_emb` cache updates.

5. **No double-crop**: Confirm `traj_features` shape equals `(feature_length, 4)` (not the original uncropped length) for a sample with `feature_length < original_length`.
