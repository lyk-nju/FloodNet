# LDF Window-Local Limited-History Training Design

This document is the implementation standard for LDF window-local
limited-history training. The goal is to make self-forcing training match
streaming runtime more closely without changing the VAE latent contract.

## 0. Implementation Status

The current v1 implementation is flag-gated by:

```yaml
stream_training:
  enabled: true
```

The default main config keeps this disabled. When enabled, the code now supports:

- precomputed latent slicing `z[S:E]`, not VAE re-encoding;
- feature padding to `latent_valid_len + horizon_tokens` while keeping
  `feature_length = latent_valid_len`;
- clean committed latent state plus a fixed history-corruption input view;
- window-local 7D trajectory reconstruction from raw 263D motion;
- per-sample `traj_start_token` and explicit `traj_num_tokens`;
- segmented text endpoint cropping from global to local token coordinates;
- segmented text schedule validation rejects mismatched segment/end counts and
  non-monotonic endpoints instead of silently dropping text spans;
- horizon masking in global token coordinates;
- diffusion time is represented in the shifted-local coordinate system,
  equivalent to the global token clock for the sliced window;
- `sample_policy: variable_history` and `sample_policy: fixed_window`;
- `motion_aux_loss: latent_only` as the default bring-up path;
- `motion_aux_loss: full_prefix` as an opt-in path that splices local predicted
  latents back into `[0:E]` before VAE decode, then compares decoded active
  root poses and GT root poses in the same GT-anchor window-local frame.
- config validation rejects unsupported `latent_source`, `sample_policy`,
  `motion_aux_loss`, and `anchor_move_in_rollout` values before training starts.
- training log metrics that summarize sampled window start, window length,
  trajectory token length, and whether fixed-window sampling is active.

The remaining validation work is not more index plumbing. It is real-data
training validation: a smoke run, a small overfit run, and a decision on whether
`motion_aux_loss: full_prefix` is stable enough to use beyond diagnostics.

## 1. Goal

Current self-forcing training samples an active right boundary `E`, but the
model still sees the full prefix:

```text
latent input = z[0:E]
```

Streaming runtime uses a rolling recent-history window:

```text
latent input = z[S:E]
S = body/history window start token
```

The new training mode must train the body model on the same limited-history
shape:

```text
S = window anchor token
E = active right token
H = future trajectory horizon tokens

latent input      = precomputed latent z[S:E]
trajectory input  = window-local 7D trajectory covering [S:E+H]
supervised chunk  = active chunk ending at E
body anchor       = frame at token_start_frame(S)
```

This is not just a crop. It is a window-local training contract.

## 2. Non-Goals

The first implementation must not re-encode VAE latent windows.

Do not do this in v1:

```text
raw 263 motion[S:...] -> vae.encode(...) -> new latent prefix
```

Runtime streaming keeps a continuous latent buffer and takes rolling slices from
that buffer. Re-encoding each window as a new prefix would introduce a different
reset-window distribution, especially around causal VAE first-token/cache
semantics.

The first implementation should use:

```text
latent = precomputed latent z[S:E]
```

Only trajectory, masks, text endpoints, loss indexing, and motion-space targets
become window-local.

## 3. Coordinate Contract

Window-local means:

```text
ground-plane x/z origin = root x/z at token_start_frame(S)
yaw origin              = root yaw at token_start_frame(S)
```

It does not mean root `y` height is zero. Keep the existing project convention:

```text
x/z and yaw are anchored
y is physical root height and is preserved
```

This matches `utils.local_frame.canonicalize_7d`, which anchors x/z and heading
but leaves y, fwd_delta, and yaw_delta unchanged.

## 4. Token-Space Sampling

Sample windows in token space, not frame space. LDF chunking, noise schedule,
self-forcing replacement, horizon simulation, and streaming commits are all
token-level concepts.

Recommended config:

```yaml
stream_training:
  enabled: true
  context_tokens: 30
  min_history_tokens: 8
  horizon_tokens: 20
  sample_policy: variable_history   # variable_history | fixed_window
  anchor_move_in_rollout: false
  latent_source: precomputed_slice   # v1 only
```

For `variable_history`:

```text
S ~ valid window anchor token
E0 ~ [S + min_history_tokens, min(valid_tokens, S + context_tokens)]
final_E = E0 + K - 1
require final_E - S <= context_tokens
require final_E <= valid_tokens
require min_history_tokens >= chunk_size
```

The v1 implementation realizes this as a two-stage sample:

```text
1. build the maximum valid local window from S:
   latent_valid_len = min(context_tokens, valid_tokens - S)
2. let plan_rollout() sample the active local end E0-S inside:
   [min_history_tokens, latent_valid_len - K + 1]
```

This is equivalent to sampling `E0` after `S` while keeping a fixed local window
for the rollout. The future latent padding beyond the sampled active end is not
fed to the model input or loss; it only gives the trajectory side enough length
to expose the visible horizon.

For `fixed_window`, v1 samples the final right boundary first and right-aligns
the rollout plan:

```text
sample final_E
S = max(0, final_E - context_tokens)
E0 = final_E - K + 1
require final_E - S <= context_tokens
require E0 - S >= min_history_tokens
```

This differs from sampling `E0` directly because `K > 1` would otherwise advance
the final supervised step past the intended fixed window. The implementation
therefore builds a window ending at `final_E` and makes `plan_rollout()` choose
the local start end index:

```text
local_E0 = latent_valid_len - K + 1
```

The first implementation should keep `S` fixed inside one self-forcing rollout
and advance only `E`. Anchor movement inside the K-step rollout is deferred.

## 5. Diffusion-Time Contract

Window-local training must preserve the same beta/noise schedule as the full
global prefix would assign to tokens `[S:E]`.

The two valid formulations are equivalent:

```text
global form:
  beta = clamp(1 + global_token_idx / chunk_size - t_global, 0, 1)

shifted-local form:
  local_token_idx = global_token_idx - S
  t_local = t_global - S / chunk_size
  beta = clamp(1 + local_token_idx / chunk_size - t_local, 0, 1)
```

The v1 implementation uses the shifted-local form through
`utils.training.self_forcing.shifted_local_time_steps()`. `plan_rollout()`
samples a local active end index inside the sliced window, and
`_get_noise_levels()` sees local token positions. This is correct only because
the local active end is the global active end shifted by `S`.

Forbidden:

```text
resetting the diffusion clock to token 0 with a newly sampled phase unrelated
to the original global position
```

If a future refactor changes time-step construction, add a regression test that
compares beta values from:

```text
full-prefix global schedule sliced to [S:E]
window-local shifted schedule for [0:E-S]
```

They should match for all valid tokens in the window.

## 6. Training Forward Length Contract

Window-local training must distinguish latent context length from trajectory /
attention length.

Definitions:

```text
latent_valid_len = E - S
attn_len         = latent_valid_len + H
traj_num_tokens  = attn_len
```

The model should see only `latent_valid_len` latent tokens, but the ControlNet /
trajectory attention must be able to expose `traj_num_tokens` trajectory tokens.
If this split is not implemented, the training path can silently regress to:

```text
latent keys: [S:E]
traj keys:   [S:E]
```

instead of the intended:

```text
latent keys: [S:E]
traj keys:   [S:E+H]
```

Allowed implementation routes:

1. **Compatibility route**: build a `feature` tensor padded to `attn_len`.
   The first `latent_valid_len` tokens are `z[S:E]`; the future latent pad is
   dummy data and must not be exposed as valid latent input or loss target.
   Set `feature_length = latent_valid_len`, while trajectory token length stays
   `traj_num_tokens = attn_len`.
2. **Refactor route**: change the model forward path to carry explicit
   `latent_pad_len` and `traj_pad_len` / `attn_len` instead of overloading one
   shared `seq_len`.

The compatibility route is the recommended v1 bring-up because it is closer to
the existing `_forward_single_window()` structure.

The future dummy latent pad is only shape padding. It is forbidden to use it in:

```text
noise_ref / feature_ref targets
latent reconstruction loss
scheduled-sampling or self-forcing replacement
motion-space VAE decode
body/control auxiliary losses
```

All active-token indexing must clamp to `latent_valid_len`. Only trajectory
attention may use `traj_num_tokens`.

The batch contract for window-local training should be explicit:

```text
model_batch["feature"] = feature_window_padded       # [B, attn_len, D]
model_batch["feature_length"] = latent_valid_len     # [B], valid latent only

model_batch["traj_features"] = traj7_window_local    # frame-level 7D for [S:E+H]
model_batch["traj_start_token"] = S                  # scalar or [B]
model_batch["traj_num_tokens"] = attn_len            # scalar or [B]
model_batch["traj_features_length"] = attn_len       # token length, not frame length
model_batch["traj_length"] = valid_frame_len         # frame length, for frame masks/artifacts
```

`_get_traj_seq_lens()` must not prefer `feature_length` when
`traj_num_tokens` / `traj_features_length` is present. In this mode,
`feature_length` is the latent valid length, while trajectory sequence length is
`traj_num_tokens`.

Required priority:

```text
traj_num_tokens
  -> traj_features_length
  -> traj_length-derived token count
  -> feature_length fallback
```

The current full-prefix path historically used `feature_length` first; the
window-local implementation must change that for this mode.

## 7. Token-Frame Mapping Rules

Never hand-write `4 * token`.

Use `utils.token_frame`:

```text
token_start_frame(token)
token_end_frame(token)
token_range_to_frame_slice(start_token, num_tokens)
```

Important rule:

```text
prefix [0:N)        -> 4N - 3 frames
arbitrary [S:S+N)   -> 4N frames when S > 0
```

This means a window-local tensor may start at local frame 0, but its grouping
still uses the global token start `S`.

The trajectory payload must preserve:

```text
traj_start_token = S
```

Do not reset it to 0 unless `S == 0`.

## 8. Window-Local Trajectory Construction

Do not directly crop full-clip `traj_cond_7d` and treat it as window-local.

Wrong:

```text
traj7_window = full_traj_cond_7d[S_frame:...]
```

Even if x/z/yaw are later canonicalized, the first frame's fwd_delta and
yaw_delta still come from the outside-window transition `S-1 -> S`.

Preferred v1 construction:

```text
frame_origin = token_start_frame(S)
frame_slice  = token_range_to_frame_slice(S, num_traj_tokens)
raw_window   = raw_feature_263[frame_slice.start:frame_slice.stop]

root_quat, root_xyz = recover_root_rot_pos(raw_window)
traj7_local = root_to_traj_feats_7d(root_quat, root_xyz)
```

This guarantees:

```text
first x/z       = 0
first yaw       = 0
first fwd_delta = 0
first yaw_delta = 0
```

Fallback construction is allowed only if raw 263 is unavailable:

```text
traj7_slice = full_traj7[frame_slice]
traj7_local = canonicalize_7d(traj7_slice, anchor_xz_at_S, anchor_yaw_at_S)
recompute fwd_delta and yaw_delta from local x/z/yaw for the whole slice
```

Only setting `traj7_local[0, 5:7] = 0` is not sufficient as the main path.

When `E + H` extends past the valid clip end, v1 should allow the window but
mask the unavailable future tail as invalid:

```text
expected_frame_slice = token_range_to_frame_slice(S, traj_num_tokens)
available_raw_window = raw_feature_263[expected_frame_slice.start:clip_end]
traj_features        = zero-padded / padded-to-expected frame length
traj_cond_mask       = 1 for available frames, 0 for unavailable tail frames
```

Do not read past the raw feature length. Do not hold-last as the default GT
training target; hold-last is a runtime plan fallback, not a ground-truth future
motion label.

`raw_feature_length` itself must be valid for the provided tensor:

```text
0 <= raw_feature_length[b] <= raw_feature_263.shape[1]
```

If this invariant is violated, fail fast. Do not rely on tensor slicing silently
truncating an out-of-range length, because that desynchronizes `traj_length`,
`traj_cond_mask`, and the actual recovered raw window.

The sampled window must still have a valid origin:

```text
token_start_frame(S) < raw_feature_length
```

If this is not true, resample the training window instead of producing an empty
raw window.

## 9. Raw 263 Preservation

`prepare_model_input()` currently maps:

```python
model_batch["feature"] = batch["token"]
model_batch["feature_length"] = batch["token_length"]
```

Therefore the window builder must preserve raw motion before this overwrite or
copy it into explicit fields:

```text
raw_feature_263
raw_feature_length
```

Any implementation that reconstructs window-local 7D without raw 263 must state
which fallback path it uses and must test delta recomputation.

The v1 implementation is intentionally strict: the window-local trajectory
builder accepts raw HumanML3D motion with feature dimension `263` only. A
precomputed 7D trajectory tensor, latent tensor, or already prepared
`model_batch["feature"]` must fail fast instead of being interpreted as raw
motion.

## 10. Trajectory Start Token And Batch Semantics

`traj_start_token` is load-bearing. It tells trajectory grouping and horizon
masking that the local frame tensor corresponds to global token `S`.

`traj_num_tokens` is also load-bearing. For arbitrary windows, frame length
cannot safely be converted back to token length with prefix formulas.

Example:

```text
S = 5
N = 20
frame length = 80
correct traj_num_tokens = 20
prefix inverse would infer 21 tokens
```

Therefore window-local training must pass an explicit trajectory token length:

```text
traj_num_tokens
traj_features_length   # token length alias for model code that uses this name
```

Frame length fields such as `traj_length` are for frame masks, artifacts, and
motion-space comparisons; they are not the source of trajectory attention length
when `traj_num_tokens` is present.

Current trajectory helpers historically accepted a scalar start token. The v1
implementation now supports per-sample `[B]` `traj_start_token` in trajectory
grouping and masking helpers, and that support is load-bearing for
window-local training.

Forbidden:

```text
per-sample S values + one scalar traj_start_token
```

Supported bring-up choices:

1. sample one shared `S` for the whole batch and pass a scalar
   `traj_start_token`;
2. sample per-sample `S` and pass `[B]` `traj_start_token`.

The production target is option 2, and current tests should keep covering it.

Do not sample per-sample `S` while passing a scalar `traj_start_token` just to
make the batch fit existing helpers. That creates silently wrong frame grouping
for every sample whose real `S` differs from the scalar.

## 11. Horizon Masking

Horizon masking must use one token coordinate system.

If:

```text
traj_start_token = S
active_end_token = E
stream_training.horizon_tokens = H
```

then visible trajectory ends at:

```text
token_start_frame(E + H) - token_start_frame(S)
```

`stream_training.horizon_tokens` is the visible horizon contract, not only the
constructed trajectory-buffer length. Therefore it must be applied even when
`horizon_sim.enabled` is false. If `horizon_sim.enabled` is true and samples a
larger horizon than `stream_training.horizon_tokens`, the sampled value is
clamped to `stream_training.horizon_tokens` for window-local training. This
keeps the model from seeing beyond `[S:E+H]` while still allowing horizon
curriculum inside that bound.

Do not mix local `E-S` with global `S` in the same horizon calculation.

If `E + H` passes the clip end, the unavailable future tail must be masked out
instead of being treated as valid trajectory. `traj_num_tokens` may still equal
`latent_valid_len + H`; the token/frame mask decides which future frames/tokens
are valid.

## 12. Active Loss Frame Slices

Active loss frame ranges must be derived from global token slices and then
shifted into the window-local frame tensor.

Example:

```text
A = E - chunk_size
origin_frame = token_start_frame(S)
global_active = token_range_to_frame_slice(A, chunk_size)
local_active = [
  global_active.start - origin_frame,
  global_active.stop  - origin_frame
)
```

Do not use:

```text
token_range_to_frame_slice(A - S, chunk_size)
```

That treats a non-prefix window as a local prefix and can be off by three frames.

## 13. Self-Forcing State And Corruption

Split committed latent state from corrupted model-input view:

```text
z_state = clean committed latent window
z_input = z_state with fixed corruption overlay
```

History corruption in v1:

```text
sample once per rollout
fixed across all K self-forcing steps
region = initial history only
do not corrupt the active chunk
do not corrupt generated/replaced tokens later in the rollout
```

Initial corruption region:

```text
local_E0 = E0 - S
initial_active_left = local_E0 - chunk_size
corrupt tokens [0:initial_active_left)
```

Self-forcing replacement writes to `z_state`, not to the corrupted input view:

```text
local_end = E - S
replace_idx = local_end - chunk_size
z_state[:, replace_idx] = predicted_x0[:, replace_idx].detach()
```

The current implementation keeps a clean committed latent state and builds a
fixed corrupted input view for each rollout step. Self-forcing replacement writes
predicted tokens into the clean state, while the corruption mask/value overlay
remains a read-only view over initial history.

Do not mutate `z_state` with `mask_emb` or additive corruption noise. Corruption
is an observation/input view; self-forcing replacement writes only generated
latent predictions into the committed state.

## 14. Anchor Movement Inside Rollout

The first implementation should not move `S` inside one self-forcing rollout.

Instead:

```text
one rollout: fixed S, E advances by step
across batches: S is sampled from many valid anchors
```

This keeps the implementation tractable and still trains many body-window
anchors.

Add `anchor_move_in_rollout: true` only as a later ablation. That version would
need to rebuild trajectory, masks, text context, active frame offsets, and VAE
loss decode context at every self-forcing step.

## 15. Text Context Cropping

For single-caption HumanML3D, text cropping is usually trivial.

For segmented text data, token endpoints must be converted from global to local
window coordinates:

```text
global segment [g0, g1)
window [S, E)
intersection = [max(g0, S), min(g1, E))
local segment = [intersection.start - S, intersection.end - S)
```

Do not pass global `token_text_end` into a local window model batch.

## 16. Motion-Space Loss Decode

Do not decode latent `[S:E]` from an empty VAE state and treat it as a valid
motion prefix. That resets causal VAE state and does not match streaming.

Allowed v1 paths:

1. latent-only bring-up: disable motion-space body/control loss while validating
   indexing, masks, and self-forcing;
2. full-prefix decode for loss: splice predicted local latents back into the
   full prefix `[0:E]`, decode with normal VAE context, then canonicalize the
   decoded active root to the same window-local anchor before comparing;
3. VAE stream-cache warm-up: warm cache with `[0:S]`, stream-decode predicted
   `[S:E]`, then compare in window-local coordinates.

Current implementation status:

```text
option 1 latent-only      -> implemented, default when stream_training is enabled
option 2 full-prefix loss -> implemented behind motion_aux_loss: full_prefix
option 3 stream-cache     -> not implemented; later optimization
```

Recommended production path: validate option 2 on a small real-data run first,
then consider option 3 only if decode cost becomes a bottleneck.

Recommended validation order:

1. latent-only real-data smoke: window sampling, length contracts, trajectory
   grouping, horizon masks, text crop, corruption behavior, and local
   replacement;
2. small overfit with `motion_aux_loss: latent_only`;
3. small overfit with `motion_aux_loss: full_prefix`;
4. optional VAE stream-cache warm-up optimization.

During smoke / overfit runs, check the `stream_training/*` metrics:

```text
stream_training/enabled
stream_training/sample_policy_fixed_window
stream_training/window_start_mean
stream_training/window_len_mean
stream_training/window_len_min
stream_training/window_len_max
stream_training/traj_tokens_mean
stream_training/active_history_len_mean
stream_training/active_history_len_min
stream_training/active_history_len_max
stream_training/active_abs_end_mean
```

These metrics are not model-selection metrics. They are run-control diagnostics
that confirm the window sampler is producing the intended limited-history
distribution before interpreting motion quality.

Use the checked smoke runner rather than hand-writing a long override list:

```bash
python scripts/stream_training_smoke.py \
  --resume-ckpt /path/to/body.ckpt \
  --vae-ckpt /path/to/vae.ckpt \
  --raw-data-root /path/to/raw_data \
  --train-split /path/to/raw_data/HumanML3D/train_min.txt \
  --val-split /path/to/raw_data/HumanML3D/val.txt \
  --z-stats-dir deps/body_stats \
  --max-steps ABSOLUTE_TARGET_STEP \
  --sample-policy variable_history \
  --motion-aux-loss latent_only \
  --print-only
```

`--raw-data-root` must point to the `raw_data` root directory, not the
`HumanML3D` subdirectory. The dataset loads samples relative to the split file
directory, but the config also resolves shared assets such as
`${dirs.raw_data}/HumanML3D/t5_text_embeddings.pt`; passing the dataset
subdirectory as `dirs.raw_data` would produce a broken
`HumanML3D/HumanML3D/...` path.

To print the whole recommended validation matrix, use:

```bash
python scripts/stream_training_smoke.py \
  --resume-ckpt /path/to/body.ckpt \
  --vae-ckpt /path/to/vae.ckpt \
  --raw-data-root /path/to/raw_data \
  --train-split /path/to/raw_data/HumanML3D/train_min.txt \
  --val-split /path/to/raw_data/HumanML3D/val.txt \
  --z-stats-dir deps/body_stats \
  --max-steps ABSOLUTE_TARGET_STEP \
  --validation-plan \
  --manifest outputs/stream_training_smoke/validation_plan.json \
  --print-only
```

The plan expands into:

```text
01_smoke_latent          variable_history + latent_only
02_overfit_latent        variable_history + latent_only
03_overfit_full_prefix   variable_history + full_prefix
04_smoke_fixed_window    fixed_window     + latent_only
```

The manifest also records the post-training stream-eval closure for each stage:

```text
stream_eval.baseline
  -> python -m eval.ldf.stream_metrics --run_name baseline_full_prefix ...
  -> <output-dir>/stream_eval/baseline_full_prefix/summary.json

stages[i].post_training_eval.candidate_eval
  -> python -m eval.ldf.stream_metrics --ckpt {candidate_ckpt}
     --run_name <stage> ...
  -> <output-dir>/stream_eval/<stage>/summary.json

stages[i].post_training_eval.comparison
  -> python -m eval.ldf.report --baseline ... --candidate ...
  -> <output-dir>/stream_eval/<stage>_comparison.json
```

`{candidate_ckpt}` is intentionally a placeholder because Lightning checkpoint
file names depend on the actual run. Use `expected_candidate_ckpt_glob` from the
manifest to locate the checkpoint after a stage finishes, then replace the
placeholder in the recorded `argv` / `command`.

After the training stages finish, run the manifest-recorded post-training stream
eval and comparison reports with:

```bash
python scripts/stream_training_post_eval.py \
  --manifest outputs/stream_training_smoke/validation_plan.json \
  --out outputs/stream_training_smoke/post_eval_status.json
```

The runner is idempotent by default: it skips `summary.json` / comparison files
that already exist, resolves the newest checkpoint for each stage from
`expected_candidate_ckpt_glob`, replaces `{candidate_ckpt}` in candidate eval
commands, and then writes the collected validation status.

If the post-eval commands have already been run manually or by a scheduler,
collect the matrix status without executing anything:

```bash
python scripts/stream_training_collect.py \
  --manifest outputs/stream_training_smoke/validation_plan.json \
  --out outputs/stream_training_smoke/validation_status.json
```

The collector is read-only. It resolves each stage's newest checkpoint from
`expected_candidate_ckpt_glob`, checks the recorded `summary.json` and
comparison JSON paths, and writes:

```text
overall.status = passed | failed | pending
stages[i].status = passed | failed | pending
stages[i].commands.candidate_eval.argv = argv with {candidate_ckpt} resolved
stages[i].commands.comparison.argv = report argv from the manifest
```

Use `stages[i].commands.candidate_eval.command` when a stage checkpoint exists
but its stream eval summary is still pending. This avoids manually replacing
`{candidate_ckpt}` in the manifest.

Both scripts use the same final status convention:

```text
0 -> all stages passed and baseline summary exists
1 -> at least one comparison report failed
2 -> outputs are still pending or missing
```

Then remove `--print-only` on the data machine. For a single second validation
pass, repeat with:

```bash
--motion-aux-loss full_prefix
```

For the fixed-window sampler pass, use:

```bash
--sample-policy fixed_window
```

`max_steps` is the absolute Lightning target step, not the number of extra
steps. When resuming from step `N`, a one-step smoke uses `--max-steps N+1`.

Starting with motion-space loss enabled at the same time as the window-forward
refactor is discouraged because failures become hard to localize.

## 17. Suggested Patch Order

The small-patch implementation order was:

1. Range-aware batch plumbing:
   `traj_start_token` scalar / `[B]`, explicit `traj_num_tokens`,
   range-aware masks, and `_get_traj_seq_lens()` semantics.
2. Window-local trajectory builder:
   raw 263 window -> recovered local 7D, with unavailable future tail masked.
3. Latent-only window-local self-forcing:
   limited latent input, fixed corruption view, local replacement, no
   motion-space aux loss yet.
4. Motion-space loss decode:
   full-prefix decode for loss first; stream-cache warm-up later.

Items 1-4 are implemented for v1 except stream-cache warm-up. The remaining
patches should focus on real-data validation and metrics.

## 18. Required Tests

Required unit tests before enabling the mode in a real run:

- `S=0` frame count uses prefix semantics `4N - 3`.
- `S>0` frame count uses arbitrary-window semantics `4N`.
- `S=5`, `N=20`, frame length `80` still yields `traj_num_tokens=20`, not 21.
- `latent_valid_len != traj_num_tokens`: latent input/loss uses
  `latent_valid_len`, while ControlNet trajectory attention uses
  `traj_num_tokens`.
- Shifted-local diffusion time matches the full-prefix global beta schedule
  sliced to `[S:E]`.
- Window-local 7D reconstructed from raw 263 has first x/z near zero, first yaw
  near zero, first fwd_delta zero, and first yaw_delta zero.
- Directly cropped full-clip 7D is not accepted as the main reconstruction path.
- `traj_start_token=S` is threaded into trajectory token grouping and horizon
  masking.
- `E+H` past the clip end masks the unavailable future tail and performs no
  out-of-range raw feature read.
- If per-sample `S` is supported, `traj_start_token` tensor `[B]` works in
  `encode_traj_batch`, token masks, and horizon masks.
- Per-sample `S=[0,5]` in one batch applies prefix grouping to the first sample
  and arbitrary-window grouping to the second sample.
- Active frame slices are computed as global slice minus origin frame.
- Self-forcing replacement writes local index `E-S-chunk_size`.
- Corruption is sampled once and fixed across K steps, with replacement written
  into clean `z_state` rather than compared against or committed into the
  corrupted input view.
- `final_E - S <= context_tokens` is enforced.
- `min_history_tokens >= chunk_size` is enforced.
- Segmented text endpoints are converted from global to local window coordinates.
- Motion-space loss never decodes `[S:E]` as an empty-state prefix.

## 19. Acceptance Criteria

The implementation is acceptable for flag-gated v1 only when:

- model latent input is limited to `[S:E]`;
- latent is a precomputed continuation slice, not a re-encoded window prefix;
- training forward distinguishes `latent_valid_len` from `traj_num_tokens` /
  `attn_len`;
- trajectory sequence length is explicit and is not inferred from arbitrary
  window frame count with prefix formulas;
- trajectory condition is window-local and reconstructed from raw 263 or an
  explicitly tested delta-recompute fallback;
- `traj_start_token` preserves the global token start `S`;
- diffusion time is global-clock equivalent through shifted-local indexing;
- unavailable future trajectory beyond the clip end is masked false;
- self-forcing replacement, horizon, loss, and text indices are all local/global
  consistent;
- history corruption remains fixed for the rollout as a corrupted input view over
  clean `z_state`;
- existing full-prefix training remains available when `stream_training.enabled`
  is false.

Before using this as a production training default, additionally require:

- real-data smoke run with `stream_training.enabled: true`;
- small overfit with `motion_aux_loss: latent_only`;
- small overfit or targeted numerical validation with
  `motion_aux_loss: full_prefix`;
- stream eval comparison against the current full-prefix checkpoint.
- `scripts/stream_training_collect.py` reports `overall.status == "passed"` for
  the validation manifest.

Training-time async stream eval is available but default-off:

```yaml
validation:
  test_mode: async
  stream_eval:
    enabled: true
```

When this gate is enabled, async request JSONs carry a `stream_eval` block and
the watcher dispatches `python -m eval.ldf.stream_metrics`. Keep it disabled for
ordinary runs until the validation matrix above has passed on real data.

After both checkpoints have `eval/ldf/stream_metrics.py` `summary.json` files,
write a strict comparison report:

```bash
python -m eval.ldf.report \
  --baseline /path/to/full_prefix_stream_eval/summary.json \
  --candidate /path/to/window_local_stream_eval/summary.json \
  --out outputs/stream_training_smoke/stream_eval_comparison.json \
  --max-root-ade-regression 0.05 \
  --max-root-fde-regression 0.05 \
  --max-path-arc-regression 0.05 \
  --max-yaw-error-regression 0.20 \
  --max-jitter-regression 0.01 \
  --max-foot-skating-regression 0.03
```

`eval/ldf/stream_metrics.py` writes design-aligned aliases such as
`stream_gt/root_ADE`, `stream_gt/jitter`, and `stream_gt/yaw_error` directly into
`summary.json` while keeping legacy keys such as `traj/ADE_mean`.
`eval.ldf.report` accepts either naming style, so checkpoint selection does not
depend on remembering legacy key names. The report can gate regressions in
root ADE/FDE, path arc ADE, yaw error, jitter, foot skating, and chunk-boundary
root jump.
