# LDF Streaming TODO

This note summarizes the current LDF / RootRefiner / streaming integration
gaps and the practical route to fix them. It focuses on the body model runtime
and LDF training objective, not on RootRefiner architecture details.

## 0. Target Streaming State Machine

The target runtime should be a single explicit loop:

```text
head_state: current world root pose and commit_idx
body_anchor_state: world pose at the left edge of the body window
latent rolling buffer: recent generated latent tokens
root plan: future trajectory plan, in plan-anchor-local coordinates
text state: current text / text schedule
        |
        v
RootPlan -> body-window-local 7D trajectory condition
        |
        v
Denoise with recent latent context W and future trajectory horizon H
        |
        v
Commit one latent token
        |
        v
VAE stream_decode -> motion frames
        |
        v
Recover local root delta / yaw delta
        |
        v
Advance world-space head_state via dual-anchor rule
        |
        v
Next step
```

The code already has many pieces of this state machine:

- `InferenceGlueState` and `InferenceGlueTimeline`
- `advance_head_from_body_window`
- `RootPlan`
- `TrajStreamBuffer.get_body_traj_cond`
- 7D `LocalTrajEncoder` / `TrajEncoder`
- body-window canonicalization in self-forcing training
- horizon simulation and history corruption

The main problem is no longer missing concepts. The web-demo runtime now has a
RootPlan / direct-7D bridge for the normal route path, runtime horizon attention
is wired through explicit trajectory token lengths, and a flag-gated
window-local limited-history training path now exists. The remaining large gaps
are validation and rollout:

- the new training path is still default-off and needs real-data smoke /
  small-overfit runs before it should become a production training default;
- stream eval exists, but it still needs to become the primary checkpoint
  selection signal for LDF/web-demo checkpoints;
- RootPlan/runtime behavior should stay covered by regression tests while the
  training path changes.

## 1. P0 Issues

### P0-1: Web Demo RootPlan / 7D Path Needs Production Lock-In

**Status: partially done**

`web_demo/model_manager.py` now has a RootPlan / direct 7D payload path through
`_build_rootplan_stream_traj_input()` and `utils.runtime_rootplan`. The payload
carries:

```text
traj_cond_7d_frame
traj_cond_frame_mask
traj_start_token
traj_num_tokens
```

Tests cover RootPlan payload construction and prevent silent fallback to legacy
XYZ in important update paths.

The historical issue was that the demo built a future trajectory and passed it
as:

```python
{
    "traj": future[None, :, :],
    "token_mask": token_mask,
}
```

`stream_generate_step()` then calls:

```python
self._traj_buf.update(x, self.commit_index, device)
traj_emb = self._traj_buf.build_traj_emb(end_index, self.seq_len, device)
```

This was the legacy XYZ path. It is not the intended 7D RootPlan path.

**Remaining work**

The remaining work is production lock-in, not greenfield implementation:

- ensure every normal user-route / RootRefiner route update goes through
  RootPlan 7D;
- keep legacy `"traj"` XYZ only as explicit debug / ablation behavior;
- keep stream eval coverage for rootplan payloads after future training changes.

**Why this matters**

The current LDF trajectory encoder is 7D-only:

```text
[x, y, z, cos(yaw), sin(yaw), fwd_delta, yaw_delta]
```

`TrajStreamBuffer._build_from_xyz()` already fail-fasts for the 7D model and
tells callers to use `get_body_traj_cond()` / RootPlan instead. So the demo path
and the trained 7D ControlNet contract are not aligned.

**Solution**

Make RootPlan / 7D body-window-local condition the runtime main path:

```text
manual path or RootRefiner output
  -> RootPlan(waypoints_local_7d)
  -> InferenceGlueTimeline.head + body_anchor_state
  -> TrajStreamBuffer.get_body_traj_cond(...)
  -> frame-level 7D traj condition + frame mask
  -> encode_traj_batch / LocalTrajEncoder / TrajEncoder
  -> ControlNet
```

Implementation checklist:

- Keep the explicit `traj_cond_7d_frame` + `traj_cond_frame_mask` payload for
  `stream_generate_step()`.
- Keep routing that payload through the same frame-to-token 7D encoding logic
  used by training.
- Carry the global trajectory/window start token, e.g. `traj_start_token = S`.
  The encoder must know whether a frame slice is a causal-VAE prefix or an
  arbitrary rolling range; otherwise token/frame grouping will be off by three
  frames for non-prefix windows.
- Keep the old `"traj"` XYZ path only as a debug / ablation path.

### P0-2: Streaming Runtime Future-Horizon Path Needs Continued Verification

**Status: partially done**

The direct 7D stream path now supports `traj_num_tokens` and can extend
trajectory attention beyond the latent context via `attn_sl`. The runtime also
preserves global diffusion timing when `attn_sl > model_sl`.

The historical behavior was:

`stream_generate_step()` built trajectory embeddings with:

```python
traj_emb = self._traj_buf.build_traj_emb(end_index, self.seq_len, device)
model_sl = min(end_index, self.seq_len)
```

That window is effectively:

```text
traj keys: [end_index - W, end_index)
```

It does not include future trajectory keys after the active right boundary.

**Why this matters**

Training horizon simulation is designed around this idea:

```text
latent context:      up to active_end
trajectory context: up to active_end + horizon_tokens
```

The runtime should let the model see future route intent:

```text
latent keys: [S, E)
traj keys:   [S, E + H)
```

That made runtime closer to:

```text
latent keys: [S, E)
traj keys:   [S, E)
```

That made trajectory control short-sighted even when `traj_horizon_tokens=20`
existed in the demo configuration.

**Remaining work**

- Keep tests proving `latent_len < traj/attn_len` actually exposes `[S:E+H)`
  trajectory keys.
- Keep the window-local training length split:
  `feature_length = latent_valid_len`, while `traj_num_tokens = attn_len`.
- Keep fail-fast guards for direct 7D payloads whose `traj_start_token` begins
  after the current latent window.

**Solution**

Separate latent context length from trajectory condition length:

```text
W = context_tokens
H = horizon_tokens
S = max(0, E - W)

latent input = z[S:E]
traj input   = traj_cond[S:E+H]
```

Implementation notes:

- `WanModel` / `WanControlNet` FlexTraj attention already supports separate
  `latent_lens` and `traj_lens`.
- Some forward paths still assume one shared `seq_len` padding length for the
  latent and trajectory halves. Window-local training must explicitly split
  `latent_valid_len` from `traj_num_tokens` / `attn_len`; otherwise the future
  horizon silently collapses back to the active latent length.
- Use explicit local indexing for updates:

```text
local_start = start_index - S
local_end   = end_index - S
generated[..., start_index:end_index] += pred[..., local_start:local_end] * dt
```

- Preserve the global diffusion clock when slicing a rolling window. Either
  compute beta from global token positions or shift the local time by
  `-S / chunk_size`. Do not reset the diffusion schedule at local window start.
- Future trajectory horizon requires FlexTraj attention to expose future traj
  keys to latent queries. Add a fail-fast if a future-horizon runtime is used
  with incompatible attention semantics, e.g. `causal=True`.

### P0-3: Window-Local Limited-History Training Needs Real-Run Validation

**Status: flag-gated v1 implemented; default disabled**

The default training path remains full-prefix because `stream_training.enabled`
is false in the main config. When enabled, the new window-local path now:

```text
latent input      = precomputed continuation tokens z[S:E]
feature tensor    = padded to latent_valid_len + horizon_tokens
feature_length    = latent_valid_len only
trajectory input  = window-local 7D for [S:E+H]
traj_start_token  = S
traj_num_tokens   = latent_valid_len + horizon_tokens
```

The implementation reconstructs 7D trajectory conditioning from raw 263D motion
instead of directly cropping full-clip `traj_cond_7d`, supports per-sample
`traj_start_token`, and preserves global horizon coordinates by adding `S` to
the local active end token. It also crops segmented text endpoints from global
token coordinates to the local window, supports both `variable_history` and
`fixed_window` sampling policies, and keeps history corruption as a fixed input
view over a clean committed latent state.

**Remaining work**

The default v1 mode is latent-only. Motion-space body/control auxiliary losses
are disabled for `stream_training.motion_aux_loss: latent_only`, because
decoding `[S:E]` from an empty VAE state would not match runtime.

An opt-in `stream_training.motion_aux_loss: full_prefix` path now splices local
predictions back into the full latent prefix `[0:E]` before body aux loss decode.
This gives the causal VAE the correct prefix context. Unit smoke tests cover both
the latent-only tiny-model path and the full-prefix splice path, but this has not
yet been validated as a real training run on the project dataset.

Remaining bring-up work:

- run real training smoke / small overfit with `stream_training.enabled: true`
  via `scripts/stream_training_smoke.py`;
- verify `stream_training/*` run-control metrics show the intended window
  distribution before judging motion quality;
- validate `motion_aux_loss: full_prefix` numerically on a small real-data run;
- decide later whether to support moving `S` inside one self-forcing rollout.

`scripts/stream_training_smoke.py --validation-plan --manifest ...` also writes
the post-training stream-eval closure for each stage:

```text
stream_eval/baseline_full_prefix/summary.json
stream_eval/<stage>/summary.json
stream_eval/<stage>_comparison.json
```

The candidate checkpoint is recorded as `{candidate_ckpt}` plus an
`expected_candidate_ckpt_glob`, because the exact Lightning checkpoint file name
is only known after the stage finishes.

`eval/ldf/stream_metrics.py` writes both legacy summary keys
(`traj/ADE_mean`, `traj/FDE_mean`, etc.) and design-aligned keys
(`stream_gt/root_ADE`, `stream_gt/root_FDE`, `stream_gt/path_arc_ADE`,
`stream_gt/jitter`, `stream_gt/yaw_error`, `stream_gt/foot_skating`,
`stream_no_traj/root_ADE`, `control_gain/root_ADE_delta`, etc.). The
comparison report accepts either form and can fail on regressions in root
ADE/FDE, path arc ADE, yaw error, jitter, foot skating, and chunk-boundary root
jump.

After the manifest-recorded training commands run on a data machine, run the
missing post-training stream eval and comparison commands with:

```bash
python scripts/stream_training_post_eval.py \
  --manifest outputs/stream_training_smoke/validation_plan.json \
  --out outputs/stream_training_smoke/post_eval_status.json
```

The runner is idempotent by default: it skips outputs that already exist,
resolves `{candidate_ckpt}` from each stage's `expected_candidate_ckpt_glob`, and
then writes the collected validation status. If stream eval/report commands have
already been run manually or by a scheduler, collect status without executing
anything:

```bash
python scripts/stream_training_collect.py \
  --manifest outputs/stream_training_smoke/validation_plan.json \
  --out outputs/stream_training_smoke/validation_status.json
```

This read-only collector resolves the newest stage checkpoint from
`expected_candidate_ckpt_glob`, checks every recorded `summary.json` and
comparison report, and returns exit code `0` for passed, `1` for failed
comparison, and `2` for pending or missing outputs. When a checkpoint exists
but the stream eval summary is still pending, the status JSON includes
`stages[i].commands.candidate_eval.command` with `{candidate_ckpt}` already
resolved.

**Historical behavior**

Training did not feed future latent tokens, but it usually fed the complete
prefix up to the active right boundary:

```text
latent input = tokens [0, E)
```

Runtime streaming uses a rolling window:

```text
latent input = tokens [max(0, E - W), E)
```

So the past history context distribution was not the same.

**Why this matters**

The model can rely on a long prefix during training but only receives a fixed
recent window during web-demo inference. Body-window canonicalization also tends
to anchor to the clip/window start when `body_window_tokens=seq_len`, while the
runtime anchor is the rolling body-window left edge.

**Solution / Current Contract**

Use the flag-gated window-local limited-history self-forcing path and keep it
aligned with the normative design in
[ldf_window_local_training_design.md](ldf_window_local_training_design.md).

```yaml
stream_training:
  enabled: true
  context_tokens: 30
  min_history_tokens: 8
  horizon_tokens: 20
  anchor_source: gt_root
  latent_source: precomputed_slice
```

For each training step:

```text
Choose token-space window anchor S and active end E
Keep final_E - S <= context_tokens across the self-forcing rollout
latent input = precomputed continuation tokens z[S:E]
trajectory condition = window-local 7D for [S:E+horizon_tokens]
traj_start_token = S
latent_valid_len = E - S
attn_len = traj_num_tokens = latent_valid_len + horizon_tokens
loss window = active chunk around E, mapped into local frame offsets
```

Important: do not only change `body_window_tokens`. The latent window, trajectory
window, body anchor, and frame/token offset used by the loss must all change
together.

Do not re-encode the VAE latent window in the first implementation. Runtime uses
a continuous latent buffer; training should slice precomputed continuation tokens
`[S:E]`. Reconstruct the window-local 7D trajectory / motion-space target from
raw 263 motion, or use an explicitly tested fallback that canonicalizes and
recomputes delta channels. See P0-6 for loss decode context.

### P0-4: Arbitrary-Window Frame-To-Token Grouping Is A Regression Guard

**Status: implemented for current runtime and window-local training paths**

Range-aware helpers such as `frames_to_tokens_range`,
`frames_to_token_mask_range`, `traj_start_token`, and runtime direct 7D payload
tests already exist. The window-local training helper now supports per-sample
`S` and explicit `traj_num_tokens`. Keep this section as a regression guard:
future refactors must not infer arbitrary-window token count from frame count
using prefix-only formulas.

**Current behavior**

The legacy prefix helper `utils.traj_batch.frames_to_tokens()` assumes a
causal-VAE prefix:

```text
token 0      -> frame 0, padded to 4 copies
token k >= 1 -> frames [4k - 3, 4k]
```

This is correct for prefix ranges like `[0, N)`.

**Why this matters**

Streaming and window-local limited-history training use arbitrary token ranges:

```text
[S, S + N)
```

When `S > 0`, the first local token is not the special global token 0. It should
consume a full 4-frame token span, not one effective frame padded to four.

Example:

```text
prefix range [0, 20)       -> 77 frames
arbitrary range [5, 25)    -> 80 frames
```

If an 80-frame arbitrary window is fed to the prefix-only `frames_to_tokens()`
with `seq_len=20`, it will silently use the wrong causal grouping.

**Solution**

Keep and extend range-aware helpers:

```python
frames_to_tokens_range(
    feats_frame,
    *,
    start_token_idx: int,
    num_tokens: int,
    frames_per_token: int = 4,
)
```

and the matching mask conversion helper.

Rules:

```text
start_token_idx == 0:
  use prefix convention, length = 4N - 3

start_token_idx > 0:
  every local token has 4 frames, length = 4N
```

Use this in:

- runtime 7D trajectory encoding;
- window-local limited-history self-forcing training;
- RootPlan body-window-local condition ingestion.

Additional requirement for training:

```text
traj_start_token: scalar or [B]
traj_num_tokens: scalar or [B]
```

Do not infer trajectory token length from arbitrary-window frame count with
prefix formulas.

### P0-5: Rolling-Window Diffusion Time Must Preserve Global Position

**Status: implemented by equivalent shifted-local time; keep as regression
guard**

The existing streaming path preserves global time by computing:

```python
noise_level_full = self._get_noise_levels(device, end_index, time_steps)
noise_level = noise_level_full[:, -self.seq_len:]
```

That means the beta schedule is still indexed by global token position.

Window-local training uses the equivalent shifted-local form. If the global
window is `[S, E)`, the rollout time step is computed from local `E - S`, which
is equivalent to subtracting `S / chunk_size` from the global diffusion clock:

```text
t_local = t_global - S / chunk_size
```

This is correct because `_get_noise_levels()` receives local token positions
inside the sliced window. Future refactors should keep this equivalence explicit.

**Risk**

When refactoring to fixed windows, it is easy to call:

```python
_get_noise_levels(device, model_seq_len, local_time_steps)
```

and accidentally make the rolling window look like a new clip starting at token
0. That shifts beta/noise levels and changes the active-window semantics.

**Solution**

For a rolling window `[S, E)`, keep one of these equivalent formulations:

```text
beta = clamp(1 + global_token_idx / chunk_size - t_global, 0, 1)
```

or:

```text
t_local = t_global - S / chunk_size
beta = clamp(1 + local_token_idx / chunk_size - t_local, 0, 1)
```

Do not reset diffusion time at the window start in a way that changes beta.
Local-time indexing is allowed only when it is the shifted-local equivalent of
the global token clock.

### P0-6: Motion-Space Loss Needs Causal VAE Decode Context

**Status: latent-only default plus opt-in full-prefix splice implemented**

**Current behavior**

The control/body auxiliary losses decode predicted latents with the VAE and then
recover root position / yaw:

```python
decoded = vae.decode(pred_latent.unsqueeze(0))[0]
```

This is acceptable while `pred_latent` is effectively a prefix. It is unsafe when
the model input is a fixed suffix/window `[S, E)` unless the loss path first
restores the missing causal prefix context.

**Why this matters**

The VAE is causal and has a `stream_decode()` cache path. Decoding only `[S, E)`
from an empty VAE state treats token `S` as a new sequence start. Runtime does
not do that; it decodes after previous context/cache.

**Solution**

For window-local limited-history training, do not decode `[S, E)` from an empty
VAE state as if token `S` were a new prefix. The current implementation supports
two safe v1 paths:

1. **Full-prefix decode for loss**: model forward sees `[S, E)`, but
   motion-space loss splices predicted local latents back into the full prefix
   `[0:E]`, decodes with proper VAE context, canonicalizes decoded active root
   to the window anchor `S`, and compares in window-local coordinates.
2. **Latent-only bring-up**: disable motion-space control/body losses
   while validating limited-history indexing, then re-enable them with option 1
   after real-run validation.

The later optimization path is still VAE stream-cache warm-up: warm the VAE cache
with `[0:S]`, then `stream_decode()` predicted `[S:E]` and compare in
window-local coordinates.

## 2. P1 Issues

### P1-1: Training Anchor Is GT, Runtime Anchor Is Generated

**Current behavior**

Training canonicalizes 7D trajectory conditions using GT root pose:

```text
traj_local = Canon(traj_world; anchor_gt)
```

Runtime canonicalizes using the generated timeline state:

```text
traj_local = Canon(traj_world; anchor_generated)
```

**Risk**

If generated root pose drifts, the trajectory condition distribution seen by
the body model differs from training.

**Solution**

Use a staged curriculum:

1. Keep `anchor_source: gt_root` while window-local limited-history training is
   first brought up.
2. Add anchor noise as a cheap robustness step:

```text
anchor_xz  = gt_xz  + noise_xz
anchor_yaw = gt_yaw + noise_yaw
```

3. Later add mixed generated anchors:

```yaml
anchor_canonicalize:
  source: mixed
  pred_anchor_prob:
    early: 0.0
    mid: 0.3
    late: 0.7
```

Generated-anchor training is more expensive because it needs decoded/recovered
poses from the self-forcing rollout.

### P1-2: No-Trajectory Branch Is Unified To ControlNet(null)

**Status: done for the current code path**

Training with dropped trajectory runs:

```text
backbone + ControlNet(null)
```

Inference no-traj now also runs ControlNet(null), including:

- text-CFG double-batch path;
- text-CFG fallback path;
- no-CFG path;
- separated trajectory-CFG unconditioned branch.

`cfg_scale_traj` does not apply when trajectory is absent. Keep this section as
a regression guard: do not reintroduce a backbone-only inference no-traj branch
unless it is an explicit ablation.

### P1-3: Stream Evaluation Must Become The Main Model-Selection Path

**Current behavior**

Inline eval still primarily calls:

```python
module.model.generate(model_batch)
```

There are now stream diagnostic/eval scripts and LDF stream-step eval paths, but
stream eval is still not the primary training-time model-selection signal.

The async validation path now has a default-off gate:

```yaml
validation:
  stream_eval:
    enabled: false
    stream_mode: stream_generate_step
    max_samples: 5
    num_runs: 1
```

When enabled together with `validation.test_mode: async`, emitted eval requests
carry a `stream_eval` block, and `eval/eval_watcher.py` dispatches
`python -m eval.ldf.stream_metrics` instead of the offline `run_eval.py` path.
The request is only marked complete after the stream eval `summary.json` exists;
missing summaries remain retryable because stream eval is a checkpoint gate.
The stream-eval gate is validated at training startup, including stream mode,
`num_runs`, `max_samples`, and `max_batches`.

**Risk**

Offline generation can improve while long-horizon web-demo streaming still
drifts, jitters, or under-runs.

**Solution**

Promote a stream eval runner to a first-class validation path:

```text
initialize stream state
loop stream_generate_step()
commit one token per step
stream_decode
recover root / joints / yaw
advance InferenceGlueTimeline
compute long-horizon metrics
```

Minimum LDF model-selection metrics:

- `stream_gt/root_ADE`
- `stream_gt/root_FDE`
- `stream_gt/yaw_error`
- `stream_gt/path_arc_ADE`
- `stream_gt/jitter`
- `stream_gt/foot_skating`
- text-update reaction delay
- trajectory-update delay / blend correctness

Runtime / performance diagnostics should still be saved, but they should not be
the primary LDF checkpoint-ranking metrics:

- `stream/latency_ms_per_token`
- `stream/buffer_underflow_rate`

## 3. P2 Issues

### P2-1: RootRefiner Path-Feature Stats Are Wired, Keep As Regression Guard

**Status: done for the current main config**

The R2 RootRefiner design expects path geometry and path summary features to
have separate normalization:

```text
path geometry   -> waypoint x/z stats
path_features   -> path-feature own stats
```

The implementation supports `data.path_feature_stats_dir`, and
`configs/root_refiner_train.yaml` now sets:

```yaml
data:
  path_feature_stats_dir: deps/refiner_path_stats
```

Keep this section as a regression guard. Do not remove the separate path-feature
stats unless the RootRefiner input adapter is changed deliberately.

If the stats directory is missing for a new environment, generate it with:

```bash
python scripts/compute_path_stats.py \
  --config configs/root_refiner_train.yaml \
  --output-dir deps/refiner_path_stats
```

Keep the config-hash validation enabled.

### P2-2: RootRefiner Runtime Boundary Exists, Keep Covered

**Status: done for the current web-demo boundary**

RootRefiner predicts normalized 5D waypoints. The LDF body model requires
physical 7D trajectory condition.

The current runtime boundary is:

```text
RootRefiner normalized 5D
  -> unnormalize xyz
  -> normalize heading cos/sin
  -> append fwd_delta / yaw_delta
  -> RootPlan(waypoints_local_7d)
```

`utils.refiner.runtime.RootRefinerRuntime.build_root_plan()` performs this
conversion with `utils.motion_process.build_physical_7d_from_normalized_5d`,
and `web_demo.model_manager.ModelManager` uses it when RootRefiner runtime is
enabled.

Remaining work is coverage and production lock-in:

- keep tests that assert web-demo route updates use RootRefiner / RootPlan 7D
  when available;
- keep manual fallback RootPlan conversion only as fallback/debug behavior;
- keep eval artifacts that compare RootRefiner-produced 7D with GT 7D route
  control.

## 4. Recommended Implementation Order

### Phase 1: Runtime 7D Main Path + Range-Aware Tokenization

Goal: keep the RootPlan / direct-7D web-demo path locked in and prevent fallback
to legacy XYZ in normal user-route / RootRefiner route updates.

Tasks:

1. Keep RootPlan state and `InferenceGlueTimeline` in `ModelManager`.
2. Keep manual path / RootRefiner output converted into `RootPlan`.
3. Keep arbitrary-window frame/token grouping helpers for non-prefix ranges.
4. At each step, compute:

```text
head_state = timeline.head
body_anchor_commit = max(0, head_state.commit_idx - context_tokens)
body_anchor_state = timeline.at_commit(body_anchor_commit)
```

5. Build body-window-local 7D condition with
   `TrajStreamBuffer.get_body_traj_cond(...)`.
6. Feed the 7D condition through the training-compatible trajectory encoder
   using range-aware frame-to-token grouping.
7. Keep tests that fail if normal web-demo RootPlan routes silently fall back to
   the legacy `"traj"` XYZ path.

### Phase 2: Future-Horizon Trajectory In Streaming + Global Time

Goal: keep verifying that runtime ControlNet sees `[S, E+H)` trajectory, not
only `[S, E)`.

Tasks:

1. Keep separate `context_tokens` and `horizon_tokens` in stream runtime.
2. Keep `stream_generate_step()` accepting a trajectory condition longer than the
   latent context.
3. Keep latent update indexing explicit and local-window aware.
4. Preserve diffusion beta/noise levels using global token positions or shifted
   local time.
5. Add fail-fast guards for incompatible attention settings.
6. Add unit tests for off-by-one token/frame slices and beta schedule alignment.

### Phase 3: Stream Eval As A Regression Gate

Goal: validate runtime and training changes with GT trajectory / RootPlan stream
eval before trusting offline metrics.

Tasks:

1. Keep the numeric stream eval runner using GT trajectory / RootPlan.
2. Loop `stream_generate_step()` with one-token commits.
3. `stream_decode`, recover root/yaw, and compute root/path/yaw metrics.
4. Use stream eval to compare full-prefix checkpoints against window-local
   checkpoints.

### Phase 4: Window-Local Limited-History Self-Forcing Training

Goal: train on the same rolling-window shape as streaming inference.

Tasks:

1. Add `stream_training` config. **Done; default disabled.**
2. Sample token-space `S` and build a local latent window with
   `final_E - S <= context_tokens`.
   **Latent-only v1 done for `variable_history` and `fixed_window`.**
3. Slice precomputed latent continuation context `[S, E)`. **Done.**
4. Reconstruct window-local 7D trajectory / loss target from raw 263 motion for
   `[S, E+H)`.
   **Trajectory conditioning done; motion-space aux loss uses full-prefix splice
   when explicitly enabled.**
5. Preserve `traj_start_token=S`; do not reset arbitrary windows to prefix
   token 0. **Done.**
6. Keep self-forcing replacement in local window coordinates. **Done through
   local `feature_length` / `plan_rollout`.**
7. Compute losses with explicit global-to-local frame offsets and preserve the
   diffusion clock by shifted-local time.
   **Diffusion latent loss done; motion-space aux path exists behind
   `motion_aux_loss: full_prefix`, pending real-run validation.**
8. Decode motion-space losses with full-prefix context or VAE stream-cache
   warm-up. **Full-prefix splice path done; real training validation remains.**

### Phase 5: Robustness Curriculum

Goal: reduce long-horizon closed-loop drift.

Tasks:

1. Keep `horizon_sim.enabled: true`.
2. Keep `history_corruption.enabled: true`.
3. Add anchor noise or mixed generated anchors.
4. Try larger self-forcing K only after window-local limited-history training is
   stable.

Example later-stage K schedule:

```yaml
self_forcing_k_schedule:
  - [0.0, 1]
  - [0.2, 3]
  - [0.5, 5]
  - [0.8, 9]
```

### Phase 6: Stream Eval As Model Selection

Goal: select checkpoints by streaming behavior, not only offline generation.

Tasks:

1. Promote stream eval to validation / async eval. **Async request plumbing is
   now available behind `validation.stream_eval.enabled`; real-data use remains
   default-off until smoke / overfit validation is complete.**
2. Save both summaries:

```text
offline_eval/...
stream_eval/...
```

3. Use stream metrics to choose web-demo checkpoints.

## 5. Current Config Notes

For the current main `configs/ldf.yaml`, these are already enabled:

```yaml
horizon_sim:
  enabled: true

history_corruption:
  enabled: true
```

If a specific experiment disables them through `configs/stream.yaml` or command
line overrides, that is an experiment-local setting, not the main LDF training
default.

## 6. Highest-Impact Next Three Tasks

1. Lock and continuously test web-demo trajectory control through RootPlan / 7D
   body-window-local, including arbitrary-window frame-to-token grouping.
2. Keep verifying `stream_generate_step()` future-horizon behavior:
   `[S, E+H)` trajectory attention, explicit `traj_num_tokens`, and preserved
   global diffusion time.
3. Validate the flag-gated window-local limited-history training path on real
   data with `scripts/stream_training_smoke.py --validation-plan --manifest ...`:
   one-step smoke, small overfit with `latent_only`, targeted check with
   `full_prefix`, fixed-window smoke, then stream eval against the current
   full-prefix checkpoint and write the comparison with the manifest-recorded
   `python -m eval.ldf.report` command. Finish by running
   `scripts/stream_training_post_eval.py --manifest ... --out ...` or the
   read-only `scripts/stream_training_collect.py --manifest ... --out ...`, and
   require `overall.status == "passed"` before considering
   `stream_training.enabled` for production training.
