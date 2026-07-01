# Multi-Step Detached Root Loss Design

## Goal

Add a diagnostic self-forcing objective that gives every rollout step direct
7D body/root supervision while keeping rollout history replacement detached.
The objective tests whether poor stream tracking is caused by the current
self-forcing loss being too indirect: today only the final rollout step receives
diffusion and body-aux gradients, while earlier steps only produce detached
history replacements.

The new mode should train:

```text
L = final_diffusion_loss
  + control_loss_weight * multistep_commit_body_aux.weight
    * mean_{valid committed rollout steps}(L_commit_step)
```

where each `L_commit_step` is computed only on the frame range covered by that
step's committed token. The local `weight` lets the diagnostic term be tuned
without changing the run's global `control_loss_weight`.

## Non-Goals

- Do not add full cross-step BPTT. Replacements written into history remain
  detached.
- Do not supervise the whole decoded prefix. Only committed-token frames should
  contribute to the new auxiliary loss.
- Do not add per-step diffusion losses. Diffusion loss remains final-step only.
- Do not make K-decode strict training the default. That is a later debug
  comparison.

## Existing Behavior

`SelfForcingTrainer._run_rollout()` currently runs steps `0..K-2` under
`torch.no_grad()`, writes `result["x0_latent_list"][b][replace_idx].detach()`
back into `clean_feature_state`, then runs only the final step with gradients.

`DiffForcingWanModel._forward_single_window()` returns two x0 variants:

- `pred_x0_latent_list`: full-gradient estimate for motion/control losses.
- `x0_latent_list`: low-variance estimate for self-forcing history replacement.

The new objective must preserve that split: root loss uses
`pred_x0_latent_list`, replacement uses detached `x0_latent_list`.

## Rollout Data Flow

For every rollout step, including non-final steps:

1. Run `_forward_single_window()` with gradients enabled.
2. Compute `local_commit_idx = end_idx - model.chunk_size`.
3. Store, per sample:
   - `batch_idx`
   - `local_commit_idx`
   - `global_commit_idx`
   - `pred_token = result["pred_x0_latent_list"][b][local_commit_idx]`
4. If this is not the final step, write detached replacement into history:
   `clean_feature_state[b, local_commit_idx] =
   result["x0_latent_list"][b][local_commit_idx].detach()`.
5. Keep `final_step_result` from the final step for the diffusion loss.

Invalid commit indices should be skipped consistently with the current
replacement path.

## Prefix Assembly

The decode prefix source depends on `_window_local_body_aux_mode`.

For `full_prefix_splice`:

- `model_batch["feature"]` is window-local, not a full prefix.
- Start from the original full `batch["token"]` detached clone.
- Replace `global_commit_idx = _window_global_start_token + local_commit_idx`.
  Do not use `_window_local_latent_start_token` for this; local and global start
  metadata diverge between precomputed slicing and online encoding.
- Decode the assembled full prefix.
- Build one loss frame mask per commit record in full/global frame coordinates.

For `local_decode`:

- `model_batch["feature"]` is the local prefix produced by online encoding.
- Start from `model_batch["feature"]` detached clone.
- Replace `local_commit_idx`.
- Decode the assembled local prefix.
- Build one loss frame mask per commit record in local frame coordinates.

Only the K committed tokens carry gradients in the assembled decode latent.
All non-committed prefix tokens are detached.

## Commit-Frame Mask

Use the canonical causal VAE mapping in `utils.token_frame`:
`token_range_to_frame_slice(commit_idx, 1)`.

The body-aux loss must be computed over the decoded full/local prefix, then
masked to committed-token frames. Do not slice out only the committed frames
before deriving `fwd_delta` and `yaw_delta`; those delta terms need the complete
decoded prefix so the first committed frame can see the preceding frame.

Each commit record keeps its own frame mask. The implementation must not collapse
all commit masks into one union mask before loss computation, because that would
change the reduction from step mean to frame mean. Each mask should include only
that commit token's frame range; it should not include earlier history frames or
uncommitted future frames.

## Loss Reduction

`commit_7d_body_aux_loss` is reduced as a mean over valid committed rollout
steps:

```text
commit_7d_body_aux_loss = mean_s L_commit_step_s
```

Inside each `L_commit_step_s`, the 7D body terms use the existing frame-level
masked mean over that single committed token's frames. Across rollout steps,
each valid commit decision has equal weight. This keeps the loss scale
comparable across K values and lets K=1 recover the single-step commit
objective. A union-frame-mean variant may be added later for ablation, but it is
not the first diagnostic target.

## 7D Body-Aux Weights

The first diagnostic mode uses conservative commit-loss weights:

```yaml
root_xz: 1.0
root_y: 0.0
heading: 0.2
fwd_delta: 0.05
yaw_delta: 0.05
end_xz: 0.0
```

`end_xz` is disabled because the current implementation's
`last_valid_smooth_l1()` would only supervise the final valid frame across all
committed tokens, not each commit token's endpoint. A later variant may add
`per_commit_end_xz`.

## Loss Interaction

When the new mode is enabled, commit 7D body aux replaces the existing final-step
body aux. The commit loss includes the final rollout step by default, so the old
final-step body aux is replaced by a per-commit-token body aux over all K rollout
steps, including the final step. This avoids double-counting the final committed
token and keeps the diagnostic variable isolated.

When `replace_final_body_aux=true`, do not fall back to the old final-step body
aux if the commit loss is `None`. Returning only the final diffusion loss in
that edge case keeps the diagnostic comparison clean.

When the new mode is disabled, existing self-forcing and body-aux behavior must
remain unchanged.

## Configuration

Add a dedicated config block, default disabled:

```yaml
multistep_commit_body_aux:
  enabled: false
  decode_mode: single
  replace_final_body_aux: true
  include_final_step: true
  reduction: step_mean
  weight: 1.0
  weights:
    root_xz: 1.0
    root_y: 0.0
    heading: 0.2
    fwd_delta: 0.05
    yaw_delta: 0.05
    end_xz: 0.0
```

`decode_mode: single` is the initial implementation. `decode_mode: per_step`
is reserved for the stricter K-decode comparison.

## Runtime Cost

Single decode avoids K VAE decodes, but this mode still keeps K gradient-enabled
LDF forward graphs. K=3 should be the first diagnostic setting. K=5 may require
smaller batch size or additional memory checks.

## Expected Gradient Path

The root loss flows back through the single VAE decode into the K stored
`pred_x0_latent_list` commit tokens, and from those tokens into their respective
rollout-step model forwards.

It must not flow through the self-forcing history replacement chain, because
history writes use detached `x0_latent_list` values.

Single decode may still allow decoder-internal temporal coupling between
committed tokens. This is acceptable for the cheap diagnostic. The stricter
per-step decode comparison can measure whether that approximation matters.

## Tests

Add focused tests for:

- Non-final rollout steps run with gradients when the new mode is enabled.
- Root-loss commit tokens are sourced from `pred_x0_latent_list`.
- Replacement tokens are sourced from detached `x0_latent_list`.
- `full_prefix_splice` uses original full `batch["token"]` and global commit
  indices, not `model_batch["feature"]`.
- `full_prefix_splice` computes global commit indices from
  `_window_global_start_token`, not `_window_local_latent_start_token`.
- `local_decode` uses local `model_batch["feature"]` and local commit indices.
- Each per-record commit-frame mask covers only
  `token_range_to_frame_slice(idx, 1)` for that committed token.
- Prefix assembly returns sample indices so decoded-list positions map back to
  the correct original batch rows.
- Scalar `_window_global_start_token` metadata expands to batch size before
  indexing.
- The commit body-aux reduction is a step mean over valid committed rollout
  steps and does not scale linearly with K.
- Enabling the new mode replaces final-step body aux rather than adding a second
  body-aux loss.
- Replacement values are sourced from detached `x0_latent_list`, while root loss
  tokens are sourced from `pred_x0_latent_list`.

## Experiment Plan

1. K=3, `model.params.traj_dropout=0.0`, single decode, commit-token 7D body aux.
2. K=3, per-step decode comparison.
3. K=5, `model.params.traj_dropout=0.0`, single decode.
4. Consider full BPTT or no-detach only after the diagnostic results are clear.
