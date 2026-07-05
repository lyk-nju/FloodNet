# Streaming Best-of-K State Selector Design

## Goal

Implement a minimal, eval/runtime-only streaming best-of-K selector for LDF
stream generation. The selector should test whether sampling multiple legal next
stream states under the same text/trajectory condition and conservatively
choosing one can improve root XZ tracking without destabilizing rollout history.

This is not a training change and must not alter checkpoint structure.

## Non-Goals

- No H>1 rollout or MPC.
- No root replacement, root correction, or re-encoding inside best-of-K.
- No controller-aware or semantic verifier.
- No batch K optimization in the first version.
- No changes to training loss or training data flow.

## Core Requirements

1. `best_of_k <= 1` must completely bypass best-of-K code and run the original
   `stream_generate_step` path unchanged.
2. For `best_of_k > 1`, candidate 0 is the exact legal next state produced by
   the original K=1 runtime from the current state. Candidates 1..K-1 are extra
   proposal states generated from the same base state with different sampling
   noise/RNG.
3. Selection operates on complete candidate runtime states, not on latent tensors.
4. The selector must score only the clean ready-to-commit decoded chunk for the
   current step, never a mixed-noise active window.
5. The first selector is conservative: keep candidate 0 unless another candidate
   has a clear tracking improvement and does not worsen continuity beyond a
   tolerance.
6. Candidate rollout must be transactional. Candidate generation must not
   permanently mutate global eval return buffers unless those buffers are part
   of the candidate snapshot and are restored with the selected state.
7. Candidate 0 must use the unmodified base RNG state. Candidates 1..K-1 may use
   perturbed RNG states. If candidate 0 is selected, the post-step RNG state must
   match the original K=1 path.

## Proposed File Structure

- `eval/ldf/stream_generation.py`
  - Keep the public `run_stream_generate_step_sample` entrypoint.
  - Preserve the existing K=1 path exactly.
  - Delegate only `best_of_k > 1` to the new selector.

- `eval/ldf/stream_state.py`
  - Define `StreamRuntimeSnapshot`, `CandidateState`, and `StepOutput`.
  - Capture/restore model state, VAE stream cache, conditioning state, decoded
    chunk history, frame/token counters, and RNG state.

- `eval/ldf/stream_scoring.py`
  - Compute root XZ tracking metrics for the current committed chunk.
  - Compute continuity metrics between previous selected chunk and candidate
    chunk.
  - Implement conservative switch gate.

- `eval/ldf/stream_best_of_k.py`
  - Implement serial state-fork candidate generation.
  - Build candidate 0 from the original runtime step.
  - Build extra candidates from restored base state with varied RNG/noise.
  - Restore the selected candidate's complete post-step state.

- `tests/test_stream_best_of_k.py`
  - Unit tests for bypass, state restore, candidate 0 semantics, conservative
    gate, and debug records.

## State Snapshot Contract

`StreamRuntimeSnapshot` must cover at least:

- `model.generated`
- `model.commit_index`
- `model.current_step`
- `model.batch_size`
- `model.seq_len`
- `model.num_denoise_steps`
- `model.dt`
- `model.text_condition_list`
- model trajectory buffers such as `_traj_buf`
- VAE stream decode cache: `_conv_num`, `_conv_idx`, `_feat_map`
- stream conditioner timeline/root plan state when present
- decoded chunk list and latent token list used by eval
- `first_chunk`
- `generated_frames`
- `chunk_frame_ends`
- Python, NumPy, CPU torch, and CUDA RNG states

All tensor states must be deep-cloned while preserving device and dtype.
Restoring a snapshot must not share mutable tensor storage with later candidate
rollouts unless the tensor is explicitly immutable.

The selected candidate is committed by restoring its complete post-step snapshot,
not by manually writing selected latent values back into the baseline state.

## Runtime State vs Eval Return Buffers

Eval return buffers and runtime state must have one clear owner.

There are two valid implementation choices:

1. Include eval buffers such as latent tokens, decoded chunks, and chunk frame
   ends in `StreamRuntimeSnapshot`. In this case restoring the selected
   candidate's post-step snapshot is the commit, and the outer loop must not
   append the selected chunk a second time.
2. Keep eval buffers outside the snapshot. In this case candidate generation
   must write only to local candidate outputs, never to global eval buffers.
   After the selector chooses a candidate, the outer loop appends exactly once.

The first implementation should choose one of these strategies explicitly and
test that selected chunks are not double-appended.

## Step Output Contract

The one-step runtime wrapper should return a `StepOutput` with:

- `clean_committed_latent`
- `decoded_chunk`
- `commit_token_range`
- `commit_frame_range`
- `decoded_chunk_frame_range`
- `target_xz_frame_range`
- active-window debug fields available from the model

If `stream_generate_step` can expose commit ranges explicitly, use those values.
Do not infer `commit_frame_range` only from `generated_frames` or chunk-list
length when an explicit commit range is available. If the current model API
does not expose it, inference is acceptable for v1 but must be isolated in the
wrapper and logged in debug records.

## Candidate Generation Flow

For every stream step with `K > 1`:

1. Capture `base_snapshot`.
2. Candidate 0:
   - Restore `base_snapshot`.
   - Use the base RNG state without modification.
   - Run exactly the existing one-step path:
     `build_ldf_condition_provider -> model.stream_generate_step -> vae.stream_decode`.
   - Record output chunk, latent token, commit ranges, score inputs, and full
     post-step snapshot.
3. Candidates 1..K-1:
   - Restore `base_snapshot`.
   - Perturb RNG/sampling state so the proposal differs from candidate 0.
   - Run the same one-step path.
   - Record the same fields and full post-step snapshot.
4. Score candidates.
5. Apply conservative gate.
6. Restore selected candidate post-step snapshot.
7. Append selected candidate's latent/chunk into eval return buffers only if
   those buffers are not part of the restored snapshot.

The initial reference implementation is serial by design. Batch K can be added
later only after the serial selector is validated.

## Scoring and Gate

For each candidate:

```text
track_i = xz_ade_i + fde_weight * xz_fde_i
cont_i = pos_cont_i + vel_weight * vel_cont_i
```

Candidate 0 is the default. A candidate may replace it only if:

```text
improve_enough = track_i < track_0 - max(abs_margin, rel_margin * track_0)
continuity_ok = cont_i <= cont_0 + cont_tol
```

Initial values:

- `fde_weight = 1.0`
- `vel_weight = 0.5`
- `rel_margin = 0.10`
- `abs_margin = 0.03`
- `cont_tol = 0.03`

These margins assume root XZ is in meters. If debug logs show a different scale,
adjust margins before interpreting results.

`cont_weight` is not used by the conservative gate in v1. It is accepted only
for backward compatibility with the existing config surface and should not
affect selection unless a future explicit weighted-argmin mode is added.

## Debug Records

When debug is enabled, record per step:

- `best_of_k`
- `commit_index`
- `local_commit_index`
- committed token range
- committed frame range
- decoded chunk frame range
- target XZ frame range
- candidate 0 score parts
- all candidate score parts
- selected candidate index
- switch decision and reason
- active window debug fields available from the current model state, including
  `current_step`, `num_denoise_steps`, and inferred ready-to-commit token
- candidate diversity check

The debug contract should make it obvious whether selected candidates are
scored on the same committed frame range as candidate 0.

## Config

Keep existing defaults and add only conservative selector fields:

- `eval_stream_best_of_k: 1`
- `eval_stream_best_of_k_score: "xz"`
- `eval_stream_best_of_k_xz_weight: 1.0`
- `eval_stream_best_of_k_fde_weight: 1.0`
- `eval_stream_best_of_k_cont_weight: 0.0`
- `eval_stream_best_of_k_vel_weight: 0.5`
- `eval_stream_best_of_k_rel_margin: 0.10`
- `eval_stream_best_of_k_abs_margin: 0.03`
- `eval_stream_best_of_k_cont_tol: 0.03`
- `eval_stream_best_of_k_force_candidate0: false`
- `eval_stream_best_of_k_switch_cooldown_steps: 0`
- `eval_stream_best_of_k_debug: false`

`cont_weight` remains accepted for compatibility, but the conservative gate uses
explicit continuity fields rather than folding continuity into a pure argmin.
`force_candidate0` is a diagnostic mode: it still forks candidates but always
restores candidate 0 and should match K=1 metric-level behavior. Cooldown is
reserved for later anti-chatter behavior; v1 defaults it to zero.

## Validation Plan

Minimum validation on sample `000021`:

1. K=1 original baseline.
2. If a forced new-code K=1 path exists for tests, verify metric-level
   equality with original K=1. Production K=1 still bypasses best-of-K.
3. K=5 with `force_candidate0=true`; this forks candidates but always restores
   candidate 0 and should match K=1 metric-level behavior.
4. K=5 serial state-fork selector.

Success criteria:

- K=1 baseline remains unchanged.
- K=5 force-candidate0 matches K=1 metric-level behavior.
- K=5 ADE is not clearly worse than K=1.
- Selected commit frame ranges match candidate 0 for every step.
- Switch rate is much lower than the previous 37/46 greedy behavior.
- Every switch record shows a clear tracking improvement and acceptable
  continuity.

## Risks

- Full snapshotting may miss a hidden cache. Tests should use fake stateful
  components to catch obvious omissions, and debug records should expose state
  drift.
- Serial K is slower. This is acceptable for the reference implementation.
- Candidate proposals may be correlated if RNG perturbation is incomplete. Debug
  diversity records are required.
- Conservative gate may switch rarely. That is preferable for the first version;
  correctness comes before tracking gains.
