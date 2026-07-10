# StreamGenerator Execution Ownership Design

## Goal

Make `StreamGenerator` the atomic owner of one streaming commit: condition
construction, LDF generation, VAE decoding, optional root feedback, motion
recovery, generated root history, and `RootTimeline` advancement. Keep the
existing web execution path available during migration.

## Commit-Boundary Contract

Before generating commit `c`, the latest observable generated frame is the end
of the committed prefix:

```python
commit_boundary_frame(0) = 0
commit_boundary_frame(c) = token_end_frame(c - 1)  # c > 0
```

`RootSourceProposal` frame zero represents the current actor anchor and maps to
`commit_boundary_frame(anchor_commit)`. The next route frame is future. Active
window composition reads the current generated state at the boundary frame; it
must never read `token_start_frame(c)`, which has not yet been generated.

After a successful step, the decoded chunk's final recovered root advances the
timeline from commit `c` to commit `c + committed_tokens`. The timeline head is
therefore exact before the next step begins.

## Owned Dependencies and State

`StreamGenerator` accepts optional execution dependencies:

- `vae`
- `motion_recovery: StreamJointRecovery263`
- root feedback configuration

Legacy construction remains valid. `execute_step()` requires execution
dependencies and raises a clear error when they are absent. `step()` remains as
the legacy LDF-only coordinator during migration.

Owned mutable execution state:

- `first_chunk`
- `generated_frame_count`
- world-frame generated root 5D/7D history
- session anchor state
- `RootTimeline`
- VAE and motion-recovery stream state

## StreamCommitEvent

Each successful execution returns an immutable event containing:

- local and absolute commit indices before execution
- absolute commit index after execution
- committed latent token
- decoded 263D motion chunk
- recovered joint frames
- generated world root trajectory through the committed chunk
- new `RootFrameState`
- actual trajectory payload consumed by LDF
- root feedback applied flag and diagnostics

The web layer consumes joint frames and diagnostics. It does not independently
decode, re-encode, recover roots, or advance the timeline.

## Execution Order

`execute_step()` performs:

1. Snapshot LDF, VAE, recovery, and owned runtime state.
2. Read exact current commit and boundary frame.
3. Build root-source/RootPlan payload using owned generated history.
4. Build the LDF condition provider and generate one latent token.
5. Decode the token; if enabled, apply root feedback, re-encode, write back, and
   formally decode the corrected token.
6. Process every decoded frame through the owned recovery object.
7. Append joints and world root frames, then advance the timeline once.
8. Commit `first_chunk` and counters and return `StreamCommitEvent`.

On any exception, restore all snapshots and re-raise. No partial timeline,
history, VAE cache, recovery state, or LDF state may remain.

## Root Feedback

The first owned implementation preserves existing semantics:

- optional enable flag
- XZ blend alpha in `[0, 1]`
- target derived from the actual LDF trajectory payload
- temporary raw decode preserves formal decoder cache
- corrected latent is written to the committed LDF token

The helper implementation moves out of `eval` ownership into inference/runtime
code so web runtime does not depend on eval modules.

## Web Migration

Add `runtime.use_owned_stream_execution`, defaulting to `false` initially.

- `false`: existing web generation loop remains unchanged.
- `true`: `ModelManager` calls `StreamGenerator.execute_step()` and only pushes
  `event.joint_frames` to the frame buffer.

Reset configures/clears the owned VAE, recovery, history, and timeline. Once
fixed-seed parity and runtime generation are verified, the flag can become the
default and the legacy execution block can be removed separately.

## Tests

- Commit boundary mapping for commit zero and later commits.
- RootSourceProposal anchor origin uses the committed-prefix boundary.
- Active-window payload uses an existing generated history frame.
- One execution advances exactly one commit and returns coherent event fields.
- Root feedback disabled/enabled paths preserve expected decode/writeback order.
- Failures in LDF, VAE, feedback, and recovery restore all mutable state.
- Web legacy path remains available and owned path only writes event joints to
  the UI buffer.
