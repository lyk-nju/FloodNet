# Stream Runtime Session Design

## Goal

Replace the two independently stateful Web streaming paths with one authoritative
runtime session. Web and compatibility APIs submit commands and consume events;
they do not own LDF commit state, VAE caches, motion recovery, generated root
history, or the root timeline.

The design must make token/frame timing, route activation, route validity, and
world-versus-relative route behavior explicit and independently testable.

## Why the Current Problems Exist

The current implementation grew through incremental experiments with different
owners for related state:

- The legacy Web loop advances the LDF model immediately after generation, but
  advances `RootTimeline` while iterating decoded token-start frames. The model,
  frame recovery, and commit timeline therefore describe different moments.
- The newer owned execution path correctly commits after a decoded chunk, but it
  is opt-in while legacy remains the default. Two execution paths still maintain
  the same conceptual state with different rules.
- `RootSourceProposal` contains route values but no complete validity/lifecycle
  contract. Prefix padding, generated history, valid route frames, and terminal
  padding can become indistinguishable to payload construction.
- HTTP and worker threads can both change trajectory state. A route's recorded
  origin may therefore differ from the commit boundary where the worker first
  consumes it.
- The name `active_window` currently hides two different spatial policies:
  returning to an authored world route and preserving a route shape relative to
  the current actor.
- Route tracking, composition, payload canonicalization, execution, and Web UI
  adaptation are coupled through mutable objects rather than narrow contracts.

The root cause is not one arithmetic error. It is distributed ownership of a
single streaming transaction combined with implicit time and space semantics.

## Architectural Decision

`StreamRuntimeSession` becomes the only authoritative execution and state owner.
The existing legacy path remains temporarily as a compatibility wrapper, but it
must delegate to the session and must not independently decode, recover motion,
write timeline states, or manage VAE caches.

### StreamRuntimeSession

Owns:

- the commit transaction;
- `RootTimeline`;
- frame-level `GeneratedRootHistory`;
- VAE encoder and decoder streaming caches;
- motion recovery state;
- root feedback execution;
- `RootSourceManager`;
- the current session anchor;
- rollback snapshots.

`step()` is the only operation that may advance generated model state. It
activates due commands, composes the actual condition, generates and decodes one
token, recovers all frames, advances the timeline once, and returns an immutable
`StreamCommitEvent`.

### StreamGenerator

Becomes the LDF token-generation kernel. It owns text and trajectory condition
provider construction and the call to `stream_generate_step()`. It does not own
the authoritative root timeline, frame history, VAE cache, recovery, or Web
state.

During migration, the existing `execute_step()` API delegates to a
`StreamRuntimeSession`. Once all callers use the session directly, compatibility
methods may be removed separately.

### RootSourceManager

Owns:

- pending source commands;
- command versions;
- requested and actual activation commits;
- the active immutable proposal;
- route progress state;
- route lifecycle (`pending`, `active`, `exhausted`, `replaced`, `cleared`).

Only the generation worker may call `activate_due()`. The manager does not
depend on LDF, VAE, recovery, Web UI, or payload canonicalization.

### ConditionComposer

A stateless component that combines:

- generated history;
- the unique actor anchor frame;
- a bridge;
- valid route future;
- terminal numerical padding.

It returns a world-frame condition, a frame-valid mask, segment labels,
route status, and diagnostics. It never mutates runtime state.

### PayloadBuilder

Consumes a composed world condition and builds the exact model payload. It owns:

- active-window slicing;
- canonicalization against the active history window's first-frame state;
- substep payload construction;
- final 7D delta recomputation.

It does not track route progress or modify runtime state.

### RootSourceProposal

An immutable route-source DTO with at least:

```python
proposal_traj7: Tensor
proposal_frame_mask: Tensor
space_contract: Literal["world_route", "relative_route"]
source_id: str
version: int
metadata: dict
```

Sample-derived, synthetic, and RootRefiner sources all produce this same DTO.
They do not directly produce an LDF payload.

### Web and Compatibility Layers

Web/API code may:

- submit versioned commands;
- consume `StreamCommitEvent` frames and diagnostics;
- react to route lifecycle events by submitting another command.

It may not mutate the active source, timeline, VAE cache, recovery, generated
history, or model commit state. Legacy APIs translate their inputs to commands
and delegate execution to the same session.

## Time Contract

The authoritative commit meaning is:

```text
timeline.at_commit(N)
= root state after N latent tokens have completed
= the last frame of token N - 1
```

For the default causal VAE mapping:

- after token 0: timeline head is commit 1 and describes frame 0;
- after token 1: timeline head is commit 2 and describes frame 4.

Frame-level generated root states live in `GeneratedRootHistory`, not in
`RootTimeline`. All conversions between commits, generated boundary frames, and
token frame ranges use the shared token/frame mapping module.

## Proposal Anchor Contract

Every activated proposal is normalized so that:

- proposal frame 0 is the actor state at the activation commit boundary;
- proposal frame 1 is the first future route frame;
- generated history's last frame and proposal frame 0 refer to the same
  absolute frame;
- the anchor appears exactly once in the composed condition;
- sources that provide only future points receive an anchor during activation.

The recorded activation commit is the commit where the worker actually
activates the source, not the commit observed earlier by an HTTP thread.

## Space Contracts

### world_route

The composer bridges from the current generated root to a lookahead point on
the authored world route, then continues along that original world route.
Generated drift is progressively corrected. This contract supports fixed world
targets and comparable absolute-route metrics.

### relative_route

The composer translates and rotates the remaining route shape to the current
actor pose. It does not force the actor back to the authored world coordinates.
This contract supports interactive continuation and intentionally preserves
actor-relative drift.

The generic name `active_window` is not a space contract and must not select
either behavior implicitly.

## Validity and Route End

Proposal validity is explicit:

- frame 0 anchor: mask true;
- valid future route frames: mask true;
- generated history supplied by the session: mask true;
- numerical padding before/after valid route content: mask false.

At route exhaustion:

- values hold the last valid pose to keep tensors finite and continuous;
- masks are false;
- the session emits `route_exhausted` once for the active source version;
- the runtime does not implicitly loop, refresh RootRefiner, or repeat a route.

The upper layer may submit a new source, regenerate through RootRefiner, repeat
the previous source as a new command, clear control, or do nothing.

## Command Activation and Concurrency

Web/API updates create versioned commands. They never modify the active source.

At the beginning of `step()` for commit `N`, the worker atomically consumes due
commands:

- the highest version wins among all replace commands due by boundary `N`;
- an expired requested commit is promoted to `N`;
- the proposal is normalized against the real commit-`N` actor state;
- activation completes before condition composition starts;
- no source can change during the remainder of the transaction.

Commands arriving during generation of token `N` can first activate at the next
step boundary. Their recorded origin must reflect that actual boundary.

## Step Transaction

One `StreamRuntimeSession.step()` performs:

1. Snapshot all mutable session, model, VAE, recovery, source-manager, and route
   tracker state.
2. Read the exact timeline head commit `N`.
3. Activate the latest due source command at boundary `N`.
4. Compose generated history, anchor, bridge, route future, mask, and padding.
5. Build and record the exact active-window model payload.
6. Generate token `N` with the LDF kernel.
7. Decode with VAE and optionally apply/re-encode root feedback.
8. Process every decoded frame through motion recovery.
9. Append all recovered frame roots to `GeneratedRootHistory`.
10. Append one timeline state at commit `N + 1` using the chunk's final frame.
11. Emit an immutable `StreamCommitEvent`.

If any stage fails, all snapshots are restored. No partial model token, VAE
cache, recovery state, frame history, timeline commit, source activation, route
progress, UI frame, or lifecycle event may survive.

## Event Contract

`StreamCommitEvent` includes:

- local and absolute commit indices before and after execution;
- committed latent token;
- decoded 263D chunk;
- recovered joint frames;
- newly committed root frames and timeline state;
- exact payload consumed by the model;
- active source id/version and actual activation commit;
- route status and lifecycle events;
- root-feedback diagnostics.

The Web layer buffers `joint_frames` and presents diagnostics. It does not
reconstruct execution state from the event.

## Testing Strategy

### Time Mapping

- token 0 commits frame 0 as timeline commit 1;
- token 1 commits frame 4 as timeline commit 2;
- arbitrary commit boundaries agree with frame-history length.

### Pure Composition

- `world_route` bridges back to the authored route;
- `relative_route` preserves actor-relative route shape;
- segment labels and masks identify history, anchor, bridge, route, and padding;
- exact-end and past-end proposals never index out of bounds;
- final 7D deltas are recomputed from final 5D values.

### Command Concurrency

- commands submitted during a step activate only at the next boundary;
- highest-version replace wins;
- requested versus actual activation commits are recorded explicitly;
- proposal frame 0 equals the actual activation root state.

### Atomic Execution

Inject failures into LDF generation, VAE decode, root-feedback encode, recovery,
composition, and source activation. Each must restore every owned state and
produce no event or UI frame.

### Web Parity

With fixed seeds, compatibility and direct session entry points must produce the
same payloads, timeline states, generated root history, and joint frames. After
parity is verified, the duplicate legacy generation loop is removed and the
session becomes the default Web path.

## Migration Sequence

1. Add immutable proposal/command/event DTOs, `RootSourceManager`, and pure
   composer tests without changing Web execution.
2. Add `StreamRuntimeSession`, initially reusing existing decode, feedback,
   recovery, and snapshot helpers.
3. Move authoritative timeline and generated frame history into the session.
4. Route Web updates through command submission and Web generation through
   session events.
5. Convert legacy entry points into wrappers around the same session.
6. Run fixed-seed parity and real Web runtime tests.
7. Remove duplicate legacy execution and enable session execution by default.

## Non-Goals

- Automatically deciding what upper layers do after route exhaustion.
- Persisting a general event-sourcing database.
- Changing RootRefiner model architecture or training.
- Replacing the existing causal VAE streaming cache with full-prefix decoding.
- Refactoring unrelated noise-initializer or evaluation code.

## Acceptance Criteria

- Exactly one component owns model commit, VAE/recovery state, generated frame
  history, and `RootTimeline`.
- Web/API threads cannot mutate active execution state.
- Timeline commit states always describe completed token boundaries.
- Every source explicitly declares `world_route` or `relative_route`.
- Proposal validity and route exhaustion are represented by masks and status,
  never inferred from repeated endpoint values.
- Exact-end and past-end route consumption cannot throw indexing errors.
- The actual payload consumed by LDF is present in the corresponding event.
- All failure stages are transactionally reversible.
- Legacy compatibility delegates to the same session and contains no duplicate
  execution state machine.
