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
- the frozen per-step runtime configuration;
- `RuntimeCommandQueue` integration;
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

It receives an already-built frame-level trajectory payload and may encode that
payload into `traj_emb`/`LDFCondition`. It may not choose absolute frame slices,
history anchors, horizon extent, route validity, or route masks.

During migration, the existing `execute_step()` API delegates to a
`StreamRuntimeSession`. Once all callers use the session directly, compatibility
methods may be removed separately.

### RootSourceManager

Owns:

- the active immutable proposal;
- its `ActivatedRootSource` metadata;
- committed immutable route progress state;
- route lifecycle (`active`, `exhausted`, `replaced`, `cleared`). Command
  `pending` state belongs to `RuntimeCommandQueue`.

Only the generation worker may commit a prepared source transition. The manager
does not mutate progress during composition and does not depend on LDF, VAE,
recovery, Web UI, or payload canonicalization.

### RuntimeCommandQueue

Owns every versioned command that can affect token generation, including:

- `SetRootSource` and `ClearRootSource`;
- `SetText`;
- `SetGuidance`;
- `SetRootFeedback`;
- `SetRuntimeControls`;
- `ResetSession`.

Commands share one global monotonic sequence, independent of command type. The
worker prepares due commands at a step boundary and acknowledges them only after
the step commits. Commands submitted after prepare remain in the queue and are
never removed by rollback.

### ConditionComposer

A stateless component that combines:

- generated history;
- the activation boundary state as a geometric input;
- a bridge;
- valid route future;
- terminal numerical padding.

It returns a world-frame condition, a frame-valid mask, segment labels,
route status, diagnostics, and a proposed next route progress value. It never
mutates runtime state. The session commits proposed progress only after the full
token transaction succeeds.

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
future_traj7: Tensor
future_frame_mask: Tensor
source_id: str
version: int
metadata: dict
```

Sample-derived, synthetic, and RootRefiner sources all produce this same DTO.
`future_traj7[0]` is the first real future route frame, never an activation
anchor. Proposals do not directly produce an LDF payload and can be activated
under either supported space contract.

Source adapters convert model-specific/local outputs into physical authored
world coordinates before constructing the DTO. `future_frame_mask` is a boolean
vector with exactly the same frame length. Adapters that receive an output with
an explicit current-state anchor remove that anchor from `future_traj7`; runtime
validation rejects ambiguous or mismatched shapes.

### RootSourceCommand and ActivatedRootSource

`RootSourceCommand` carries the proposal (or clear intent), global command
version, requested activation commit, and explicit `world_route` or
`relative_route` contract.

At a real worker boundary it becomes an immutable `ActivatedRootSource` with:

- requested and actual activation commits;
- the boundary `RootFrameState` stored separately from route frames;
- `first_future_frame_abs`;
- the selected space contract;
- committed `RouteProgressState`.

```python
@dataclass(frozen=True)
class RootSourceCommand:
    proposal: RootSourceProposal | None
    command_version: int
    requested_activation_commit: int
    space_contract: Literal["world_route", "relative_route"] | None
    kind: Literal["replace", "clear"]

@dataclass(frozen=True)
class ActivatedRootSource:
    proposal: RootSourceProposal
    requested_activation_commit: int
    actual_activation_commit: int
    boundary_state: RootFrameState
    first_future_frame_abs: int
    space_contract: Literal["world_route", "relative_route"]
    progress: RouteProgressState
```

No caller computes route-local indices by subtracting an anchor frame from an
absolute frame. Dedicated activated-source APIs perform checked conversions.

### Progress Policies

Progress is immutable transaction state. Pure policies implement:

```python
project(route, actor_state, previous_progress) -> RouteProjection
```

`WorldRouteProgressPolicy` performs heading-aware monotonic projection and
lookahead on authored world-route arc length. `RelativeRouteProgressPolicy`
advances proposal-local phase/arc length and does not project the actor onto the
untransformed authored world coordinates.

Composition returns, but does not commit, progress:

```python
@dataclass(frozen=True)
class ComposeResult:
    frame_start_abs: int
    world_condition_7d: Tensor
    frame_mask: Tensor
    segment_labels: Tensor
    proposed_route_progress: RouteProgressState
    route_status: str
    diagnostics: dict
```

### GeneratedRootHistory

Frame history is a bounded absolute-indexed container:

```python
base_frame_abs: int
frames_7d: Tensor

slice_abs(start_frame_abs, end_frame_abs)
append(...)
trim_before(frame_abs)
```

Consumers never assume `frames_7d[i]` is absolute frame `i`. Trimming preserves
the absolute base. `RootTimeline` similarly exposes its earliest and head
absolute commit indices.

### Web and Compatibility Layers

Web/API code may:

- submit versioned commands;
- consume `RuntimeEvent` frames and diagnostics;
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

The mapping exposes two distinct concepts:

```text
first_future_frame_abs(N) = number of frames generated by N committed tokens
last_generated_frame_abs(N) = first_future_frame_abs(N) - 1, only when N > 0
```

At commit 10, the last generated frame is 36 and the first future frame is 37.
At commit 0, generated history is empty, there is an initial boundary state,
and the first future frame is frame 0. The initial state is not motion frame 0.

Rolling latent buffers do not change absolute runtime time. The session records:

- `latent_buffer_start_commit_abs`;
- `latent_buffer_epoch`;
- local commit before/after;
- absolute commit before/after.

Runtime modules outside the LDF kernel use absolute commits. Every successful
one-token step enforces:

```text
absolute_commit_after = absolute_commit_before + 1
timeline.head.commit_idx = absolute_commit_after
```

## Activation Boundary Contract

The activation boundary is state, not a route frame. An activated source stores
the boundary `RootFrameState` separately, while `future_traj7[0]` is the first
real future route frame at `first_future_frame_abs`.

For commits after cold start, generated history already ends at the boundary
root frame; the composer does not append a duplicate anchor. At commit 0,
generated history is empty and the initial boundary state is a virtual geometric
start, not frame 0. A bridge may use it as its `t=0` endpoint, but emitted future
samples exclude that endpoint so frame 0 remains the first generated motion
frame.

The recorded activation commit is the commit where the worker actually
activates the source, not the commit observed earlier by an HTTP thread. Checked
mapping APIs expose boundary state, first future absolute frame, and future-local
indices without implicit `+1` or `-1` arithmetic in callers.

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

- valid future route frames: mask true;
- generated history supplied by the session: mask true;
- numerical padding before/after valid route content: mask false.

The activation boundary state has no route-frame mask because it is not stored
inside the proposal. If it already exists in generated history, that history
frame is valid. At cold start it is only a virtual bridge endpoint.

At route exhaustion:

- values hold the last valid pose to keep tensors finite and continuous;
- masks are false;
- the session emits `route_exhausted` once for the active source version;
- the runtime does not implicitly loop, refresh RootRefiner, or repeat a route.

The upper layer may submit a new source, regenerate through RootRefiner, repeat
the previous source as a new command, clear control, or do nothing.

## Command Activation and Concurrency

Web/API updates create globally versioned `RuntimeCommand` values. They never
modify active execution state directly. Every setting that can change one token's
result is frozen at a worker boundary: route source/clear, text, text/traj CFG,
root-feedback policy, history/horizon/denoise controls, and reset intent.

Pause is worker scheduling rather than model state and takes effect only between
steps. Reset requires an exclusive quiescent boundary and starts a new session
epoch; it cannot partially apply during a token transaction. When a reset is the
final prepared command, the worker performs an exclusive reset transaction,
emits `SessionResetEvent`, and does not generate a motion token in that call.

At the beginning of `step()` for commit `N`, the worker prepares all due
commands:

- commands are ordered by one global monotonic version across all kinds;
- commands are reduced in that order into a proposed per-step configuration;
  the last write to each state field wins;
- route replace/clear ordering is therefore unambiguous (`replace A`, then
  `clear`, then `replace B` results in B);
- superseded writes to the same field produce diagnostic
  `command_superseded` records;
- an expired requested commit is promoted to `N`;
- a route proposal is activated against the real commit-`N` actor state;
- the resulting text/guidance/feedback/runtime configuration is immutable for
  the remainder of the transaction.

Preparation does not destructively drain the concurrent queue. A successful
session commit acknowledges the prepared command versions. On rollback they
remain due for retry. Commands arriving after prepare remain pending for the next
boundary and cannot be lost by restoring a transaction snapshot.

Commands arriving during generation of token `N` can first activate at the next
step boundary. Their recorded origin must reflect that actual boundary.

## Step Transaction

One `StreamRuntimeSession.step()` uses three phases.

### Prepare Phase

1. Read the exact timeline head commit `N` and absolute buffer metadata.
2. Prepare, but do not acknowledge, all due runtime commands.
3. Derive a proposed active-source transition and immutable per-step config.
4. Run the pure progress policy and composer using committed previous progress.
5. Build and retain the exact model payload and proposed next progress.

No authoritative runtime state changes during prepare.
Prepare is deterministic and does not consume RNG. Any future stochastic
operation belongs in the snapshotted mutating phase.

### Mutating Phase

6. Snapshot only state that generation can mutate.
7. Generate the raw latent token with the LDF kernel.
8. Decode and apply root feedback in the strict order below.
9. Process only the final decoded chunk through motion recovery.

Root-feedback order is:

1. generate raw latent;
2. preview-decode raw motion without advancing the formal decoder cache;
3. replace/blend the root in decoded motion;
4. stream-encode the corrected chunk;
5. write the corrected committed latent back to model history;
6. formally stream-decode the corrected latent exactly once;
7. pass only corrected decoded motion to recovery.

Preview decode uses a cache snapshot/restore or an explicit preview cache. Raw
and corrected decode must never advance the same formal decoder cache twice.

### Commit Phase

10. Append recovered frame roots to `GeneratedRootHistory`.
11. Append one timeline state at commit `N + 1` using the chunk's final frame.
12. Commit proposed source transition and route progress.
13. Acknowledge prepared commands and route-exhaustion emission state.
14. Publish an immutable `StreamCommitEvent` to external sinks.

Mutable snapshots include, as applicable:

- LDF generated buffer, local commit/current step, text-condition history,
  rolling-buffer absolute offset/epoch, token update counters, trajectory buffer,
  and other formal stream state;
- torch CPU RNG, RNG for every CUDA device used by the step, and Python/NumPy
  RNG when code inside the transaction consumes them;
- VAE decoder and encoder cache counters and feature maps;
- recovery accumulators, smoothing, and previous-frame state;
- session timeline/history, active source, committed progress, lifecycle flags,
  and session anchor.

Model and VAE components should expose `snapshot_stream_state()` and
`restore_stream_state()` so the session does not permanently depend on private
field names. Compatibility adapters may enumerate current fields during
migration.

If any stage fails, mutable snapshots are restored, prepared commands remain
unacknowledged, and no commit-phase external effects run. No partial model token,
VAE cache, recovery state, frame history, timeline commit, source activation,
route progress, UI frame, metric, diagnostic file, or lifecycle event may
survive.

## Event Contract

```python
RuntimeEvent = StreamCommitEvent | SessionResetEvent
```

`StreamCommitEvent` includes:

- local and absolute commit indices before and after execution;
- latent-buffer absolute start commit and epoch;
- committed latent token;
- decoded 263D chunk;
- recovered joint frames;
- newly committed root frames and timeline state;
- exact payload consumed by the model;
- active source id/version and actual activation commit;
- route status and lifecycle events;
- root-feedback diagnostics.

At minimum its reproducibility fields are:

```python
@dataclass(frozen=True)
class StreamCommitEvent:
    absolute_commit_before: int
    absolute_commit_after: int
    local_commit_before: int
    local_commit_after: int
    latent_buffer_start_commit_abs: int
    latent_buffer_epoch: int
    committed_latent: Tensor
    decoded_chunk: Tensor
    joint_frames: Tensor
    root_frames_start_abs: int
    root_frames: Tensor
    timeline_state: RootFrameState
    actual_payload: dict | None
    source_id: str | None
    source_version: int | None
    actual_activation_commit: int | None
    lifecycle_events: tuple

@dataclass(frozen=True)
class SessionResetEvent:
    previous_session_epoch: int
    session_epoch: int
    applied_command_version: int
```

All tensor/array payloads are detached clones, moved to CPU where appropriate,
so frozen dataclasses cannot expose mutable aliases to runtime state. The session
publishes events but does not retain an unbounded history of full payloads;
diagnostic recorders explicitly choose what to persist.

The Web layer buffers `joint_frames` and presents diagnostics. It does not
reconstruct execution state from the event. UI buffers, metrics, and diagnostic
files are written only after transaction commit.

## Testing Strategy

### Time Mapping

- cold start has no generated motion frame and retains a separate initial
  boundary state;
- token 0 commits frame 0 as timeline commit 1;
- token 1 commits frame 4 as timeline commit 2;
- arbitrary commit boundaries agree with frame-history length.
- local commit rollback in a rolling buffer does not change monotonically
  increasing absolute commits or payload absolute frame ranges.

### Pure Composition

- `world_route` bridges back to the authored route;
- `relative_route` preserves actor-relative route shape;
- segment labels and masks identify history, bridge, route, and padding;
- exact-end and past-end proposals never index out of bounds;
- final 7D deltas are recomputed from final 5D values.
- generated boundary, virtual activation state, and first future frame produce no
  duplicate zero-speed anchor seam.
- the same proposal and generated drift produce intentionally different,
  contract-correct `world_route` and `relative_route` results.

### Command Concurrency

- commands submitted during a step activate only at the next boundary;
- global version ordering resolves replace/clear/replace deterministically and
  reports superseded commands;
- requested versus actual activation commits are recorded explicitly;
- `ActivatedRootSource.boundary_state` equals the actual activation root state;
- text submitted during token `N` affects token `N + 1`, not token `N`.

### Atomic Execution

Inject failures into LDF generation, VAE decode, root-feedback encode, recovery,
composition, and source activation. Each must restore every owned state and
produce no event or UI frame.

An RNG rollback test compares a failed-then-retried session with a clean session
from identical initial RNG states. Their committed latent and decoded output must
be bitwise equal.

Route exhaustion tests verify exact-end safety, one-shot `route_exhausted`,
hold-last numerical values, and false terminal masks.

### Web Parity

With fixed seeds, compatibility and direct session entry points must produce the
same payloads, timeline states, generated root history, and joint frames. After
parity is verified, the duplicate legacy generation loop is removed and the
session becomes the default Web path.

## Migration Sequence

1. Add immutable proposal/activated-source/command/event DTOs, absolute-indexed
   history, progress policies, and pure composer tests without changing Web
   execution.
2. Add the global `RuntimeCommandQueue` and transactional `RootSourceManager`.
3. Add formal model/VAE stream snapshot interfaces and compatibility adapters.
4. Add `StreamRuntimeSession`, initially reusing existing decode, feedback, and
   recovery helpers.
5. Move authoritative timeline and generated frame history into the session.
6. Route all generation-affecting Web updates through command submission and Web
   generation through session events.
7. Convert legacy entry points into wrappers around the same session.
8. Run fixed-seed parity, rolling-buffer, concurrency, and real Web runtime tests.
9. Remove duplicate legacy execution and enable session execution by default.

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
- All token-affecting settings are frozen from globally ordered commands at a
  worker boundary.
- Timeline commit states always describe completed token boundaries.
- Activation boundary state is never stored as a normal future route frame.
- Every activated source explicitly declares `world_route` or `relative_route`.
- Proposal validity and route exhaustion are represented by masks and status,
  never inferred from repeated endpoint values.
- Exact-end and past-end route consumption cannot throw indexing errors.
- World and relative routes use separate progress policies.
- Rolling-buffer local commits cannot leak into absolute runtime contracts.
- Generated frame-history trimming preserves its absolute frame base.
- The actual payload consumed by LDF is present in the corresponding event.
- Published events contain detached data and are emitted only after commit.
- All failure stages are transactionally reversible.
- Failed-and-retried stochastic steps reproduce clean-session outputs from the
  same initial RNG state.
- Legacy compatibility delegates to the same session and contains no duplicate
  execution state machine.
