# RootSourceProposal Time-Origin Contract

## Goal

Prevent absolute runtime frame indices from being used directly against
anchor-relative RootRefiner proposals. Synthetic and dataset routes that already
represent a full global timeline must retain their current behavior.

## Data Contract

`RootSourceProposal` gains three first-class fields:

- `start_frame_abs: int = 0`
- `start_commit_abs: int = 0`
- `timeline_mode: Literal["absolute_timeline", "anchor_relative"] = "absolute_timeline"`

Direct construction remains backward compatible and therefore defaults to a
full absolute route beginning at frame and commit zero.

`from_root_plan()` creates an `anchor_relative` proposal. Its absolute commit
origin is `root_plan.anchor_commit_idx`, and its absolute frame origin is derived
from that commit with the shared token/frame mapping utility.

The two origins must be non-negative and mutually consistent. Unsupported
timeline modes and inconsistent origins raise `ValueError`; they are never
silently clamped.

## Indexing Boundary

`RootSourceProposal` owns absolute/local conversion through explicit methods:

- `absolute_to_local_frame(absolute_frame)`
- `local_to_absolute_frame(local_frame)`

Runtime keeps two frame indices explicit:

- `current_frame_abs` indexes `generated_history_traj7` and the materialized
  world timeline.
- `route_frame_local` indexes `proposal_traj7` and constrains route progress.

`StreamGenerator` derives both before calling the composer. Route tracking only
receives `route_frame_local`; generated-root lookup only receives
`current_frame_abs`. The composer does not inspect proposal metadata.

For `absolute_timeline`, local and absolute indices are identical because the
default origin is zero. Non-zero absolute origins are still supported through
the same conversion methods.

## Runtime Behavior

When an anchor-relative proposal starts at absolute frame `F`:

1. Runtime computes the current absolute frame from the absolute commit.
2. The proposal converts it to `route_frame_local = current_frame_abs - F`.
3. `compose_active_window_segment()` reads the current root at
   `generated_history_traj7[current_frame_abs]`, while route projection and
   progress tracking use `route_frame_local`.
4. `compose_active_window_world_condition()` materializes the local proposal at
   `start_frame_abs`, then overwrites its past with generated history and its
   future from `current_frame_abs` with the composed segment.
5. A current frame before the proposal origin is rejected. A current frame past
   the proposal length is also rejected instead of selecting the final route
   point.

This prevents a short RootRefiner future route from being indexed by a large
global frame number and clamped to its endpoint.

## Compatibility

- Existing synthetic/sample proposals require no call-site changes.
- Existing `metadata["root_plan_anchor_commit_idx"]` is retained for artifact
  compatibility, but runtime indexing no longer reads it.
- `absolute_route` and `active_window` both use the proposal's explicit time
  conversion before route slicing or composition.

## Tests

Add regression coverage for:

- legacy direct construction defaults to an absolute origin at zero;
- `from_root_plan()` records anchor-relative frame and commit origins;
- absolute/local frame conversion is reversible;
- active-window composition at a large absolute commit reads the generated root
  at that absolute frame but uses the matching local route frame;
- materialized world conditions place an anchor-relative proposal at its
  absolute origin and remain continuous at the update frame;
- frames before the proposal origin or after its valid range fail explicitly;
- existing absolute synthetic payload tests remain unchanged.
