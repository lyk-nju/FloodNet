# Web Trajectory Diagnostics Design

## Goal

Make Web runtime trajectory failures observable by rendering the three distinct
control layers alongside the generated motion:

- the authored user route;
- the world-frame RootRefiner proposal;
- the exact valid 7D payload consumed by LDF for the current token.

The Web demo must also retain one bounded diagnostic snapshot per activated
root-source version so an update that introduced divergence remains visible.
Root feedback is enabled with XZ alpha `1.0` by default.

## Display Contract

The scene uses stable colors and labels:

- cyan: authored user route;
- green: current RootRefiner/root-source proposal;
- orange: current token's actual LDF payload;
- red: generated actor root trail;
- translucent green/orange: proposal and payload snapshots from prior source
  activations.

Current proposal and payload lines update after every committed runtime event.
Historical snapshots are added only when a new source version emits
`route_active`. At most the latest 32 source versions are retained. A compact
legend provides independent visibility toggles for authored, proposal, payload,
history, and generated-root layers.

## Authoritative Data Flow

`ModelManager._generate_once()` remains the publication boundary. After
`StreamRuntimeSession.step()` commits successfully, it derives diagnostics only
from committed state:

1. Read the active immutable `RootSourceProposal` from `RootSourceManager`.
2. Read the exact `StreamCommitEvent.actual_payload` passed to LDF.
3. Apply `future_frame_mask` and `traj_cond_frame_mask`; terminal numerical
   padding is never rendered as valid control.
4. Convert the payload from active-window local coordinates back to world
   coordinates using `body_anchor_abs_token` and the matching `RootTimeline`
   state.
5. Publish immutable CPU arrays through the frame API.

The frontend never reconstructs canonicalization, route progress, or masks. It
only renders world-space XYZ polylines supplied by the backend.

## Backend State

A Web-only `TrajectoryDiagnosticsStore` owns presentation diagnostics:

- current authored route;
- current proposal, source id, source version, and activation commit;
- current actual payload, payload commit, and valid-frame count;
- a bounded ordered collection of activation snapshots.

It does not own or mutate runtime route state. Reset and committed
`route_cleared` events clear current and historical diagnostics. Route
exhaustion preserves the latest geometry but marks its status as exhausted.

The response shape is:

```text
trajectory_debug:
  current:
    authored_route: [[x, y, z], ...]
    root_source_proposal: [[x, y, z], ...]
    actual_payload: [[x, y, z], ...]
    source_id: string | null
    source_version: int | null
    activation_commit: int | null
    payload_commit: int | null
    route_status: string
  snapshots:
    - source_version: int
      activation_commit: int
      proposal: [[x, y, z], ...]
      payload: [[x, y, z], ...]
```

`/api/get_frame` includes this object with successful frame responses. Status
may expose counts and identifiers but does not duplicate large arrays.

## Payload World Conversion

The exact top-level payload is used because it is the payload consumed for the
current generation call. Its `traj_cond_7d_frame[0]` is canonicalized against
the timeline state at `body_anchor_abs_token`.

Conversion rules:

- use `uncanonicalize_7d()` with the matching anchor XZ and yaw;
- keep only the contiguous valid prefix from `traj_cond_frame_mask[0]`;
- publish XYZ columns only;
- reject malformed payloads without failing generation;
- record a concise diagnostic error in status when conversion is impossible.

The proposal is already world-frame physical 7D and requires only mask
application and XYZ extraction.

## Root Feedback Defaults

Defaults become:

- `root_feedback_enabled: true`;
- `root_feedback_xz_blend_alpha: 1.0`.

The Web reset form and backend fallback both use alpha `1.0`. Explicit API
values continue to override the default. The status display starts at
`On - 1.00` and is reconciled from committed session configuration.

## Failure Handling

Diagnostic extraction is non-authoritative. A malformed proposal, missing
timeline anchor, or malformed payload must not roll back or stop motion
generation. The store keeps the previous valid line, records the extraction
error, and lets the next committed event retry.

No diagnostics are published from a failed runtime transaction. Frame and
diagnostic publication occur together after a successful commit.

## Testing

Backend tests cover:

- proposal masks exclude terminal padding;
- payload local-to-world conversion uses the declared body anchor;
- current diagnostics update every commit;
- snapshots are created once per activated source version and capped at 32;
- reset and route clear remove diagnostic state;
- malformed diagnostics do not fail `_generate_once()`;
- default root feedback is enabled with alpha `1.0`.

Frontend tests or static contract tests cover:

- all trajectory layers have distinct materials;
- API diagnostics update current lines and activation snapshots;
- visibility controls independently hide each layer;
- reset disposes lines and markers without leaking Three.js objects.

## Non-Goals

- Editing RootRefiner or payload geometry from the browser.
- Persisting diagnostic history across server restarts.
- Rendering invalid numerical padding.
- Sending latent tensors, 263D motion features, or complete runtime events to
  the browser.
