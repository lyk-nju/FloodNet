# Stream Runtime Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace independently stateful Web/legacy streaming paths with one transactional `StreamRuntimeSession` that owns absolute time, commands, route composition, VAE/recovery state, and commit publication.

**Architecture:** Build immutable contracts and pure route/condition components first, then add formal model/VAE snapshots and a three-phase runtime transaction. Web and legacy entry points finally become command/event adapters around the same session; duplicate execution is deleted only after fixed-seed parity passes.

**Tech Stack:** Python 3.10 dataclasses and enums, threading locks, NumPy, PyTorch, causal 1D WAN VAE, pytest.

## Global Constraints

- `StreamRuntimeSession` is the only authoritative owner of model commit, VAE/recovery state, generated root history, and `RootTimeline`.
- Activation boundary state is separate from `RootSourceProposal.future_traj7`; future frame zero is a real future frame.
- Runtime code outside the LDF kernel uses absolute commit/frame indices.
- Every activated source explicitly selects `world_route` or `relative_route`.
- All generation-affecting changes enter through a globally ordered command queue and freeze at a worker boundary.
- Composition and payload construction are pure; route progress commits only after a successful token transaction.
- Route end is hold-last numerically, invalid by mask, and emits `route_exhausted` once.
- Preview root-feedback decode must not advance the formal decoder cache.
- Events publish detached clones only after commit and are not retained indefinitely by the session.
- Existing `eval/ldf/runtime_update/` imports remain compatibility re-exports during migration.
- Preserve unrelated dirty noise-initializer/config/eval-output changes; stage only files named by each task.
- Use `apply_patch` for manual edits and TDD for every behavior change.

---

## Phase 1: Absolute Time and Pure Runtime Contracts

### Task 1: Absolute Frame Mapping and GeneratedRootHistory

**Files:**
- Modify: `utils/token_frame.py`
- Create: `utils/inference/stream_runtime/history.py`
- Create: `utils/inference/stream_runtime/__init__.py`
- Test: `tests/test_stream_runtime_history.py`
- Modify test: `tests/test_token_frame.py`

**Interfaces:**
- Produces: `first_future_frame_abs(commit_idx, frames_per_token=4) -> int`.
- Produces: `last_generated_frame_abs(commit_idx, frames_per_token=4) -> int | None`.
- Produces: `GeneratedRootHistory(base_frame_abs: int, frames_7d: Tensor)` with `next_frame_abs`, `slice_abs()`, `append()`, and `trim_before()`.

- [ ] **Step 1: Write failing causal-boundary tests**

```python
def test_future_and_last_generated_frames_cover_cold_start_and_commit10():
    assert first_future_frame_abs(0) == 0
    assert last_generated_frame_abs(0) is None
    assert first_future_frame_abs(10) == 37
    assert last_generated_frame_abs(10) == 36

def test_generated_history_preserves_absolute_base_after_trim():
    history = GeneratedRootHistory.empty(dtype=torch.float32)
    history.append(torch.arange(35, dtype=torch.float32).view(5, 7), start_frame_abs=0)
    history.trim_before(3)
    assert history.base_frame_abs == 3
    assert history.next_frame_abs == 5
    assert torch.equal(history.slice_abs(3, 5), history.frames_7d)
```

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_token_frame.py tests/test_stream_runtime_history.py -q`

Expected: import failures for the two mapping helpers and `GeneratedRootHistory`.

- [ ] **Step 3: Implement mapping and bounded absolute history**

```python
def first_future_frame_abs(commit_idx: int, frames_per_token: int = 4) -> int:
    return num_frames_for_tokens(max(0, int(commit_idx)), frames_per_token)

def last_generated_frame_abs(commit_idx: int, frames_per_token: int = 4) -> int | None:
    future = first_future_frame_abs(commit_idx, frames_per_token)
    return None if future == 0 else future - 1
```

`GeneratedRootHistory.append()` must require `start_frame_abs == next_frame_abs`; `slice_abs()` raises when requested data was trimmed or is not generated; `trim_before()` advances `base_frame_abs` without renumbering frames.

- [ ] **Step 4: Verify GREEN and public exports**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_token_frame.py tests/test_stream_runtime_history.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/token_frame.py utils/inference/stream_runtime/__init__.py utils/inference/stream_runtime/history.py tests/test_token_frame.py tests/test_stream_runtime_history.py
git commit -m "feat: add absolute generated root history"
```

### Task 2: Immutable Source, Activation, Progress, and Event DTOs

**Files:**
- Create: `utils/inference/stream_runtime/contracts.py`
- Modify: `utils/inference/stream_runtime/__init__.py`
- Test: `tests/test_stream_runtime_contracts.py`

**Interfaces:**
- Produces: `SpaceContract`, `RouteStatus`, `SegmentLabel`, `RouteProgressState`, `RootSourceProposal`, `RootSourceCommand`, `ActivatedRootSource`, `ComposeResult`, `RuntimeStepConfig`, `KernelStepResult`, `StreamCommitEvent`, `SessionResetEvent`, and `RuntimeEvent`.
- Consumes: `RootFrameState` and absolute history conventions from Task 1.

- [ ] **Step 1: Write failing DTO validation and alias tests**

```python
def test_proposal_contains_future_only_and_validates_mask():
    future = torch.zeros(4, 7)
    future[:, 3] = 1.0
    proposal = RootSourceProposal(
        future_traj7=future,
        future_frame_mask=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        source_id="route-a",
        version=3,
        metadata={},
    )
    future[0, 0] = 99
    assert proposal.future_traj7[0, 0] == 0

def test_event_clones_mutable_payloads():
    latent = torch.ones(1, 4)
    event = StreamCommitEvent(
        absolute_commit_before=0,
        absolute_commit_after=1,
        local_commit_before=0,
        local_commit_after=1,
        latent_buffer_start_commit_abs=0,
        latent_buffer_epoch=0,
        committed_latent=latent,
        decoded_chunk=torch.zeros(1, 263),
        joint_frames=torch.zeros(1, 22, 3),
        root_frames_start_abs=0,
        root_frames=torch.zeros(1, 7),
        timeline_state=RootFrameState.initial(dtype=torch.float32),
        actual_payload=None,
        source_id=None,
        source_version=None,
        actual_activation_commit=None,
        lifecycle_events=(),
    )
    latent.zero_()
    assert event.committed_latent.eq(1).all()
```

Also test mask length/dtype, empty future rejection, optional clear contract, `actual_activation_commit`, `first_future_frame_abs`, and event rolling-buffer fields.

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_contracts.py -q`

Expected: module/type import failures.

- [ ] **Step 3: Implement immutable contracts**

Use `Enum` subclasses for space/status and frozen dataclasses. In each DTO `__post_init__`, replace tensor/array fields with detached CPU clones using `object.__setattr__`. `RootSourceCommand(kind=CLEAR)` requires `proposal=None` and `space_contract=None`; `REPLACE` requires both.

- [ ] **Step 4: Verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_contracts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/inference/stream_runtime/contracts.py utils/inference/stream_runtime/__init__.py tests/test_stream_runtime_contracts.py
git commit -m "feat: define stream runtime contracts"
```

### Task 3: Global RuntimeCommandQueue

**Files:**
- Create: `utils/inference/stream_runtime/commands.py`
- Modify: `utils/inference/stream_runtime/contracts.py`
- Modify: `utils/inference/stream_runtime/__init__.py`
- Test: `tests/test_stream_runtime_commands.py`

**Interfaces:**
- Produces: typed commands `SetRootSource`, `ClearRootSource`, `SetText`, `SetGuidance`, `SetRootFeedback`, `SetRuntimeControls`, `ResetSession`.
- Produces: `RuntimeCommandQueue.submit(command)`, `prepare_due(commit_abs) -> PreparedCommandBatch`, and `ack(batch)`.
- Produces: `reduce_commands(base_config, batch, boundary_state) -> PreparedRuntimeTransition`.

- [ ] **Step 1: Write failing queue concurrency/reduction tests**

```python
def test_prepare_is_non_destructive_and_ack_removes_exact_versions():
    queue = RuntimeCommandQueue()
    queue.submit(SetText(version=1, requested_commit_abs=0, text="walk"))
    batch = queue.prepare_due(0)
    queue.submit(SetText(version=2, requested_commit_abs=0, text="run"))
    assert [c.version for c in queue.prepare_due(0).commands] == [1, 2]
    queue.ack(batch)
    assert [c.version for c in queue.prepare_due(0).commands] == [2]

def test_replace_clear_replace_reduces_by_global_version():
    future = torch.zeros(2, 7)
    future[:, 3] = 1.0
    mask = torch.ones(2, dtype=torch.bool)
    proposal_a = RootSourceProposal(future, mask, "a", version=10, metadata={})
    proposal_b = RootSourceProposal(future, mask, "b", version=12, metadata={})
    boundary = RootFrameState.initial(dtype=torch.float32)
    batch = PreparedCommandBatch(commands=(
        SetRootSource(version=10, requested_commit_abs=0, proposal=proposal_a, space_contract=SpaceContract.WORLD_ROUTE),
        ClearRootSource(version=11, requested_commit_abs=0),
        SetRootSource(version=12, requested_commit_abs=0, proposal=proposal_b, space_contract=SpaceContract.WORLD_ROUTE),
    ))
    transition = reduce_commands(RuntimeStepConfig.default(), batch, boundary)
    assert transition.root_source_command.proposal.source_id == "b"
    assert transition.superseded_versions == (10, 11)
```

Add a thread test where submit occurs after `prepare_due()` and before `ack()`.

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_commands.py -q`

Expected: queue/commands missing.

- [ ] **Step 3: Implement queue and reducer**

Use one `threading.Lock`, strict increasing versions, immutable prepared batches, exact-version acknowledgement, and last-write-per-field reduction. `ResetSession` wins as an exclusive transition only if it is the final due command; later commands remain applicable to the new epoch.

- [ ] **Step 4: Verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_commands.py -q`

Expected: PASS without timing sleeps.

- [ ] **Step 5: Commit**

```bash
git add utils/inference/stream_runtime/contracts.py utils/inference/stream_runtime/commands.py utils/inference/stream_runtime/__init__.py tests/test_stream_runtime_commands.py
git commit -m "feat: add boundary-applied runtime commands"
```

## Phase 2: Pure Route Semantics and Payload Construction

### Task 4: Separate World and Relative Progress Policies

**Files:**
- Create: `utils/inference/stream_runtime/progress.py`
- Modify: `utils/inference/stream_runtime/__init__.py`
- Test: `tests/test_stream_runtime_progress.py`

**Interfaces:**
- Produces: `WorldRouteProgressPolicy.project(...) -> RouteProjection`.
- Produces: `RelativeRouteProgressPolicy.project(...) -> RouteProjection`.
- Both consume immutable `RouteProgressState` and never mutate policy instances.

- [ ] **Step 1: Write failing policy tests**

```python
def test_world_policy_is_monotonic_and_heading_aware():
    first = policy.project(route, actor_at_20, yaw_forward, RouteProgressState.initial())
    second = policy.project(route, actor_at_5, yaw_forward, first.proposed_progress)
    assert second.proposed_progress.route_index >= first.proposed_progress.route_index

def test_relative_policy_advances_by_absolute_future_phase_not_world_projection():
    projection = policy.project(
        activated,
        current_first_future_frame_abs=activated.first_future_frame_abs + 8,
        previous_progress=RouteProgressState.initial(),
    )
    assert projection.proposed_progress.route_index == 8
```

Include exact-end and single-future-frame cases.

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_progress.py -q`

Expected: policy imports fail.

- [ ] **Step 3: Implement pure policies**

Port heading-aware/arc-length math from `runtime_update/route_tracker.py` without `_last_index`. Clamp exact end to the last valid future index. Relative progress derives consumed future phase from absolute frame difference and uses route-local arc lookahead.

- [ ] **Step 4: Verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_progress.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/inference/stream_runtime/progress.py utils/inference/stream_runtime/__init__.py tests/test_stream_runtime_progress.py
git commit -m "feat: separate route progress policies"
```

### Task 5: Stateless ConditionComposer and Route Exhaustion

**Files:**
- Create: `utils/inference/stream_runtime/composer.py`
- Modify: `utils/inference/stream_runtime/contracts.py`
- Modify: `utils/inference/stream_runtime/__init__.py`
- Test: `tests/test_stream_runtime_composer.py`

**Interfaces:**
- Produces: `ConditionComposer.compose(activated, history, boundary_state, first_future_frame_abs, previous_progress, horizon_frames, bridge_frames=8) -> ComposeResult`.
- Consumes: Tasks 1, 2, and 4 contracts.

- [ ] **Step 1: Write failing cold-start, anchor uniqueness, and space-contract tests**

```python
def test_cold_start_uses_virtual_boundary_without_emitting_duplicate_anchor():
    result = composer.compose(activated_at_commit0, empty_history, initial_state, 0, initial_progress, 16)
    assert result.frame_start_abs == 0
    assert result.world_condition_7d[0, :5].equal(proposal.future_traj7[0, :5]) is False
    assert result.segment_labels[0] != SegmentLabel.BOUNDARY

def test_world_and_relative_contracts_diverge_after_generated_drift():
    world = composer.compose(world_activated, drifted_history, boundary, next_frame, progress, 40)
    relative = composer.compose(relative_activated, drifted_history, boundary, next_frame, progress, 40)
    target_xz = proposal.future_traj7[-1, [0, 2]]
    world_error = torch.linalg.norm(world.world_condition_7d[-1, [0, 2]] - target_xz)
    relative_error = torch.linalg.norm(relative.world_condition_7d[-1, [0, 2]] - target_xz)
    assert world_error < relative_error

def test_route_end_holds_value_with_false_mask_and_one_exhaustion_transition():
    result = composer.compose(ended_source, history, boundary, next_frame, progress_at_end, 20)
    assert torch.equal(result.world_condition_7d[-1], result.world_condition_7d[-2])
    assert not result.frame_mask[-1]
    assert result.route_status is RouteStatus.EXHAUSTED

def test_bridge_frames_controls_real_bridge_span():
    short = composer.compose(source, history, boundary, next_frame, progress, 40, bridge_frames=4)
    long = composer.compose(source, history, boundary, next_frame, progress, 40, bridge_frames=12)
    assert short.segment_labels.eq(SegmentLabel.BRIDGE.value).sum() == 4
    assert long.segment_labels.eq(SegmentLabel.BRIDGE.value).sum() == 12
    assert not torch.equal(short.world_condition_7d[:12], long.world_condition_7d[:12])
```

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_composer.py -q`

Expected: composer missing.

- [ ] **Step 3: Implement composition**

Build Hermite bridge geometry from boundary state, sample exactly `bridge_frames`
points excluding `t=0`, preserve authored coordinates for `WORLD_ROUTE`, rebase
deltas for `RELATIVE_ROUTE`, overlay generated history by absolute slices, return
explicit labels/mask, and recompute all deltas with
`build_physical_7d_from_5d()` only after final 5D assembly. Reject negative
`bridge_frames`; zero means direct continuity to the first selected future frame.

- [ ] **Step 4: Verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_composer.py -q`

Expected: PASS, including one-frame and past-end sources.

- [ ] **Step 5: Commit**

```bash
git add utils/inference/stream_runtime/contracts.py utils/inference/stream_runtime/composer.py utils/inference/stream_runtime/__init__.py tests/test_stream_runtime_composer.py
git commit -m "feat: add transactional route composer"
```

### Task 6: Exact PayloadBuilder with Absolute Bases

**Files:**
- Create: `utils/inference/stream_runtime/payload_builder.py`
- Modify: `utils/inference/runtime_update/payload_builder.py`
- Modify: `eval/ldf/runtime_update/payload_builder.py`
- Test: `tests/test_stream_runtime_payload.py`
- Modify test: `tests/test_ldf_runtime_active_update.py`

**Interfaces:**
- Produces: `PayloadBuilder.build(composed, timeline, local_commit_before, absolute_commit_before, chunk_size, history_tokens, horizon_tokens) -> dict | None`.
- Existing payload-builder modules become compatibility re-exports/adapters.

- [ ] **Step 1: Write failing trimmed-history and delta tests**

```python
def test_payload_slices_by_compose_frame_start_not_tensor_index():
    world = torch.zeros(80, 7)
    world[:, 3] = 1.0
    composed = ComposeResult(
        frame_start_abs=117,
        world_condition_7d=world,
        frame_mask=torch.ones(80, dtype=torch.bool),
        segment_labels=torch.full((80,), SegmentLabel.ROUTE.value),
        proposed_route_progress=RouteProgressState.initial(),
        route_status=RouteStatus.ACTIVE,
        diagnostics={},
    )
    payload = builder.build(composed, timeline, local_commit_before=31, absolute_commit_before=61, chunk_size=5, history_tokens=30, horizon_tokens=20)
    assert payload["traj_abs_start_token"] == 32
    assert payload["debug_world_frame_start_abs"] == 117

def test_payload_recomputes_delta_after_history_and_padding_overlay():
    anchor = timeline.at_commit(payload["body_anchor_abs_token"])
    world = uncanonicalize_7d(
        payload["traj_cond_7d_frame"],
        anchor.world_xz.unsqueeze(0),
        anchor.world_yaw.reshape(1),
    )[0]
    expected = build_physical_7d_from_5d(world[:, :5])
    assert torch.allclose(world[:, 5:7], expected[:, 5:7])
```

Also assert masks for generated history, valid route, and terminal hold padding across every `traj_substep_payloads` entry.

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_payload.py tests/test_ldf_runtime_active_update.py -q`

Expected: new builder missing or absolute-base assertion fails.

- [ ] **Step 3: Implement exact builder and compatibility wrappers**

Move absolute slicing/canonicalization from `runtime_update/payload_builder.py`; consume `ComposeResult.frame_start_abs` and `frame_mask`; never infer validity from tensor length. Keep old function names as deprecated wrappers that construct a `ComposeResult` explicitly.

- [ ] **Step 4: Verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_payload.py tests/test_ldf_runtime_active_update.py tests/test_runtime_update_migration.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/inference/stream_runtime/payload_builder.py utils/inference/runtime_update/payload_builder.py eval/ldf/runtime_update/payload_builder.py tests/test_stream_runtime_payload.py tests/test_ldf_runtime_active_update.py
git commit -m "feat: build payloads from absolute composed conditions"
```

## Phase 3: Formal Stream State and Transactional Session

### Task 7: Formal Model/VAE/RNG Snapshot APIs and Rolling Metadata

**Files:**
- Modify: `models/diffusion_forcing_wan.py`
- Modify: `models/vae_wan_1d.py`
- Modify: `models/tools/wan_vae_1d.py`
- Create: `utils/inference/stream_runtime/snapshots.py`
- Modify: `utils/inference/stream_execution.py`
- Test: `tests/test_stream_runtime_snapshots.py`
- Modify test: `tests/test_stream_generate_step_horizon.py`

**Interfaces:**
- Produces: model/VAE `snapshot_stream_state()` and `restore_stream_state(state)`.
- Produces: model `stream_buffer_metadata() -> StreamBufferMetadata`.
- Produces: `snapshot_rng_state(devices)` and `restore_rng_state(state)`.

- [ ] **Step 1: Write failing rolling-buffer and RNG restoration tests**

```python
def test_model_snapshot_restores_roll_and_absolute_buffer_metadata():
    before = model.snapshot_stream_state()
    force_one_roll(model)
    model.restore_stream_state(before)
    assert model.latent_buffer_epoch == before.latent_buffer_epoch
    assert model.latent_buffer_start_commit_abs == before.latent_buffer_start_commit_abs
    assert torch.equal(model.generated, before.generated)

def test_rng_snapshot_restores_python_numpy_torch_and_cuda():
    state = snapshot_rng_state(devices=[])
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
```

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_snapshots.py tests/test_stream_generate_step_horizon.py -q`

Expected: formal methods/metadata missing.

- [ ] **Step 3: Implement minimal formal snapshots**

Track `latent_buffer_start_commit_abs` and `latent_buffer_epoch` in `init_generated()` and increment both when rolling by `seq_len`. Snapshot generated buffer, local step/commit, text conditions, CFG values, denoise configuration, rolling metadata, and optional trajectory-buffer state protocol. VAE snapshots clone both encoder/decoder counters and feature maps. Retain `stream_execution.py` helpers as delegates to formal APIs with a temporary legacy fallback.

- [ ] **Step 4: Verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_snapshots.py tests/test_stream_generate_step_horizon.py tests/test_stream_execution.py -q`

Expected: PASS on CPU; CUDA RNG assertions run only when CUDA is available.

- [ ] **Step 5: Commit**

```bash
git add models/diffusion_forcing_wan.py models/vae_wan_1d.py models/tools/wan_vae_1d.py utils/inference/stream_runtime/snapshots.py utils/inference/stream_execution.py tests/test_stream_runtime_snapshots.py tests/test_stream_generate_step_horizon.py
git commit -m "feat: formalize streaming state snapshots"
```

### Task 8: Reduce StreamGenerator to the LDF Kernel

**Files:**
- Modify: `utils/inference/stream_generator.py`
- Modify: `utils/inference/stream_execution.py`
- Test: `tests/test_stream_generator_kernel.py`
- Modify test: `tests/test_stream_generator_execution.py`

**Interfaces:**
- Produces: `StreamGenerator.generate_token(text: str, payload: dict | None, first_chunk: bool, num_denoise_steps: int) -> KernelStepResult`.
- `KernelStepResult` contains detached `raw_latent`, the exact `actual_payload`, local commit before/after, `latent_buffer_start_commit_abs`, and `latent_buffer_epoch`; it contains no decoded/recovered/timeline state.
- Removes route slicing/tracking/payload decisions from the kernel.
- Existing `execute_step()` remains temporarily but delegates to an attached session added in Task 9.

- [ ] **Step 1: Write failing kernel boundary test**

```python
def test_kernel_consumes_exact_payload_without_route_or_timeline_access(monkeypatch):
    payload = {"traj_cond_7d_frame": torch.zeros(1, 5, 7)}
    monkeypatch.setattr(generator, "build_root_plan_stream_payload", Mock(side_effect=AssertionError("kernel must not build payloads")))
    result = generator.generate_token("walk", payload, first_chunk=True, num_denoise_steps=10)
    assert result.actual_payload is payload
    assert result.local_commit_after == result.local_commit_before + 1
```

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_generator_kernel.py -q`

Expected: `generate_token` missing and old route ownership still present.

- [ ] **Step 3: Implement kernel result and move route ownership out**

`generate_token()` builds text/trajectory embeddings from the supplied payload and invokes `stream_generate_step()`. It returns detached raw latent plus local/rolling metadata but performs no VAE decode, recovery, timeline write, source activation, progress update, or absolute slicing. Leave old ownership fields deprecated but intact until Task 9 provides the compatibility session and Task 14 removes them.

- [ ] **Step 4: Verify GREEN and compatibility tests**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_generator_kernel.py tests/test_stream_benchmark_rootplan.py tests/test_stream_generator_execution.py -q`

Expected: kernel tests PASS; compatibility tests remain green through wrappers.

- [ ] **Step 5: Commit**

```bash
git add utils/inference/stream_generator.py utils/inference/stream_execution.py tests/test_stream_generator_kernel.py tests/test_stream_generator_execution.py
git commit -m "refactor: isolate LDF streaming kernel"
```

### Task 9: StreamRuntimeSession Prepare/Mutate/Commit Transaction

**Files:**
- Create: `utils/inference/stream_runtime/source_manager.py`
- Create: `utils/inference/stream_runtime/session.py`
- Modify: `utils/inference/stream_runtime/__init__.py`
- Modify: `utils/inference/stream_execution.py`
- Test: `tests/test_stream_runtime_session.py`
- Modify test: `tests/test_stream_generator_execution.py`

**Interfaces:**
- Produces: `StreamRuntimeSession.submit(command) -> int`, `step() -> RuntimeEvent`, `reset(...) -> SessionResetEvent`.
- Constructor dependencies are explicit: `StreamRuntimeSession(kernel, vae, recovery, timeline, generated_history, command_queue, source_manager, composer, payload_builder, initial_config)`.
- Produces: `RootSourceManager.prepare_transition(batch, boundary_state, commit_abs)`, `commit_transition(prepared, proposed_progress, route_status)`, and one-shot exhaustion state.
- `prepare_transition()` injects the activation boundary into `ActivatedRootSource`; it never inserts that boundary into `RootSourceProposal.future_traj7`.
- Consumes all Phase 1-3 components.

- [ ] **Step 1: Write failing two-token timeline and transaction tests**

```python
def test_two_steps_commit_frame0_then_frame4():
    first = session.step()
    second = session.step()
    assert first.absolute_commit_after == 1
    assert first.timeline_state.world_xz.equal(first.root_frames[-1, [0, 2]])
    assert second.absolute_commit_after == 2
    assert session.timeline.head.commit_idx == 2
    assert session.generated_history.next_frame_abs == 5

def test_progress_and_commands_commit_only_after_success():
    command = session.submit(root_command_v1)
    session.recovery.fail = True
    with pytest.raises(RuntimeError):
        session.step()
    assert session.source_manager.active is None
    assert command in session.command_queue.pending_versions
```

- [ ] **Step 2: Add failing stochastic retry equivalence test**

Create two identical fake sessions. Session A fails once after model generation,
then retries; session B succeeds once. Assert committed latent, decoded chunk,
timeline, history, command acknowledgement, and RNG state are bitwise equal.

- [ ] **Step 3: Add failing root-feedback cache-order test**

Instrument a fake VAE so preview decode, replacement encode, and formal decode
are distinguishable. Assert preview does not change the formal decoder snapshot,
replacement encode occurs before the sole formal decode, and the committed VAE
snapshot contains exactly one token of formal decoder advancement.

- [ ] **Step 4: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_session.py -q`

Expected: session/source manager missing.

- [ ] **Step 5: Implement three-phase session**

Prepare commands/config/source/progress/payload without mutation; snapshot formal
stream and RNG state; invoke the kernel; perform root-feedback preview on a
non-committing decoder state; encode replacement; then perform exactly one
formal decode. Recover frames into temporary arrays; then atomically append
history/timeline, commit source progress/lifecycle, ack commands, and construct
cloned events. On exception restore snapshots and publish nothing.

- [ ] **Step 6: Make old `StreamGenerator.execute_step()` a session delegate**

Add `attach_runtime_session(session)` and have compatibility `execute_step()` call the attached session. It must raise a migration error if no session is attached rather than silently running the old state machine.

- [ ] **Step 7: Verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_session.py tests/test_stream_generator_execution.py tests/test_stream_execution.py -q`

Expected: PASS, including all failure injection stages and RNG retry.

- [ ] **Step 8: Commit**

```bash
git add utils/inference/stream_runtime/source_manager.py utils/inference/stream_runtime/session.py utils/inference/stream_runtime/__init__.py utils/inference/stream_execution.py utils/inference/stream_generator.py tests/test_stream_runtime_session.py tests/test_stream_generator_execution.py
git commit -m "feat: add authoritative stream runtime session"
```

## Phase 4: Web Commands, Compatibility, and Cutover

### Task 10: Build Session in ModelBundle and Convert Web Updates to Commands

**Files:**
- Modify: `web_demo/runtime/model_bundle.py`
- Modify: `web_demo/runtime/model_loader.py`
- Modify: `web_demo/runtime/contracts.py`
- Modify: `web_demo/model_manager.py`
- Test: `tests/test_model_manager_runtime_session.py`
- Modify test: `tests/test_model_manager_rootplan.py`

**Interfaces:**
- `ModelBundle.runtime_session: StreamRuntimeSession` is required.
- Web methods submit typed commands and do not mutate generator/session state.

- [ ] **Step 1: Write failing bundle and command-submission tests**

```python
def test_model_bundle_constructs_one_authoritative_session():
    bundle = load_fake_bundle()
    assert bundle.runtime_session.kernel is bundle.stream_generator
    assert bundle.runtime_session.vae is bundle.vae

def test_text_and_route_updates_submit_without_mutating_active_state():
    before = manager.runtime_session.source_manager.active
    manager.update_text("turn left")
    manager.update_trajectory(points, route_mode="world_route")
    assert manager.runtime_session.source_manager.active is before
    assert [type(c) for c in manager.runtime_session.command_queue.snapshot()] == [SetText, SetRootSource]
```

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_model_manager_runtime_session.py -q`

Expected: bundle/session and command methods missing.

- [ ] **Step 3: Wire one session and adapt all token-affecting APIs**

Construct recovery/session once in `model_loader.py`. Convert text, route replace/clear, CFG, root feedback, history/horizon/denoise, and reset APIs to typed commands. Pause/resume only control `GenerationWorker`; reset waits for a quiescent worker and consumes `SessionResetEvent`.

- [ ] **Step 4: Verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_model_manager_runtime_session.py tests/test_model_manager_rootplan.py -q`

Expected: PASS and no test observes direct active-source mutation from the caller thread.

- [ ] **Step 5: Commit**

```bash
git add web_demo/runtime/model_bundle.py web_demo/runtime/model_loader.py web_demo/runtime/contracts.py web_demo/model_manager.py tests/test_model_manager_runtime_session.py tests/test_model_manager_rootplan.py
git commit -m "refactor: route web updates through runtime commands"
```

### Task 11: Replace Web Generation Loop and Repair Compatibility Controllers

**Files:**
- Modify: `web_demo/model_manager.py`
- Modify: `web_demo/runtime/rootplan_controller.py`
- Modify: `web_demo/runtime/trajectory_controller.py`
- Modify: `configs/stream.yaml`
- Test: `tests/test_model_manager_runtime_session.py`
- Modify test: `tests/test_model_manager_rootplan.py`

**Interfaces:**
- Web generation worker calls only `runtime_session.step()` and buffers committed event frames.
- Legacy controller context managers restore plan, source, contract, progress, and version atomically.

- [ ] **Step 1: Write failing no-duplicate-execution and context restoration tests**

```python
def test_web_loop_only_consumes_session_event(monkeypatch):
    event = fake_commit_event(joint_frames=torch.zeros(4, 22, 3))
    manager.runtime_session.step = Mock(return_value=event)
    manager._generate_once()
    manager.runtime_session.step.assert_called_once_with()
    assert manager.frame_buffer.size() == 4
    assert manager.vae.stream_decode.call_count == 0

def test_temporary_plan_restores_previous_source_contract_and_progress():
    with controller.temporarily_active(plan):
        assert controller.active_plan is plan
    assert controller.active_source is previous_source
    assert controller.source_contract is previous_contract
    assert controller.progress == previous_progress
```

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_model_manager_runtime_session.py tests/test_model_manager_rootplan.py -q`

Expected: legacy loop still decodes/recovers or temporary plan loses source.

- [ ] **Step 3: Replace loop and make config session-first**

Extract `_generate_once()` that handles `StreamCommitEvent` and `SessionResetEvent`; delete per-frame legacy VAE/recovery/timeline code. Set `traj_mask.runtime.use_owned_stream_execution: true` only after these tests pass. Keep the old flag parser for one release but both values delegate to the session.

- [ ] **Step 4: Verify GREEN and static ownership audit**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_model_manager_runtime_session.py tests/test_model_manager_rootplan.py -q
rg -n '_append_root_state_from_stream_recovery|_apply_root_feedback_to_latent|vae\.stream_decode|stream_recovery\.process_frame' web_demo/model_manager.py
```

Expected: tests PASS; `rg` returns no legacy execution calls.

- [ ] **Step 5: Commit only the runtime config hunk plus code/tests**

Because `configs/stream.yaml` may contain unrelated local checkpoint/CFG edits,
stage only the `use_owned_stream_execution` hunk. Use `apply_patch` to create
`/tmp/enable-owned-runtime.patch` with this cached patch:

```diff
diff --git a/configs/stream.yaml b/configs/stream.yaml
--- a/configs/stream.yaml
+++ b/configs/stream.yaml
@@ -26,7 +26,7 @@ traj_mask:
     runtime:
         # Gradual migration switch. False preserves the legacy web execution loop.
-        use_owned_stream_execution: false
+        use_owned_stream_execution: true
```

Then stage and commit:

```bash
git add web_demo/model_manager.py web_demo/runtime/rootplan_controller.py web_demo/runtime/trajectory_controller.py tests/test_model_manager_runtime_session.py tests/test_model_manager_rootplan.py
git apply --cached /tmp/enable-owned-runtime.patch
git commit -m "refactor: make web generation session-owned"
```

### Task 12: Source Adapters and Eval/Legacy Compatibility Wrappers

**Files:**
- Modify: `utils/inference/runtime_update/root_source.py`
- Modify: `utils/inference/runtime_update/route_tracker.py`
- Modify: `utils/inference/runtime_update/active_condition.py`
- Modify: `utils/inference/runtime_update/__init__.py`
- Modify: `eval/ldf/runtime_update/` wrappers
- Modify: `tools/run_ldf_condition_update_eval.py`
- Test: `tests/test_runtime_update_migration.py`
- Modify test: `tests/test_ldf_condition_update_cli.py`
- Modify test: `tests/test_stream_benchmark_rootplan.py`

**Interfaces:**
- Dataset, synthetic, and RootRefiner adapters output future-only `RootSourceProposal`.
- Old `absolute_route` maps to `world_route`; old `active_window` requires an explicit migration choice and emits a deprecation warning.

- [ ] **Step 1: Write failing adapter tests**

```python
def test_rootplan_adapter_removes_current_anchor_from_future():
    root_plan = _root_plan_with_six_frames()
    world = uncanonicalize_7d(
        root_plan.waypoints_local_7d[: root_plan.valid_frames].unsqueeze(0),
        root_plan.anchor_world_xz.unsqueeze(0),
        root_plan.anchor_world_yaw.reshape(1),
    )[0]
    proposal = root_plan_to_proposal(root_plan)
    assert torch.equal(proposal.future_traj7[0, :5], world[1, :5])
    assert proposal.future_traj7.shape[0] == root_plan.valid_frames - 1
    assert proposal.future_frame_mask.all()

def test_eval_wrapper_is_identity_reexport():
    assert eval_RootSourceProposal is runtime_RootSourceProposal
```

- [ ] **Step 2: Verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_runtime_update_migration.py tests/test_ldf_condition_update_cli.py tests/test_stream_benchmark_rootplan.py -q`

Expected: old anchor-containing DTO assertions fail.

- [ ] **Step 3: Implement adapters and compatibility mappings**

Convert all source-specific physical/local outputs to authored world 7D, strip known anchor frames, create boolean masks, and require explicit space contract in new APIs. Keep eval modules as re-exports and update CLI choices to `world_route`/`relative_route` with legacy aliases.

- [ ] **Step 4: Verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_runtime_update_migration.py tests/test_ldf_condition_update_cli.py tests/test_stream_benchmark_rootplan.py tests/test_ldf_runtime_active_update.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/inference/runtime_update eval/ldf/runtime_update tools/run_ldf_condition_update_eval.py tests/test_runtime_update_migration.py tests/test_ldf_condition_update_cli.py tests/test_stream_benchmark_rootplan.py tests/test_ldf_runtime_active_update.py
git commit -m "refactor: adapt root sources to runtime session contracts"
```

## Phase 5: Parity Gates and Legacy Removal

### Task 13: Fixed-Seed, Rolling-Buffer, and Mid-Step Concurrency Parity

**Files:**
- Create: `tests/test_stream_runtime_parity.py`
- Create: `tools/check_stream_runtime_parity.py`
- Modify: `tests/test_stream_runtime_session.py`
- Modify: `docs/superpowers/specs/2026-07-10-stream-runtime-session-design.md` only if observed behavior requires an approved contract correction.

**Interfaces:**
- Produces a machine-readable parity report comparing compatibility and direct session entry points.

- [ ] **Step 1: Add deterministic fake-model parity tests**

Test 70 tokens so the fake/realistic model rolls at least once. Compare every event's absolute/local commits, epoch/base, actual payload, latent, decoded chunk, root frames, timeline state, source version, and lifecycle events.

- [ ] **Step 2: Add controlled mid-step command test**

Use barriers, not sleeps: block kernel generation for token `N`, submit `SetText` and `SetRootSource`, release generation, and assert token `N` uses old config while `N+1` activates both commands at the same real boundary.

- [ ] **Step 3: Verify deterministic suite**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_runtime_parity.py tests/test_stream_runtime_session.py -q`

Expected: PASS.

- [ ] **Step 4: Implement parity CLI**

`tools/check_stream_runtime_parity.py` accepts config, checkpoint, sample, seed, token count, and output JSON. It runs both public entry points against fresh identical model/VAE instances and exits nonzero on the first field mismatch.

- [ ] **Step 5: Run a real checkpoint smoke on an available GPU**

First inspect GPU memory, then run sample `001168`, seed `1234`, 40 tokens, root feedback disabled and enabled separately. Use the configured checkpoint rather than hard-coding a new path.

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
CUDA_VISIBLE_DEVICES=1 /home/yuankai/.conda/envs/flooddiffusion/bin/python tools/check_stream_runtime_parity.py --config configs/stream.yaml --sample 001168 --seed 1234 --tokens 40 --output /tmp/stream_runtime_parity_001168.json
```

Expected: exit 0 and JSON `"matched": true`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_stream_runtime_parity.py tests/test_stream_runtime_session.py tools/check_stream_runtime_parity.py
git commit -m "test: gate stream runtime parity"
```

### Task 14: Delete Duplicate Legacy State and Finalize Runtime Ownership

**Files:**
- Modify: `utils/inference/stream_generator.py`
- Modify: `utils/inference/stream_execution.py`
- Modify: `web_demo/model_manager.py`
- Modify: `web_demo/runtime/rootplan_controller.py`
- Modify: `configs/stream.yaml`
- Modify tests: all stream runtime/Web compatibility tests above

**Interfaces:**
- `StreamRuntimeSession.step()` is the only execution API that advances state.
- Legacy methods are thin delegates or removed when no callers remain.

- [ ] **Step 1: Prove duplicate symbols have no callers**

Run:

```bash
rg -n 'append_timeline_state_at_token_start_frame|_generated_root_5d|_active_root_source_tracker|_apply_root_feedback_to_latent|decode_token_with_root_feedback' web_demo utils/inference tests
```

Classify every result as session implementation, compatibility wrapper, or stale caller. Do not delete a symbol with a live non-wrapper caller.

- [ ] **Step 2: Remove the duplicate legacy execution state machine**

Delete Web-owned timeline/history/recovery/cache advancement, mutable route tracker ownership in `StreamGenerator`, and old direct root-source payload construction. Keep only documented compatibility delegates that call the session.

- [ ] **Step 3: Run full focused regression and ownership audit**

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest \
  tests/test_token_frame.py \
  tests/test_stream_runtime_history.py \
  tests/test_stream_runtime_contracts.py \
  tests/test_stream_runtime_commands.py \
  tests/test_stream_runtime_progress.py \
  tests/test_stream_runtime_composer.py \
  tests/test_stream_runtime_payload.py \
  tests/test_stream_runtime_snapshots.py \
  tests/test_stream_generator_kernel.py \
  tests/test_stream_runtime_session.py \
  tests/test_model_manager_runtime_session.py \
  tests/test_stream_runtime_parity.py \
  tests/test_runtime_update_migration.py \
  tests/test_model_manager_rootplan.py -q

/home/yuankai/.conda/envs/flooddiffusion/bin/python -m py_compile \
  utils/inference/stream_runtime/*.py \
  utils/inference/stream_generator.py \
  web_demo/model_manager.py

git diff --check
```

Expected: all tests PASS, py_compile exit 0, diff check empty.

- [ ] **Step 4: Audit dependency direction and dirty-worktree scope**

```bash
if rg -n '(^|\s)(from|import) eval' utils/inference/stream_runtime utils/inference/stream_generator.py; then exit 1; fi
git status --short
```

Expected: no runtime import from `eval`; unrelated pre-existing noise-initializer/output changes remain unstaged.

- [ ] **Step 5: Commit**

```bash
git add utils/inference/stream_runtime utils/inference/stream_generator.py utils/inference/stream_execution.py web_demo/model_manager.py web_demo/runtime/rootplan_controller.py \
  tests/test_token_frame.py tests/test_stream_runtime_history.py tests/test_stream_runtime_contracts.py tests/test_stream_runtime_commands.py \
  tests/test_stream_runtime_progress.py tests/test_stream_runtime_composer.py tests/test_stream_runtime_payload.py tests/test_stream_runtime_snapshots.py \
  tests/test_stream_generator_kernel.py tests/test_stream_runtime_session.py tests/test_model_manager_runtime_session.py tests/test_stream_runtime_parity.py \
  tests/test_runtime_update_migration.py tests/test_model_manager_rootplan.py
git commit -m "refactor: finalize authoritative stream runtime"
```

## Final Acceptance Gate

The migration is complete only when all of the following evidence exists:

- two-token timeline test proves commit 1/frame 0 and commit 2/frame 4;
- cold-start tests prove initial state is not motion frame 0;
- command barrier tests prove no mid-step mutation;
- exact-end/past-end route tests pass for both space contracts;
- one-shot exhaustion and masks are verified;
- rolling-buffer absolute time remains monotonic;
- RNG failed/retry output is bitwise equal to a clean run;
- Web loop contains no direct decode/recovery/timeline advancement;
- compatibility and direct session entry points pass fixed-seed parity;
- real checkpoint parity JSON reports `matched: true`;
- no owned runtime module imports `eval`;
- unrelated working-tree changes were not staged or reverted.
