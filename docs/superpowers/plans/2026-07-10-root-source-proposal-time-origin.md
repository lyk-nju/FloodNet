# RootSourceProposal Time-Origin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RootSourceProposal time origin explicit so runtime never indexes an anchor-relative RootRefiner proposal with an absolute frame number.

**Architecture:** RootSourceProposal owns absolute/local frame conversion and absolute-timeline materialization. Active-window composition receives separate generated-history and route indices. StreamGenerator performs the conversion once at the proposal boundary before route tracking or payload construction.

**Tech Stack:** Python, dataclasses, PyTorch, pytest, FloodNet token/frame utilities.

## Global Constraints

- Direct RootSourceProposal construction remains backward compatible with an absolute timeline starting at frame/commit zero.
- RootPlan conversion produces an anchor-relative proposal whose origin comes from `anchor_commit_idx`.
- Absolute and proposal-local frame indices must never be silently mixed or clamped.
- Existing unrelated worktree changes must remain untouched.

---

### Task 1: Add the proposal time-origin contract

**Files:**
- Modify: `utils/inference/runtime_update/root_source.py`
- Test: `tests/test_runtime_update_migration.py`

**Interfaces:**
- Produces: `RootSourceProposal.start_frame_abs`, `start_commit_abs`, `timeline_mode`, `absolute_to_local_frame()`, `local_to_absolute_frame()`, and `to_absolute_timeline()`.
- Consumes: `utils.token_frame.token_start_frame()` and `utils.motion_process.build_physical_7d_from_5d()`.

- [ ] **Step 1: Write failing tests for defaults, RootPlan origins, conversion, bounds, and materialization**

Add tests that assert direct construction defaults to `(0, 0, "absolute_timeline")`, `from_root_plan(anchor_commit_idx=3)` starts at `token_start_frame(3)`, conversion is reversible, out-of-range conversion raises `ValueError`, and materialization places local frame zero at the declared absolute origin.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_runtime_update_migration.py -q`

Expected: failures because the new fields and methods do not exist.

- [ ] **Step 3: Implement the minimal typed contract**

Use `Literal["absolute_timeline", "anchor_relative"]`, validate non-negative and token/frame-consistent origins in `__post_init__`, set origins in `from_root_plan()`, and implement strict conversion/materialization methods. Recompute physical 7D deltas after prefix materialization.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_runtime_update_migration.py -q`

Expected: all tests pass.

### Task 2: Separate generated absolute time from proposal-local time

**Files:**
- Modify: `utils/inference/runtime_update/active_condition.py`
- Test: `tests/test_ldf_runtime_active_update.py`

**Interfaces:**
- Consumes: `current_frame` as the generated-history absolute frame and `route_frame_local` as the proposal-local frame.
- Produces: `compose_active_window_segment(..., route_frame_local: int | None = None)` and `compose_active_window_world_condition(..., route_start_frame_abs: int = 0)`.

- [ ] **Step 1: Write a failing regression for a short local route at a large absolute frame**

Create generated history long enough to contain absolute frame 37 and a short route whose local frame zero matches that generated root. Assert composition reads generated frame 37 while route tracking begins near local frame zero, not the route endpoint.

- [ ] **Step 2: Write a failing world-timeline materialization test**

Assert an anchor-relative route with `route_start_frame_abs=37` appears at frame 37, generated history is preserved through the update frame, the segment starts continuously there, and output 7D deltas match recomputation from final 5D values.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_ldf_runtime_active_update.py -q`

Expected: failures because composer still uses one frame index for both timelines.

- [ ] **Step 4: Implement separate indices and materialization**

Keep `route_frame_local=None` backward compatible by defaulting to `current_frame`. Use the local value for desired route length and tracker minimum, the absolute value for generated-root lookup, and `route_start_frame_abs` when constructing the world timeline. Reject invalid origins instead of clamping.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_ldf_runtime_active_update.py -q`

Expected: all tests pass.

### Task 3: Convert time once at the StreamGenerator boundary

**Files:**
- Modify: `utils/inference/stream_generator.py`
- Test: `tests/test_stream_benchmark_rootplan.py`
- Test: `tests/test_model_manager_rootplan.py`

**Interfaces:**
- Consumes: RootSourceProposal conversion/materialization APIs from Task 1 and dual-index composer APIs from Task 2.
- Produces: correct `absolute_route` and `active_window` payloads for anchor-relative RootRefiner proposals.

- [ ] **Step 1: Write failing StreamGenerator tests**

Build an anchor-relative proposal starting at a non-zero commit. Assert `absolute_route` materializes it at the correct absolute frame. For `active_window`, provide generated frame history and assert the tracker remains near the proposal beginning rather than jumping to its endpoint.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_benchmark_rootplan.py tests/test_model_manager_rootplan.py -q`

Expected: anchor-relative payload assertions fail under absolute indexing.

- [ ] **Step 3: Implement boundary conversion**

For both contracts, derive `current_frame_abs` from the absolute commit, validate it through `proposal.absolute_to_local_frame()`, and materialize absolute world conditions with `proposal.to_absolute_timeline()`. For active-window composition, pass `current_frame=current_frame_abs` and `route_frame_local=proposal_local_frame`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_benchmark_rootplan.py tests/test_model_manager_rootplan.py -q`

Expected: all tests pass.

- [ ] **Step 5: Run the complete runtime-update regression set**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_runtime_update_migration.py tests/test_ldf_runtime_active_update.py tests/test_stream_benchmark_rootplan.py tests/test_model_manager_rootplan.py tests/test_ldf_condition_update_cli.py -q`

Expected: all tests pass with no warnings introduced by this change.

- [ ] **Step 6: Inspect the final diff**

Run: `git diff --check` and `git diff -- utils/inference/runtime_update/root_source.py utils/inference/runtime_update/active_condition.py utils/inference/stream_generator.py tests/test_runtime_update_migration.py tests/test_ldf_runtime_active_update.py tests/test_stream_benchmark_rootplan.py tests/test_model_manager_rootplan.py`

Expected: only the time-origin contract, integration, and regression tests are present.
