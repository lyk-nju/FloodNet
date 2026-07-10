# StreamGenerator Execution Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an atomic `StreamGenerator.execute_step()` that owns LDF generation through decoded/recovered commit state while preserving the legacy web path behind a migration flag.

**Architecture:** Correct commit-boundary frame semantics first. Add focused execution DTO/state helpers under `utils/inference`, then make StreamGenerator own VAE, recovery, generated root history, feedback, rollback, and timeline advancement. Finally add an opt-in web path that consumes `StreamCommitEvent.joint_frames` only.

**Tech Stack:** Python dataclasses, PyTorch, NumPy, pytest, FloodNet LDF/VAE streaming APIs.

## Global Constraints

- `traj_mask.runtime.use_owned_stream_execution` defaults to `false`.
- Existing `StreamGenerator.step()` remains available as the LDF-only compatibility API.
- Existing ModelManager generation remains available while the owned path is opt-in.
- No partial LDF/VAE/recovery/timeline state may survive a failed owned step.
- Existing unrelated noise-initializer worktree changes remain untouched.

---

### Task 1: Correct commit-boundary frame semantics

**Files:**
- Modify: `utils/token_frame.py`
- Modify: `utils/inference/runtime_update/root_source.py`
- Modify: `utils/inference/stream_generator.py`
- Test: `tests/test_token_frame_mapping.py`
- Test: `tests/test_runtime_update_migration.py`
- Test: `tests/test_stream_benchmark_rootplan.py`

**Interfaces:**
- Produces: `commit_boundary_frame(commit_idx, frames_per_token=4) -> int`.
- Changes: RootPlan proposals map frame zero to the committed-prefix boundary.

- [ ] Add failing tests for commit 0, commit 1, commit 10, proposal origin, and active-window history bounds.
- [ ] Run the focused tests and verify failures against the current token-start behavior.
- [ ] Implement the boundary helper and switch proposal/runtime current-frame derivation.
- [ ] Run focused tests and verify they pass.

### Task 2: Add execution contracts and state snapshots

**Files:**
- Create: `utils/inference/stream_execution.py`
- Test: `tests/test_stream_execution.py`

**Interfaces:**
- Produces: `RootFeedbackConfig`, `StreamCommitEvent`, LDF/VAE/recovery snapshot and restore helpers, root-feedback target/re-encode helpers.
- Consumes: existing VAE stream APIs and 263D root replacement utilities.

- [ ] Add failing tests for config validation, immutable event fields, full VAE encoder/decoder cache restore, and recovery state restore.
- [ ] Run tests and verify RED.
- [ ] Implement minimal DTOs and clone/snapshot/restore helpers.
- [ ] Add failing tests for disabled feedback decode and enabled feedback corrected-latent writeback behavior.
- [ ] Implement feedback helper without importing `eval` modules.
- [ ] Run tests and verify GREEN.

### Task 3: Implement atomic StreamGenerator.execute_step

**Files:**
- Modify: `utils/inference/stream_generator.py`
- Test: `tests/test_stream_generator_execution.py`
- Test: `tests/test_stream_benchmark_rootplan.py`

**Interfaces:**
- Produces: `configure_execution(...)`, `reset_execution_state(...)`, `execute_step(...) -> StreamCommitEvent`, `generated_history_traj7`.
- Consumes: Task 2 execution contracts.

- [ ] Add fake LDF/VAE/recovery tests showing one step advances exactly one commit and returns coherent latent, decoded, joints, root history, payload, and timeline state.
- [ ] Run tests and verify RED.
- [ ] Implement dependency configuration, owned state reset, payload construction with owned generated history, decode/recovery, history append, and exact timeline append.
- [ ] Add a failing rollback test where recovery raises after VAE decode.
- [ ] Implement rollback of LDF, VAE, recovery, histories, counters, first-chunk flag, and timeline.
- [ ] Run execution and root-source tests and verify GREEN.

### Task 4: Add opt-in web migration path

**Files:**
- Modify: `configs/stream.yaml`
- Modify: `web_demo/runtime/model_loader.py`
- Modify: `web_demo/model_manager.py`
- Test: `tests/test_model_manager_rootplan.py`
- Test: `tests/test_stream_model_loader.py`

**Interfaces:**
- Consumes: `StreamGenerator.configure_execution()` and `execute_step()`.
- Produces: `traj_mask.runtime.use_owned_stream_execution` web switch.

- [ ] Add failing tests that the flag defaults false, model loading wires the VAE, reset wires the current recovery object, and the owned loop path only buffers event joints.
- [ ] Run tests and verify RED.
- [ ] Implement opt-in configuration and a small `_execute_owned_stream_step()` ModelManager adapter.
- [ ] Keep the legacy block unchanged for `false`; call the owned adapter for `true`.
- [ ] Run web/runtime tests and verify GREEN.

### Task 5: Final regression and audit

**Files:**
- Verify all files above.

- [ ] Run focused runtime suites, py_compile, and `git diff --check`.
- [ ] Audit that no owned runtime module imports from `eval`.
- [ ] Confirm unrelated dirty files are not staged or modified by these commits.
