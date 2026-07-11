# Web Trajectory Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render authored, RootRefiner proposal, exact LDF payload, and generated-root trajectories in the Web demo while defaulting root feedback to alpha 1.0.

**Architecture:** Add a Web-only diagnostics store that derives immutable world-space polylines from committed runtime state. `ModelManager` publishes those diagnostics atomically with frame events, the API serializes them, and Three.js owns independent current/history layers with visibility controls.

**Tech Stack:** Python, PyTorch, Flask, Three.js, pytest.

## Global Constraints

- Runtime route ownership remains in `StreamRuntimeSession` and `RootSourceManager`.
- Diagnostics never mutate proposals, payloads, timeline, or route progress.
- Invalid proposal/payload padding is masked and never rendered.
- Historical snapshots are bounded to the latest 32 source versions.
- Diagnostic extraction errors cannot fail a generation transaction.
- Root feedback defaults to enabled with XZ alpha `1.0` unless explicitly overridden.

---

### Task 1: World-Space Diagnostics Store

**Files:**
- Create: `web_demo/runtime/trajectory_diagnostics.py`
- Create: `tests/test_trajectory_diagnostics.py`

**Interfaces:**
- Consumes: `RootSourceManager.active`, `StreamCommitEvent.actual_payload`, and `RootTimeline`.
- Produces: `TrajectoryDiagnosticsStore.update_from_commit(event, source_manager, timeline)`, `set_authored_route(points)`, `clear()`, and `to_payload()`.

- [ ] Write failing tests for proposal mask handling, payload local-to-world conversion, one snapshot per source version, 32-version retention, clear, and malformed payload isolation.
- [ ] Run `python -m pytest -q tests/test_trajectory_diagnostics.py` and confirm the tests fail because the module is absent.
- [ ] Implement `TrajectoryDiagnosticsStore` using `uncanonicalize_7d`, `future_frame_mask`, `traj_cond_frame_mask`, and `body_anchor_abs_token`.
- [ ] Run the test file and confirm all tests pass.

### Task 2: Runtime Publication and API Contract

**Files:**
- Modify: `web_demo/model_manager.py`
- Modify: `web_demo/api/routes.py`
- Modify: `tests/test_model_manager_runtime_session.py`

**Interfaces:**
- Consumes: `TrajectoryDiagnosticsStore` from Task 1.
- Produces: `ModelManager.get_trajectory_debug()` and successful `/api/get_frame` responses containing `trajectory_debug`.

- [ ] Write failing tests proving committed events update diagnostics, route clear/reset clears diagnostics, malformed extraction does not fail `_generate_once()`, and frame responses serialize diagnostics.
- [ ] Run the focused tests and confirm expected failures.
- [ ] Initialize the store in `ModelManager`, update authored routes on accepted updates, update committed diagnostics after `session.step()`, and include diagnostics in frame responses.
- [ ] Run focused ModelManager/API tests and confirm they pass.

### Task 3: Root Feedback Defaults

**Files:**
- Modify: `web_demo/model_manager.py`
- Modify: `web_demo/app.py`
- Modify: `web_demo/templates/index.html`
- Modify: `web_demo/static/js/main.js`
- Modify: `web_demo/runtime/model_loader.py`
- Modify: `tests/test_model_manager_rootplan.py`
- Modify: `tests/test_web_demo_root_feedback_ui.py`

**Interfaces:**
- Consumes: existing `SetRootFeedback` command flow.
- Produces: default status `On - 1.00` and start/reset requests with alpha `1.0`.

- [ ] Write failing tests for backend/model-loader defaults and static UI defaults.
- [ ] Run focused tests and verify they fail on the current `False/0.5` values.
- [ ] Change fallback defaults to `enabled=True`, `alpha=1.0`; keep explicit request values authoritative.
- [ ] Run focused tests and confirm they pass.

### Task 4: Three.js Diagnostic Layers

**Files:**
- Modify: `web_demo/static/js/main.js`
- Modify: `web_demo/templates/index.html`
- Modify: `web_demo/static/css/style.css`
- Modify: `tests/test_web_demo_root_feedback_ui.py`

**Interfaces:**
- Consumes: `trajectory_debug.current` and `trajectory_debug.snapshots` from `/api/get_frame`.
- Produces: independently toggleable current authored/proposal/payload lines and translucent activation snapshots.

- [ ] Add failing static contract tests for stable layer names, colors, toggles, snapshot disposal, and frame-response updates.
- [ ] Run the UI contract tests and verify expected failures.
- [ ] Implement reusable Three.js polyline layers with cyan `0x00c8ff`, green `0x35d07f`, orange `0xff9f43`, and translucent history materials.
- [ ] Add a compact legend with authored, proposal, payload, and history checkboxes; reset disposes history geometries/materials.
- [ ] Run UI contract and Web tests and confirm they pass.

### Task 5: Verification and Web Smoke Test

**Files:**
- Verify only; no new production file required.

**Interfaces:**
- Consumes: completed backend and frontend implementation.
- Produces: test evidence and a running GPU 1 Web demo.

- [ ] Run runtime/Web tests including trajectory diagnostics, ModelManager, API, session, and root feedback.
- [ ] Run `python -m py_compile` for modified Python modules and `git diff --check`.
- [ ] Restart Web demo on GPU 1 with LDF `step_500000.ckpt` and RootRefiner `refiner_step_700000.ckpt`.
- [ ] Trigger `001168`, consume frames, and verify `/api/get_frame` returns non-empty proposal and actual-payload diagnostics with root feedback `On - 1.00`.
- [ ] Report the URL, checkpoint identities, diagnostic counts, and any remaining extraction errors.
