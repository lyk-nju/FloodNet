# Eval Refactor Design

## Goal

Refactor evaluation code so each evaluator has a clear ownership boundary and a
small reusable core. The immediate goal is not to rewrite metrics or change
model behavior. The goal is to stop mixing LDF offline eval, RootRefiner eval,
streaming runtime eval, and shared orchestration in one flat `eval/` directory.

## Target Package Layout

```text
eval/
  common/
    context.py
    artifacts.py
    loaders.py
    summary.py
    gates.py
    runner.py
    watcher.py

  ldf/
    generation_metrics.py
    stream_metrics.py
    t2m_metrics.py

  root_refiner/
    benchmark.py
    metrics.py

  runtime/
    rollout.py
    benchmark.py
    cases.py
    metrics.py
    diagnose_control.py
```

Top-level legacy entrypoints may remain temporarily as thin wrappers so existing
commands and training-time watcher hooks do not break during migration.

## Evaluation Boundaries

### RootRefiner Evaluator

Input:

- text
- user route or dataset path condition
- root/history context

Output:

- predicted duration
- predicted normalized 5D waypoints
- physical 7D `RootPlan`

Metrics:

- duration top-k / MAE
- xyz ADE / FDE
- heading error
- fwd/yaw delta error
- smoothness

This evaluator must not load or run the LDF body model.

### LDF Evaluator

Input:

- text
- known 7D trajectory condition from dataset or controlled ablation
- optional no-traj branch

Output:

- latent sequence
- decoded body motion
- recovered root trajectory

Metrics:

- trajectory following
- motion quality / T2M metrics
- offline generation metrics
- LDF-only streaming metrics when the trajectory condition is already supplied

This evaluator may use `stream_generate()` / `stream_generate_step()`, but it
must not call RootRefiner or web-demo state management.

### Runtime Evaluator

Input:

- text schedule
- user route update events
- runtime settings matching the web demo

Pipeline:

```text
user route + text
  -> RootRefinerRuntime
  -> RootPlan 7D
  -> LDF stream_generate_step
  -> VAE stream_decode
  -> root recovery
  -> InferenceGlueTimeline advance
```

Metrics:

- closed-loop route ADE / FDE
- root drift
- yaw error
- update delay / blend correctness
- jitter and foot skating
- latency per token
- buffer underflow rate

This is the only eval layer that should measure the real web-demo runtime
contract.

## Common Abstractions

### EvalContext

Holds shared run state:

- config
- device
- seed
- output directory
- logger / rank metadata

### ModelBundle

Groups loaded components:

- LDF model
- VAE
- RootRefiner
- text encoder
- normalization stats

Each evaluator requests only the components it needs.

### EvalSample

Standard sample boundary:

- text
- ground-truth motion
- ground-truth root
- user path / route
- metadata

### EvalPrediction

Standard prediction boundary:

- motion
- root
- latent
- root plan
- debug state

### MetricResult

Standard output boundary:

- aggregate summary
- per-sample records
- artifact paths

## RuntimeRollout Boundary

The runtime evaluator should not duplicate web-demo internals. It should use a
testable rollout object:

```python
class RuntimeRollout:
    def reset(self, *, text: str, route=None): ...
    def update_text(self, text: str): ...
    def update_route(self, route, *, mode: str = "replace_future"): ...
    def step(self) -> RuntimeStepOutput: ...
    def run(self, num_tokens: int) -> RuntimeRolloutResult: ...
```

The rollout owns:

- `InferenceGlueTimeline`
- root 5D history for RootRefiner
- active/pending route events
- RootPlan activation
- LDF streaming step
- VAE stream decode
- root recovery

The web demo and runtime eval should share this boundary as much as practical.
The web demo adds frontend/socket state; the evaluator adds datasets, cases, and
metrics.

## Migration Plan

### Phase 1: Package Structure And Wrappers

- Create `eval/common`, `eval/ldf`, `eval/root_refiner`, and `eval/runtime`.
- Move or copy code behind new package modules.
- Keep old top-level scripts as wrappers importing the new modules.
- Add smoke tests that old entrypoints still import.

### Phase 2: Extract Common Loading And Artifacts

- Move duplicated config/device/seed/checkpoint loading into `eval/common`.
- Move summary and artifact writing into `eval/common`.
- Keep metric formulas unchanged.

### Phase 3: RootRefiner Standalone Cleanup

- Move `root_refiner_benchmark.py` to `eval/root_refiner/benchmark.py`.
- Split reusable metric helpers into `eval/root_refiner/metrics.py`.
- Keep CLI behavior compatible.

### Phase 4: LDF Eval Cleanup

- Move offline generation metrics into `eval/ldf/generation_metrics.py`.
- Move LDF-only stream metrics into `eval/ldf/stream_metrics.py`.
- Keep RootRefiner out of this layer.

### Phase 5: Runtime Rollout

- Introduce `eval/runtime/rollout.py`.
- Port runtime benchmark cases to use RootRefinerRuntime + RootPlan + LDF
  streaming path.
- Add end-to-end smoke tests with fake model/refiner/vae before using real
  checkpoints.

## Compatibility Rules

- Do not break current CLI commands during the first migration phase.
- Do not change metric definitions while moving files.
- Do not mix RootRefiner standalone metrics with runtime closed-loop metrics.
- Keep old legacy XYZ trajectory eval paths only as explicit ablations, not as
  the default runtime evaluator.
- Prefer small importable functions/classes over script-global logic.

## Success Criteria

- `eval/root_refiner/benchmark.py` can run RootRefiner-only evaluation.
- `eval/ldf/generation_metrics.py` can run the current LDF offline evaluation.
- `eval/runtime/benchmark.py` can run a closed-loop RootRefiner + LDF streaming
  smoke case.
- Old top-level commands either still work through wrappers or print a clear
  migration message.
- Tests cover import compatibility and at least one fake closed-loop runtime
  rollout.
