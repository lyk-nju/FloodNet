# Dataset NoiseInitializer Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `train_initializer.py`, a resumable dataset-level Stage 1 NoiseInitializer trainer over the first 300 `train_difficult.txt` samples with dynamic Gaussian roll-ins and a finite replay buffer.

**Architecture:** Reuse the frozen LDF/VAE execution and context-building code from the single-sample prototype, but move dataset training into focused contracts, collector, buffer, trainer, validation, and checkpoint modules. Replay snapshots own immutable original base noise and are stored on CPU; sampled snapshots are restored to GPU for effect-aligned differentiable rollouts.

**Tech Stack:** Python 3.10, PyTorch, OmegaConf, existing HumanML3D dataset/LDF/VAE stream APIs, pytest.

## Global Constraints

- Formal entry point is `train_initializer.py`; `tools/train_noise_initializer.py` remains diagnostic.
- Train split is `HumanML3D/train_difficult.txt`, limited to the first 300 loaded samples.
- Validation split is `HumanML3D/val.txt`, limited to the first 50 loaded samples.
- Stage 1 collection is Gaussian-only with a new checkpointed seed per roll-in.
- Collect two to four valid commits without replacement per roll-in.
- Every assignment is `original_base_zT + alpha * clipped_delta`; never accumulate on assigned noise.
- The pilot uses ten desired affected tokens and adds the first frontier offset to the rollout length.
- Replay payload tensors are stored on CPU.
- Do not modify Web/runtime files or unrelated dirty configuration files.

---

### Task 1: Immutable Replay Contracts And Objective Semantics

**Files:**
- Create: `utils/training/noise_initializer/contracts.py`
- Create: `utils/training/noise_initializer/objectives.py`
- Modify: `utils/training/noise_initializer/lightning_module.py`
- Test: `tests/test_noise_initializer_dataset_contracts.py`
- Modify: `tests/test_noise_initializer_lightning_module.py`

**Interfaces:**
- Produces: `DatasetReplaySnapshot`, `FrontierNoiseAssignment`, `tree_to_cpu`, `tree_to_device`.
- Produces: `resolve_effect_aligned_horizon(context, desired_affected_tokens) -> EffectAlignedHorizon`.
- Produces: `raw_delta_ratio_regularization(raw_delta, original_base) -> Tensor`.
- `NoiseInitializerLightningModule.training_step` consumes batch key `rollout_tokens` and loss key `lambda_raw_delta_ratio`.

- [ ] **Step 1: Write failing immutable-assignment and CPU-storage tests**

```python
def test_frontier_assignment_always_uses_immutable_original_base():
    base = torch.tensor([[[1.0, 2.0]]])
    state = FrontierNoiseAssignment(original_base_zT=base)
    first = state.assign(torch.tensor([[[0.5, 0.5]]]), version=1)
    second = state.assign(torch.tensor([[[-0.5, 1.0]]]), version=2)
    assert torch.equal(first, torch.tensor([[[1.5, 2.5]]]))
    assert torch.equal(second, torch.tensor([[[0.5, 3.0]]]))

def test_dataset_snapshot_moves_nested_tensors_to_cpu():
    snapshot = _snapshot_on_device("cpu")
    assert all(t.device.type == "cpu" for t in snapshot.tensors())
```

- [ ] **Step 2: Run contracts tests and verify import failures**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_noise_initializer_dataset_contracts.py -q`

Expected: collection fails because `contracts.py` does not exist.

- [ ] **Step 3: Implement frozen contracts and recursive tensor movement**

```python
@dataclass(frozen=True)
class FrontierNoiseAssignment:
    original_base_zT: torch.Tensor
    assigned_zT: torch.Tensor | None = None
    assignment_version: int = 0

    def assign(self, delta_zT: torch.Tensor, *, version: int, alpha: float = 1.0):
        return self.original_base_zT + float(alpha) * delta_zT

@dataclass(frozen=True)
class DatasetReplaySnapshot:
    sample_id: str
    caption_id: int
    seed: int
    commit_index: int
    target_tokens: int
    valid_affected_frames: int
    requested_affected_frames: int
    original_base_zT: torch.Tensor
    model_state: dict
    vae_state: object
    context: NoiseInitializerContext
    batch: dict
    conditioner: object
    recovery: object
    first_chunk: bool
```

- [ ] **Step 4: Write failing effect-aligned horizon tests**

```python
def test_effect_aligned_horizon_adds_frontier_offset():
    context = replace(_context(), frontier_offsets=torch.tensor([4, 5, 6, 7, 8]))
    horizon = resolve_effect_aligned_horizon(context, desired_affected_tokens=10)
    assert horizon.no_effect_tokens == 4
    assert horizon.rollout_tokens == 14
```

- [ ] **Step 5: Implement effect-aligned horizon and raw-ratio objective**

```python
@dataclass(frozen=True)
class EffectAlignedHorizon:
    no_effect_tokens: int
    desired_affected_tokens: int
    rollout_tokens: int

def raw_delta_ratio_regularization(raw_delta, original_base):
    raw = raw_delta.reshape(raw_delta.shape[0], -1).norm(dim=1)
    base = original_base.reshape(original_base.shape[0], -1).norm(dim=1)
    return (raw / base.clamp(min=1e-12)).pow(2).mean()
```

- [ ] **Step 6: Write failing Lightning saturation-gradient test**

```python
def test_raw_ratio_regularization_has_radial_gradient_after_clip_saturates():
    module = _module(max_delta_norm_ratio=0.1, lambda_raw_delta_ratio=1.0)
    loss = module.training_step(_batch(rollout_tokens=14), 0)
    loss.backward()
    assert module.initializer.scale.grad.abs().item() > 0
    assert module.last_step_diagnostics["raw_delta_ratio"] > 0.1
```

- [ ] **Step 7: Update Lightning to use batch rollout length and raw-ratio loss**

```python
rollout_tokens = int(batch.get(
    "rollout_tokens",
    rollout_cfg.get("loss_horizon_tokens", 1),
))
raw_ratio_reg = raw_delta_ratio_regularization(
    result.raw_delta_zT,
    batch.get("original_base_zT", batch["context"].frontier_base_zT),
)
loss = traj_loss + lambda_raw_delta_ratio * raw_ratio_reg
```

- [ ] **Step 8: Run focused tests**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_noise_initializer_dataset_contracts.py tests/test_noise_initializer_lightning_module.py -q`

Expected: all pass.

### Task 2: Finite Weighted Replay Buffer

**Files:**
- Create: `utils/training/noise_initializer/replay_buffer.py`
- Test: `tests/test_noise_initializer_replay_buffer.py`

**Interfaces:**
- Produces: `ReplayBufferConfig` and `NoiseInitializerReplayBuffer`.
- Consumes: `DatasetReplaySnapshot` from Task 1.
- Supports: `extend`, `sample`, `state_dict`, `load_state_dict`, `coverage`.

- [ ] **Step 1: Write failing FIFO, CPU, and weighted-sampling tests**

```python
def test_replay_buffer_evicts_oldest_and_keeps_cpu_snapshots():
    buffer = NoiseInitializerReplayBuffer(capacity=2, seed=7)
    buffer.extend([_snapshot("a", 0), _snapshot("b", 5), _snapshot("c", 10)])
    assert [item.sample_id for item in buffer.items] == ["b", "c"]
    assert buffer.eviction_count == 1

def test_late_commits_receive_configured_sampling_weight():
    buffer = NoiseInitializerReplayBuffer(
        capacity=10, seed=7, late_progress_start=0.55,
        late_weight_multiplier=3.0,
    )
    assert buffer.weight(_snapshot("a", 30, target_tokens=46)) == pytest.approx(3.0)
```

- [ ] **Step 2: Verify tests fail for missing module**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_noise_initializer_replay_buffer.py -q`

- [ ] **Step 3: Implement bounded deque storage and deterministic RNG**

```python
class NoiseInitializerReplayBuffer:
    def extend(self, snapshots):
        for snapshot in snapshots:
            cpu_snapshot = snapshot.to("cpu")
            if len(self._items) == self.capacity:
                self._items.popleft()
                self.eviction_count += 1
            self._items.append(cpu_snapshot)

    def sample(self):
        weights = [self.weight(item) for item in self._items]
        return self._rng.choices(tuple(self._items), weights=weights, k=1)[0]
```

- [ ] **Step 4: Implement state round trip and coverage diagnostics**

Coverage must report unique samples, captions, seeds, commits, early/late counts,
size, capacity, and eviction count. `state_dict` stores snapshots and
`random.Random.getstate()`.

- [ ] **Step 5: Run replay buffer tests**

Expected: all pass.

### Task 3: HumanML3D Dataset Builder

**Files:**
- Create: `utils/training/noise_initializer/dataset_builder.py`
- Test: `tests/test_noise_initializer_dataset_builder.py`

**Interfaces:**
- Produces: `build_initializer_datasets(cfg) -> (Dataset, Dataset)`.
- Produces: `normalize_initializer_sample(raw, caption_id) -> dict`.
- Reuses: `build_humanml3d_dataset_cfg` and `HumanML3DDataset`.

- [ ] **Step 1: Write failing split-limit and overlap tests**

```python
def test_dataset_builder_uses_difficult_train_and_deterministic_limits(monkeypatch):
    train, val = build_initializer_datasets(_cfg(max_train_samples=300, max_val_samples=50))
    assert len(train) == 300
    assert len(val) == 50
    assert train.split_file.name == "train_difficult.txt"
    assert val.split_file.name == "val.txt"

def test_dataset_builder_rejects_sample_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_split_identity(["a", "b"], ["b", "c"])
```

- [ ] **Step 2: Run test and confirm missing-module failure**

- [ ] **Step 3: Implement a deterministic limited dataset view**

```python
class LimitedDataset(Dataset):
    def __init__(self, dataset, limit, split_file):
        self.dataset = dataset
        self.indices = tuple(range(min(int(limit), len(dataset))))
        self.split_file = Path(split_file)
    def __len__(self): return len(self.indices)
    def __getitem__(self, index): return self.dataset[self.indices[index]]
```

- [ ] **Step 4: Normalize one raw sample into the existing single-sample stream batch contract**

The normalized batch must retain `name`, selected `caption_id`, text schedule,
263D feature, feature length, token length, `traj_cond_7d`, and trajectory mask.
Use the existing HumanML3D collate or a one-item collate helper rather than
manual padding.

- [ ] **Step 5: Run dataset builder tests**

Expected: all pass.

### Task 4: Dynamic Gaussian Snapshot Collector

**Files:**
- Create: `utils/training/noise_initializer/collector.py`
- Modify: `utils/training/noise_initializer/replay_runner.py`
- Test: `tests/test_noise_initializer_collector.py`

**Interfaces:**
- Produces: `CollectorConfig`, `DynamicGaussianCollector.collect(sample_batch) -> list[DatasetReplaySnapshot]`.
- Produces: `state_dict/load_state_dict` for exact seed and commit-selection resume.
- Reuses a refactored one-seed Gaussian collection primitive from `replay_runner.py`.

- [ ] **Step 1: Write failing dynamic-seed and commit-selection tests**

```python
def test_collector_draws_new_seed_and_two_to_four_commits():
    collector = _collector(seed=123)
    first = collector.collect(_sample("a"))
    second = collector.collect(_sample("a"))
    assert first[0].seed != second[0].seed
    assert 2 <= len(first) <= 4
    assert len({item.commit_index for item in first}) == len(first)
```

- [ ] **Step 2: Write failing exact-resume RNG test**

```python
state = collector.state_dict()
expected = collector.next_seed_and_count()
restored = _collector(seed=999)
restored.load_state_dict(state)
assert restored.next_seed_and_count() == expected
```

- [ ] **Step 3: Extract a one-seed all-valid-snapshots primitive**

Refactor `collect_replay_snapshots` so dataset collection can request one seed
and receive all valid decision snapshots without changing existing two-stage
behavior.

- [ ] **Step 4: Implement random selection without replacement and immutable source capture**

```python
count = self._rng.randint(min_commits, min(max_commits, len(candidates)))
selected = self._rng.sample(candidates, k=count)
return [
    DatasetReplaySnapshot.from_diagnostic(
        item,
        sample_id=sample_id,
        caption_id=caption_id,
        target_tokens=target_tokens,
    )
    for item in selected
]
```

- [ ] **Step 5: Verify context base noise equals stored original base noise**

The collector must clone `context.frontier_base_zT` before any assignment and
store it as `original_base_zT`.

- [ ] **Step 6: Run collector and existing replay tests**

Run: `/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_noise_initializer_collector.py tests/test_noise_initializer_snapshot_replay.py -q`

### Task 5: Checkpointing And Dataset Trainer

**Files:**
- Create: `utils/training/noise_initializer/checkpointing.py`
- Create: `utils/training/noise_initializer/trainer.py`
- Test: `tests/test_noise_initializer_checkpointing.py`
- Test: `tests/test_noise_initializer_dataset_trainer.py`

**Interfaces:**
- Produces: `NoiseInitializerCheckpointManager`.
- Produces: `NoiseInitializerDatasetTrainer.fit()` and `train_one_step(snapshot)`.
- Consumes dataset, collector, replay buffer, Lightning objective wrapper, frozen model/VAE, and optimizer.

- [ ] **Step 1: Write failing atomic checkpoint round-trip test**

```python
manager.save_latest(trainer.state_dict())
restored = manager.load_latest()
assert restored["global_step"] == 17
assert restored["collector"]["rng_state"] == state["collector"]["rng_state"]
assert not (tmp_path / "latest.pt.tmp").exists()
```

- [ ] **Step 2: Implement atomic latest/best/failure checkpoint writes**

Write to `*.tmp`, flush via `torch.save`, and replace with `Path.replace`.
Checkpoint initializer, optimizer, RNG, collector, replay buffer, counters,
configuration, and validation history.

- [ ] **Step 3: Write failing alternating-cycle and logging tests**

```python
trainer.fit(max_steps=2)
assert collector.calls == 8
assert trainer.global_step == 2
assert len(buffer) >= 16
assert logs[0]["replay_size"] == len(buffer)
```

- [ ] **Step 4: Implement collection cycles and one-snapshot optimizer updates**

Restore model/VAE state, move the sampled snapshot to device, install the
effect-aligned shadow rollout callback, backpropagate only initializer params,
and record finite diagnostics. A non-finite loss or gradient writes a failure
checkpoint and raises.

- [ ] **Step 5: Implement exact trainer state restoration**

Resume must continue the dataset cursor, collector seed stream, replay sample
stream, optimizer, and global step.

- [ ] **Step 6: Run trainer/checkpoint tests**

Expected: all pass.

### Task 6: Held-Out Validation

**Files:**
- Create: `utils/training/noise_initializer/validation.py`
- Test: `tests/test_noise_initializer_validation.py`

**Interfaces:**
- Produces: `NoiseInitializerValidator.evaluate(initializer) -> dict`.
- Produces aggregate and per-sample Gaussian, alpha-zero, and configured-alpha metrics.

- [ ] **Step 1: Write failing alpha-zero and harm-rate aggregation tests**

```python
result = aggregate_validation_rows(_rows())
assert result["alpha_zero_max_l2"] == 0.0
assert result["sample_ade_harm_rate"] == pytest.approx(0.25)
assert result["mean_delta_ade"] < 0
```

- [ ] **Step 2: Implement fixed sample/seed schedule and aggregate metrics**

Use configured validation seeds, never the collector RNG. Preserve per-sample
ADE/FDE/MSE/heading/path ratio, affected windows, clip statistics, and alpha-zero
diffs.

- [ ] **Step 3: Add best-checkpoint eligibility validation**

Reject non-finite metrics and any alpha-zero max L2 above `1e-7`. Return mean
ADE as the selection metric only when eligibility checks pass.

- [ ] **Step 4: Run validation tests**

Expected: all pass.

### Task 7: Formal Entry Point, Configuration, And Integration Smoke

**Files:**
- Create: `train_initializer.py`
- Create: `configs/noise_initializer_train.yaml`
- Modify: `utils/training/noise_initializer/config_validate.py`
- Modify: `utils/training/noise_initializer/__init__.py`
- Test: `tests/test_train_initializer_entrypoint.py`

**Interfaces:**
- `python train_initializer.py --config configs/noise_initializer_train.yaml --override training.max_steps=10000` launches or resumes formal training.

- [ ] **Step 1: Write failing configuration and dry-build tests**

```python
cfg = load_initializer_train_config("configs/noise_initializer_train.yaml")
assert cfg["data"]["train_split_file"] == "train_difficult.txt"
assert cfg["data"]["max_train_samples"] == 300
assert cfg["training"]["stage"] == "gaussian"
```

- [ ] **Step 2: Implement formal config with explicit pilot defaults**

Set buffer capacity 10,000; collect 8 roll-ins; train 32 steps per cycle;
desired affected tokens 10; log every 50; validate/checkpoint every 500; fixed
validation seeds; Stage 1 only.

- [ ] **Step 3: Implement a thin `train_initializer.py`**

```python
def main():
    args = parse_args()
    cfg = load_initializer_train_config(args.config, args.override)
    runtime = build_frozen_initializer_runtime(cfg)
    train_data, val_data = build_initializer_datasets(cfg)
    trainer = build_dataset_trainer(cfg, runtime, train_data, val_data)
    trainer.fit(resume_path=cfg.get("resume_ckpt"))
```

- [ ] **Step 4: Run all NoiseInitializer unit tests and py_compile**

Run all `tests/test_noise_initializer_*.py`, `tests/test_train_initializer_entrypoint.py`, and compile every new module.

- [ ] **Step 5: Run a GPU integration smoke**

Override to two training samples, two validation samples, one collection
roll-in, two optimizer steps, buffer capacity eight, and validation at step two.
Verify nonzero initializer gradients, zero frozen-model gradients, CPU replay
storage, alpha-zero equivalence, and latest checkpoint creation.

- [ ] **Step 6: Run resume smoke**

Resume the smoke checkpoint for one additional optimizer step and verify the
next collector seed and sampled snapshot match an uninterrupted three-step run.

- [ ] **Step 7: Review task-owned diff only**

Run `git diff --check`, confirm Web/runtime dirty files are untouched, and
summarize the exact pilot launch command without starting the 300-sample run.
