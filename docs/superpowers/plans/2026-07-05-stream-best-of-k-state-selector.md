# Stream Best-of-K State Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current latent argmin best-of-K eval path with a serial state-fork runtime selector that keeps K=1 unchanged and conservatively switches only when a full candidate state is clearly better.

**Architecture:** Keep `run_stream_generate_step_sample` as the public entrypoint and make `best_of_k <= 1` return through the original branch. Add focused modules for state capture, one-step candidate output, scoring/gating, and serial selector orchestration. The v1 selector keeps eval return buffers outside snapshots, so candidate rollout returns local `StepOutput` objects and the outer loop appends selected chunks exactly once.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest, existing FloodNet LDF eval/runtime helpers.

---

## File Structure

- Create `eval/ldf/stream_state.py`
  - Owns `StepOutput`, `CandidateState`, `StreamRuntimeSnapshot`, RNG capture/restore, deep clone helpers, and model/VAE/conditioner state restore.
- Create `eval/ldf/stream_scoring.py`
  - Owns `ConservativeGateConfig`, score dataclasses, root XZ/continuity scoring, and conservative switch decision.
- Create `eval/ldf/stream_best_of_k.py`
  - Owns `StreamBestOfKConfig`, serial candidate generation, force-candidate0 mode, and selected-state restore.
- Modify `eval/ldf/stream_generation.py`
  - Remove the old batch latent argmin helpers from the public path.
  - Preserve K=1 branch exactly.
  - Call `run_best_of_k_step` only when `best_of_k > 1`.
- Modify `utils/training/ldf/validation_eval_runtime.py`
  - Add the new selector config defaults.
- Modify `utils/training/ldf/validation_generation.py`
  - Pass the new selector config through validation eval.
- Modify `eval/ldf/stream_metrics.py`
  - Parse and pass the new config fields.
  - Include selector debug records only when debug is enabled.
- Modify `tools/sweep_stream_step_cfg.py`
  - Expose the new fields for quick 000021 experiments.
- Create `tests/test_stream_best_of_k.py`
  - Focused tests for state snapshotting, scoring/gating, K=1 bypass, force-candidate0, and transaction-safe append.
- Modify `tests/test_validation_eval_interface.py`
  - Extend config default/override coverage.
- Modify `tests/test_stream_eval_metrics.py`
  - Update or replace the old batch-argmin test so it matches state-fork semantics.

---

### Task 1: Config Surface

**Files:**
- Modify: `utils/training/ldf/validation_eval_runtime.py`
- Modify: `utils/training/ldf/validation_generation.py`
- Modify: `eval/ldf/stream_metrics.py`
- Modify: `tools/sweep_stream_step_cfg.py`
- Test: `tests/test_validation_eval_interface.py`

- [ ] **Step 1: Extend config tests first**

Add these assertions to `tests/test_validation_eval_interface.py::test_generation_eval_cfg_exposes_stream_best_of_k_defaults_and_overrides`:

```python
assert default_cfg["stream_best_of_k_vel_weight"] == 0.5
assert default_cfg["stream_best_of_k_rel_margin"] == 0.10
assert default_cfg["stream_best_of_k_abs_margin"] == 0.03
assert default_cfg["stream_best_of_k_cont_tol"] == 0.03
assert default_cfg["stream_best_of_k_force_candidate0"] is False
assert default_cfg["stream_best_of_k_switch_cooldown_steps"] == 0
```

Add these override inputs to the custom config in the same test:

```python
"eval_stream_best_of_k_vel_weight": 0.75,
"eval_stream_best_of_k_rel_margin": 0.20,
"eval_stream_best_of_k_abs_margin": 0.05,
"eval_stream_best_of_k_cont_tol": 0.07,
"eval_stream_best_of_k_force_candidate0": True,
"eval_stream_best_of_k_switch_cooldown_steps": 2,
```

Add these assertions to the custom config block:

```python
assert custom_cfg["stream_best_of_k_vel_weight"] == 0.75
assert custom_cfg["stream_best_of_k_rel_margin"] == 0.20
assert custom_cfg["stream_best_of_k_abs_margin"] == 0.05
assert custom_cfg["stream_best_of_k_cont_tol"] == 0.07
assert custom_cfg["stream_best_of_k_force_candidate0"] is True
assert custom_cfg["stream_best_of_k_switch_cooldown_steps"] == 2
```

- [ ] **Step 2: Run the config test and verify it fails**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_validation_eval_interface.py::test_generation_eval_cfg_exposes_stream_best_of_k_defaults_and_overrides -q
```

Expected: FAIL with missing keys such as `stream_best_of_k_vel_weight`.

- [ ] **Step 3: Add config defaults to validation runtime**

In `utils/training/ldf/validation_eval_runtime.py`, extend `build_generation_eval_cfg` with:

```python
"stream_best_of_k_vel_weight": float(
    validation_cfg.get("eval_stream_best_of_k_vel_weight", 0.5)
),
"stream_best_of_k_rel_margin": float(
    validation_cfg.get("eval_stream_best_of_k_rel_margin", 0.10)
),
"stream_best_of_k_abs_margin": float(
    validation_cfg.get("eval_stream_best_of_k_abs_margin", 0.03)
),
"stream_best_of_k_cont_tol": float(
    validation_cfg.get("eval_stream_best_of_k_cont_tol", 0.03)
),
"stream_best_of_k_force_candidate0": bool(
    validation_cfg.get("eval_stream_best_of_k_force_candidate0", False)
),
"stream_best_of_k_switch_cooldown_steps": int(
    validation_cfg.get("eval_stream_best_of_k_switch_cooldown_steps", 0)
),
```

- [ ] **Step 4: Pass config through validation generation**

In `utils/training/ldf/validation_generation.py`, add parameters to `generate_t2m_sample`:

```python
stream_best_of_k_vel_weight: float = 0.5,
stream_best_of_k_rel_margin: float = 0.10,
stream_best_of_k_abs_margin: float = 0.03,
stream_best_of_k_cont_tol: float = 0.03,
stream_best_of_k_force_candidate0: bool = False,
stream_best_of_k_switch_cooldown_steps: int = 0,
```

Pass them into `run_stream_generate_step_sample`:

```python
best_of_k_vel_weight=float(stream_best_of_k_vel_weight),
best_of_k_rel_margin=float(stream_best_of_k_rel_margin),
best_of_k_abs_margin=float(stream_best_of_k_abs_margin),
best_of_k_cont_tol=float(stream_best_of_k_cont_tol),
best_of_k_force_candidate0=bool(stream_best_of_k_force_candidate0),
best_of_k_switch_cooldown_steps=int(stream_best_of_k_switch_cooldown_steps),
```

In the validation loop, read the new keys from `eval_cfg` and pass them to `generate_t2m_sample`.

- [ ] **Step 5: Parse config in stream metrics**

In `eval/ldf/stream_metrics.py`, add reads near the existing best-of-K config:

```python
best_of_k_vel_weight = float(
    cfg.get("eval.stream_best_of_k_vel_weight", cfg.get("eval_stream_best_of_k_vel_weight", 0.5))
)
best_of_k_rel_margin = float(
    cfg.get("eval.stream_best_of_k_rel_margin", cfg.get("eval_stream_best_of_k_rel_margin", 0.10))
)
best_of_k_abs_margin = float(
    cfg.get("eval.stream_best_of_k_abs_margin", cfg.get("eval_stream_best_of_k_abs_margin", 0.03))
)
best_of_k_cont_tol = float(
    cfg.get("eval.stream_best_of_k_cont_tol", cfg.get("eval_stream_best_of_k_cont_tol", 0.03))
)
best_of_k_force_candidate0 = bool(
    cfg.get("eval.stream_best_of_k_force_candidate0", cfg.get("eval_stream_best_of_k_force_candidate0", False))
)
best_of_k_switch_cooldown_steps = int(
    cfg.get("eval.stream_best_of_k_switch_cooldown_steps", cfg.get("eval_stream_best_of_k_switch_cooldown_steps", 0))
)
```

Pass these to `run_stream_generate_step_sample`.

- [ ] **Step 6: Add CLI flags to sweep script**

In `tools/sweep_stream_step_cfg.py`, add parser arguments:

```python
parser.add_argument("--best_of_k_vel_weight", type=float, default=0.5)
parser.add_argument("--best_of_k_rel_margin", type=float, default=0.10)
parser.add_argument("--best_of_k_abs_margin", type=float, default=0.03)
parser.add_argument("--best_of_k_cont_tol", type=float, default=0.03)
parser.add_argument("--best_of_k_force_candidate0", action="store_true")
parser.add_argument("--best_of_k_switch_cooldown_steps", type=int, default=0)
```

Pass the values into `run_stream_generate_step_sample` with the same keyword names.

- [ ] **Step 7: Run config tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_validation_eval_interface.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit config surface**

Run:

```bash
git add utils/training/ldf/validation_eval_runtime.py utils/training/ldf/validation_generation.py eval/ldf/stream_metrics.py tools/sweep_stream_step_cfg.py tests/test_validation_eval_interface.py
git commit -m "feat: add stream best-of-k selector config"
```

---

### Task 2: Scoring and Conservative Gate

**Files:**
- Create: `eval/ldf/stream_scoring.py`
- Create: `tests/test_stream_best_of_k.py`

- [ ] **Step 1: Write scoring tests**

Create `tests/test_stream_best_of_k.py` with:

```python
from __future__ import annotations

import torch

from eval.ldf.stream_scoring import (
    ConservativeGateConfig,
    CandidateScore,
    choose_candidate,
    compute_chunk_xz_score,
    compute_continuity_score,
)


def test_conservative_gate_keeps_candidate0_when_improvement_is_small():
    cfg = ConservativeGateConfig(
        fde_weight=1.0,
        vel_weight=0.5,
        rel_margin=0.10,
        abs_margin=0.03,
        cont_tol=0.03,
        force_candidate0=False,
        switch_cooldown_steps=0,
    )
    scores = [
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
        CandidateScore(index=1, xz_ade=0.19, xz_fde=0.19, pos_cont=0.01, vel_cont=0.01),
    ]

    decision = choose_candidate(scores, cfg, steps_since_switch=100)

    assert decision.selected_index == 0
    assert decision.reason == "keep_candidate0"
    assert decision.candidate0_track == 0.40


def test_conservative_gate_switches_for_clear_tracking_gain_with_good_continuity():
    cfg = ConservativeGateConfig(
        fde_weight=1.0,
        vel_weight=0.5,
        rel_margin=0.10,
        abs_margin=0.03,
        cont_tol=0.03,
        force_candidate0=False,
        switch_cooldown_steps=0,
    )
    scores = [
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
        CandidateScore(index=1, xz_ade=0.10, xz_fde=0.10, pos_cont=0.02, vel_cont=0.01),
    ]

    decision = choose_candidate(scores, cfg, steps_since_switch=100)

    assert decision.selected_index == 1
    assert decision.reason == "switch_improved_continuous"
    assert decision.selected_track == 0.20


def test_conservative_gate_rejects_bad_continuity():
    cfg = ConservativeGateConfig(
        fde_weight=1.0,
        vel_weight=0.5,
        rel_margin=0.10,
        abs_margin=0.03,
        cont_tol=0.03,
        force_candidate0=False,
        switch_cooldown_steps=0,
    )
    scores = [
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
        CandidateScore(index=1, xz_ade=0.05, xz_fde=0.05, pos_cont=0.20, vel_cont=0.20),
    ]

    decision = choose_candidate(scores, cfg, steps_since_switch=100)

    assert decision.selected_index == 0
    assert decision.reason == "keep_candidate0"


def test_force_candidate0_overrides_better_candidates():
    cfg = ConservativeGateConfig(
        fde_weight=1.0,
        vel_weight=0.5,
        rel_margin=0.10,
        abs_margin=0.03,
        cont_tol=0.03,
        force_candidate0=True,
        switch_cooldown_steps=0,
    )
    scores = [
        CandidateScore(index=0, xz_ade=0.20, xz_fde=0.20, pos_cont=0.01, vel_cont=0.01),
        CandidateScore(index=1, xz_ade=0.01, xz_fde=0.01, pos_cont=0.01, vel_cont=0.01),
    ]

    decision = choose_candidate(scores, cfg, steps_since_switch=100)

    assert decision.selected_index == 0
    assert decision.reason == "force_candidate0"


def test_chunk_xz_score_uses_ade_and_fde():
    pred = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0], [1.5, 0.0], [1.0, 0.0]], dtype=torch.float32)

    ade, fde = compute_chunk_xz_score(pred, target)

    assert torch.isclose(torch.tensor(ade), torch.tensor(0.5))
    assert torch.isclose(torch.tensor(fde), torch.tensor(1.0))


def test_continuity_score_uses_position_and_velocity_jump():
    prev = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    candidate = torch.tensor([[1.2, 0.0], [2.5, 0.0]], dtype=torch.float32)

    pos_jump, vel_jump = compute_continuity_score(prev, candidate)

    assert torch.isclose(torch.tensor(pos_jump), torch.tensor(0.2), atol=1e-6)
    assert torch.isclose(torch.tensor(vel_jump), torch.tensor(0.3), atol=1e-6)
```

- [ ] **Step 2: Run scoring tests and verify they fail**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'eval.ldf.stream_scoring'`.

- [ ] **Step 3: Implement scoring module**

Create `eval/ldf/stream_scoring.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConservativeGateConfig:
    fde_weight: float = 1.0
    vel_weight: float = 0.5
    rel_margin: float = 0.10
    abs_margin: float = 0.03
    cont_tol: float = 0.03
    force_candidate0: bool = False
    switch_cooldown_steps: int = 0


@dataclass(frozen=True)
class CandidateScore:
    index: int
    xz_ade: float
    xz_fde: float
    pos_cont: float
    vel_cont: float

    def track(self, cfg: ConservativeGateConfig) -> float:
        return float(self.xz_ade) + float(cfg.fde_weight) * float(self.xz_fde)

    def continuity(self, cfg: ConservativeGateConfig) -> float:
        return float(self.pos_cont) + float(cfg.vel_weight) * float(self.vel_cont)


@dataclass(frozen=True)
class SwitchDecision:
    selected_index: int
    reason: str
    selected_track: float
    candidate0_track: float
    selected_continuity: float
    candidate0_continuity: float


def compute_chunk_xz_score(pred_xz: torch.Tensor, target_xz: torch.Tensor) -> tuple[float, float]:
    valid = min(int(pred_xz.shape[0]), int(target_xz.shape[0]))
    if valid <= 0:
        return 0.0, 0.0
    pred = pred_xz[:valid].float().cpu()
    target = target_xz[:valid].float().cpu()
    per_frame = torch.linalg.norm(pred - target, dim=-1)
    return float(per_frame.mean().item()), float(per_frame[-1].item())


def compute_continuity_score(previous_xz: torch.Tensor | None, candidate_xz: torch.Tensor) -> tuple[float, float]:
    if previous_xz is None or int(previous_xz.shape[0]) == 0 or int(candidate_xz.shape[0]) == 0:
        return 0.0, 0.0
    prev = previous_xz.float().cpu()
    cand = candidate_xz.float().cpu()
    pos_jump = float(torch.linalg.norm(cand[0] - prev[-1]).item())
    if int(prev.shape[0]) < 2 or int(cand.shape[0]) < 2:
        return pos_jump, 0.0
    prev_vel = prev[-1] - prev[-2]
    cand_vel = cand[1] - cand[0]
    vel_jump = float(torch.linalg.norm(cand_vel - prev_vel).item())
    return pos_jump, vel_jump


def choose_candidate(
    scores: list[CandidateScore],
    cfg: ConservativeGateConfig,
    *,
    steps_since_switch: int,
) -> SwitchDecision:
    if not scores:
        raise ValueError("choose_candidate requires at least candidate 0")
    candidate0 = scores[0]
    track0 = candidate0.track(cfg)
    cont0 = candidate0.continuity(cfg)
    if bool(cfg.force_candidate0):
        return SwitchDecision(0, "force_candidate0", track0, track0, cont0, cont0)
    if int(cfg.switch_cooldown_steps) > 0 and int(steps_since_switch) < int(cfg.switch_cooldown_steps):
        return SwitchDecision(0, "cooldown", track0, track0, cont0, cont0)

    best = candidate0
    best_track = track0
    best_cont = cont0
    reason = "keep_candidate0"
    margin = max(float(cfg.abs_margin), float(cfg.rel_margin) * track0)
    for score in scores[1:]:
        track = score.track(cfg)
        cont = score.continuity(cfg)
        improve_enough = track < track0 - margin
        continuity_ok = cont <= cont0 + float(cfg.cont_tol)
        if improve_enough and continuity_ok and track < best_track:
            best = score
            best_track = track
            best_cont = cont
            reason = "switch_improved_continuous"
    return SwitchDecision(
        selected_index=int(best.index),
        reason=reason,
        selected_track=float(best_track),
        candidate0_track=float(track0),
        selected_continuity=float(best_cont),
        candidate0_continuity=float(cont0),
    )
```

- [ ] **Step 4: Run scoring tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py -q
```

Expected: PASS for the scoring tests in this file.

- [ ] **Step 5: Commit scoring**

Run:

```bash
git add eval/ldf/stream_scoring.py tests/test_stream_best_of_k.py
git commit -m "feat: add conservative stream best-of-k scoring"
```

---

### Task 3: Runtime Snapshot and StepOutput

**Files:**
- Create: `eval/ldf/stream_state.py`
- Modify: `tests/test_stream_best_of_k.py`

- [ ] **Step 1: Add snapshot tests**

Append to `tests/test_stream_best_of_k.py`:

```python
import copy
import random
import numpy as np

from eval.ldf.stream_state import (
    StepOutput,
    capture_runtime_snapshot,
    restore_runtime_snapshot,
)


class _StatefulVAEModel:
    def __init__(self):
        self._conv_num = 1
        self._conv_idx = [3]
        self._feat_map = [torch.tensor([1.0, 2.0])]


class _StatefulVAE:
    def __init__(self):
        self.model = _StatefulVAEModel()


class _StatefulConditioner:
    def __init__(self):
        self.timeline = {"head": torch.tensor([1.0])}
        self.root_plan = {"anchor": torch.tensor([2.0])}


class _StatefulModel:
    def __init__(self):
        self.generated = torch.ones(1, 4, 5, 1, 1)
        self.commit_index = 2
        self.current_step = 7
        self.batch_size = 1
        self.seq_len = 4
        self.num_denoise_steps = 10
        self.dt = 0.1
        self.text_condition_list = [[torch.tensor([3.0])]]
        self._traj_buf = {"value": torch.tensor([4.0])}


def test_runtime_snapshot_deep_clones_mutable_state():
    model = _StatefulModel()
    vae = _StatefulVAE()
    conditioner = _StatefulConditioner()

    snapshot = capture_runtime_snapshot(
        model=model,
        vae=vae,
        stream_conditioner=conditioner,
        first_chunk=False,
        generated_frames=12,
        chunk_frame_ends=[4, 8, 12],
    )
    model.generated.zero_()
    model.text_condition_list[0][0].fill_(9.0)
    vae.model._feat_map[0].fill_(8.0)
    conditioner.timeline["head"].fill_(7.0)

    restore_runtime_snapshot(model=model, vae=vae, stream_conditioner=conditioner, snapshot=snapshot)

    assert torch.allclose(model.generated, torch.ones(1, 4, 5, 1, 1))
    assert torch.allclose(model.text_condition_list[0][0], torch.tensor([3.0]))
    assert torch.allclose(vae.model._feat_map[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(conditioner.timeline["head"], torch.tensor([1.0]))
    assert snapshot.first_chunk is False
    assert snapshot.generated_frames == 12
    assert snapshot.chunk_frame_ends == [4, 8, 12]
    assert model.generated.data_ptr() != snapshot.model_state["generated"].data_ptr()


def test_runtime_snapshot_restores_rng_state():
    model = _StatefulModel()
    vae = _StatefulVAE()
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    snapshot = capture_runtime_snapshot(
        model=model,
        vae=vae,
        stream_conditioner=None,
        first_chunk=True,
        generated_frames=0,
        chunk_frame_ends=[],
    )
    expected_py = random.random()
    expected_np = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_runtime_snapshot(model=model, vae=vae, stream_conditioner=None, snapshot=snapshot)

    assert random.random() == expected_py
    assert float(np.random.rand()) == expected_np
    assert float(torch.rand(1).item()) == expected_torch


def test_step_output_records_explicit_ranges():
    output = StepOutput(
        clean_committed_latent=torch.zeros(1, 4),
        decoded_chunk=torch.zeros(4, 263),
        commit_token_range=(2, 3),
        commit_frame_range=(8, 12),
        decoded_chunk_frame_range=(8, 12),
        target_xz_frame_range=(8, 12),
        debug={"ready_to_commit_token": 2},
    )

    assert output.commit_token_range == (2, 3)
    assert output.commit_frame_range == (8, 12)
    assert output.debug["ready_to_commit_token"] == 2
```

- [ ] **Step 2: Run snapshot tests and verify they fail**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'eval.ldf.stream_state'`.

- [ ] **Step 3: Implement stream state module**

Create `eval/ldf/stream_state.py` with:

```python
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class StepOutput:
    clean_committed_latent: torch.Tensor
    decoded_chunk: torch.Tensor
    commit_token_range: tuple[int, int]
    commit_frame_range: tuple[int, int]
    decoded_chunk_frame_range: tuple[int, int]
    target_xz_frame_range: tuple[int, int]
    debug: dict[str, Any]


@dataclass(frozen=True)
class StreamRuntimeSnapshot:
    model_state: dict[str, Any]
    vae_state: dict[str, Any] | None
    conditioner_state: dict[str, Any] | None
    rng_state: dict[str, Any]
    first_chunk: bool
    generated_frames: int
    chunk_frame_ends: list[int]


@dataclass(frozen=True)
class CandidateState:
    index: int
    step_output: StepOutput
    snapshot: StreamRuntimeSnapshot


def deep_clone_state(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [deep_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(deep_clone_state(item) for item in value)
    if isinstance(value, dict):
        return {key: deep_clone_state(item) for key, item in value.items()}
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _capture_model_state(model) -> dict[str, Any]:
    names = (
        "generated",
        "commit_index",
        "current_step",
        "batch_size",
        "seq_len",
        "num_denoise_steps",
        "dt",
        "text_condition_list",
        "_traj_buf",
    )
    return {
        name: deep_clone_state(getattr(model, name))
        for name in names
        if hasattr(model, name)
    }


def _restore_model_state(model, state: dict[str, Any]) -> None:
    for name, value in state.items():
        setattr(model, name, deep_clone_state(value))


def _capture_vae_state(vae) -> dict[str, Any] | None:
    model = getattr(vae, "model", None)
    if model is None:
        return None
    names = ("_conv_num", "_conv_idx", "_feat_map")
    return {
        name: deep_clone_state(getattr(model, name))
        for name in names
        if hasattr(model, name)
    }


def _restore_vae_state(vae, state: dict[str, Any] | None) -> None:
    if state is None:
        return
    model = getattr(vae, "model", None)
    if model is None:
        return
    for name, value in state.items():
        setattr(model, name, deep_clone_state(value))


def _capture_conditioner_state(stream_conditioner) -> dict[str, Any] | None:
    if stream_conditioner is None:
        return None
    names = ("timeline", "root_plan", "_anchor_xz", "_anchor_yaw")
    return {
        name: deep_clone_state(getattr(stream_conditioner, name))
        for name in names
        if hasattr(stream_conditioner, name)
    }


def _restore_conditioner_state(stream_conditioner, state: dict[str, Any] | None) -> None:
    if stream_conditioner is None or state is None:
        return
    for name, value in state.items():
        setattr(stream_conditioner, name, deep_clone_state(value))


def capture_runtime_snapshot(
    *,
    model,
    vae,
    stream_conditioner,
    first_chunk: bool,
    generated_frames: int,
    chunk_frame_ends: list[int],
) -> StreamRuntimeSnapshot:
    return StreamRuntimeSnapshot(
        model_state=_capture_model_state(model),
        vae_state=_capture_vae_state(vae),
        conditioner_state=_capture_conditioner_state(stream_conditioner),
        rng_state=_capture_rng_state(),
        first_chunk=bool(first_chunk),
        generated_frames=int(generated_frames),
        chunk_frame_ends=list(chunk_frame_ends),
    )


def restore_runtime_snapshot(*, model, vae, stream_conditioner, snapshot: StreamRuntimeSnapshot) -> None:
    _restore_model_state(model, snapshot.model_state)
    _restore_vae_state(vae, snapshot.vae_state)
    _restore_conditioner_state(stream_conditioner, snapshot.conditioner_state)
    _restore_rng_state(snapshot.rng_state)
```

- [ ] **Step 4: Run snapshot tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit stream state module**

Run:

```bash
git add eval/ldf/stream_state.py tests/test_stream_best_of_k.py
git commit -m "feat: add stream runtime state snapshots"
```

---

### Task 4: One-Step Runtime Wrapper and K=1 Bypass

**Files:**
- Create: `eval/ldf/stream_best_of_k.py`
- Modify: `eval/ldf/stream_generation.py`
- Modify: `tests/test_stream_best_of_k.py`
- Modify: `tests/test_stream_eval_metrics.py`

- [ ] **Step 1: Add K=1 bypass test**

Append to `tests/test_stream_best_of_k.py`:

```python
def test_best_of_k_disabled_uses_original_single_step_path(monkeypatch):
    import eval.ldf.stream_generation as stream_generation

    called = {"selector": 0}

    def _selector_should_not_run(**kwargs):
        called["selector"] += 1
        raise AssertionError("best_of_k <= 1 must bypass selector")

    monkeypatch.setattr(stream_generation, "run_best_of_k_step", _selector_should_not_run, raising=False)

    assert stream_generation._should_use_best_of_k(1) is False
    assert stream_generation._should_use_best_of_k(0) is False
    assert called["selector"] == 0
```

- [ ] **Step 2: Add StepOutput helper test with fake model**

Append to `tests/test_stream_best_of_k.py`:

```python
class _OneStepModel(_StatefulModel):
    input_dim = 4
    chunk_size = 1

    def stream_generate_step(self, step_payload, first_chunk=True, condition=None):
        self.commit_index += 1
        latent = torch.full((1, 1, self.input_dim), float(self.commit_index))
        return {"generated": latent}


class _OneStepVAE:
    def __init__(self):
        self.model = _StatefulVAEModel()
        self.calls = []

    def stream_decode(self, latent, first_chunk=True):
        self.calls.append((latent.detach().clone(), bool(first_chunk)))
        frames = torch.zeros(1, 4, 263, dtype=torch.float32)
        frames[:, :, 0] = latent[:, :, 0].view(1, 1)
        return frames


class _OneStepStream:
    def build_ldf_condition_provider(self, step_payload, first_chunk=True, device=None):
        return lambda **kwargs: None


def test_run_one_stream_step_returns_clean_commit_ranges():
    from eval.ldf.stream_best_of_k import run_one_stream_step

    model = _OneStepModel()
    vae = _OneStepVAE()
    output = run_one_stream_step(
        model=model,
        vae=vae,
        stream=_OneStepStream(),
        step_payload={"text": "walk"},
        first_chunk=True,
        device=torch.device("cpu"),
        local_commit_index=0,
        generated_frames=0,
        frames_per_token=4,
    )

    assert output.clean_committed_latent.shape == (1, 4)
    assert output.decoded_chunk.shape == (4, 263)
    assert output.commit_token_range == (0, 1)
    assert output.commit_frame_range == (0, 4)
    assert output.decoded_chunk_frame_range == (0, 4)
    assert output.debug["ready_to_commit_token"] == 0
```

- [ ] **Step 3: Run new tests and verify they fail**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py::test_best_of_k_disabled_uses_original_single_step_path tests/test_stream_best_of_k.py::test_run_one_stream_step_returns_clean_commit_ranges -q
```

Expected: FAIL because `_should_use_best_of_k` and `eval.ldf.stream_best_of_k` do not exist.

- [ ] **Step 4: Implement one-step wrapper**

Create the initial `eval/ldf/stream_best_of_k.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from eval.ldf.stream_state import StepOutput


@dataclass(frozen=True)
class StreamBestOfKConfig:
    k: int = 1
    score: str = "xz"
    xz_weight: float = 1.0
    fde_weight: float = 1.0
    cont_weight: float = 0.0
    vel_weight: float = 0.5
    rel_margin: float = 0.10
    abs_margin: float = 0.03
    cont_tol: float = 0.03
    force_candidate0: bool = False
    switch_cooldown_steps: int = 0
    debug: bool = False

    @property
    def enabled(self) -> bool:
        return int(self.k) > 1


def run_one_stream_step(
    *,
    model,
    vae,
    stream,
    step_payload: dict[str, Any],
    first_chunk: bool,
    device: torch.device,
    local_commit_index: int,
    generated_frames: int,
    frames_per_token: int,
) -> StepOutput:
    condition_provider = stream.build_ldf_condition_provider(
        step_payload,
        first_chunk=first_chunk,
        device=device,
    )
    output = model.stream_generate_step(
        step_payload,
        first_chunk=first_chunk,
        condition=condition_provider,
    )
    latent = output["generated"][0].detach().cpu()
    decoded = vae.stream_decode(
        output["generated"][0][None, :],
        first_chunk=first_chunk,
    )[0].float().detach().cpu()
    chunk_frames = int(decoded.shape[0])
    frame_start = int(generated_frames)
    frame_end = frame_start + chunk_frames
    token_start = int(local_commit_index)
    token_end = token_start + 1
    return StepOutput(
        clean_committed_latent=latent,
        decoded_chunk=decoded,
        commit_token_range=(token_start, token_end),
        commit_frame_range=(frame_start, frame_end),
        decoded_chunk_frame_range=(frame_start, frame_end),
        target_xz_frame_range=(frame_start, frame_end),
        debug={
            "ready_to_commit_token": token_start,
            "model_commit_index_after_step": int(getattr(model, "commit_index", token_end)),
            "model_current_step_after_step": float(getattr(model, "current_step", 0.0)),
            "num_denoise_steps": int(getattr(model, "num_denoise_steps", 0)),
            "frames_per_token": int(frames_per_token),
        },
    )
```

- [ ] **Step 5: Add bypass helper to stream generation**

In `eval/ldf/stream_generation.py`, add:

```python
def _should_use_best_of_k(best_of_k: int) -> bool:
    return int(best_of_k) > 1
```

Do not change the K=1 branch behavior in this task.

- [ ] **Step 6: Run tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py -q
```

Expected: PASS for new state/scoring/one-step tests.

- [ ] **Step 7: Commit one-step wrapper**

Run:

```bash
git add eval/ldf/stream_best_of_k.py eval/ldf/stream_generation.py tests/test_stream_best_of_k.py
git commit -m "feat: add stream best-of-k one-step wrapper"
```

---

### Task 5: Serial State-Fork Selector

**Files:**
- Modify: `eval/ldf/stream_best_of_k.py`
- Modify: `eval/ldf/stream_generation.py`
- Modify: `tests/test_stream_best_of_k.py`
- Modify: `tests/test_stream_eval_metrics.py`

- [ ] **Step 1: Add selector tests for candidate0 and force-candidate0**

Append to `tests/test_stream_best_of_k.py`:

```python
def test_serial_selector_force_candidate0_restores_candidate0_state(monkeypatch):
    from eval.ldf.stream_best_of_k import StreamBestOfKConfig, run_best_of_k_step

    model = _OneStepModel()
    vae = _OneStepVAE()
    stream = _OneStepStream()
    cfg = StreamBestOfKConfig(k=3, force_candidate0=True, debug=True)
    sample_batch = {
        "traj_cond_7d": torch.zeros(1, 4, 7, dtype=torch.float32),
    }
    sample_batch["traj_cond_7d"][0, :, 3] = 1.0

    selected, record = run_best_of_k_step(
        model=model,
        vae=vae,
        stream=stream,
        step_payload={"text": "walk"},
        sample_batch=sample_batch,
        first_chunk=True,
        device=torch.device("cpu"),
        local_commit_index=0,
        generated_frames=0,
        previous_decoded_chunks=[],
        chunk_frame_ends=[],
        frames_per_token=4,
        cfg=cfg,
        steps_since_switch=100,
    )

    assert selected.commit_token_range == (0, 1)
    assert record["selected_idx"] == 0
    assert record["switch_reason"] == "force_candidate0"
    assert model.commit_index == 1
    assert len(record["candidate_scores"]) == 3


def test_serial_selector_records_same_frame_range_for_all_candidates():
    from eval.ldf.stream_best_of_k import StreamBestOfKConfig, run_best_of_k_step

    model = _OneStepModel()
    vae = _OneStepVAE()
    stream = _OneStepStream()
    cfg = StreamBestOfKConfig(k=2, force_candidate0=True, debug=True)
    sample_batch = {
        "traj_cond_7d": torch.zeros(1, 4, 7, dtype=torch.float32),
    }
    sample_batch["traj_cond_7d"][0, :, 3] = 1.0

    selected, record = run_best_of_k_step(
        model=model,
        vae=vae,
        stream=stream,
        step_payload={"text": "walk"},
        sample_batch=sample_batch,
        first_chunk=True,
        device=torch.device("cpu"),
        local_commit_index=0,
        generated_frames=0,
        previous_decoded_chunks=[],
        chunk_frame_ends=[],
        frames_per_token=4,
        cfg=cfg,
        steps_since_switch=100,
    )

    ranges = [tuple(item["commit_frame_range"]) for item in record["candidate_scores"]]
    assert selected.commit_frame_range == (0, 4)
    assert ranges == [(0, 4), (0, 4)]
```

- [ ] **Step 2: Run selector tests and verify they fail**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py::test_serial_selector_force_candidate0_restores_candidate0_state tests/test_stream_best_of_k.py::test_serial_selector_records_same_frame_range_for_all_candidates -q
```

Expected: FAIL because `run_best_of_k_step` is not implemented.

- [ ] **Step 3: Implement serial selector**

In `eval/ldf/stream_best_of_k.py`, add helper functions and `run_best_of_k_step`:

```python
import random
import numpy as np

from eval.ldf.stream_scoring import (
    CandidateScore,
    ConservativeGateConfig,
    choose_candidate,
    compute_chunk_xz_score,
    compute_continuity_score,
)
from eval.ldf.stream_state import (
    CandidateState,
    capture_runtime_snapshot,
    restore_runtime_snapshot,
)
from utils.motion_process import recover_root_rot_pos


def _target_xz_slice(sample_batch: dict, frame_range: tuple[int, int]) -> torch.Tensor:
    traj = sample_batch["traj_cond_7d"]
    if traj.ndim == 3:
        traj = traj[0]
    start, end = frame_range
    if int(traj.shape[0]) < end:
        pad = traj[-1:].expand(end - int(traj.shape[0]), -1)
        traj = torch.cat([traj, pad], dim=0)
    return traj[start:end, [0, 2]].float().cpu()


def _decoded_root_xz(decoded_chunk: torch.Tensor, previous_decoded_chunks: list[torch.Tensor]) -> torch.Tensor:
    full = torch.cat(list(previous_decoded_chunks) + [decoded_chunk], dim=0)
    _, root_xyz = recover_root_rot_pos(full.unsqueeze(0))
    start = int(full.shape[0] - decoded_chunk.shape[0])
    return root_xyz[0, start:start + decoded_chunk.shape[0], [0, 2]].float().cpu()


def _previous_root_xz(previous_decoded_chunks: list[torch.Tensor]) -> torch.Tensor | None:
    if not previous_decoded_chunks:
        return None
    full = torch.cat(previous_decoded_chunks, dim=0)
    _, root_xyz = recover_root_rot_pos(full.unsqueeze(0))
    return root_xyz[0, :, [0, 2]].float().cpu()


def _seed_extra_candidate(base_seed: int, candidate_idx: int) -> None:
    seed = int(base_seed) + int(candidate_idx) * 1000003
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _score_step_output(
    *,
    candidate_idx: int,
    step_output: StepOutput,
    sample_batch: dict,
    previous_decoded_chunks: list[torch.Tensor],
    cfg: StreamBestOfKConfig,
) -> tuple[CandidateScore, dict]:
    pred_xz = _decoded_root_xz(step_output.decoded_chunk, previous_decoded_chunks)
    target_xz = _target_xz_slice(sample_batch, step_output.target_xz_frame_range)
    xz_ade, xz_fde = compute_chunk_xz_score(pred_xz, target_xz)
    pos_cont, vel_cont = compute_continuity_score(_previous_root_xz(previous_decoded_chunks), pred_xz)
    score = CandidateScore(
        index=int(candidate_idx),
        xz_ade=float(xz_ade),
        xz_fde=float(xz_fde),
        pos_cont=float(pos_cont),
        vel_cont=float(vel_cont),
    )
    return score, {
        "index": int(candidate_idx),
        "xz_ade": float(xz_ade),
        "xz_fde": float(xz_fde),
        "pos_cont": float(pos_cont),
        "vel_cont": float(vel_cont),
        "commit_token_range": list(step_output.commit_token_range),
        "commit_frame_range": list(step_output.commit_frame_range),
        "decoded_chunk_frame_range": list(step_output.decoded_chunk_frame_range),
        "target_xz_frame_range": list(step_output.target_xz_frame_range),
    }


def run_best_of_k_step(
    *,
    model,
    vae,
    stream,
    step_payload: dict,
    sample_batch: dict,
    first_chunk: bool,
    device: torch.device,
    local_commit_index: int,
    generated_frames: int,
    previous_decoded_chunks: list[torch.Tensor],
    chunk_frame_ends: list[int],
    frames_per_token: int,
    cfg: StreamBestOfKConfig,
    steps_since_switch: int,
) -> tuple[StepOutput, dict]:
    base_snapshot = capture_runtime_snapshot(
        model=model,
        vae=vae,
        stream_conditioner=None,
        first_chunk=first_chunk,
        generated_frames=generated_frames,
        chunk_frame_ends=chunk_frame_ends,
    )
    base_seed = int(torch.initial_seed())
    candidates: list[CandidateState] = []
    score_objects: list[CandidateScore] = []
    score_records: list[dict] = []

    for candidate_idx in range(int(cfg.k)):
        restore_runtime_snapshot(model=model, vae=vae, stream_conditioner=None, snapshot=base_snapshot)
        if candidate_idx > 0:
            _seed_extra_candidate(base_seed, candidate_idx)
        step_output = run_one_stream_step(
            model=model,
            vae=vae,
            stream=stream,
            step_payload=step_payload,
            first_chunk=first_chunk,
            device=device,
            local_commit_index=local_commit_index,
            generated_frames=generated_frames,
            frames_per_token=frames_per_token,
        )
        post_snapshot = capture_runtime_snapshot(
            model=model,
            vae=vae,
            stream_conditioner=None,
            first_chunk=False,
            generated_frames=generated_frames + int(step_output.decoded_chunk.shape[0]),
            chunk_frame_ends=chunk_frame_ends + [step_output.commit_frame_range[1]],
        )
        candidates.append(CandidateState(candidate_idx, step_output, post_snapshot))
        score_obj, score_record = _score_step_output(
            candidate_idx=candidate_idx,
            step_output=step_output,
            sample_batch=sample_batch,
            previous_decoded_chunks=previous_decoded_chunks,
            cfg=cfg,
        )
        score_objects.append(score_obj)
        score_records.append(score_record)

    gate_cfg = ConservativeGateConfig(
        fde_weight=float(cfg.fde_weight),
        vel_weight=float(cfg.vel_weight),
        rel_margin=float(cfg.rel_margin),
        abs_margin=float(cfg.abs_margin),
        cont_tol=float(cfg.cont_tol),
        force_candidate0=bool(cfg.force_candidate0),
        switch_cooldown_steps=int(cfg.switch_cooldown_steps),
    )
    decision = choose_candidate(score_objects, gate_cfg, steps_since_switch=steps_since_switch)
    selected = candidates[int(decision.selected_index)]
    restore_runtime_snapshot(model=model, vae=vae, stream_conditioner=None, snapshot=selected.snapshot)
    record = {
        "selected_idx": int(decision.selected_index),
        "switch_reason": decision.reason,
        "candidate_scores": score_records,
        "candidate0_track": float(decision.candidate0_track),
        "selected_track": float(decision.selected_track),
        "candidate0_continuity": float(decision.candidate0_continuity),
        "selected_continuity": float(decision.selected_continuity),
    }
    return selected.step_output, record
```

- [ ] **Step 4: Wire selector into stream generation K>1 path**

In `eval/ldf/stream_generation.py`:

1. Import `StreamBestOfKConfig` and `run_best_of_k_step` from `eval.ldf.stream_best_of_k`.
2. Add the new keyword arguments to `run_stream_generate_step_sample`.
3. Build `best_of_k_cfg` with the expanded fields.
4. Replace the old `_stream_generate_step_best_of_k` branch with:

```python
if _should_use_best_of_k(best_of_k_cfg.k):
    step_output, record = run_best_of_k_step(
        model=model,
        vae=vae,
        stream=stream,
        step_payload=step_payload,
        sample_batch=sample_batch,
        first_chunk=first_chunk,
        device=device,
        local_commit_index=local_commit_index,
        generated_frames=generated_frames,
        previous_decoded_chunks=decoded_chunks,
        chunk_frame_ends=chunk_frame_ends,
        frames_per_token=int(frames_per_token),
        cfg=best_of_k_cfg,
        steps_since_switch=steps_since_switch,
    )
    latent_token = step_output.clean_committed_latent
    decoded_chunk_raw = step_output.decoded_chunk
    best_of_k_records.append(record)
    if int(record["selected_idx"]) == 0:
        steps_since_switch += 1
    else:
        steps_since_switch = 0
else:
    condition_provider = stream.build_ldf_condition_provider(
        step_payload,
        first_chunk=first_chunk,
        device=device,
    )
    output = model.stream_generate_step(
        step_payload,
        first_chunk=first_chunk,
        condition=condition_provider,
    )
    latent_token = output["generated"][0].detach().cpu()
    decoded_chunk_raw = vae.stream_decode(
        output["generated"][0][None, :],
        first_chunk=first_chunk,
    )[0].float().detach().cpu()
```

Initialize `steps_since_switch = 10**9` before the rollout loop.

- [ ] **Step 5: Remove old batch-argmin test expectation**

In `tests/test_stream_eval_metrics.py`, replace `test_stream_best_of_k_batches_candidates_and_commits_selected_latent` with a test named `test_stream_best_of_k_force_candidate0_does_not_batch_mutate_state`. The new test should assert:

```python
assert stream_out["stream_best_of_k"]["k"] == 3
assert stream_out["stream_best_of_k"]["records"][0]["selected_idx"] == 0
assert stream_out["latent_stream"].shape[0] == 1
```

Do not assert that model batch size becomes K; serial state-fork should keep runtime batch size at 1.

- [ ] **Step 6: Run selector tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py tests/test_stream_eval_metrics.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit serial selector**

Run:

```bash
git add eval/ldf/stream_best_of_k.py eval/ldf/stream_generation.py tests/test_stream_best_of_k.py tests/test_stream_eval_metrics.py
git commit -m "feat: add serial stream best-of-k selector"
```

---

### Task 6: Debug Records and Metric Plumbing

**Files:**
- Modify: `eval/ldf/stream_best_of_k.py`
- Modify: `eval/ldf/stream_generation.py`
- Modify: `eval/ldf/stream_metrics.py`
- Modify: `tools/sweep_stream_step_cfg.py`
- Modify: `tests/test_stream_best_of_k.py`

- [ ] **Step 1: Add debug record test**

Append to `tests/test_stream_best_of_k.py`:

```python
def test_selector_debug_record_contains_commit_and_target_ranges():
    from eval.ldf.stream_best_of_k import StreamBestOfKConfig, run_best_of_k_step

    model = _OneStepModel()
    vae = _OneStepVAE()
    cfg = StreamBestOfKConfig(k=2, force_candidate0=True, debug=True)
    sample_batch = {"traj_cond_7d": torch.zeros(1, 4, 7, dtype=torch.float32)}
    sample_batch["traj_cond_7d"][0, :, 3] = 1.0

    _, record = run_best_of_k_step(
        model=model,
        vae=vae,
        stream=_OneStepStream(),
        step_payload={"text": "walk"},
        sample_batch=sample_batch,
        first_chunk=True,
        device=torch.device("cpu"),
        local_commit_index=0,
        generated_frames=0,
        previous_decoded_chunks=[],
        chunk_frame_ends=[],
        frames_per_token=4,
        cfg=cfg,
        steps_since_switch=100,
    )

    first_score = record["candidate_scores"][0]
    assert first_score["commit_token_range"] == [0, 1]
    assert first_score["commit_frame_range"] == [0, 4]
    assert first_score["decoded_chunk_frame_range"] == [0, 4]
    assert first_score["target_xz_frame_range"] == [0, 4]
    assert "switch_reason" in record
```

- [ ] **Step 2: Run debug test**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py::test_selector_debug_record_contains_commit_and_target_ranges -q
```

Expected: PASS if Task 5 record already includes ranges. If it fails, add the missing fields exactly as asserted.

- [ ] **Step 3: Add aggregate switch metrics**

In `eval/ldf/stream_generation.py`, include these fields in `stream_best_of_k` return payload:

```python
"switch_count": int(sum(1 for item in best_of_k_records if int(item.get("selected_idx", 0)) != 0)),
"step_count": int(len(best_of_k_records)),
"records": best_of_k_records if bool(best_of_k_cfg.debug) else [],
```

In `eval/ldf/stream_metrics.py`, add sample metrics when best-of-K is enabled:

```python
stream_metric["stream_best_of_k_switch_count"] = float(best_of_k_info.get("switch_count", 0))
stream_metric["stream_best_of_k_step_count"] = float(best_of_k_info.get("step_count", 0))
```

In the sample summary section, aggregate these two keys with `_average_scalar_metric`.

- [ ] **Step 4: Run metrics tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py tests/test_stream_eval_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit debug records**

Run:

```bash
git add eval/ldf/stream_best_of_k.py eval/ldf/stream_generation.py eval/ldf/stream_metrics.py tools/sweep_stream_step_cfg.py tests/test_stream_best_of_k.py
git commit -m "feat: add stream best-of-k debug records"
```

---

### Task 7: Local Regression Verification

**Files:**
- No required source edits.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_stream_best_of_k.py tests/test_stream_eval_metrics.py tests/test_validation_eval_interface.py -q
```

Expected: PASS.

- [ ] **Step 2: Run py_compile on touched modules**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m py_compile eval/ldf/stream_state.py eval/ldf/stream_scoring.py eval/ldf/stream_best_of_k.py eval/ldf/stream_generation.py eval/ldf/stream_metrics.py utils/training/ldf/validation_eval_runtime.py utils/training/ldf/validation_generation.py tools/sweep_stream_step_cfg.py
```

Expected: exit code 0.

- [ ] **Step 3: Run 000021 K=1 baseline**

Run:

```bash
cd /home/yuankai/Text2Motion/FloodNet
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/yuankai/.conda/envs/flooddiffusion/bin/python tools/sweep_stream_step_cfg.py \
  --sample_name 000021 \
  --probe_tag bestk_ref_000021_k1 \
  --cfg_text_values 1.25 \
  --cfg_traj_values 3.0 \
  --history_length 30 \
  --horizon_tokens 20 \
  --num_runs 1 \
  --best_of_k 1
```

Expected: command completes and writes a summary under `eval/output_eval`.

- [ ] **Step 4: Run 000021 K=5 force-candidate0**

Run:

```bash
cd /home/yuankai/Text2Motion/FloodNet
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/yuankai/.conda/envs/flooddiffusion/bin/python tools/sweep_stream_step_cfg.py \
  --sample_name 000021 \
  --probe_tag bestk_ref_000021_k5_force0 \
  --cfg_text_values 1.25 \
  --cfg_traj_values 3.0 \
  --history_length 30 \
  --horizon_tokens 20 \
  --num_runs 1 \
  --best_of_k 5 \
  --best_of_k_force_candidate0 \
  --best_of_k_debug
```

Expected: command completes. ADE/FDE should be metric-level close to K=1. The summary should report `stream_best_of_k_switch_count` equal to 0.

- [ ] **Step 5: Run 000021 K=5 selector**

Run:

```bash
cd /home/yuankai/Text2Motion/FloodNet
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/yuankai/.conda/envs/flooddiffusion/bin/python tools/sweep_stream_step_cfg.py \
  --sample_name 000021 \
  --probe_tag bestk_ref_000021_k5_selector \
  --cfg_text_values 1.25 \
  --cfg_traj_values 3.0 \
  --history_length 30 \
  --horizon_tokens 20 \
  --num_runs 1 \
  --best_of_k 5 \
  --best_of_k_debug
```

Expected: command completes. ADE must not be clearly worse than K=1. Switch count should be materially lower than 37/46.

- [ ] **Step 6: Report results**

Summarize:

```text
K=1 ADE/FDE/FPS:
K=5 force0 ADE/FDE/FPS/switch_count:
K=5 selector ADE/FDE/FPS/switch_count:
```

If force-candidate0 differs materially from K=1, stop and inspect snapshot/restore before interpreting selector quality.

- [ ] **Step 7: Commit verification notes if a script or doc was added**

If no files changed during verification, do not create a commit. If a helper report file is added intentionally, run:

```bash
git add path/to/report
git commit -m "docs: record stream best-of-k verification"
```

---

## Self-Review Checklist

- Spec coverage:
  - K=1 bypass: Task 4 and Task 7.
  - Candidate0 baseline: Task 5 and Task 7 force-candidate0.
  - Serial state-fork: Task 5.
  - Transaction-safe buffers: Task 5 chooses local `StepOutput`; Task 5 tests one append.
  - Deep clone state: Task 3.
  - Clean committed chunk scoring: Task 4 `StepOutput`; Task 5 scorer uses `commit_frame_range`.
  - Conservative gate: Task 2.
  - Debug records: Task 6.
- No batch optimization appears in implementation tasks.
- No root replacement or re-encoding is added to best-of-K.
- The plan keeps all code changes local to eval/runtime and validation config plumbing.
