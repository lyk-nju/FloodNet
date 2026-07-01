# Multi-Step Detached Root Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional diagnostic self-forcing mode where every rollout step's committed token receives direct 7D body/root supervision, while history replacement remains detached.

**Architecture:** Keep the rollout orchestration in `utils/training/self_forcing.py`, add one pure decoded-frame-mask body-aux helper in `utils/training/control_loss.py`, and gate everything behind `multistep_commit_body_aux.enabled`. The rollout stores gradient-bearing commit tokens from `pred_x0_latent_list`, writes detached replacements from `x0_latent_list`, assembles one decode prefix per sample, decodes once, and computes a frame-level masked-mean 7D body aux over committed-token frames only.

**Tech Stack:** Python, PyTorch, Lightning manual optimization, existing causal VAE token/frame helpers in `utils.token_frame`, existing 7D body-aux primitives in `utils.training.control_loss`.

---

## File Structure

- Modify `configs/ldf.yaml`: add disabled-by-default `multistep_commit_body_aux` config block with local `weight` and conservative 7D weights.
- Modify `utils/training/control_loss.py`: add `compute_body_aux_loss_on_frame_mask()` that accepts already-decoded motion features and a frame mask, derives 7D body terms over the complete decoded prefix, and reduces by masked frame mean.
- Modify `utils/training/self_forcing.py`: add a commit record dataclass, collect records during rollout when the new mode is enabled, assemble decode prefixes using full-prefix or local-prefix coordinates, compute commit body aux, and replace final-step body aux when configured.
- Modify `tests/test_body_aux_loss.py`: add focused tests for the new decoded-frame-mask helper and frame-mean reduction.
- Create `tests/test_multistep_commit_body_aux.py`: add rollout and integration tests for gradient-enabled non-final forwards, source tensor split, full/local prefix assembly, and body-aux replacement semantics.

---

### Task 1: Add Config Defaults

**Files:**
- Modify: `configs/ldf.yaml`

- [ ] **Step 1: Add the disabled config block**

Add this block near `body_aux_loss`:

```yaml
multistep_commit_body_aux:
    enabled: false
    decode_mode: single
    replace_final_body_aux: true
    weight: 1.0
    weights:
        root_xz: 1.0
        root_y: 0.0
        heading: 0.2
        fwd_delta: 0.05
        yaw_delta: 0.05
        end_xz: 0.0
```

- [ ] **Step 2: Verify YAML parses**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/ldf.yaml")
node = cfg.multistep_commit_body_aux
assert node.enabled is False
assert node.decode_mode == "single"
assert node.replace_final_body_aux is True
assert float(node.weight) == 1.0
assert float(node.weights.end_xz) == 0.0
print("ok")
PY
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add configs/ldf.yaml
git commit -m "config: add multistep commit body aux defaults"
```

---

### Task 2: Add Decoded Frame-Mask Body-Aux Helper

**Files:**
- Modify: `utils/training/control_loss.py`
- Modify: `tests/test_body_aux_loss.py`

- [ ] **Step 1: Write failing helper tests**

Append these tests to `tests/test_body_aux_loss.py`:

```python
def test_body_aux_on_frame_mask_uses_frame_mean(monkeypatch):
    import utils.training.control_loss as cl

    def fake_recover_root_rot_pos(decoded):
        b, t, _ = decoded.shape
        quat = decoded.new_zeros(b, t, 4)
        quat[..., 0] = 1.0
        xyz = decoded[..., :3]
        return quat, xyz

    monkeypatch.setattr(cl, "recover_root_rot_pos", fake_recover_root_rot_pos)

    decoded = [torch.zeros(6, 263)]
    gt = torch.zeros(1, 6, 7)
    gt[..., 3] = 1.0
    gt[0, 1, 0] = 2.0
    gt[0, 2, 0] = 2.0
    gt[0, 4, 0] = 2.0
    mask = torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.float32)
    weights = {
        "root_xz": 1.0,
        "root_y": 0.0,
        "heading": 0.0,
        "fwd_delta": 0.0,
        "yaw_delta": 0.0,
        "end_xz": 0.0,
    }

    loss, terms = cl.compute_body_aux_loss_on_frame_mask(
        decoded,
        gt,
        torch.tensor([6]),
        mask,
        torch.device("cpu"),
        weights,
        heading_form="cosine",
    )

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(1.5))
    assert abs(terms["root_xz"] - 1.5) < 1e-6


def test_body_aux_on_frame_mask_delta_uses_full_prefix(monkeypatch):
    import utils.training.control_loss as cl

    def fake_recover_root_rot_pos(decoded):
        b, t, _ = decoded.shape
        quat = decoded.new_zeros(b, t, 4)
        quat[..., 0] = 1.0
        xyz = decoded[..., :3]
        return quat, xyz

    monkeypatch.setattr(cl, "recover_root_rot_pos", fake_recover_root_rot_pos)

    decoded = [torch.zeros(4, 263)]
    decoded[0][1, 0] = 1.0
    gt = torch.zeros(1, 4, 7)
    gt[..., 3] = 1.0
    mask = torch.tensor([[0, 1, 0, 0]], dtype=torch.float32)
    weights = {
        "root_xz": 0.0,
        "root_y": 0.0,
        "heading": 0.0,
        "fwd_delta": 1.0,
        "yaw_delta": 0.0,
        "end_xz": 0.0,
    }

    loss, terms = cl.compute_body_aux_loss_on_frame_mask(
        decoded,
        gt,
        torch.tensor([4]),
        mask,
        torch.device("cpu"),
        weights,
        heading_form="cosine",
    )

    assert loss is not None
    assert terms["fwd_delta"] > 0.0
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_body_aux_loss.py::test_body_aux_on_frame_mask_uses_frame_mean tests/test_body_aux_loss.py::test_body_aux_on_frame_mask_delta_uses_full_prefix -q
```

Expected: fails with `AttributeError: module 'utils.training.control_loss' has no attribute 'compute_body_aux_loss_on_frame_mask'`.

- [ ] **Step 3: Implement the helper**

In `utils/training/control_loss.py`, add this function after `compute_body_aux_loss()`:

```python
def compute_body_aux_loss_on_frame_mask(
    decoded_list,
    gt_traj_7d,
    traj_length,
    frame_mask,
    device,
    weights: dict,
    heading_form: str = "cosine",
    sample_loss_mask=None,
    window_start_tokens=None,
    token_to_frame: int = 4,
):
    """Compute 7D body aux over a caller-provided frame mask.

    `decoded_list` contains full/local decoded prefixes, not cropped commit
    slices. Deltas are derived over each full decoded prefix before `frame_mask`
    selects committed frames, so the first committed frame can see the previous
    frame.
    """
    from utils.local_frame import root_quat_to_physical_yaw
    from utils.token_frame import token_start_frame

    if frame_mask is None:
        raise ValueError("frame_mask is required for commit body aux")
    if not torch.is_tensor(frame_mask):
        frame_mask = torch.as_tensor(frame_mask, device=device, dtype=torch.float32)
    else:
        frame_mask = frame_mask.to(device=device, dtype=torch.float32)
    if frame_mask.ndim != 2:
        raise ValueError(f"frame_mask must be [B,T], got {tuple(frame_mask.shape)}")

    if not torch.is_tensor(traj_length):
        traj_length = torch.as_tensor(traj_length, device=device, dtype=torch.long)
    else:
        traj_length = traj_length.to(device=device, dtype=torch.long)
    traj_length = traj_length.view(-1)

    starts = None
    if window_start_tokens is not None:
        if not torch.is_tensor(window_start_tokens):
            starts = torch.as_tensor(window_start_tokens, device=device, dtype=torch.long)
        else:
            starts = window_start_tokens.to(device=device, dtype=torch.long)
        starts = starts.view(-1)
        if starts.numel() == 1 and len(decoded_list) > 1:
            starts = starts.expand(len(decoded_list))

    weighted_losses = []
    term_sums = {
        k: 0.0
        for k in ("root_xz", "root_y", "heading", "fwd_delta", "yaw_delta", "end_xz")
    }
    total_n = 0.0
    for i, decoded in enumerate(decoded_list):
        decoded = decoded.to(device=device).float()
        quat, xyz = recover_root_rot_pos(decoded.unsqueeze(0))
        yaw = root_quat_to_physical_yaw(quat)

        gt_len = min(int(traj_length[i].item()), gt_traj_7d.shape[1])
        end_f = min(decoded.shape[0], gt_len, frame_mask.shape[1])
        if end_f <= 0:
            continue
        mask_i = frame_mask[i : i + 1, :end_f].to(device=device, dtype=xyz.dtype)
        if mask_i.sum().item() <= 0:
            continue

        gt7 = gt_traj_7d[i : i + 1, :end_f, :].to(device=device, dtype=xyz.dtype)
        gt_xyz = gt7[..., :3]
        gt_yaw = torch.atan2(gt7[..., 4], gt7[..., 3])
        pred_xyz = xyz[:, :end_f, :]
        pred_yaw = yaw[:, :end_f]

        if starts is not None:
            anchor_f = token_start_frame(int(starts[i].item()), token_to_frame)
            if anchor_f >= gt_len:
                raise ValueError(
                    "window_start_tokens must reference a valid GT anchor frame; "
                    f"sample={i}, start_token={int(starts[i].item())}, "
                    f"anchor_frame={anchor_f}, traj_length={gt_len}"
                )
            anchor7 = gt_traj_7d[i : i + 1, anchor_f : anchor_f + 1, :].to(
                device=device, dtype=xyz.dtype
            )
            anchor_xyz = anchor7[..., :3]
            anchor_yaw = torch.atan2(anchor7[..., 4], anchor7[..., 3])
            pred_xyz, pred_yaw = canonicalize_pose_to_anchor(
                pred_xyz, pred_yaw, anchor_xyz, anchor_yaw
            )
            gt_xyz, gt_yaw = canonicalize_pose_to_anchor(
                gt_xyz, gt_yaw, anchor_xyz, anchor_yaw
            )

        slm_i = None
        sample_w = 1.0
        if sample_loss_mask is not None:
            sample_w = float(sample_loss_mask[i])
            slm_i = sample_loss_mask[i : i + 1].to(device)
        n_eff = float(mask_i.sum().item()) * sample_w
        if n_eff <= 0:
            continue

        total_i, terms_i = body_aux_loss_terms(
            pred_xyz,
            pred_yaw,
            gt_xyz,
            gt_yaw,
            mask_i,
            weights,
            heading_form=heading_form,
            sample_loss_mask=slm_i,
        )
        weighted_losses.append(total_i * n_eff)
        for k in term_sums:
            term_sums[k] += float(terms_i[k].detach()) * n_eff
        total_n += n_eff

    if total_n <= 0 or not weighted_losses:
        return None, {}
    loss = torch.stack(weighted_losses).sum() / total_n
    metrics = {k: v / total_n for k, v in term_sums.items()}
    return loss, metrics
```

- [ ] **Step 4: Run helper tests and existing body aux tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_body_aux_loss.py -q
```

Expected: all tests in `tests/test_body_aux_loss.py` pass.

- [ ] **Step 5: Commit**

```bash
git add utils/training/control_loss.py tests/test_body_aux_loss.py
git commit -m "test: cover commit-frame body aux helper"
```

---

### Task 3: Add Commit Record Collection Without Changing Default Behavior

**Files:**
- Modify: `utils/training/self_forcing.py`
- Create: `tests/test_multistep_commit_body_aux.py`

- [ ] **Step 1: Write failing default-behavior tests**

Create `tests/test_multistep_commit_body_aux.py` with:

```python
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from utils.training.self_forcing import RolloutPlan, SelfForcingTrainer


def _cfg(commit_enabled: bool):
    def get(key, default=None):
        values = {
            "anchor_canonicalize": {"enabled": False},
            "history_corruption": {},
            "horizon_sim": {"enabled": False},
            "self_forcing_disable_replace": True,
            "multistep_commit_body_aux": {
                "enabled": commit_enabled,
                "decode_mode": "single",
                "replace_final_body_aux": True,
                "weight": 1.0,
                "weights": {
                    "root_xz": 1.0,
                    "root_y": 0.0,
                    "heading": 0.2,
                    "fwd_delta": 0.05,
                    "yaw_delta": 0.05,
                    "end_xz": 0.0,
                },
            },
        }
        return values.get(key, default)
    return SimpleNamespace(get=get)


def _trainer(commit_enabled: bool, k: int = 3):
    batch = 1
    seq_len = 5
    hidden = 4
    feature = torch.zeros(batch, seq_len, hidden)
    model = MagicMock(name="model")
    model.chunk_size = 1
    model.self_forcing_stride_tokens = 1
    model.self_forcing_k_schedule = [(0.0, k)]
    model._decide_text_dropout.return_value = torch.zeros(batch, dtype=torch.bool)
    model._prepare_text_context.return_value = None
    model._decide_traj_dropout.return_value = False
    model._prepare_traj_condition.return_value = (None, None, False, None)

    grad_flags = []

    def forward(*args, **kwargs):
        grad_flags.append(torch.is_grad_enabled())
        pred = torch.ones(seq_len, hidden, requires_grad=torch.is_grad_enabled())
        repl = torch.full((seq_len, hidden), 2.0)
        return {
            "loss": pred.sum() * 0.0,
            "pred_x0_latent_list": [pred],
            "x0_latent_list": [repl],
        }

    model._forward_single_window.side_effect = forward
    module = SimpleNamespace(model=model, cfg=_cfg(commit_enabled))
    trainer = SelfForcingTrainer.__new__(SelfForcingTrainer)
    trainer._module = module
    trainer._last_replace_diff = None
    trainer._last_sample_loss_mask = None
    trainer._last_horizon_tokens = -1.0
    trainer._last_corruption_applied = 0.0
    trainer.plan_rollout = MagicMock(
        return_value=RolloutPlan(
            effective_k=k,
            start_end_indices=torch.tensor([1], dtype=torch.long),
            phase_offset=torch.tensor([0.0]),
        )
    )
    model_batch = {
        "feature": feature,
        "feature_length": torch.tensor([seq_len], dtype=torch.long),
    }
    return trainer, model_batch, grad_flags


def test_default_rollout_keeps_non_final_steps_no_grad():
    trainer, model_batch, grad_flags = _trainer(commit_enabled=False, k=3)

    final_result, k = trainer._run_rollout(model_batch, progress=1.0)

    assert k == 3
    assert final_result is not None
    assert grad_flags == [False, False, True]


def test_commit_aux_rollout_enables_grad_for_every_step_and_records_commits():
    trainer, model_batch, grad_flags = _trainer(commit_enabled=True, k=3)

    final_result, k = trainer._run_rollout(model_batch, progress=1.0)

    assert k == 3
    assert final_result is not None
    assert grad_flags == [True, True, True]
    records = trainer._last_commit_token_records
    assert [int(r.local_commit_idx) for r in records] == [0, 1, 2]
    assert [int(r.global_commit_idx) for r in records] == [0, 1, 2]
    assert all(r.pred_token.requires_grad for r in records)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py -q
```

Expected: second test fails because `_last_commit_token_records` does not exist and non-final steps remain `no_grad`.

- [ ] **Step 3: Add commit record dataclass and config helpers**

In `utils/training/self_forcing.py`, add near `RolloutPlan`:

```python
@dataclass(frozen=True)
class CommitTokenRecord:
    batch_idx: int
    local_commit_idx: int
    global_commit_idx: int
    pred_token: torch.Tensor
```

Add helper methods inside `SelfForcingTrainer`:

```python
    def _commit_body_aux_cfg(self) -> dict:
        return self._module.cfg.get("multistep_commit_body_aux", {}) or {}

    def _commit_body_aux_enabled(self) -> bool:
        cfg = self._commit_body_aux_cfg()
        return bool(cfg.get("enabled", False))
```

- [ ] **Step 4: Modify `_run_rollout()` to collect records**

In `_run_rollout()`, initialize before the loop:

```python
        commit_aux_enabled = self._commit_body_aux_enabled()
        commit_records: list[CommitTokenRecord] = []
        global_starts = model_batch.get("_window_global_start_token")
        if global_starts is None:
            global_starts = torch.zeros(feature.shape[0], device=device, dtype=torch.long)
        elif not torch.is_tensor(global_starts):
            global_starts = torch.as_tensor(global_starts, device=device, dtype=torch.long)
        else:
            global_starts = global_starts.to(device=device, dtype=torch.long)
        global_starts = global_starts.view(-1)
        if global_starts.numel() == 1 and feature.shape[0] > 1:
            global_starts = global_starts.expand(feature.shape[0])
```

Replace the non-final `with torch.no_grad()` branch with:

```python
            if is_final_step or commit_aux_enabled:
                rollout_result = model._forward_single_window(
                    model_batch,
                    current_feature,
                    time_steps,
                    all_text_context,
                    traj_emb,
                    traj_seq_lens,
                    traj_dropped,
                    enable_scheduled_sampling=False,
                    traj_token_mask=traj_token_mask,
                )
                if is_final_step:
                    final_step_result = rollout_result
            else:
                with torch.no_grad():
                    rollout_result = model._forward_single_window(
                        model_batch,
                        current_feature,
                        time_steps,
                        all_text_context,
                        traj_emb,
                        traj_seq_lens,
                        traj_dropped,
                        enable_scheduled_sampling=False,
                        traj_token_mask=traj_token_mask,
                    )
```

Immediately after `rollout_result` exists, collect records:

```python
            if commit_aux_enabled:
                pred_list = rollout_result.get("pred_x0_latent_list")
                if pred_list is not None:
                    for b in range(feature.shape[0]):
                        local_idx = int(end_indices[b].item()) - int(model.chunk_size)
                        if local_idx < 0:
                            continue
                        pred_seq = pred_list[b]
                        if pred_seq is None or local_idx >= pred_seq.shape[0]:
                            continue
                        commit_records.append(
                            CommitTokenRecord(
                                batch_idx=b,
                                local_commit_idx=local_idx,
                                global_commit_idx=int(global_starts[b].item()) + local_idx,
                                pred_token=pred_seq[local_idx],
                            )
                        )
            if is_final_step:
                break
```

Keep the existing replacement code after this block for non-final steps. At the end of `_run_rollout()`, before returning, set:

```python
        self._last_commit_token_records = commit_records
```

- [ ] **Step 5: Run rollout tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py -q
```

Expected: both tests pass.

- [ ] **Step 6: Run existing self-forcing tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_self_forcing_traj_token_mask.py -q
```

Expected: existing self-forcing tests pass.

- [ ] **Step 7: Commit**

```bash
git add utils/training/self_forcing.py tests/test_multistep_commit_body_aux.py
git commit -m "feat: collect multistep commit tokens"
```

---

### Task 4: Assemble Decode Prefixes and Commit Frame Masks

**Files:**
- Modify: `utils/training/self_forcing.py`
- Modify: `tests/test_multistep_commit_body_aux.py`

- [ ] **Step 1: Add failing prefix assembly tests**

Append to `tests/test_multistep_commit_body_aux.py`:

```python
def test_full_prefix_splice_assembly_uses_global_commit_indices():
    trainer, _, _ = _trainer(commit_enabled=True, k=1)
    batch = {
        "token": torch.zeros(1, 6, 2),
        "token_length": torch.tensor([6]),
        "traj_cond_7d": torch.zeros(1, 24, 7),
        "traj_length": torch.tensor([24]),
    }
    model_batch = {
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([2]),
        "_window_local_latent_start_token": torch.tensor([99]),
        "feature": torch.zeros(1, 4, 2),
        "feature_length": torch.tensor([4]),
    }
    record = torch.ones(2, requires_grad=True)
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=0,
            local_commit_idx=1,
            global_commit_idx=3,
            pred_token=record,
        )
    ]

    decoded_latents, frame_mask, window_starts = trainer._assemble_commit_decode_inputs(
        batch, model_batch
    )

    assert torch.equal(decoded_latents[0][3], record)
    assert decoded_latents[0][3].requires_grad
    assert not decoded_latents[0][2].requires_grad
    assert window_starts.tolist() == [2]
    assert frame_mask[0].sum().item() == 4


def test_local_decode_assembly_uses_local_commit_indices():
    trainer, _, _ = _trainer(commit_enabled=True, k=1)
    batch = {
        "token": torch.zeros(1, 6, 2),
        "token_length": torch.tensor([6]),
        "traj_cond_7d": torch.zeros(1, 24, 7),
        "traj_length": torch.tensor([24]),
    }
    model_batch = {
        "_window_local_body_aux_mode": "local_decode",
        "_window_global_start_token": torch.tensor([5]),
        "_window_local_latent_start_token": torch.tensor([0]),
        "feature": torch.zeros(1, 4, 2),
        "feature_length": torch.tensor([4]),
    }
    record = torch.ones(2, requires_grad=True)
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=0,
            local_commit_idx=1,
            global_commit_idx=6,
            pred_token=record,
        )
    ]

    decoded_latents, frame_mask, window_starts = trainer._assemble_commit_decode_inputs(
        batch, model_batch
    )

    assert torch.equal(decoded_latents[0][1], record)
    assert decoded_latents[0][1].requires_grad
    assert not decoded_latents[0][0].requires_grad
    assert window_starts is None
    assert frame_mask[0].sum().item() == 4
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py::test_full_prefix_splice_assembly_uses_global_commit_indices tests/test_multistep_commit_body_aux.py::test_local_decode_assembly_uses_local_commit_indices -q
```

Expected: fails because `_assemble_commit_decode_inputs` is not defined.

- [ ] **Step 3: Implement `_assemble_commit_decode_inputs()`**

In `utils/training/self_forcing.py`, import:

```python
from utils.token_frame import token_range_to_frame_slice
```

Add method inside `SelfForcingTrainer`:

```python
    def _assemble_commit_decode_inputs(self, batch: dict, model_batch: dict):
        records = getattr(self, "_last_commit_token_records", [])
        if not records:
            return [], None, None
        mode = str(
            model_batch.get(
                "_window_local_body_aux_mode",
                batch.get("_window_local_body_aux_mode", "full_prefix_splice"),
            )
        )
        device = self._module.device if hasattr(self._module, "device") else records[0].pred_token.device

        if mode == "local_decode":
            source = model_batch["feature"]
            lengths = model_batch["feature_length"]
            use_global = False
            window_starts = None
        elif mode == "full_prefix_splice":
            source = batch["token"]
            lengths = batch.get("token_length")
            if lengths is None:
                lengths = torch.full(
                    (source.shape[0],), source.shape[1], device=source.device, dtype=torch.long
                )
            use_global = True
            starts = model_batch.get("_window_global_start_token", batch.get("_window_global_start_token"))
            if starts is None:
                starts = torch.zeros(source.shape[0], device=source.device, dtype=torch.long)
            elif not torch.is_tensor(starts):
                starts = torch.as_tensor(starts, device=source.device, dtype=torch.long)
            window_starts = starts.to(device=device, dtype=torch.long).view(-1)
        else:
            raise ValueError(f"Unsupported _window_local_body_aux_mode={mode!r}")

        source = source.to(device)
        if not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)
        lengths = lengths.view(-1)
        if lengths.numel() == 1 and source.shape[0] > 1:
            lengths = lengths.expand(source.shape[0])

        records_by_sample: dict[int, list] = {}
        for record in records:
            records_by_sample.setdefault(int(record.batch_idx), []).append(record)

        decoded_latents = []
        max_frames = 0
        sample_indices = sorted(records_by_sample)
        for b in sample_indices:
            token_len = int(lengths[b].item())
            decoded_latents.append(source[b, :token_len, :].detach().clone())
            max_frames = max(
                max_frames,
                token_range_to_frame_slice(0, token_len).stop,
            )

        frame_mask = source.new_zeros(len(sample_indices), max_frames)
        sample_pos = {b: pos for pos, b in enumerate(sample_indices)}
        for record in records:
            b = int(record.batch_idx)
            pos = sample_pos[b]
            token_idx = int(record.global_commit_idx if use_global else record.local_commit_idx)
            local_idx_for_write = token_idx if use_global else int(record.local_commit_idx)
            if token_idx < 0 or local_idx_for_write < 0:
                continue
            if local_idx_for_write >= decoded_latents[pos].shape[0]:
                continue
            decoded_latents[pos][local_idx_for_write] = record.pred_token.to(
                device=device, dtype=decoded_latents[pos].dtype
            )
            sl = token_range_to_frame_slice(token_idx, 1)
            if sl.start < frame_mask.shape[1]:
                frame_mask[pos, sl.start : min(sl.stop, frame_mask.shape[1])] = 1.0

        if window_starts is not None:
            window_starts = window_starts[sample_indices]
        return decoded_latents, frame_mask, window_starts
```

Important correction while implementing: in `full_prefix_splice`, write to
`decoded_latents[pos][global_commit_idx]`, not local commit index.

- [ ] **Step 4: Run prefix assembly tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py::test_full_prefix_splice_assembly_uses_global_commit_indices tests/test_multistep_commit_body_aux.py::test_local_decode_assembly_uses_local_commit_indices -q
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add utils/training/self_forcing.py tests/test_multistep_commit_body_aux.py
git commit -m "feat: assemble commit decode prefixes"
```

---

### Task 5: Compute Commit 7D Body Aux and Replace Final Body Aux

**Files:**
- Modify: `utils/training/self_forcing.py`
- Modify: `tests/test_multistep_commit_body_aux.py`

- [ ] **Step 1: Add failing loss integration tests**

Append to `tests/test_multistep_commit_body_aux.py`:

```python
def test_commit_aux_replaces_final_body_aux(monkeypatch):
    trainer, _, _ = _trainer(commit_enabled=True, k=1)
    trainer._module.device = torch.device("cpu")
    trainer._module.vae = MagicMock()
    trainer._module.cfg.model = SimpleNamespace(
        params={"control_loss_weight": 5.0}
    )
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=0,
            local_commit_idx=0,
            global_commit_idx=0,
            pred_token=torch.ones(4, requires_grad=True),
        )
    ]
    final_step_result = {
        "loss": torch.tensor(2.0, requires_grad=True),
        "pred_x0_latent_list": [torch.ones(1, 4, requires_grad=True)],
    }
    batch = {
        "token": torch.zeros(1, 1, 4),
        "token_length": torch.tensor([1]),
        "traj_cond_7d": torch.zeros(1, 1, 7),
        "traj_length": torch.tensor([1]),
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([0]),
    }
    model_batch = {
        "feature": torch.zeros(1, 1, 4),
        "feature_length": torch.tensor([1]),
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([0]),
    }

    monkeypatch.setattr(
        "utils.training.self_forcing._compute_body_aux_loss",
        MagicMock(side_effect=AssertionError("final body aux should be skipped")),
    )
    monkeypatch.setattr(
        "utils.training.self_forcing.compute_body_aux_loss_on_frame_mask",
        MagicMock(return_value=(torch.tensor(0.5), {"root_xz": 0.5})),
    )
    monkeypatch.setattr(
        trainer,
        "_decode_commit_latents",
        MagicMock(return_value=[torch.zeros(1, 263)]),
    )

    total, diff, control = trainer._compute_losses(
        final_step_result, batch, model_batch
    )

    assert torch.isclose(total, torch.tensor(4.5))
    assert torch.isclose(diff, torch.tensor(2.0))
    assert torch.isclose(control, torch.tensor(0.5))
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py::test_commit_aux_replaces_final_body_aux -q
```

Expected: fails because `_compute_losses()` does not accept `model_batch` and commit aux is not integrated.

- [ ] **Step 3: Import helper and add decode wrapper**

In `utils/training/self_forcing.py`, update imports:

```python
from .control_loss import (
    compute_body_aux_loss,
    compute_body_aux_loss_on_frame_mask,
    compute_control_loss_xz,
)
```

Add method inside `SelfForcingTrainer`:

```python
    def _decode_commit_latents(self, decoded_latents: list[torch.Tensor]):
        return [self._module.vae.decode(latent.unsqueeze(0))[0].float() for latent in decoded_latents]
```

- [ ] **Step 4: Add commit aux compute method**

Add inside `SelfForcingTrainer`:

```python
    def _compute_commit_body_aux_loss(self, batch: dict, model_batch: dict):
        cfg = self._commit_body_aux_cfg()
        if not bool(cfg.get("enabled", False)):
            return None, {}
        if str(cfg.get("decode_mode", "single")) != "single":
            raise NotImplementedError(
                "multistep_commit_body_aux.decode_mode currently supports only 'single'"
            )
        if "traj_cond_7d" not in batch:
            return None, {}
        decoded_latents, frame_mask, window_starts = self._assemble_commit_decode_inputs(
            batch, model_batch
        )
        if not decoded_latents or frame_mask is None:
            return None, {}
        decoded = self._decode_commit_latents(decoded_latents)
        ba_cfg = self._module.cfg.get("body_aux_loss", {}) or {}
        weights = {
            **_DEFAULT_BODY_AUX_WEIGHTS,
            **(ba_cfg.get("weights", {}) or {}),
            **(cfg.get("weights", {}) or {}),
        }
        loss, terms = compute_body_aux_loss_on_frame_mask(
            decoded,
            batch["traj_cond_7d"],
            batch["traj_length"],
            frame_mask,
            self._module.device,
            weights,
            heading_form=ba_cfg.get("heading_form", "cosine"),
            sample_loss_mask=getattr(self, "_last_sample_loss_mask", None),
            window_start_tokens=window_starts,
        )
        if loss is None:
            return None, {}
        return loss * float(cfg.get("weight", 1.0)), terms
```

- [ ] **Step 5: Change `_compute_losses()` signature and logic**

Change:

```python
    def _compute_losses(self, final_step_result: dict, batch: dict):
```

to:

```python
    def _compute_losses(self, final_step_result: dict, batch: dict, model_batch: dict):
```

At the top after `control_weight`, add:

```python
        commit_cfg = self._commit_body_aux_cfg()
        commit_enabled = bool(commit_cfg.get("enabled", False))
        replace_final = bool(commit_cfg.get("replace_final_body_aux", True))
        if control_weight > 0.0 and commit_enabled:
            step_control_loss, self._last_body_aux_terms = (
                self._compute_commit_body_aux_loss(batch, model_batch)
            )
            if step_control_loss is not None:
                total_loss = total_loss + control_weight * step_control_loss
                if replace_final:
                    return total_loss, step_diff_loss, step_control_loss
```

Keep the existing final-step body aux logic below this. In `_self_forcing_step()`,
update the call:

```python
        total_loss, step_diff_loss, step_control_loss = self._compute_losses(
            final_step_result, batch, model_batch
        )
```

- [ ] **Step 6: Run loss integration tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py -q
```

Expected: all tests in the new file pass.

- [ ] **Step 7: Commit**

```bash
git add utils/training/self_forcing.py tests/test_multistep_commit_body_aux.py
git commit -m "feat: add multistep commit body aux loss"
```

---

### Task 6: Full Verification

**Files:**
- Modify only if needed after failures: `utils/training/self_forcing.py`, `utils/training/control_loss.py`, tests touched above.

- [ ] **Step 1: Run targeted unit tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest \
  tests/test_multistep_commit_body_aux.py \
  tests/test_body_aux_loss.py \
  tests/test_self_forcing_traj_token_mask.py \
  tests/test_stream_window_sampling.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run syntax checks**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m py_compile \
  utils/training/self_forcing.py \
  utils/training/control_loss.py \
  train_ldf.py
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Inspect git diff for accidental scope creep**

Run:

```bash
git status --short
git diff --stat
```

Expected: only intended files changed, plus any pre-existing unstaged user edits that were already present before implementation.

- [ ] **Step 4: Final commit if verification-only fixes were needed**

If Task 6 required code/test fixes, commit them:

```bash
git add utils/training/self_forcing.py utils/training/control_loss.py tests/test_multistep_commit_body_aux.py tests/test_body_aux_loss.py configs/ldf.yaml
git commit -m "fix: verify multistep commit body aux"
```

If no fixes were needed, do not create an empty commit.
