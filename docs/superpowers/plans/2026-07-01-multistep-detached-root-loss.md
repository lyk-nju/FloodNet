# Multi-Step Detached Root Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional diagnostic self-forcing mode where every rollout step's committed token receives direct 7D body/root supervision, while history replacement remains detached.

**Architecture:** Keep rollout orchestration in `utils/training/self_forcing.py`, add one decoded-prefix commit-mask helper in `utils/training/control_loss.py`, and gate the behavior behind `multistep_commit_body_aux.enabled`. The rollout stores gradient-bearing commit tokens from `pred_x0_latent_list`, writes detached replacements from `x0_latent_list`, assembles one decode prefix per sample ending at the latest committed token, decodes once, computes one 7D body/root loss per committed rollout token, and averages over valid committed steps.

**Tech Stack:** Python, PyTorch, Lightning manual optimization, existing causal VAE token/frame helpers in `utils.token_frame`, existing 7D body-aux primitives in `utils.training.control_loss`.

---

## File Structure

- Modify `configs/ldf.yaml`: add disabled-by-default `multistep_commit_body_aux` config block with `include_final_step` and `reduction: step_mean`. Commit body aux reuses `body_aux_loss.weights` and global `control_loss_weight`.
- Modify `utils/training/control_loss.py`: add `compute_body_aux_loss_on_commit_masks()` that accepts decoded prefixes plus one frame mask per commit record, derives 7D body terms over complete decoded prefixes, applies decoded-frame to GT-frame offsets, and reduces by step mean.
- Modify `utils/training/self_forcing.py`: add a commit record dataclass, collect records during rollout when the new mode is enabled, reject `chunk_size > 1` for the first implementation, assemble decode prefixes using full-prefix or local-prefix coordinates without future GT suffix, compute commit body aux, log valid commit counts, and replace final-step body aux when configured.
- Modify `tests/test_body_aux_loss.py`: add focused tests for step-mean reduction, full-prefix delta derivation, and local GT frame offset.
- Create `tests/test_multistep_commit_body_aux.py`: add rollout and integration tests for gradient-enabled non-final forwards, source tensor split, replacement detach, chunk-size guard, full/local prefix-only assembly, sample-index mapping, valid-commit logging, and final-body-aux replacement semantics.

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
    include_final_step: true
    reduction: step_mean
    strict_valid_commits: false
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
assert node.include_final_step is True
assert node.reduction == "step_mean"
assert node.strict_valid_commits is False
assert "weight" not in node
assert "weights" not in node
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

### Task 2: Add Step-Mean Commit-Mask Body-Aux Helper

**Files:**
- Modify: `utils/training/control_loss.py`
- Modify: `tests/test_body_aux_loss.py`

- [ ] **Step 1: Write failing helper tests**

Append these tests to `tests/test_body_aux_loss.py`:

```python
def test_commit_body_aux_uses_step_mean_not_frame_mean(monkeypatch):
    import utils.training.control_loss as cl

    def fake_recover_root_rot_pos(decoded):
        b, t, _ = decoded.shape
        quat = decoded.new_zeros(b, t, 4)
        quat[..., 0] = 1.0
        xyz = decoded[..., :3]
        return quat, xyz

    monkeypatch.setattr(cl, "recover_root_rot_pos", fake_recover_root_rot_pos)

    decoded = [torch.zeros(5, 263)]
    gt = torch.zeros(1, 5, 7)
    gt[..., 3] = 1.0
    gt[0, 0, 0] = 1.0
    commit_masks = torch.zeros(2, 5)
    commit_masks[0, 0] = 1.0        # one-frame commit, SmoothL1(1)=0.5
    commit_masks[1, 1:5] = 1.0      # four-frame commit, zero loss
    weights = {
        "root_xz": 1.0,
        "root_y": 0.0,
        "heading": 0.0,
        "fwd_delta": 0.0,
        "yaw_delta": 0.0,
        "end_xz": 0.0,
    }

    loss, terms = cl.compute_body_aux_loss_on_commit_masks(
        decoded,
        gt,
        torch.tensor([5]),
        commit_masks,
        torch.tensor([0, 0]),
        torch.tensor([0]),
        torch.device("cpu"),
        weights,
        heading_form="cosine",
    )

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(0.25))
    assert abs(terms["root_xz"] - 0.25) < 1e-6


def test_commit_body_aux_delta_uses_full_decoded_prefix(monkeypatch):
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
    commit_masks = torch.zeros(1, 4)
    commit_masks[0, 1] = 1.0
    weights = {
        "root_xz": 0.0,
        "root_y": 0.0,
        "heading": 0.0,
        "fwd_delta": 1.0,
        "yaw_delta": 0.0,
        "end_xz": 0.0,
    }

    loss, terms = cl.compute_body_aux_loss_on_commit_masks(
        decoded,
        gt,
        torch.tensor([4]),
        commit_masks,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.device("cpu"),
        weights,
        heading_form="cosine",
    )

    assert loss is not None
    assert terms["fwd_delta"] > 0.0


def test_commit_body_aux_uses_window_start_token_for_gt_offset(monkeypatch):
    import utils.training.control_loss as cl

    def fake_recover_root_rot_pos(decoded):
        b, t, _ = decoded.shape
        quat = decoded.new_zeros(b, t, 4)
        quat[..., 0] = 1.0
        xyz = decoded[..., :3]
        return quat, xyz

    monkeypatch.setattr(cl, "recover_root_rot_pos", fake_recover_root_rot_pos)

    decoded = [torch.zeros(1, 263)]
    gt = torch.zeros(1, 12, 7)
    gt[..., 3] = 1.0
    gt[0, 8, 0] = 1.0
    commit_masks = torch.ones(1, 1)
    weights = {
        "root_xz": 1.0,
        "root_y": 0.0,
        "heading": 0.0,
        "fwd_delta": 0.0,
        "yaw_delta": 0.0,
        "end_xz": 0.0,
    }

    loss, terms = cl.compute_body_aux_loss_on_commit_masks(
        decoded,
        gt,
        torch.tensor([12]),
        commit_masks,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.device("cpu"),
        weights,
        heading_form="cosine",
        window_start_tokens=torch.tensor([2]),
    )

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(0.5))
    assert abs(terms["root_xz"] - 0.5) < 1e-6
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_body_aux_loss.py::test_commit_body_aux_uses_step_mean_not_frame_mean tests/test_body_aux_loss.py::test_commit_body_aux_delta_uses_full_decoded_prefix tests/test_body_aux_loss.py::test_commit_body_aux_uses_window_start_token_for_gt_offset -q
```

Expected: fails with `AttributeError: module 'utils.training.control_loss' has no attribute 'compute_body_aux_loss_on_commit_masks'`.

- [ ] **Step 3: Implement the helper**

In `utils/training/control_loss.py`, add this function after `compute_body_aux_loss()`:

```python
def compute_body_aux_loss_on_commit_masks(
    decoded_list,
    gt_traj_7d,
    traj_length,
    commit_frame_masks,
    commit_sample_positions,
    sample_indices,
    device,
    weights: dict,
    heading_form: str = "cosine",
    sample_loss_mask=None,
    window_start_tokens=None,
    token_to_frame: int = 4,
):
    """Compute 7D body aux as a mean over committed rollout steps.

    `decoded_list` contains full/local decoded prefixes. `commit_frame_masks`
    has one row per commit record, not a union mask. Deltas are derived over the
    full decoded prefix before a per-commit mask selects frames.
    """
    from utils.local_frame import root_quat_to_physical_yaw
    from utils.token_frame import token_start_frame

    if not torch.is_tensor(commit_frame_masks):
        commit_frame_masks = torch.as_tensor(
            commit_frame_masks, device=device, dtype=torch.float32
        )
    else:
        commit_frame_masks = commit_frame_masks.to(device=device, dtype=torch.float32)
    if commit_frame_masks.ndim != 2:
        raise ValueError(
            f"commit_frame_masks must be [R,T], got {tuple(commit_frame_masks.shape)}"
        )
    if not torch.is_tensor(commit_sample_positions):
        commit_sample_positions = torch.as_tensor(
            commit_sample_positions, device=device, dtype=torch.long
        )
    else:
        commit_sample_positions = commit_sample_positions.to(device=device, dtype=torch.long)
    commit_sample_positions = commit_sample_positions.view(-1)
    if commit_sample_positions.numel() != commit_frame_masks.shape[0]:
        raise ValueError(
            "commit_sample_positions must have one entry per commit mask; "
            f"got {commit_sample_positions.numel()} for {commit_frame_masks.shape[0]}"
        )
    if not torch.is_tensor(sample_indices):
        sample_indices = torch.as_tensor(sample_indices, device=device, dtype=torch.long)
    else:
        sample_indices = sample_indices.to(device=device, dtype=torch.long)
    sample_indices = sample_indices.view(-1)
    if sample_indices.numel() != len(decoded_list):
        raise ValueError(
            "sample_indices must map each decoded prefix to an original batch row; "
            f"got {sample_indices.numel()} for {len(decoded_list)} decoded prefixes"
        )
    if not torch.is_tensor(traj_length):
        traj_length = torch.as_tensor(traj_length, device=device, dtype=torch.long)
    else:
        traj_length = traj_length.to(device=device, dtype=torch.long)
    traj_length = traj_length.view(-1)

    starts = None
    if window_start_tokens is None:
        starts = torch.zeros(len(decoded_list), device=device, dtype=torch.long)
    elif not torch.is_tensor(window_start_tokens):
        starts = torch.as_tensor(window_start_tokens, device=device, dtype=torch.long)
    else:
        starts = window_start_tokens.to(device=device, dtype=torch.long)
    starts = starts.view(-1)
    if starts.numel() == 1 and len(decoded_list) > 1:
        starts = starts.expand(len(decoded_list))
    if starts.numel() != len(decoded_list):
        raise ValueError(
            "window_start_tokens must provide one GT offset per decoded prefix; "
            f"got {starts.numel()} starts for {len(decoded_list)} decoded prefixes"
        )

    decoded_cache = []
    for decoded_pos, decoded in enumerate(decoded_list):
        orig_i = int(sample_indices[decoded_pos].item())
        decoded = decoded.to(device=device).float()
        quat, xyz = recover_root_rot_pos(decoded.unsqueeze(0))
        yaw = root_quat_to_physical_yaw(quat)
        gt_len = min(int(traj_length[orig_i].item()), gt_traj_7d.shape[1])
        window_start_token = int(starts[decoded_pos].item())
        gt_start_f = token_start_frame(window_start_token, token_to_frame)
        if gt_start_f >= gt_len:
            raise ValueError(
                "window_start_tokens must reference a valid GT start frame; "
                f"sample={orig_i}, start_token={window_start_token}, "
                f"start_frame={gt_start_f}, traj_length={gt_len}"
            )
        end_f = min(
            decoded.shape[0],
            gt_len - gt_start_f,
            commit_frame_masks.shape[1],
        )
        if end_f <= 0:
            decoded_cache.append(None)
            continue
        gt7 = gt_traj_7d[
            orig_i : orig_i + 1, gt_start_f : gt_start_f + end_f, :
        ].to(
            device=device, dtype=xyz.dtype
        )
        gt_xyz = gt7[..., :3]
        gt_yaw = torch.atan2(gt7[..., 4], gt7[..., 3])
        pred_xyz = xyz[:, :end_f, :]
        pred_yaw = yaw[:, :end_f]
        anchor7 = gt_traj_7d[
            orig_i : orig_i + 1, gt_start_f : gt_start_f + 1, :
        ].to(device=device, dtype=xyz.dtype)
        anchor_xyz = anchor7[..., :3]
        anchor_yaw = torch.atan2(anchor7[..., 4], anchor7[..., 3])
        pred_xyz, pred_yaw = canonicalize_pose_to_anchor(
            pred_xyz, pred_yaw, anchor_xyz, anchor_yaw
        )
        gt_xyz, gt_yaw = canonicalize_pose_to_anchor(
            gt_xyz, gt_yaw, anchor_xyz, anchor_yaw
        )
        decoded_cache.append((orig_i, end_f, pred_xyz, pred_yaw, gt_xyz, gt_yaw))

    losses = []
    term_sums = {
        k: 0.0
        for k in ("root_xz", "root_y", "heading", "fwd_delta", "yaw_delta", "end_xz")
    }
    for record_idx in range(commit_frame_masks.shape[0]):
        decoded_pos = int(commit_sample_positions[record_idx].item())
        if decoded_pos < 0 or decoded_pos >= len(decoded_cache):
            continue
        cached = decoded_cache[decoded_pos]
        if cached is None:
            continue
        orig_i, end_f, pred_xyz, pred_yaw, gt_xyz, gt_yaw = cached
        mask_i = commit_frame_masks[record_idx : record_idx + 1, :end_f].to(
            device=device, dtype=pred_xyz.dtype
        )
        if mask_i.sum().item() <= 0:
            continue
        slm_i = None
        if sample_loss_mask is not None:
            if float(sample_loss_mask[orig_i]) <= 0:
                continue
            slm_i = sample_loss_mask[orig_i : orig_i + 1].to(device)
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
        losses.append(total_i)
        for key in term_sums:
            term_sums[key] += float(terms_i[key].detach())

    if not losses:
        return None, {}
    loss = torch.stack(losses).mean()
    denom = float(len(losses))
    metrics = {key: value / denom for key, value in term_sums.items()}
    metrics["valid_count"] = denom
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
git commit -m "test: cover step-mean commit body aux helper"
```

---

### Task 3: Collect Commit Records and Preserve Replacement Semantics

**Files:**
- Modify: `utils/training/self_forcing.py`
- Create: `tests/test_multistep_commit_body_aux.py`

- [ ] **Step 1: Write failing rollout tests**

Create `tests/test_multistep_commit_body_aux.py` with:

```python
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from utils.training.self_forcing import RolloutPlan, SelfForcingTrainer


def _cfg(
    commit_enabled: bool,
    disable_replace: bool = True,
    strict_valid_commits: bool = False,
):
    def get(key, default=None):
        values = {
            "anchor_canonicalize": {"enabled": False},
            "history_corruption": {},
            "horizon_sim": {"enabled": False},
            "self_forcing_disable_replace": disable_replace,
            "multistep_commit_body_aux": {
                "enabled": commit_enabled,
                "decode_mode": "single",
                "replace_final_body_aux": True,
                "include_final_step": True,
                "reduction": "step_mean",
                "strict_valid_commits": strict_valid_commits,
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


def _trainer(
    commit_enabled: bool,
    k: int = 3,
    disable_replace: bool = True,
    chunk_size: int = 1,
    strict_valid_commits: bool = False,
):
    batch = 1
    seq_len = 5
    hidden = 4
    feature = torch.zeros(batch, seq_len, hidden)
    model = MagicMock(name="model")
    model.chunk_size = chunk_size
    model.self_forcing_stride_tokens = 1
    model.self_forcing_k_schedule = [(0.0, k)]
    model._decide_text_dropout.return_value = torch.zeros(batch, dtype=torch.bool)
    model._prepare_text_context.return_value = None
    model._decide_traj_dropout.return_value = False
    model._prepare_traj_condition.return_value = (None, None, False, None)

    grad_flags = []
    seen_features = []

    def forward(_model_batch, current_feature, *args, **kwargs):
        grad_flags.append(torch.is_grad_enabled())
        seen_features.append(current_feature.detach().clone())
        pred = torch.ones(seq_len, hidden, requires_grad=torch.is_grad_enabled())
        repl = torch.full((seq_len, hidden), 2.0)
        return {
            "loss": pred.sum() * 0.0,
            "pred_x0_latent_list": [pred],
            "x0_latent_list": [repl],
        }

    model._forward_single_window.side_effect = forward
    module = SimpleNamespace(
        model=model,
        cfg=_cfg(commit_enabled, disable_replace, strict_valid_commits),
    )
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
    return trainer, model_batch, grad_flags, seen_features


def test_default_rollout_keeps_non_final_steps_no_grad():
    trainer, model_batch, grad_flags, _ = _trainer(commit_enabled=False, k=3)

    final_result, k = trainer._run_rollout(model_batch, progress=1.0)

    assert k == 3
    assert final_result is not None
    assert grad_flags == [False, False, True]


def test_commit_aux_rollout_enables_grad_for_every_step_and_records_commits():
    trainer, model_batch, grad_flags, _ = _trainer(commit_enabled=True, k=3)

    final_result, k = trainer._run_rollout(model_batch, progress=1.0)

    assert k == 3
    assert final_result is not None
    assert grad_flags == [True, True, True]
    records = trainer._last_commit_token_records
    assert [int(r.local_commit_idx) for r in records] == [0, 1, 2]
    assert [int(r.global_commit_idx) for r in records] == [0, 1, 2]
    assert all(r.pred_token.requires_grad for r in records)


def test_replacement_uses_detached_x0_latent_list_not_pred_x0():
    trainer, model_batch, _, seen_features = _trainer(
        commit_enabled=True, k=2, disable_replace=False
    )

    trainer._run_rollout(model_batch, progress=1.0)

    assert torch.equal(seen_features[1][0, 0], torch.full((4,), 2.0))
    assert not seen_features[1].requires_grad


def test_commit_aux_rejects_chunk_size_greater_than_one_for_first_version():
    trainer, model_batch, _, _ = _trainer(commit_enabled=True, k=2, chunk_size=2)

    try:
        trainer._run_rollout(model_batch, progress=1.0)
    except NotImplementedError as exc:
        assert "chunk_size == 1" in str(exc)
    else:
        raise AssertionError("expected chunk_size > 1 to be rejected")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py -q
```

Expected: commit-aux tests fail because `_last_commit_token_records` does not exist, non-final steps remain `no_grad`, and the chunk-size guard does not exist.

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
        commit_cfg = self._commit_body_aux_cfg()
        commit_aux_enabled = bool(commit_cfg.get("enabled", False))
        if commit_aux_enabled and int(model.chunk_size) != 1:
            raise NotImplementedError(
                "multistep_commit_body_aux initial implementation requires "
                "model.chunk_size == 1; add commit span records before enabling "
                "chunk_size > 1"
            )
        include_final_commit = bool(commit_cfg.get("include_final_step", True))
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

Replace the final/no-grad split with this pattern:

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
            should_record_commit = commit_aux_enabled and (
                include_final_commit or not is_final_step
            )
            if should_record_commit:
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

Expected: rollout tests pass.

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

### Task 4: Assemble Decode Prefixes and Per-Commit Frame Masks

**Files:**
- Modify: `utils/training/self_forcing.py`
- Modify: `tests/test_multistep_commit_body_aux.py`

- [ ] **Step 1: Add failing prefix assembly tests**

Append to `tests/test_multistep_commit_body_aux.py`:

```python
def test_full_prefix_splice_assembly_decodes_only_to_global_commit_prefix():
    from utils.token_frame import token_range_to_frame_slice

    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
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

    decoded_latents, commit_masks, commit_positions, sample_indices, window_starts = (
        trainer._assemble_commit_decode_inputs(batch, model_batch)
    )

    assert decoded_latents[0].shape[0] == 4
    assert torch.equal(decoded_latents[0][3], record)
    assert decoded_latents[0][3].requires_grad
    assert not decoded_latents[0][2].requires_grad
    assert commit_positions.tolist() == [0]
    assert sample_indices.tolist() == [0]
    assert window_starts.tolist() == [0]
    sl = token_range_to_frame_slice(3, 1)
    assert commit_masks.shape[0] == 1
    assert commit_masks[0, sl.start:sl.stop].sum().item() == 4
    assert commit_masks[0, :sl.start].sum().item() == 0
    assert commit_masks[0, sl.stop:].sum().item() == 0


def test_local_decode_assembly_decodes_only_to_local_commit_prefix():
    from utils.token_frame import token_range_to_frame_slice

    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
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

    decoded_latents, commit_masks, commit_positions, sample_indices, window_starts = (
        trainer._assemble_commit_decode_inputs(batch, model_batch)
    )

    assert decoded_latents[0].shape[0] == 2
    assert torch.equal(decoded_latents[0][1], record)
    assert decoded_latents[0][1].requires_grad
    assert not decoded_latents[0][0].requires_grad
    assert commit_positions.tolist() == [0]
    assert sample_indices.tolist() == [0]
    assert window_starts.tolist() == [5]
    sl = token_range_to_frame_slice(1, 1)
    assert commit_masks[0, sl.start:sl.stop].sum().item() == 4
    assert commit_masks[0, :sl.start].sum().item() == 0
    assert commit_masks[0, sl.stop:].sum().item() == 0


def test_local_prefix_assembly_preserves_sample_indices_and_expands_scalar_starts():
    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    batch = {
        "token": torch.zeros(3, 6, 2),
        "token_length": torch.tensor([6, 6, 6]),
        "traj_cond_7d": torch.zeros(3, 24, 7),
        "traj_length": torch.tensor([24, 24, 24]),
    }
    model_batch = {
        "_window_local_body_aux_mode": "local_decode",
        "_window_global_start_token": torch.tensor([2]),
        "feature": torch.zeros(3, 4, 2),
        "feature_length": torch.tensor([4, 4, 4]),
    }
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=2,
            local_commit_idx=1,
            global_commit_idx=3,
            pred_token=torch.ones(2, requires_grad=True),
        )
    ]

    _, _, commit_positions, sample_indices, window_starts = (
        trainer._assemble_commit_decode_inputs(batch, model_batch)
    )

    assert commit_positions.tolist() == [0]
    assert sample_indices.tolist() == [2]
    assert window_starts.tolist() == [2]


def test_prefix_assembly_does_not_include_future_gt_suffix():
    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    batch = {
        "token": torch.zeros(1, 10, 2),
        "token_length": torch.tensor([10]),
        "traj_cond_7d": torch.zeros(1, 40, 7),
        "traj_length": torch.tensor([40]),
    }
    batch["token"][0, 4:, :] = 99.0
    model_batch = {
        "_window_local_body_aux_mode": "full_prefix_splice",
        "_window_global_start_token": torch.tensor([2]),
        "feature": torch.zeros(1, 4, 2),
        "feature_length": torch.tensor([4]),
    }
    trainer._last_commit_token_records = [
        SimpleNamespace(
            batch_idx=0,
            local_commit_idx=1,
            global_commit_idx=3,
            pred_token=torch.ones(2, requires_grad=True),
        )
    ]

    decoded_latents, _, _, _, _ = trainer._assemble_commit_decode_inputs(
        batch, model_batch
    )

    assert decoded_latents[0].shape[0] == 4
    assert not torch.any(decoded_latents[0] == 99.0)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py::test_full_prefix_splice_assembly_decodes_only_to_global_commit_prefix tests/test_multistep_commit_body_aux.py::test_local_decode_assembly_decodes_only_to_local_commit_prefix tests/test_multistep_commit_body_aux.py::test_local_prefix_assembly_preserves_sample_indices_and_expands_scalar_starts tests/test_multistep_commit_body_aux.py::test_prefix_assembly_does_not_include_future_gt_suffix -q
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
            return [], None, None, None, None
        mode = str(
            model_batch.get(
                "_window_local_body_aux_mode",
                batch.get("_window_local_body_aux_mode", "full_prefix_splice"),
            )
        )
        device = (
            self._module.device
            if hasattr(self._module, "device")
            else records[0].pred_token.device
        )
        if mode == "local_decode":
            source = model_batch["feature"]
            lengths = model_batch["feature_length"]
            use_global = False
            starts = model_batch.get(
                "_window_global_start_token", batch.get("_window_global_start_token")
            )
            if starts is None:
                starts = torch.zeros(source.shape[0], device=source.device, dtype=torch.long)
            elif not torch.is_tensor(starts):
                starts = torch.as_tensor(starts, device=source.device, dtype=torch.long)
            starts = starts.to(device=device, dtype=torch.long).view(-1)
            if starts.numel() == 1 and source.shape[0] > 1:
                starts = starts.expand(source.shape[0])
            window_starts = starts
        elif mode == "full_prefix_splice":
            source = batch["token"]
            lengths = batch.get("token_length")
            if lengths is None:
                lengths = torch.full(
                    (source.shape[0],),
                    source.shape[1],
                    device=source.device,
                    dtype=torch.long,
                )
            use_global = True
            window_starts = torch.zeros(source.shape[0], device=device, dtype=torch.long)
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

        def record_commit_idx(record):
            return int(record.global_commit_idx if use_global else record.local_commit_idx)

        sample_indices_list = []
        decoded_latents = []
        max_frames = 0
        for batch_idx in sorted(records_by_sample):
            max_commit_idx = max(record_commit_idx(r) for r in records_by_sample[batch_idx])
            decode_end_token = max_commit_idx + 1
            token_len = int(lengths[batch_idx].item())
            decode_end_token = min(decode_end_token, token_len)
            if decode_end_token <= 0:
                continue
            sample_indices_list.append(batch_idx)
            decoded_latents.append(
                source[batch_idx, :decode_end_token, :].detach().clone()
            )
            max_frames = max(
                max_frames,
                token_range_to_frame_slice(0, decode_end_token).stop,
            )

        sample_pos = {batch_idx: pos for pos, batch_idx in enumerate(sample_indices_list)}
        commit_masks = []
        commit_positions = []
        for record in records:
            batch_idx = int(record.batch_idx)
            if batch_idx not in sample_pos:
                continue
            decoded_pos = sample_pos[batch_idx]
            token_idx = int(record.global_commit_idx if use_global else record.local_commit_idx)
            write_idx = token_idx if use_global else int(record.local_commit_idx)
            if token_idx < 0 or write_idx < 0:
                continue
            if write_idx >= decoded_latents[decoded_pos].shape[0]:
                continue
            decoded_latents[decoded_pos][write_idx] = record.pred_token.to(
                device=device, dtype=decoded_latents[decoded_pos].dtype
            )
            mask = source.new_zeros(max_frames)
            sl = token_range_to_frame_slice(token_idx, 1)
            if sl.start < max_frames:
                mask[sl.start : min(sl.stop, max_frames)] = 1.0
            if mask.sum().item() <= 0:
                continue
            commit_masks.append(mask)
            commit_positions.append(decoded_pos)

        if not commit_masks:
            return decoded_latents, None, None, torch.as_tensor(
                sample_indices_list, device=device, dtype=torch.long
            ), None
        sample_indices = torch.as_tensor(sample_indices_list, device=device, dtype=torch.long)
        commit_frame_masks = torch.stack(commit_masks, dim=0)
        commit_sample_positions = torch.as_tensor(
            commit_positions, device=device, dtype=torch.long
        )
        if window_starts is not None:
            window_starts = window_starts[sample_indices]
        return (
            decoded_latents,
            commit_frame_masks,
            commit_sample_positions,
            sample_indices,
            window_starts,
        )
```

- [ ] **Step 4: Run prefix assembly tests**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py::test_full_prefix_splice_assembly_decodes_only_to_global_commit_prefix tests/test_multistep_commit_body_aux.py::test_local_decode_assembly_decodes_only_to_local_commit_prefix tests/test_multistep_commit_body_aux.py::test_local_prefix_assembly_preserves_sample_indices_and_expands_scalar_starts tests/test_multistep_commit_body_aux.py::test_prefix_assembly_does_not_include_future_gt_suffix -q
```

Expected: all four prefix assembly tests pass.

- [ ] **Step 5: Commit**

```bash
git add utils/training/self_forcing.py tests/test_multistep_commit_body_aux.py
git commit -m "feat: assemble commit decode prefixes"
```

---

### Task 5: Compute Step-Mean Commit 7D Body Aux and Replace Final Body Aux

**Files:**
- Modify: `utils/training/self_forcing.py`
- Modify: `tests/test_multistep_commit_body_aux.py`

- [ ] **Step 1: Add failing loss integration tests**

Append to `tests/test_multistep_commit_body_aux.py`:

```python
def test_commit_aux_replaces_final_body_aux(monkeypatch):
    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    trainer._module.device = torch.device("cpu")
    trainer._module.vae = MagicMock()
    trainer._module.cfg.model = SimpleNamespace(params={"control_loss_weight": 5.0})
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
        "utils.training.self_forcing.compute_body_aux_loss_on_commit_masks",
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
    assert trainer._last_commit_body_aux_valid_count == 1


def test_commit_aux_none_does_not_fallback_to_final_body_aux(monkeypatch):
    trainer, _, _, _ = _trainer(commit_enabled=True, k=1)
    trainer._module.device = torch.device("cpu")
    trainer._module.vae = MagicMock()
    trainer._module.cfg.model = SimpleNamespace(params={"control_loss_weight": 5.0})
    trainer._last_commit_token_records = []
    final_step_result = {
        "loss": torch.tensor(2.0, requires_grad=True),
        "pred_x0_latent_list": [torch.ones(1, 4, requires_grad=True)],
    }
    batch = {
        "token": torch.zeros(1, 1, 4),
        "token_length": torch.tensor([1]),
        "traj_cond_7d": torch.zeros(1, 1, 7),
        "traj_length": torch.tensor([1]),
    }
    model_batch = {
        "feature": torch.zeros(1, 1, 4),
        "feature_length": torch.tensor([1]),
    }
    monkeypatch.setattr(
        "utils.training.self_forcing._compute_body_aux_loss",
        MagicMock(side_effect=AssertionError("must not fallback")),
    )

    total, diff, control = trainer._compute_losses(
        final_step_result, batch, model_batch
    )

    assert torch.isclose(total, torch.tensor(2.0))
    assert torch.isclose(diff, torch.tensor(2.0))
    assert control is None
    assert trainer._last_commit_body_aux_valid_count == 0


def test_commit_aux_none_raises_in_strict_mode(monkeypatch):
    trainer, _, _, _ = _trainer(
        commit_enabled=True, k=1, strict_valid_commits=True
    )
    trainer._module.device = torch.device("cpu")
    trainer._module.vae = MagicMock()
    trainer._module.cfg.model = SimpleNamespace(params={"control_loss_weight": 5.0})
    trainer._last_commit_token_records = []
    final_step_result = {
        "loss": torch.tensor(2.0, requires_grad=True),
        "pred_x0_latent_list": [torch.ones(1, 4, requires_grad=True)],
    }
    batch = {
        "token": torch.zeros(1, 1, 4),
        "token_length": torch.tensor([1]),
        "traj_cond_7d": torch.zeros(1, 1, 7),
        "traj_length": torch.tensor([1]),
    }
    model_batch = {
        "feature": torch.zeros(1, 1, 4),
        "feature_length": torch.tensor([1]),
    }
    monkeypatch.setattr(
        "utils.training.self_forcing._compute_body_aux_loss",
        MagicMock(side_effect=AssertionError("must not fallback")),
    )

    try:
        trainer._compute_losses(final_step_result, batch, model_batch)
    except RuntimeError as exc:
        assert "no valid commit body aux loss" in str(exc)
    else:
        raise AssertionError("expected strict mode to reject missing commit loss")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest tests/test_multistep_commit_body_aux.py::test_commit_aux_replaces_final_body_aux tests/test_multistep_commit_body_aux.py::test_commit_aux_none_does_not_fallback_to_final_body_aux tests/test_multistep_commit_body_aux.py::test_commit_aux_none_raises_in_strict_mode -q
```

Expected: fails because `_compute_losses()` does not accept `model_batch` and commit aux is not integrated.

- [ ] **Step 3: Import helper and add decode wrapper**

In `utils/training/self_forcing.py`, update imports:

```python
from .control_loss import (
    compute_body_aux_loss,
    compute_body_aux_loss_on_commit_masks,
    compute_control_loss_xz,
)
```

Add method inside `SelfForcingTrainer`:

```python
    def _decode_commit_latents(self, decoded_latents: list[torch.Tensor]):
        return [
            self._module.vae.decode(latent.unsqueeze(0))[0].float()
            for latent in decoded_latents
        ]
```

- [ ] **Step 4: Add commit aux compute method**

Add inside `SelfForcingTrainer`:

```python
    def _compute_commit_body_aux_loss(self, batch: dict, model_batch: dict):
        self._last_commit_body_aux_valid_count = 0
        cfg = self._commit_body_aux_cfg()
        if not bool(cfg.get("enabled", False)):
            return None, {}
        if str(cfg.get("decode_mode", "single")) != "single":
            raise NotImplementedError(
                "multistep_commit_body_aux.decode_mode currently supports only 'single'"
            )
        if str(cfg.get("reduction", "step_mean")) != "step_mean":
            raise NotImplementedError(
                "multistep_commit_body_aux.reduction currently supports only 'step_mean'"
            )
        if "traj_cond_7d" not in batch:
            return None, {}
        (
            decoded_latents,
            commit_frame_masks,
            commit_sample_positions,
            sample_indices,
            window_starts,
        ) = self._assemble_commit_decode_inputs(batch, model_batch)
        if not decoded_latents or commit_frame_masks is None:
            return None, {}
        candidate_commit_count = int(commit_frame_masks.shape[0])
        decoded = self._decode_commit_latents(decoded_latents)
        ba_cfg = self._module.cfg.get("body_aux_loss", {}) or {}
        weights = {
            **_DEFAULT_BODY_AUX_WEIGHTS,
            **(ba_cfg.get("weights", {}) or {}),
        }
        loss, terms = compute_body_aux_loss_on_commit_masks(
            decoded,
            batch["traj_cond_7d"],
            batch["traj_length"],
            commit_frame_masks,
            commit_sample_positions,
            sample_indices,
            self._module.device,
            weights,
            heading_form=ba_cfg.get("heading_form", "cosine"),
            sample_loss_mask=getattr(self, "_last_sample_loss_mask", None),
            window_start_tokens=window_starts,
        )
        if loss is None:
            self._last_commit_body_aux_valid_count = 0
            return None, {}
        valid_count = int(float(terms.get("valid_count", candidate_commit_count)))
        self._last_commit_body_aux_valid_count = valid_count
        return loss, terms
```

- [ ] **Step 5: Change `_compute_losses()` signature and no-fallback logic**

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
            step_control_loss, commit_terms = self._compute_commit_body_aux_loss(
                batch, model_batch
            )
            self._last_body_aux_terms = commit_terms
            if step_control_loss is not None:
                total_loss = total_loss + control_weight * step_control_loss
            elif bool(commit_cfg.get("strict_valid_commits", False)):
                raise RuntimeError(
                    "multistep_commit_body_aux produced no valid commit body aux loss"
                )
            if replace_final:
                return total_loss, step_diff_loss, step_control_loss
```

Keep the existing final-step body aux logic below this for disabled mode or
`replace_final_body_aux=false`. In `_self_forcing_step()`, update the call:

```python
        total_loss, step_diff_loss, step_control_loss = self._compute_losses(
            final_step_result, batch, model_batch
        )
```

Also add commit-validity metrics before logging:

```python
        commit_valid_count = getattr(self, "_last_commit_body_aux_valid_count", None)
        if commit_valid_count is not None:
            runtime_metrics["body_aux/commit_valid_count"] = float(commit_valid_count)
            runtime_metrics["body_aux/commit_loss_skipped"] = (
                1.0 if int(commit_valid_count) == 0 else 0.0
            )
            self._last_commit_body_aux_valid_count = None
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
git commit -m "feat: add step-mean commit body aux loss"
```

---

### Task 6: Full Verification

**Files:**
- Modify only if needed after failures: `configs/ldf.yaml`, `utils/training/self_forcing.py`, `utils/training/control_loss.py`, tests touched above.

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

- [ ] **Step 4: Document diagnostic run overrides**

Use these overrides for the first experiment:

```yaml
model:
  params:
    self_forcing_k_schedule:
      - [0.0, 3]
    traj_dropout: 0.0
multistep_commit_body_aux:
  enabled: true
```

Expected: K=3, no trajectory-condition dropout, single-decode step-mean commit body aux replaces old final-step body aux.

The commit body-aux term scale is controlled by `model.params.control_loss_weight`
and `body_aux_loss.weights`. Do not add `multistep_commit_body_aux.weight` or
`multistep_commit_body_aux.weights` for this diagnostic.

- [ ] **Step 5: Final commit if verification-only fixes were needed**

If Task 6 required code/test fixes, commit them:

```bash
git add configs/ldf.yaml utils/training/self_forcing.py utils/training/control_loss.py tests/test_multistep_commit_body_aux.py tests/test_body_aux_loss.py
git commit -m "fix: verify step-mean commit body aux"
```

If no fixes were needed, do not create an empty commit.
