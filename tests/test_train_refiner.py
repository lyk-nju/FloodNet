"""Unit tests for RootRefiner training helpers."""

from __future__ import annotations

import torch
import pytest

from datasets.humanml3d_refiner import HumanML3DRefinerDataset
from datasets.humanml3d_refiner import HumanML3DRefinerDataset as RefinerDataset
from train_refiner import (
    FrozenStubTextEncoder,
    RefinerLightningModule,
    masked_mean,
    refiner_collate,
    second_order_diff_l2,
    smooth_l1_masked,
)


# ---------------------------------------------------------------------------
# Masked loss helpers
# ---------------------------------------------------------------------------


def test_smooth_l1_masked_ignores_masked_frames():
    pred = torch.zeros(2, 4, 3)
    gt = torch.ones(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, :2] = True   # only first 2 frames valid
    loss = smooth_l1_masked(pred, gt, mask)
    # SmoothL1(0, 1) = 0.5 for each element; all valid frames have same diff.
    assert abs(loss.item() - 0.5) < 1e-6


def test_smooth_l1_masked_zero_when_no_valid():
    pred = torch.randn(2, 4, 3)
    gt = torch.randn(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    assert smooth_l1_masked(pred, gt, mask).item() == 0.0


def test_masked_mean_basic():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, True, False, False]])
    assert abs(masked_mean(values, mask).item() - 1.5) < 1e-6


def test_second_order_diff_l2_zero_on_linear_ramp():
    """A linear ramp has zero 2nd-order difference."""
    B, T, C = 1, 6, 2
    ramp = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(B, T, C).contiguous()
    mask = torch.ones(B, T, dtype=torch.bool)
    assert second_order_diff_l2(ramp, mask).item() < 1e-9


def test_second_order_diff_l2_nonzero_on_curved():
    B, T = 1, 6
    curve = (torch.arange(T, dtype=torch.float32) ** 2).view(1, T, 1)
    mask = torch.ones(B, T, dtype=torch.bool)
    # 2nd diff of t^2 is constant 2 → L2 = 4
    assert abs(second_order_diff_l2(curve, mask).item() - 4.0) < 1e-5


# ---------------------------------------------------------------------------
# Stub text encoder
# ---------------------------------------------------------------------------


def test_stub_text_encoder_deterministic_and_frozen():
    enc = FrozenStubTextEncoder(emb_dim=32)
    a = enc.encode(["walk forward", "turn left"])
    b = enc.encode(["walk forward", "turn left"])
    assert a.shape == (2, 32)
    assert torch.equal(a, b)   # deterministic
    # frozen: no trainable params
    assert all(not p.requires_grad for p in enc.parameters())


def test_stub_text_encoder_id_is_process_stable_hashlib_not_builtin_hash():
    """Lock-in for the review fix: _stable_id must use hashlib (process-stable),
    NOT builtin hash() (PYTHONHASHSEED-salted). Verify against a precomputed
    hashlib md5 value so a regression back to hash() is caught.
    """
    import hashlib

    vocab = 4096
    text = "a person walks forward"
    expected = int.from_bytes(
        hashlib.md5(text.encode("utf-8")).digest()[:8], "big",
    ) % vocab
    assert FrozenStubTextEncoder._stable_id(text, vocab) == expected
    # And it must NOT equal builtin hash()'s result mapping (which is salted) —
    # we can't assert inequality reliably, but we CAN assert determinism here.
    assert FrozenStubTextEncoder._stable_id(text, vocab) == \
        FrozenStubTextEncoder._stable_id(text, vocab)


# ---------------------------------------------------------------------------
# build_datasets: train/val tuple via the real-layout loader (fake HumanML3D)
# ---------------------------------------------------------------------------


def _make_fake_humanml3d(root, train_names, val_names):
    import numpy as np
    ds = root / "HumanML3D"
    (ds / "new_joint_vecs").mkdir(parents=True)
    (ds / "texts").mkdir(parents=True)
    (ds / "train.txt").write_text("\n".join(train_names) + "\n")
    (ds / "val.txt").write_text("\n".join(val_names) + "\n")
    for n in set(train_names) | set(val_names):
        arr = np.zeros((50, 263), dtype=np.float32)
        arr[:, 2] = 0.05
        arr[:, 3] = 1.0
        np.save(ds / "new_joint_vecs" / f"{n}.npy", arr)
        (ds / "texts" / f"{n}.txt").write_text(f"a person does {n}#x#0#0\n")
    return ds


def test_build_datasets_returns_train_and_val(tmp_path):
    from train_refiner import build_datasets

    _make_fake_humanml3d(tmp_path, ["t1", "t2", "t3"], ["v1", "v2"])
    cfg = _tiny_cfg()
    cfg["data"] = {
        "raw_data_dir": str(tmp_path),
        "dataset": "humanml3d",
        "train_split_file": "train.txt",
        "val_split_file": "val.txt",
        "feature_path": "new_joint_vecs",
        "text_path": "texts",
        # no stats_dir → normalize=False
    }
    train_ds, val_ds = build_datasets(cfg)
    assert len(train_ds) == 3
    assert val_ds is not None and len(val_ds) == 2


def test_module_raises_when_no_encoder_and_no_debug_stub():
    """Real-training guard: without an explicit encoder and without
    text_encoder.debug_stub, init must raise (not silently use the stub)."""
    import pytest

    cfg = _tiny_cfg()
    cfg["text_encoder"] = {"share_with": "ldf"}   # debug_stub absent/false
    with pytest.raises(NotImplementedError):
        RefinerLightningModule(cfg)


def test_module_uses_explicit_encoder_over_stub():
    cfg = _tiny_cfg()
    cfg["text_encoder"] = {"share_with": "ldf"}   # no debug_stub
    enc = FrozenStubTextEncoder(emb_dim=cfg["model"]["params"]["text_emb_dim"])
    module = RefinerLightningModule(cfg, text_encoder=enc)
    assert module.text_encoder is enc


def test_module_requires_waypoint_stats_when_normalizing(tmp_path):
    cfg = _tiny_cfg()
    cfg["data"]["normalize"] = True
    cfg["data"]["stats_dir"] = str(tmp_path / "missing_stats")

    with pytest.raises(FileNotFoundError, match="stats_dir"):
        RefinerLightningModule(cfg)


def test_build_datasets_val_none_when_no_val_split(tmp_path):
    from train_refiner import build_datasets

    _make_fake_humanml3d(tmp_path, ["t1", "t2"], ["v1"])
    cfg = _tiny_cfg()
    cfg["data"] = {
        "raw_data_dir": str(tmp_path),
        "dataset": "humanml3d",
        "train_split_file": "train.txt",
        # val_split_file omitted
        "feature_path": "new_joint_vecs",
        "text_path": "texts",
    }
    train_ds, val_ds = build_datasets(cfg)
    assert len(train_ds) == 2
    assert val_ds is None


# ---------------------------------------------------------------------------
# Loss dict key alignment
# ---------------------------------------------------------------------------


def _tiny_cfg():
    return {
        "model": {
            "target": "models.root_refiner.RootRefiner",
            "ema_decay": None,
            "params": {
                "d_model": 32, "n_layers": 2, "n_heads": 4, "ff_dim": 64,
                "max_tokens": 8, "min_tokens": 2, "frames_per_token": 4,
                "n_path": 16, "n_hist": 8, "text_emb_dim": 16,
                "path_features_dim": 5, "dropout": 0.0,
            },
        },
        "data": {
            "target": "datasets.humanml3d_refiner.HumanML3DRefinerDataset",
            "collate_fn": "datasets.humanml3d_refiner.refiner_collate",
            "train_bs": 4,
            "val_bs": 4,
            "num_workers": 0,
        },
        "optimizer": {
            "target": "AdamW",
            "params": {"lr": 1e-3, "weight_decay": 0.01},
        },
        "lr_scheduler": {"target": None, "params": {}},
        "loss": {"heading_form": "cosine"},
        "loss_weights": {
            "num_token": 1.0, "num_token_soft": 0.1, "xyz": 5.0, "heading": 1.0,
            "fwd_delta": 0.5, "yaw_delta": 0.5, "path_control": 0.0,
            "smoothness": 0.0,
        },
        # tests opt into the debug stub explicitly (real training must wire LDF).
        "text_encoder": {"debug_stub": True},
    }


def _make_batch(module, B=2):
    m = module.refiner
    g = torch.Generator().manual_seed(0)
    waypoints = torch.zeros(B, m.max_frames, 7)
    yaw = torch.randn(B, m.max_frames, generator=g) * 0.3
    waypoints[..., 3] = torch.cos(yaw)
    waypoints[..., 4] = torch.sin(yaw)
    return {
        "text": ["walk"] * B,
        "mode": ["full"] * B,
        "path": torch.randn(B, m.n_path, 2, generator=g),
        "path_valid_mask": torch.ones(B, m.n_path, dtype=torch.bool),
        "path_features": torch.randn(B, m.path_features_dim, generator=g),
        "history_motion": torch.randn(B, m.n_hist, 5, generator=g),
        "history_mask": torch.ones(B, m.n_hist, dtype=torch.bool),
        "waypoints": waypoints[..., :5],
        "waypoints_mask": torch.ones(B, m.max_frames, dtype=torch.bool),
        "num_tokens": torch.tensor([3, 5]),
    }


def test_lightning_module_rejects_legacy_refiner_batch_keys():
    module = RefinerLightningModule(_tiny_cfg())
    batch = _make_batch(module)
    path = batch.pop("path")
    path_valid_mask = batch.pop("path_valid_mask")
    path_features = batch.pop("path_features")
    history_motion = batch.pop("history_motion")
    legacy_batch = dict(batch)
    legacy_batch.update(
        {
            "xz_path": path,
            "path_mask": path_valid_mask,
            "path_stats": path_features,
            "current_motion": history_motion,
        }
    )

    with pytest.raises(KeyError):
        module(legacy_batch)


def test_loss_dict_keys_match_config_weights_and_no_speed():
    module = RefinerLightningModule(_tiny_cfg())
    batch = _make_batch(module)
    out = module(batch)
    losses = module._compute_loss(out, batch)
    # Loss-term keys + the logged-only num_token diagnostic metrics.
    expected = {"loss", "num_token", "num_token_soft", "xyz", "heading",
                "fwd_delta", "yaw_delta", "path_control", "smoothness"}
    expected |= set(RefinerLightningModule.METRIC_KEYS)
    assert set(losses.keys()) == expected
    assert "speed" not in losses
    # all finite
    for k, v in losses.items():
        assert torch.isfinite(v).all(), f"{k} not finite"


def test_num_token_metrics_follow_actual_pred_num_tokens_not_argmax():
    module = RefinerLightningModule(_tiny_cfg())
    batch = _make_batch(module, B=2)
    out = {
        "num_token_logits": torch.tensor(
            [
                [3.0, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9],
                [3.0, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9],
            ],
        ),
        "expected_num_tokens": torch.tensor([5.0, 5.0]),
        "pred_num_tokens": torch.tensor([5, 5]),
        "used_num_tokens": batch["num_tokens"],
        "waypoints": batch["waypoints"].clone(),
    }
    batch["num_tokens"] = torch.tensor([5, 5])

    losses = module._compute_loss(out, batch)

    assert losses["num_token_mae"].item() == 0.0
    assert losses["num_token_argmax_mae"].item() > 0.0


def test_delta_helper_unnormalizes_waypoints_before_deriving_physical_7d():
    module = RefinerLightningModule(_tiny_cfg())
    module._wp_mean = torch.tensor([10.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    module._wp_std = torch.tensor([2.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0])
    module._wp_norm_idx = torch.tensor([0, 1, 2])
    wp5 = torch.zeros(1, 2, 5)
    wp5[..., 3] = 1.0
    wp5[0, 1, 0] = 1.0
    wp5[0, 1, 2] = 1.0

    physical = module._to_physical_7d(wp5)

    assert torch.allclose(physical[0, 1, :3], torch.tensor([12.0, 0.0, -1.0]))


def test_num_token_soft_term_present_and_differentiable():
    """The soft-argmax expected-token aux term is a weighted loss term, finite,
    and its gradient reaches num_token_head (so it actually shapes the logits)."""
    module = RefinerLightningModule(_tiny_cfg())
    batch = _make_batch(module)
    losses = module._compute_loss(module(batch), batch)
    assert "num_token_soft" in losses and "num_token_soft_mae" in losses
    assert torch.isfinite(losses["num_token_soft"]).all()
    assert "num_token_soft" not in RefinerLightningModule.METRIC_KEYS      # weighted loss term
    assert "num_token_soft_mae" in RefinerLightningModule.METRIC_KEYS      # logged-only metric
    losses["num_token_soft"].backward()
    assert any(
        p.grad is not None
        for p in module.refiner.num_token_head.parameters()
    )


def test_soft_argmax_expected_value_is_distance_aware():
    """Soft-argmax expected class = sum_k p_k * k tracks the logit peak (so the
    Huber aux penalizes by token distance, not 0/1 like CE)."""
    K = 7
    logits = torch.full((1, K), -10.0)
    logits[0, 3] = 10.0
    probs = logits.softmax(dim=-1)
    expected = (probs * torch.arange(K, dtype=probs.dtype)).sum(-1)
    assert abs(expected.item() - 3.0) < 1e-2
    # SmoothL1 to a far target is larger than to a near target (distance-aware).
    import torch.nn.functional as F
    near = F.smooth_l1_loss(expected, torch.tensor([3.0]))
    far = F.smooth_l1_loss(expected, torch.tensor([0.0]))
    assert far > near


def test_loss_weights_keys_align_with_compute_loss_keys():
    """The weighted-sum keys must be a subset of the produced loss keys
    (so no weight silently has no matching loss term, and vice versa)."""
    cfg = _tiny_cfg()
    module = RefinerLightningModule(cfg)
    batch = _make_batch(module)
    losses = module._compute_loss(module(batch), batch)
    weight_keys = set(cfg["loss_weights"].keys())
    # Exclude "loss" (the total) and the logged-only diagnostic metric keys.
    loss_keys = set(losses.keys()) - {"loss"} - set(RefinerLightningModule.METRIC_KEYS)
    assert weight_keys == loss_keys, (
        f"weight keys {weight_keys} != loss term keys {loss_keys}"
    )


# ---------------------------------------------------------------------------
# Forward + backward step
# ---------------------------------------------------------------------------


def test_training_step_produces_finite_loss_and_grads():
    module = RefinerLightningModule(_tiny_cfg())
    batch = _make_batch(module)
    out = module(batch)
    losses = module._compute_loss(out, batch)
    losses["loss"].backward()
    n_bad = sum(
        1 for p in module.parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    )
    assert n_bad == 0


def test_training_step_skips_batch_on_nonfinite_loss(monkeypatch):
    """A non-finite loss must NOT propagate to the optimizer step: training_step
    returns None (Lightning's skip-batch signal) so a single bad batch cannot
    poison every parameter with NaN."""
    module = RefinerLightningModule(_tiny_cfg())
    batch = _make_batch(module)

    def _nan_loss(out, b):
        losses = RefinerLightningModule._compute_loss(module, out, b)
        losses["loss"] = losses["loss"] * float("nan")
        return losses

    monkeypatch.setattr(module, "_compute_loss", _nan_loss)
    monkeypatch.setattr(module, "log", lambda *a, **k: None)   # no Trainer attached
    assert module.training_step(batch, 0) is None


def test_training_step_returns_finite_loss_on_good_batch():
    """The guard must not regress the normal path: a finite loss is returned."""
    module = RefinerLightningModule(_tiny_cfg())
    batch = _make_batch(module)
    import types

    logged = {}
    module.log = types.MethodType(
        lambda self, k, v, **kw: logged.__setitem__(k, v), module)
    loss = module.training_step(batch, 0)
    assert loss is not None and torch.isfinite(loss).all()
    assert "train/loss" in logged and "train/nonfinite_skip" not in logged


def test_configure_optimizers_returns_adamw():
    module = RefinerLightningModule(_tiny_cfg())
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _make_clip(T: int):
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 2] = 0.05
    motion[:, 3] = 1.0
    return {"motion_263": motion, "text": "walk"}


def test_refiner_collate_stacks_tensors_and_keeps_text_list():
    from datasets.humanml3d_refiner import refiner_collate as dataset_refiner_collate

    ds = HumanML3DRefinerDataset([_make_clip(40) for _ in range(3)], full_plan_ratio=1.0, seed=0)
    samples = [ds[0], ds[1], ds[2]]
    assert dataset_refiner_collate is refiner_collate
    batch = dataset_refiner_collate(samples)
    assert isinstance(batch["text"], list) and len(batch["text"]) == 3
    assert batch["path"].shape[0] == 3
    assert batch["waypoints"].shape[0] == 3
    assert batch["num_tokens"].shape == (3,)


# ---------------------------------------------------------------------------
# Lightning smoke fit (single + multi step, no crash)
# ---------------------------------------------------------------------------


def test_lightning_smoke_fit_runs_a_few_steps(tmp_path):
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader

    clips = [_make_clip(50) for _ in range(8)]
    ds = HumanML3DRefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                                  full_plan_ratio=1.0, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=refiner_collate,
                         drop_last=True)
    module = RefinerLightningModule(_tiny_cfg())

    trainer = pl.Trainer(
        max_steps=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(module, loader)
    assert trainer.global_step >= 1


def test_lightning_resume_from_checkpoint(tmp_path):
    """Save a checkpoint after a short fit, then resume — must not crash."""
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader

    clips = [_make_clip(50) for _ in range(8)]
    ds = HumanML3DRefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                                  full_plan_ratio=1.0, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=refiner_collate,
                         drop_last=True)
    module = RefinerLightningModule(_tiny_cfg())

    ckpt_path = tmp_path / "ckpt.ckpt"
    trainer = pl.Trainer(
        max_steps=2, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(module, loader)
    trainer.save_checkpoint(str(ckpt_path))
    assert ckpt_path.is_file()

    # Resume.
    module2 = RefinerLightningModule(_tiny_cfg())
    trainer2 = pl.Trainer(
        max_steps=4, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    trainer2.fit(module2, loader, ckpt_path=str(ckpt_path))
    assert trainer2.global_step >= 2
