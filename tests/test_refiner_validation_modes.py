from __future__ import annotations

import types

import torch

from train_refiner import RefinerLightningModule


def _cfg() -> dict:
    return {
        "model": {
            "target": "models.root_refiner.RootRefiner",
            "params": {
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "ff_dim": 64,
                "max_tokens": 8,
                "min_tokens": 2,
                "frames_per_token": 4,
                "n_path": 16,
                "n_hist": 8,
                "text_emb_dim": 16,
                "path_features_dim": 5,
                "dropout": 0.0,
            },
        },
        "data": {
            "target": "datasets.humanml3d_refiner.HumanML3DRefinerDataset",
            "collate_fn": "datasets.humanml3d_refiner.refiner_collate",
            "train_bs": 4,
            "val_bs": 4,
            "num_workers": 0,
        },
        "optimizer": {"target": "AdamW", "params": {"lr": 1e-3, "weight_decay": 0.01}},
        "validation": {
            "eval_modes": ["groundtruth_duration", "pred_duration"],
            "suites": [
                {"name": "full_dense_max"},
                {"name": "sliding_dense_random"},
            ],
        },
        "loss": {"heading_form": "cosine"},
        "loss_weights": {
            "num_token": 1.0,
            "num_token_soft": 0.1,
            "xyz": 5.0,
            "heading": 1.0,
            "fwd_delta": 0.5,
            "yaw_delta": 0.5,
            "path_control": 0.0,
            "smoothness": 0.0,
        },
        "text_encoder": {"debug_stub": True},
    }


def _batch(module: RefinerLightningModule, B: int = 2) -> dict:
    g = torch.Generator().manual_seed(123)
    m = module.refiner
    waypoints = torch.zeros(B, m.max_frames, 5)
    waypoints[..., 3] = 1.0
    return {
        "text": ["walk"] * B,
        "path_mode": ["dense_path"] * B,
        "path": torch.randn(B, m.n_path, 2, generator=g),
        "path_valid_mask": torch.ones(B, m.n_path, dtype=torch.bool),
        "path_control_mask": torch.ones(B, m.n_path, dtype=torch.bool),
        "path_features": torch.randn(B, 5, generator=g),
        "history_motion": torch.randn(B, m.n_hist, 5, generator=g),
        "history_mask": torch.ones(B, m.n_hist, dtype=torch.bool),
        "waypoints": waypoints,
        "waypoints_mask": torch.ones(B, m.max_frames, dtype=torch.bool),
        "path_supervision_mask": torch.ones(B, m.max_frames, dtype=torch.bool),
        "offset_start_frames": torch.zeros(B, dtype=torch.long),
        "num_tokens": torch.tensor([3, 5]),
    }


def test_forward_duration_modes_choose_gt_or_predicted_horizon():
    module = RefinerLightningModule(_cfg())
    batch = _batch(module)

    gt_out = module(batch, duration_mode="groundtruth_duration")
    pred_out = module(batch, duration_mode="pred_duration")

    assert torch.equal(gt_out["used_num_tokens"], batch["num_tokens"])
    assert torch.equal(pred_out["used_num_tokens"], pred_out["pred_num_tokens"])


def test_common_prefix_mask_uses_predicted_duration_horizon():
    module = RefinerLightningModule(_cfg())
    batch = _batch(module, B=2)
    out = {
        "used_num_tokens": torch.tensor([2, 4]),
    }

    mask = module._common_prefix_mask(batch, out)

    expected_counts = torch.tensor([
        module.refiner.frames_per_token * 2 - (module.refiner.frames_per_token - 1),
        module.refiner.frames_per_token * 4 - (module.refiner.frames_per_token - 1),
    ])
    assert torch.equal(mask.sum(dim=1).cpu(), expected_counts)


def test_validation_step_logs_suite_and_duration_prefixes():
    module = RefinerLightningModule(_cfg())
    batch = _batch(module)
    logged = {}
    log_kwargs = {}
    module.log = types.MethodType(
        lambda self, key, value, **kwargs: (
            logged.__setitem__(key, value),
            log_kwargs.__setitem__(key, kwargs),
        ),
        module,
    )

    module.validation_step(batch, 0, dataloader_idx=1)

    assert "val_sliding_dense_random/groundtruth_duration/loss" in logged
    assert "val_sliding_dense_random/pred_duration/loss" in logged
    assert "val_sliding_dense_random/groundtruth_duration/xyz_ADE_m" in logged
    assert "val_sliding_dense_random/pred_duration/xyz_FDE_m" in logged
    assert not any("dataloader_idx" in key for key in logged)
    assert all(kwargs["sync_dist"] is True for kwargs in log_kwargs.values())
    assert all(kwargs["batch_size"] == 2 for kwargs in log_kwargs.values())
    assert all(kwargs["add_dataloader_idx"] is False for kwargs in log_kwargs.values())


def test_physical_xyz_metrics_report_meter_ade_fde_and_common_prefix():
    module = RefinerLightningModule(_cfg())
    B, T = 1, 5
    pred = torch.zeros(B, T, 5)
    pred[..., 0] = 1.0
    pred[..., 3] = 1.0
    gt = torch.zeros(B, T, 7)
    gt[..., 3] = 1.0
    batch = {
        "waypoints": gt[..., :5],
        "waypoints_mask": torch.ones(B, T, dtype=torch.bool),
        "waypoints_physical": gt,
    }
    out = {"waypoints": pred}
    mask = torch.tensor([[True, True, True, False, False]])

    metrics = module._compute_physical_xyz_metrics(out, batch, mask)

    assert torch.allclose(metrics["xyz_ADE_m"], torch.tensor(1.0))
    assert torch.allclose(metrics["xyz_FDE_m"], torch.tensor(1.0))
