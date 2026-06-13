"""Unit tests for eval/root_refiner_benchmark.py (T_A_10).

Smoke / metric-correctness tests. Done-criteria accuracy thresholds
(num_token top-1 > 0.5, heading < 30°) require a trained ckpt (T_A_09) and are
NOT asserted here — random weights make those numbers meaningless. We only
verify the pipeline runs end-to-end with finite metrics + JSON/CSV output, and
that the metric math is correct on hand-built predictions.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from datasets.humanml3d_refiner import HumanML3DRefinerDataset as RefinerDataset
from eval.root_refiner.benchmark import (
    _load_model_from_ckpt,
    _write_root_refiner_sample_artifacts,
    _resolve_cli_force_path_mode,
    build_eval_task_specs,
    build_full_route_task_specs,
    build_refiner_dataset_from_clips,
    compute_sample_metrics,
    resolve_suite_config,
    run_benchmark,
    run_suite_benchmark,
    validate_ckpt_eval_config_compatible,
    write_report,
)
from eval.root_refiner.adapters import (
    DURATION_GROUNDTRUTH,
    DURATION_PRED,
    ROOT_REFINER_ARTIFACT_NAMES,
)
from models.root_refiner import RootRefiner
from train_refiner import FrozenStubTextEncoder


# ---------------------------------------------------------------------------
# Per-sample metric correctness
# ---------------------------------------------------------------------------


def _unit_heading_wp(T: int, yaw_val: float = 0.0) -> torch.Tensor:
    wp = torch.zeros(T, 7)
    wp[:, 3] = math.cos(yaw_val)
    wp[:, 4] = math.sin(yaw_val)
    return wp


def test_perfect_prediction_yields_zero_errors():
    T = 10
    gt = _unit_heading_wp(T)
    gt[:, 0] = torch.arange(T, dtype=torch.float32)   # x ramp
    gt[:, 5] = 1.0   # fwd_delta
    pred = gt.clone()
    mask = torch.ones(T, dtype=torch.bool)
    m = compute_sample_metrics(pred, gt, mask)
    assert m["xyz_ADE"] < 1e-6
    assert m["xyz_FDE"] < 1e-6
    assert m["heading_error_deg"] < 1e-4
    assert m["fwd_speed_MAE"] < 1e-6
    assert m["yaw_rate_MAE"] < 1e-6


def test_xyz_ade_fde_known_offset():
    T = 5
    gt = _unit_heading_wp(T)
    pred = gt.clone()
    # Constant +1 offset in x on all frames → per-frame error = 1.
    pred[:, 0] += 1.0
    mask = torch.ones(T, dtype=torch.bool)
    m = compute_sample_metrics(pred, gt, mask)
    assert abs(m["xyz_ADE"] - 1.0) < 1e-5
    assert abs(m["xyz_FDE"] - 1.0) < 1e-5


def test_heading_error_deg_quarter_turn():
    T = 4
    gt = _unit_heading_wp(T, yaw_val=0.0)         # heading (1, 0)
    pred = _unit_heading_wp(T, yaw_val=math.pi / 2)  # heading (0, 1)
    mask = torch.ones(T, dtype=torch.bool)
    m = compute_sample_metrics(pred, gt, mask)
    assert abs(m["heading_error_deg"] - 90.0) < 1e-3


def test_masked_frames_excluded_from_metrics():
    T = 6
    gt = _unit_heading_wp(T)
    pred = gt.clone()
    pred[3:, 0] += 100.0   # large error, but those frames are masked out
    mask = torch.zeros(T, dtype=torch.bool)
    mask[:3] = True
    m = compute_sample_metrics(pred, gt, mask)
    assert m["xyz_ADE"] < 1e-6   # masked-out errors don't count


def test_empty_valid_returns_nan_metrics():
    T = 4
    gt = _unit_heading_wp(T)
    pred = gt.clone()
    mask = torch.zeros(T, dtype=torch.bool)
    m = compute_sample_metrics(pred, gt, mask)
    assert math.isnan(m["xyz_ADE"])
    assert math.isnan(m["heading_error_deg"])


# ---------------------------------------------------------------------------
# End-to-end smoke (random weights)
# ---------------------------------------------------------------------------


def _make_clip(T: int, *, raw_id: str | None = None, split_index: int | None = None):
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 2] = 0.05   # +Z velocity
    motion[:, 3] = 1.0
    clip = {"motion_263": motion, "text": "walk forward"}
    if raw_id is not None:
        clip.update(
            {
                "name": raw_id,
                "raw_id": raw_id,
                "split_index": split_index if split_index is not None else 0,
                "split_file": "test.txt",
                "dataset": "humanml3d",
            }
        )
    return clip


def _tiny_lightning_cfg():
    return {
        "model": {
            "target": "models.root_refiner.RootRefiner",
            "ema_decay": None,
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
        "optimizer": {
            "target": "AdamW",
            "params": {"lr": 1e-3, "weight_decay": 0.01},
        },
        "lr_scheduler": {"target": None, "params": {}},
        "loss": {"heading_form": "cosine"},
        "loss_weights": {
            "pace": 0.5,
            "num_token_pace": 0.1,
            "num_token_cls": 0.2,
            "num_token_soft_cls": 0.02,
            "xyz": 5.0,
            "heading": 1.0,
            "fwd_delta": 0.5,
            "yaw_delta": 0.5,
            "path_control": 0.0,
            "smoothness": 0.0,
        },
        "text_encoder": {"debug_stub": True},
    }


def test_load_model_from_ckpt_accepts_legacy_without_pace_duration(tmp_path):
    from train_refiner import RefinerLightningModule

    cfg = _tiny_lightning_cfg()
    module = RefinerLightningModule(cfg)
    state_dict = {
        key: value
        for key, value in module.state_dict().items()
        if not key.startswith(
            (
                "refiner.sample_mode_emb.",
                "refiner.pace_text_proj.",
                "refiner.pace_feature_proj.",
                "refiner.pace_head.",
            )
        )
    }
    ckpt_path = tmp_path / "legacy_no_pace.ckpt"
    torch.save({"hyper_parameters": {"cfg": cfg}, "state_dict": state_dict}, ckpt_path)

    refiner, text_encoder, loaded_cfg = _load_model_from_ckpt(str(ckpt_path), "cpu")

    assert loaded_cfg == cfg
    assert getattr(refiner, "use_pace_duration") is False
    assert text_encoder is not None


def _save_refiner_stats(tmp_path):
    cm_mean = np.zeros(5, dtype=np.float32)
    cm_std = np.ones(5, dtype=np.float32)
    cm_idx = np.array([0, 1, 2], dtype=np.int64)
    wp_mean = np.array([1.25, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    wp_std = np.array([2.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    wp_idx = np.array([0, 1, 2], dtype=np.int64)
    np.save(tmp_path / "current_motion_mean.npy", cm_mean)
    np.save(tmp_path / "current_motion_std.npy", cm_std)
    np.save(tmp_path / "current_motion_norm_indices.npy", cm_idx)
    np.save(tmp_path / "waypoint_mean.npy", wp_mean)
    np.save(tmp_path / "waypoint_std.npy", wp_std)
    np.save(tmp_path / "waypoint_norm_indices.npy", wp_idx)


def test_run_benchmark_smoke_finite_metrics_and_report(tmp_path):
    clips = [_make_clip(50) for _ in range(6)]
    ds = RefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                         full_plan_ratio=1.0, seed=0)
    model = RootRefiner(d_model=32, n_layers=2, n_heads=4, ff_dim=64,
                         max_tokens=8, min_tokens=2, n_hist=8, n_path=16,
                         text_emb_dim=16, dropout=0.0, path_features_dim=5)
    text_encoder = FrozenStubTextEncoder(emb_dim=16)

    result = run_benchmark(model, ds, text_encoder, device="cpu", max_samples=-1)
    summary = result["summary"]

    # All expected metric keys present.
    expected_keys = {
        "n_samples", "num_token_top1_accuracy", "num_token_top3_accuracy",
        "num_token_MAE", "xyz_ADE", "xyz_FDE", "heading_error_deg",
        "fwd_speed_MAE", "lateral_speed_MAE", "yaw_rate_MAE", "smoothness_acc_mean",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["n_samples"] == 6

    # Accuracy in [0, 1]; errors finite & non-negative (random weights, but
    # must not be NaN/Inf since inputs are well-formed).
    assert 0.0 <= summary["num_token_top1_accuracy"] <= 1.0
    assert 0.0 <= summary["num_token_top3_accuracy"] <= 1.0
    for k in ("xyz_ADE", "xyz_FDE", "heading_error_deg", "fwd_speed_MAE",
               "yaw_rate_MAE", "smoothness_acc_mean"):
        assert math.isfinite(summary[k]), f"{k} not finite: {summary[k]}"
        assert summary[k] >= 0.0

    # Report files written.
    write_report(result, tmp_path)
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "per_sample.csv").is_file()
    with (tmp_path / "summary.json").open() as f:
        loaded = json.load(f)
    assert loaded["n_samples"] == 6


def test_run_benchmark_writes_root_refiner_sample_artifacts(tmp_path):
    clips = [_make_clip(50) for _ in range(3)]
    ds = RefinerDataset(
        clips,
        n_hist=8,
        n_path=16,
        max_tokens=8,
        min_tokens=2,
        full_plan_ratio=1.0,
        seed=0,
    )
    model = RootRefiner(
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_dim=64,
        max_tokens=8,
        min_tokens=2,
        n_hist=8,
        n_path=16,
        text_emb_dim=16,
        dropout=0.0,
        path_features_dim=5,
    )
    text_encoder = FrozenStubTextEncoder(emb_dim=16)
    artifact_dir = tmp_path / "samples"

    result = run_benchmark(
        model,
        ds,
        text_encoder,
        device="cpu",
        max_samples=2,
        artifact_dir=artifact_dir,
        artifact_max_samples=1,
    )

    assert result["summary"]["n_samples"] == 2
    sample_dir = artifact_dir / "sample_000000"
    shared = ROOT_REFINER_ARTIFACT_NAMES["shared"]
    assert (sample_dir / "route_input.npy").is_file()
    assert (sample_dir / "gt_root_7d.npy").is_file()
    assert (sample_dir / shared["metadata"]).is_file()
    assert (sample_dir / shared["metrics"]).is_file()
    assert (sample_dir / shared["plot_xz"]).is_file()
    assert (sample_dir / shared["plot_yaw"]).is_file()
    assert (
        sample_dir
        / ROOT_REFINER_ARTIFACT_NAMES[DURATION_PRED]["pred_root_7d"]
    ).is_file()
    assert (
        sample_dir
        / ROOT_REFINER_ARTIFACT_NAMES[DURATION_GROUNDTRUTH]["pred_root_7d"]
    ).is_file()
    assert (
        sample_dir / ROOT_REFINER_ARTIFACT_NAMES[DURATION_PRED]["rootplan"]
    ).is_file()
    assert (
        sample_dir
        / ROOT_REFINER_ARTIFACT_NAMES[DURATION_GROUNDTRUTH]["rootplan"]
    ).is_file()
    assert (sample_dir / "route_valid_mask.npy").is_file()
    assert (sample_dir / "route_control_mask.npy").is_file()


def test_root_refiner_artifact_metadata_includes_raw_id_and_split(tmp_path):
    clips = [
        _make_clip(60, raw_id="000021", split_index=25),
        _make_clip(60, raw_id="004792", split_index=21),
    ]
    ds = RefinerDataset(
        clips,
        n_hist=8,
        n_path=16,
        max_tokens=8,
        min_tokens=2,
        full_plan_ratio=1.0,
        seed=0,
    )
    model = RootRefiner(
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_dim=64,
        max_tokens=8,
        min_tokens=2,
        n_hist=8,
        n_path=16,
        text_emb_dim=16,
        dropout=0.0,
        path_features_dim=5,
    )
    text_encoder = FrozenStubTextEncoder(emb_dim=16)

    run_benchmark(
        model,
        ds,
        text_encoder,
        device="cpu",
        task_specs=[
            {
                "idx": 0,
                "raw_id": "000021",
                "split_index": 25,
                "split_file": "test.txt",
                "mode": "full",
                "num_tokens": 8,
                "anchor_frame": 0,
                "task_key": "000021:full:8:0",
            }
        ],
        artifact_dir=tmp_path / "samples",
        artifact_max_samples=1,
    )

    metadata = json.loads(
        (
            tmp_path
            / "samples"
            / "sample_000000"
            / ROOT_REFINER_ARTIFACT_NAMES["shared"]["metadata"]
        ).read_text()
    )
    assert metadata["sample_id"] == "sample_000000"
    assert metadata["raw_id"] == "000021"
    assert metadata["split_index"] == 25
    assert metadata["split_file"] == "test.txt"
    assert metadata["dataset"] == "humanml3d"
    assert metadata["task_key"] == "000021:full:8:0"


def test_root_refiner_artifacts_use_physical_path_and_real_anchor(tmp_path):
    _save_refiner_stats(tmp_path)
    clips = [_make_clip(80)]
    common = dict(
        n_hist=8,
        n_path=16,
        max_tokens=8,
        min_tokens=2,
        full_plan_ratio=1.0,
        seed=0,
    )
    ds_raw = RefinerDataset(clips, normalize=False, **common)
    ds_norm = RefinerDataset(clips, normalize=True, stats_dir=tmp_path, **common)
    task_specs = [
        {
            "idx": 0,
            "mode": "full",
            "num_tokens": 5,
            "anchor_frame": 5,
            "task_key": "sample-anchor-5",
        }
    ]
    raw_sample = ds_raw.get_sample(
        0,
        force_mode="full",
        force_num_tokens=5,
        force_anchor_frame=5,
        force_path_mode="dense_path",
        force_no_path_aug=True,
    )
    norm_sample = ds_norm.get_sample(
        0,
        force_mode="full",
        force_num_tokens=5,
        force_anchor_frame=5,
        force_path_mode="dense_path",
        force_no_path_aug=True,
    )
    model = RootRefiner(
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_dim=64,
        max_tokens=8,
        min_tokens=2,
        n_hist=8,
        n_path=16,
        text_emb_dim=16,
        dropout=0.0,
        path_features_dim=5,
    )
    text_encoder = FrozenStubTextEncoder(emb_dim=16)

    run_benchmark(
        model,
        ds_norm,
        text_encoder,
        device="cpu",
        force_path_mode="dense_path",
        force_no_path_aug=True,
        task_specs=task_specs,
        artifact_dir=tmp_path / "samples",
        artifact_max_samples=1,
    )

    sample_dir = tmp_path / "samples" / "sample_000000"
    route_input = np.load(sample_dir / "route_input.npy")
    expected_physical_route = raw_sample["path"][
        raw_sample["path_valid_mask"].bool()
    ].numpy()
    normalized_route = norm_sample["path"][
        norm_sample["path_valid_mask"].bool()
    ].numpy()
    np.testing.assert_allclose(route_input, expected_physical_route, atol=1e-5)
    assert not np.allclose(route_input, normalized_route, atol=1e-3)

    metadata = json.loads(
        (sample_dir / ROOT_REFINER_ARTIFACT_NAMES["shared"]["metadata"]).read_text()
    )
    expected_anchor_xz = raw_sample["anchor_xz_world"].numpy()
    expected_anchor_yaw = float(raw_sample["anchor_yaw_world"].item())
    np.testing.assert_allclose(metadata["anchor_world_xz"], expected_anchor_xz)
    assert abs(metadata["anchor_world_yaw"] - expected_anchor_yaw) < 1e-6
    assert metadata["gt_slice"] == {"start": 5, "end": 5 + int(raw_sample["target_mask"].sum())}

    rootplan = json.loads(
        (
            sample_dir / ROOT_REFINER_ARTIFACT_NAMES[DURATION_PRED]["rootplan"]
        ).read_text()
    )
    assert rootplan["coordinate_frame"] == "anchor_local"
    np.testing.assert_allclose(rootplan["anchor_world_xz"], expected_anchor_xz)
    assert abs(rootplan["anchor_world_yaw"] - expected_anchor_yaw) < 1e-6


def test_root_refiner_pred_duration_artifact_is_not_clipped_to_gt_mask(tmp_path):
    sample = {
        "path": torch.arange(12, dtype=torch.float32).reshape(6, 2),
        "path_valid_mask": torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool),
        "path_control_mask": torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),
        "path_mode": "sparse_path",
        "offset_start_frames": torch.tensor(0),
        "anchor_frame": 3,
        "anchor_xz_world": torch.tensor([1.0, 2.0]),
        "anchor_yaw_world": torch.tensor(0.25),
        "mode": "full",
        "text": "walk",
    }
    gt_root = torch.zeros(12, 7)
    gt_root[:, 3] = 1.0
    gt_mask = torch.zeros(12, dtype=torch.bool)
    gt_mask[:5] = True
    pred_root = torch.zeros(12, 7)
    pred_root[:, 0] = torch.arange(12, dtype=torch.float32)
    pred_root[:, 3] = 1.0
    pred_mask = torch.zeros(12, dtype=torch.bool)
    pred_mask[:9] = True

    _write_root_refiner_sample_artifacts(
        tmp_path,
        sample=sample,
        sample_id="sample",
        metrics={"duration_mode": DURATION_PRED, "gt_num_tokens": 2},
        pred_by_duration={
            DURATION_PRED: {
                "root_7d": pred_root,
                "mask": pred_mask,
                "pred_num_tokens": 3,
            },
            DURATION_GROUNDTRUTH: {
                "root_7d": gt_root,
                "mask": gt_mask,
                "pred_num_tokens": 2,
            },
        },
        gt_root_7d=gt_root,
        gt_mask=gt_mask,
        frames_per_token=4,
    )

    pred_artifact = np.load(
        tmp_path / ROOT_REFINER_ARTIFACT_NAMES[DURATION_PRED]["pred_root_7d"]
    )
    rootplan = json.loads(
        (tmp_path / ROOT_REFINER_ARTIFACT_NAMES[DURATION_PRED]["rootplan"]).read_text()
    )
    route_valid = np.load(tmp_path / "route_valid_mask.npy")
    route_control = np.load(tmp_path / "route_control_mask.npy")

    assert pred_artifact.shape[0] == 9
    assert rootplan["valid_frames"] == 9
    assert route_valid.tolist() == [True, True, True, True]
    assert route_control.tolist() == [True, False, True, False]


def test_run_benchmark_oracle_duration_mode():
    """oracle_duration=True feeds GT num_tokens (teacher-force) so trajectory
    metrics isolate the waypoint decoder; num_token metrics (argmax) are unchanged."""
    clips = [_make_clip(50) for _ in range(6)]
    model = RootRefiner(d_model=32, n_layers=2, n_heads=4, ff_dim=64,
                         max_tokens=8, min_tokens=2, n_hist=8, n_path=16,
                         text_emb_dim=16, dropout=0.0, path_features_dim=5)
    text_encoder = FrozenStubTextEncoder(emb_dim=16)

    # A SINGLE shared dataset: run_benchmark calls dataset.reset_rng() at the start
    # so both passes see the identical sample sequence (this also exercises
    # reset_rng — otherwise the RNG would advance and the two runs would diverge).
    ds = RefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                         full_plan_ratio=1.0, seed=0)
    normal = run_benchmark(model, ds, text_encoder, device="cpu")["summary"]
    oracle = run_benchmark(model, ds, text_encoder, device="cpu",
                            oracle_duration=True)["summary"]

    assert normal["oracle_duration"] is False
    assert oracle["oracle_duration"] is True
    assert normal["duration_mode"] == "pred_duration"
    assert oracle["duration_mode"] == "groundtruth_duration"
    # num_token head metrics are argmax-based → independent of the oracle horizon.
    assert normal["num_token_top1_accuracy"] == oracle["num_token_top1_accuracy"]
    assert normal["num_token_MAE"] == oracle["num_token_MAE"]
    # trajectory metrics finite under the oracle (GT) horizon.
    assert math.isfinite(oracle["xyz_ADE"]) and oracle["xyz_ADE"] >= 0.0


def test_run_benchmark_max_samples_limit(tmp_path):
    clips = [_make_clip(50) for _ in range(10)]
    ds = RefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                         full_plan_ratio=1.0, seed=0)
    model = RootRefiner(d_model=32, n_layers=2, n_heads=4, ff_dim=64,
                         max_tokens=8, min_tokens=2, n_hist=8, n_path=16,
                         text_emb_dim=16, dropout=0.0, path_features_dim=5)
    text_encoder = FrozenStubTextEncoder(emb_dim=16)
    result = run_benchmark(model, ds, text_encoder, device="cpu", max_samples=4)
    assert result["summary"]["n_samples"] == 4
    assert len(result["per_sample"]) == 4


def test_resolve_suite_config_defines_root_refiner_eval_layers():
    smoke = resolve_suite_config("smoke")
    standard = resolve_suite_config("standard")
    standard_oracle = resolve_suite_config("standard_oracle")
    standard_groundtruth = resolve_suite_config("standard_groundtruth")
    stress = resolve_suite_config("stress")
    full_route = resolve_suite_config("full_route")

    assert smoke.default_max_samples == 50
    assert smoke.path_modes == (None,)
    assert standard.path_modes == ("dense_path", "sparse_path", "goal_point")
    assert standard_oracle.path_modes == standard.path_modes
    assert standard_oracle.oracle_duration is True
    assert standard_groundtruth.duration_mode == "groundtruth_duration"
    assert stress.path_modes == ("sparse_path", "goal_point")
    assert stress.force_no_path_aug is False
    assert full_route.path_modes == ("dense_path",)
    assert full_route.duration_mode == "groundtruth_duration"
    assert full_route.force_no_path_aug is True


def test_build_refiner_dataset_from_clips_uses_sampling_config():
    clips = [_make_clip(50) for _ in range(3)]
    cfg = {
        "data": {"normalize": False},
        "model": {
            "params": {
                "n_hist": 8,
                "n_path": 16,
                "max_tokens": 8,
                "min_tokens": 2,
                "frames_per_token": 4,
            }
        },
        "sampling": {
            "full_plan_ratio": 0.25,
            "horizon_policy": "max",
            "path_condition": {
                "policy": "mixed",
                "ratios": {
                    "dense_path": 0.2,
                    "sparse_path": 0.7,
                    "goal_point": 0.1,
                },
                "offset_start": {
                    "enabled": True,
                    "prob": 0.75,
                    "max_frames": 13,
                    "apply_to": ["sparse_path"],
                },
                "sparse_path": {"point_range": [2, 4]},
            },
        },
    }

    ds = build_refiner_dataset_from_clips(cfg, clips, dataset_cls=RefinerDataset, seed=123)

    assert ds.full_plan_ratio == 0.25
    assert ds.num_token_policy == "max"
    assert ds.path_condition_policy == "mixed"
    assert ds.path_condition_ratios == {
        "dense_path": 0.2,
        "sparse_path": 0.7,
        "goal_point": 0.1,
    }
    assert ds.offset_start_enabled is True
    assert ds.offset_start_prob == 0.75
    assert ds.offset_start_max_frames == 13
    assert ds.offset_start_apply_to == ("sparse_path",)
    assert ds.sparse_path_point_range == (2, 4)


def test_validate_ckpt_eval_config_compatible_accepts_matching_contract():
    cfg = {
        "model": {
            "params": {
                "n_hist": 8,
                "n_path": 16,
                "max_tokens": 8,
                "min_tokens": 2,
                "frames_per_token": 4,
            }
        }
    }

    validate_ckpt_eval_config_compatible(cfg, cfg)


def test_validate_ckpt_eval_config_compatible_rejects_contract_mismatch():
    ckpt_cfg = {
        "model": {
            "params": {
                "n_hist": 8,
                "n_path": 16,
                "max_tokens": 8,
                "min_tokens": 2,
                "frames_per_token": 4,
            }
        }
    }
    eval_cfg = {
        "model": {
            "params": {
                "n_hist": 12,
                "n_path": 16,
                "max_tokens": 8,
                "min_tokens": 2,
                "frames_per_token": 4,
            }
        }
    }

    with pytest.raises(ValueError, match="model.params.n_hist"):
        validate_ckpt_eval_config_compatible(ckpt_cfg, eval_cfg)


def test_build_eval_task_specs_freezes_underlying_tasks_before_path_modes():
    clips = [_make_clip(50) for _ in range(4)]
    ds = RefinerDataset(
        clips,
        n_hist=8,
        n_path=16,
        max_tokens=8,
        min_tokens=2,
        full_plan_ratio=0.5,
        seed=0,
    )

    specs = build_eval_task_specs(ds, max_samples=3)

    assert len(specs) == 3
    assert [spec["idx"] for spec in specs] == [0, 1, 2]
    assert all("mode" in spec for spec in specs)
    assert all("num_tokens" in spec for spec in specs)
    assert all("anchor_frame" in spec for spec in specs)
    assert all("task_key" in spec for spec in specs)


def test_build_full_route_task_specs_selects_raw_id_and_max_horizon():
    clips = [
        _make_clip(50, raw_id="004792", split_index=21),
        _make_clip(179, raw_id="000021", split_index=25),
    ]
    ds = RefinerDataset(
        clips,
        n_hist=8,
        n_path=16,
        max_tokens=49,
        min_tokens=2,
        full_plan_ratio=0.0,
        seed=0,
    )

    specs = build_full_route_task_specs(ds, raw_ids=["000021"])

    assert specs == [
        {
            "idx": 1,
            "raw_id": "000021",
            "split_index": 25,
            "split_file": "test.txt",
            "dataset": "humanml3d",
            "mode": "full",
            "num_tokens": 45,
            "anchor_frame": 0,
            "task_key": "000021:full:45:0",
        }
    ]


def test_cli_full_route_without_suite_forces_dense_path():
    assert _resolve_cli_force_path_mode(task_specs=[{"idx": 0}], suite=None) == "dense_path"
    assert _resolve_cli_force_path_mode(task_specs=[{"idx": 0}], suite="full_route") is None
    assert _resolve_cli_force_path_mode(task_specs=None, suite=None) is None


def test_run_suite_benchmark_standard_emits_schema_and_path_mode_buckets(tmp_path):
    clips = [_make_clip(50) for _ in range(6)]
    ds = RefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                         full_plan_ratio=1.0, seed=0)
    model = RootRefiner(d_model=32, n_layers=2, n_heads=4, ff_dim=64,
                         max_tokens=8, min_tokens=2, n_hist=8, n_path=16,
                         text_emb_dim=16, dropout=0.0, path_features_dim=5)
    text_encoder = FrozenStubTextEncoder(emb_dim=16)

    result = run_suite_benchmark(
        model,
        ds,
        text_encoder,
        suite="standard",
        device="cpu",
        max_samples=2,
    )

    assert result["schema_version"] == "root_refiner_eval.v1"
    assert result["evaluator"] == "root_refiner"
    assert result["suite"] == "standard"
    assert result["summary"]["n_samples"] == 2
    assert result["summary"]["n_records"] == 6
    assert result["summary"]["n_unique_tasks"] == 2
    assert result["summary"]["num_runs"] == 3
    assert {run["path_mode"] for run in result["runs"]} == {
        "dense_path",
        "sparse_path",
        "goal_point",
    }
    assert all("per_sample" not in run for run in result["runs"])
    assert "path_mode/dense_path/xyz_ADE" in result["summary"]
    assert all("suite" in sample for sample in result["per_sample"])
    assert all("path_mode" in sample for sample in result["per_sample"])
    grouped = {}
    for sample in result["per_sample"]:
        grouped.setdefault(sample["idx"], set()).add(sample["task_key"])
    assert all(len(task_keys) == 1 for task_keys in grouped.values())

    write_report(result, tmp_path)
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    with (tmp_path / "metrics.json").open() as f:
        payload = json.load(f)
    assert payload["schema_version"] == "root_refiner_eval.v1"
    assert payload["suite"] == "standard"


def test_run_suite_benchmark_oracle_suite_marks_duration_mode():
    clips = [_make_clip(50) for _ in range(6)]
    ds = RefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                         full_plan_ratio=1.0, seed=0)
    model = RootRefiner(d_model=32, n_layers=2, n_heads=4, ff_dim=64,
                         max_tokens=8, min_tokens=2, n_hist=8, n_path=16,
                         text_emb_dim=16, dropout=0.0, path_features_dim=5)
    text_encoder = FrozenStubTextEncoder(emb_dim=16)

    result = run_suite_benchmark(
        model,
        ds,
        text_encoder,
        suite="standard_oracle",
        device="cpu",
        max_samples=2,
    )

    assert result["suite"] == "standard_oracle"
    assert result["summary"]["oracle_duration"] is True
    assert result["summary"]["duration_mode"] == "groundtruth_duration"
    assert result["suite_config"]["duration_mode"] == "groundtruth_duration"
    assert all(run["summary"]["oracle_duration"] is True for run in result["runs"])
    assert all(
        run["summary"]["duration_mode"] == "groundtruth_duration"
        for run in result["runs"]
    )


def test_write_report_sanitizes_nan_for_strict_json(tmp_path):
    result = {
        "summary": {"n_samples": 1, "xyz_ADE": float("nan")},
        "per_sample": [{"idx": 0, "xyz_ADE": float("inf")}],
    }

    write_report(result, tmp_path)

    text = (tmp_path / "metrics.json").read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(text)
    assert payload["summary"]["xyz_ADE"] is None
    assert payload["per_sample"][0]["xyz_ADE"] is None


def test_write_report_mirrors_standard_eval_layout(tmp_path):
    result = {
        "summary": {"n_samples": 1, "xyz_ADE": 0.25},
        "per_sample": [{"idx": 0, "xyz_ADE": 0.25}],
    }

    write_report(result, tmp_path, suite="standard", run_id="step_000500")

    metrics_dir = tmp_path / "RootRefiner" / "metrics" / "standard" / "step_000500"
    per_sample_dir = (
        tmp_path / "RootRefiner" / "per_sample" / "standard" / "step_000500"
    )

    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "per_sample.csv").is_file()
    assert (metrics_dir / "metrics.json").is_file()
    assert (metrics_dir / "summary.json").is_file()
    assert (per_sample_dir / "per_sample.csv").is_file()

    payload = json.loads((metrics_dir / "metrics.json").read_text())
    assert payload["evaluator"] == "root_refiner"
    assert payload["suite"] == "standard"
    assert payload["summary"]["xyz_ADE"] == 0.25
