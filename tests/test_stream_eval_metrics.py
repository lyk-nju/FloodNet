from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

from eval.ldf.stream_metrics import _resolve_run_name, _save_sample_outputs
from metrics.stream import compute_root_path_yaw_error, summarize_stream_records
from eval.runtime.metrics import (
    build_plan_metrics,
    build_stream_eval_summary,
    compute_heading_path_error_deg,
    compute_lateral_velocity_ratio,
    compute_root_jitter,
    compute_yaw_error,
    estimate_body_yaw,
)
from eval.ldf.stream_metrics import run_stream_generate_step_sample


def test_compute_yaw_error_wraps_angles():
    pred = np.array([math.pi - 0.1], dtype=np.float32)
    target = np.array([-math.pi + 0.1], dtype=np.float32)

    err = compute_yaw_error(pred, target)

    assert abs(err - 0.2) < 1e-5


def test_compute_root_jitter_is_zero_for_constant_velocity():
    root = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0],
         [4.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    assert compute_root_jitter(root) == 0.0


def test_estimate_body_yaw_matches_physical_yaw_convention():
    motion = np.zeros((5, 263), dtype=np.float32)
    motion[0, 0] = -math.pi / 4.0

    yaw = estimate_body_yaw(motion)

    expected = np.array([0.0, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0])
    assert np.allclose(yaw, expected, atol=1e-5)


def test_lateral_velocity_ratio_zero_for_forward_plus_z_motion():
    motion = np.zeros((6, 263), dtype=np.float32)
    motion[:, 2] = 1.0

    assert compute_lateral_velocity_ratio(motion) < 1e-6


def test_heading_path_error_uses_project_yaw_convention_for_plus_z_path():
    motion = np.zeros((6, 263), dtype=np.float32)
    motion[:, 2] = 1.0
    target = np.zeros((6, 3), dtype=np.float32)
    target[:, 2] = np.arange(6, dtype=np.float32)

    assert compute_heading_path_error_deg(motion, target) < 1e-5


def test_build_plan_metrics_applies_motion_yaw_offset_for_world_heading():
    motion = np.zeros((6, 263), dtype=np.float32)
    motion[:, 2] = 1.0
    target = np.zeros((6, 3), dtype=np.float32)
    target[:, 0] = -np.arange(6, dtype=np.float32)
    plan_times = np.arange(6, dtype=np.float32) / 20.0

    without_offset = build_plan_metrics(
        target,
        original_gt_root=None,
        plan_times=plan_times,
        plan_points_xyz=target,
        target_frames=6,
        motion_fps=20.0,
        motion_263=motion,
    )
    with_offset = build_plan_metrics(
        target,
        original_gt_root=None,
        plan_times=plan_times,
        plan_points_xyz=target,
        target_frames=6,
        motion_fps=20.0,
        motion_263=motion,
        motion_yaw_offset=-math.pi / 2.0,
    )

    assert without_offset["heading_path_error_deg"] > 80.0
    assert with_offset["heading_path_error_deg"] < 1e-5


def test_build_stream_eval_summary_uses_stream_metric_keys():
    pred = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = pred.copy()
    target[-1, 0] = 4.0

    summary = build_stream_eval_summary(
        pred,
        target,
        pred_yaw=np.zeros(4, dtype=np.float32),
        target_yaw=np.zeros(4, dtype=np.float32),
    )

    assert summary["stream/root_ADE"] == 0.25
    assert summary["stream/root_FDE"] == 1.0
    assert summary["stream/yaw_error"] == 0.0
    assert "stream/jitter" in summary
    assert summary["stream/num_frames"] == 4


def test_compute_root_path_yaw_error_wraps_path_heading():
    pred = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        dtype=np.float32,
    )

    err = compute_root_path_yaw_error(pred, target)

    assert abs(err - math.pi / 2.0) < 1e-5


def test_summarize_stream_records_includes_no_traj_gain_metrics():
    records = [
        {
            "ade": 0.20,
            "fde": 0.30,
            "mse": 0.01,
            "traj_jitter": 0.001,
            "stream_yaw_error": 0.10,
            "stream_no_traj/ade": 0.35,
            "stream_no_traj/fde": 0.50,
        },
        {
            "ade": 0.30,
            "fde": 0.40,
            "mse": 0.02,
            "traj_jitter": 0.003,
            "stream_yaw_error": 0.20,
            "stream_no_traj/ade": 0.55,
            "stream_no_traj/fde": 0.70,
        },
    ]

    summary = summarize_stream_records(records)

    assert summary["stream_no_traj/root_ADE_mean"] == 0.45
    assert summary["stream_no_traj/root_FDE_mean"] == 0.60
    assert summary["control_gain/root_ADE_delta_mean"] == 0.20
    assert summary["control_gain/root_FDE_delta_mean"] == 0.25
    assert summary["stream_gt/root_ADE"] == 0.25
    assert summary["stream_gt/root_FDE"] == 0.35
    assert summary["stream_gt/jitter"] == 0.002
    assert summary["stream_gt/yaw_error"] == 0.15
    assert summary["stream_no_traj/root_ADE"] == 0.45
    assert summary["control_gain/root_ADE_delta"] == 0.20


class _FakeStepModel:
    input_dim = 4
    noise_steps = 1

    def __init__(self):
        self.payloads = []
        self.commit_index = 0
        self.chunk_size = 1

    def init_generated(self, history_length, batch_size, num_denoise_steps):
        self.history_length = history_length
        self.batch_size = batch_size
        self.num_denoise_steps = num_denoise_steps
        self.commit_index = 0

    def stream_generate_step(self, step_input, first_chunk=True):
        self.payloads.append(dict(step_input))
        self.commit_index += 1
        return {"generated": [torch.zeros(1, self.input_dim)]}


class _FakeRollingStepModel(_FakeStepModel):
    def stream_generate_step(self, step_input, first_chunk=True):
        out = super().stream_generate_step(step_input, first_chunk=first_chunk)
        if self.commit_index == 3:
            self.commit_index = 1
        return out


class _FakeStepVAE:
    def clear_cache(self):
        pass

    def stream_decode(self, latent, first_chunk=True):
        return torch.zeros(1, 1, 263)


def test_ldf_stream_generate_step_uses_direct_7d_payload_when_available():
    traj7 = torch.zeros(1, 5, 7, dtype=torch.float32)
    traj7[0, :, 2] = torch.arange(5, dtype=torch.float32)
    traj7[0, :, 3] = 1.0
    sample_batch = {
        "name": ["sample"],
        "dataset": ["HumanML3D"],
        "text": ["walk"],
        "token": torch.zeros(1, 2, 4, dtype=torch.float32),
        "token_length": torch.tensor([2], dtype=torch.long),
        "feature_length": torch.tensor([5], dtype=torch.long),
        "traj_cond_7d": traj7,
        "traj_cond": traj7[..., :3].clone(),
        "traj": traj7[..., :3].clone(),
        "traj_length": torch.tensor([5], dtype=torch.long),
        "traj_cond_mask": torch.ones(1, 5, dtype=torch.float32),
        "traj_mask": torch.ones(1, 5, dtype=torch.float32),
        "token_mask": torch.ones(1, 2, dtype=torch.float32),
    }
    model = _FakeStepModel()

    run_stream_generate_step_sample(
        model=model,
        vae=_FakeStepVAE(),
        sample_batch=sample_batch,
        device=torch.device("cpu"),
        history_length=2,
        num_denoise_steps=1,
        traj_horizon_tokens=1,
    )

    assert model.payloads
    assert all("traj_cond_7d_frame" in payload for payload in model.payloads)
    assert all("traj_cond_frame_mask" in payload for payload in model.payloads)
    assert all("traj_features" not in payload for payload in model.payloads)
    assert all("traj" not in payload for payload in model.payloads)


def test_ldf_stream_generate_step_separates_local_and_absolute_commit_after_roll():
    traj7 = torch.zeros(1, 13, 7, dtype=torch.float32)
    traj7[0, :, 2] = torch.arange(13, dtype=torch.float32)
    traj7[0, :, 3] = 1.0
    sample_batch = {
        "name": ["sample"],
        "dataset": ["HumanML3D"],
        "text": ["walk"],
        "token": torch.zeros(1, 4, 4, dtype=torch.float32),
        "token_length": torch.tensor([4], dtype=torch.long),
        "feature_length": torch.tensor([13], dtype=torch.long),
        "traj_cond_7d": traj7,
        "traj_cond": traj7[..., :3].clone(),
        "traj": traj7[..., :3].clone(),
        "traj_length": torch.tensor([13], dtype=torch.long),
        "traj_cond_mask": torch.ones(1, 13, dtype=torch.float32),
        "traj_mask": torch.ones(1, 13, dtype=torch.float32),
        "token_mask": torch.ones(1, 4, dtype=torch.float32),
    }
    model = _FakeRollingStepModel()

    run_stream_generate_step_sample(
        model=model,
        vae=_FakeStepVAE(),
        sample_batch=sample_batch,
        device=torch.device("cpu"),
        history_length=2,
        num_denoise_steps=1,
        traj_horizon_tokens=0,
    )

    assert model.payloads[-1]["traj_start_token"] == 0
    assert model.payloads[-1]["traj_abs_start_token"] == 2
    assert model.payloads[-1]["body_anchor_token"] == 0
    assert model.payloads[-1]["body_anchor_abs_token"] == 2


def test_ldf_stream_sample_outputs_include_plots_and_path_roots(tmp_path):
    frames = 8
    gt_feature = torch.zeros(frames, 263)
    stream_feature = torch.zeros(frames, 263)
    offline_feature = torch.zeros(frames, 263)
    no_traj_feature = torch.zeros(frames, 263)
    gt_feature[:, 2] = torch.arange(frames, dtype=torch.float32) * 0.1
    stream_feature[:, 2] = torch.arange(frames, dtype=torch.float32) * 0.12
    offline_feature[:, 2] = torch.arange(frames, dtype=torch.float32) * 0.11
    no_traj_feature[:, 2] = torch.arange(frames, dtype=torch.float32) * 0.04
    traj7 = torch.zeros(1, frames, 7)
    traj7[0, :, 2] = torch.arange(frames, dtype=torch.float32) * 0.1
    traj7[0, :, 3] = 1.0
    sample_batch = {
        "name": ["sample"],
        "dataset": ["HumanML3D"],
        "text": ["walk forward"],
        "traj_cond_7d": traj7,
    }

    _save_sample_outputs(
        sample_dir=tmp_path,
        sample_batch=sample_batch,
        sample_record={"name": "sample"},
        stream_feature=stream_feature,
        gt_feature=gt_feature,
        offline_feature=offline_feature,
        stream_no_traj_feature=no_traj_feature,
        stream_latent=None,
        offline_latent=None,
        save_feature_npy=True,
        save_latent_npy=False,
        save_plots=True,
        render_video=False,
    )

    assert (tmp_path / "plot_xz.png").is_file()
    assert (tmp_path / "plot_yaw.png").is_file()
    assert (tmp_path / "gt_root.npy").is_file()
    assert (tmp_path / "condition_root.npy").is_file()
    assert (tmp_path / "stream_gt_root.npy").is_file()
    assert (tmp_path / "offline_gt_root.npy").is_file()
    assert (tmp_path / "stream_no_traj_root.npy").is_file()
    assert (tmp_path / "stream_no_traj_feature.npy").is_file()


def test_ldf_stream_eval_writes_standard_eval_artifacts(tmp_path):
    from eval.ldf import stream_metrics

    frames = 6
    stream_feature = torch.zeros(frames, 263)
    stream_latent = torch.ones(3, 4)
    stream_feature[:, 2] = torch.arange(frames, dtype=torch.float32) * 0.2
    traj7 = torch.zeros(1, frames, 7)
    traj7[0, :, 2] = torch.arange(frames, dtype=torch.float32) * 0.1
    traj7[0, :, 3] = 1.0
    sample_batch = {
        "name": ["001439"],
        "dataset": ["HumanML3D"],
        "text": ["walk forward"],
        "traj_cond_7d": traj7,
        "traj_mask": torch.ones(1, frames, dtype=torch.float32),
        "feature_text_end": [torch.tensor([frames], dtype=torch.long)],
    }
    sample_record = {"name": "001439", "dataset": "HumanML3D", "ade": 0.1}

    stream_metrics._save_eval_style_sample_outputs(
        out_root=tmp_path,
        dataset_id="HumanML3D",
        probe_tag="test",
        step_tag="step_485000",
        sample_name="001439",
        sample_batch=sample_batch,
        sample_record=sample_record,
        stream_feature=stream_feature,
        stream_latent=stream_latent,
    )

    base = tmp_path / "HumanML3D"
    assert np.load(base / "feature/test/step_485000/001439.npy").shape == (frames, 263)
    assert np.load(base / "token/test/step_485000/001439.npy").shape == (3, 4)
    assert (base / "text/test/step_485000/001439.txt").read_text() == "walk forward"
    assert np.load(base / "traj_xz/test/step_485000/001439.npy").shape == (frames, 2)
    assert np.load(base / "traj_mask/test/step_485000/001439.npy").shape == (frames,)
    assert np.load(base / "frames/test/step_485000/001439.npy").tolist() == [frames]
    with open(base / "metrics/test/step_485000/001439.json") as f:
        assert json.load(f)["ade"] == 0.1
    assert (base / "condition_compare/test/step_485000/001439.png").is_file()


def test_ldf_stream_metrics_cli_can_disable_no_traj_baseline(monkeypatch):
    from eval.ldf import stream_metrics

    monkeypatch.setattr(
        sys,
        "argv",
        ["stream_metrics", "--no_compute_no_traj_baseline"],
    )

    args = stream_metrics.parse_args()

    assert args.no_compute_no_traj_baseline is True


def test_ldf_stream_metrics_defaults_run_three_design_paths():
    from eval.ldf import stream_metrics

    args = stream_metrics.parse_args_from_list([])
    offline, no_traj = stream_metrics.resolve_eval_path_flags(args, {})

    assert offline is True
    assert no_traj is True


def test_ldf_stream_metrics_can_disable_diagnostic_paths():
    from eval.ldf import stream_metrics

    args = stream_metrics.parse_args_from_list([
        "--no_compute_offline_baseline",
        "--no_compute_no_traj_baseline",
    ])
    offline, no_traj = stream_metrics.resolve_eval_path_flags(
        args,
        {
            "eval.compute_offline_baseline": True,
            "eval.compute_no_traj_baseline": True,
        },
    )

    assert offline is False
    assert no_traj is False


def test_ldf_stream_metrics_run_name_override_stabilizes_output_dir():
    assert (
        _resolve_run_name(
            ckpt_path="/ckpts/step=425000.ckpt",
            probe_tag="window_local",
            stream_mode="stream_generate_step",
            requested_run_name="03_overfit_full_prefix",
        )
        == "03_overfit_full_prefix"
    )

    generated = _resolve_run_name(
        ckpt_path="/ckpts/step=425000.ckpt",
        probe_tag="window_local",
        stream_mode="stream_generate_step",
        requested_run_name=None,
    )
    assert generated.endswith("_window_local_stream_generate_step_step_425000")


def test_ldf_stream_metrics_resolves_test_probe_meta_paths_when_test_meta_missing():
    from eval.ldf import stream_metrics
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "data": {
                "test_probe_meta_paths": {
                    "test": ["/data/HumanML3D/test_min.txt"],
                    "train": ["/data/HumanML3D/train_min.txt"],
                }
            }
        }
    )
    args = stream_metrics.parse_args_from_list([])

    meta_paths, probe_tag = stream_metrics._resolve_meta_paths_and_probe_tag(
        args, cfg
    )

    assert meta_paths == ["/data/HumanML3D/test_min.txt"]
    assert probe_tag == "test"


def test_ldf_stream_metrics_shards_batches_across_ranks():
    from eval.ldf import stream_metrics

    assert stream_metrics._parse_devices_arg("8") == list(range(8))
    assert stream_metrics._parse_devices_arg("0,1,2,3") == [0, 1, 2, 3]

    rank0 = [
        idx
        for idx in range(8)
        if stream_metrics._should_process_batch_on_rank(
            idx, rank=0, world_size=2, max_batches=0, max_samples=0
        )
    ]
    rank1 = [
        idx
        for idx in range(8)
        if stream_metrics._should_process_batch_on_rank(
            idx, rank=1, world_size=2, max_batches=0, max_samples=0
        )
    ]

    assert rank0 == [0, 2, 4, 6]
    assert rank1 == [1, 3, 5, 7]
    assert [
        idx
        for idx in range(8)
        if stream_metrics._should_process_batch_on_rank(
            idx, rank=1, world_size=2, max_batches=0, max_samples=5
        )
    ] == [1, 3]
    assert [
        idx
        for idx in range(8)
        if stream_metrics._should_process_batch_on_rank(
            idx, rank=0, world_size=2, max_batches=3, max_samples=0
        )
    ] == [0, 2]


def test_ldf_stream_metrics_aggregates_rank_payloads(tmp_path):
    from eval.ldf import stream_metrics

    rank_dir = tmp_path / "_rank_records"
    rank_dir.mkdir()
    (rank_dir / "rank_0.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "_sample_index": 2,
                        "name": "sample_2",
                        "dataset": "HumanML3D",
                        "ade": 0.30,
                        "fde": 0.50,
                        "mse": 0.03,
                        "traj_jitter": 0.003,
                    }
                ]
            }
        )
    )
    (rank_dir / "rank_1.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "_sample_index": 1,
                        "name": "sample_1",
                        "dataset": "HumanML3D",
                        "ade": 0.10,
                        "fde": 0.20,
                        "mse": 0.01,
                        "traj_jitter": 0.001,
                    }
                ]
            }
        )
    )

    payload = stream_metrics._aggregate_rank_payloads(
        run_dir=tmp_path,
        world_size=2,
        probe_tag="test",
        ckpt_path="/ckpts/step_1.ckpt",
        vae_ckpt_path="/ckpts/vae.ckpt",
        stream_mode="stream_generate_step",
        num_runs=1,
        out_root=tmp_path,
        step_tag="step_1",
    )

    assert payload["num_samples"] == 2
    assert [sample["name"] for sample in payload["samples"]] == ["sample_1", "sample_2"]
    assert payload["summary"]["stream_gt/root_ADE"] == 0.20
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "HumanML3D/metrics/test/step_1/summary.json").is_file()


def test_ldf_stream_metrics_caps_blas_threads_before_numpy_import():
    thread_env_keys = [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    env = os.environ.copy()
    for key in thread_env_keys:
        env.pop(key, None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import eval.ldf.stream_metrics; "
                f"print([os.environ.get(k) for k in {thread_env_keys!r}])"
            ),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == str(["1"] * len(thread_env_keys))
