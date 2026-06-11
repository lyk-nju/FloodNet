import numpy as np

from eval.runtime.metrics import compute_root_condition_diagnostics


def test_root_condition_diagnostics_compare_token_xyz_and_heading():
    gt_root_7d = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.20, 0.00],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.25, 0.10],
            [0.0, 0.0, 2.0, 1.0, 0.0, 0.30, 0.20],
        ],
        dtype=np.float32,
    )
    pred_root_7d = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.30, 0.00],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.35, 0.20],
            [0.0, 0.0, 4.0, 1.0, 0.0, 0.40, 0.40],
        ],
        dtype=np.float32,
    )

    metrics = compute_root_condition_diagnostics(
        gt_root_7d,
        pred_root_7d,
        gt_num_tokens=3,
        pred_num_tokens=5,
        frames_per_token=4,
    )

    assert metrics["num_token_gt"] == 3
    assert metrics["num_token_pred"] == 5
    assert metrics["num_token_abs_error"] == 2
    assert metrics["duration_frame_abs_error"] == 8
    assert np.isclose(metrics["xyz_ADE"], 1.0)
    assert np.isclose(metrics["xyz_FDE"], 2.0)
    assert np.isclose(metrics["x_AE_mean"], 1.0 / 3.0)
    assert np.isclose(metrics["z_AE_mean"], 2.0 / 3.0)
    assert np.isclose(metrics["endpoint_xz_error"], 2.0)
    assert np.isclose(metrics["heading_mae_deg"], 30.0)
    assert np.isclose(metrics["fwd_delta_mae"], 0.1)
    assert np.isclose(metrics["yaw_delta_mae"], 0.1)


def test_root_condition_diagnostics_handles_empty_inputs():
    metrics = compute_root_condition_diagnostics(
        np.zeros((0, 7), dtype=np.float32),
        np.zeros((0, 7), dtype=np.float32),
        gt_num_tokens=0,
        pred_num_tokens=0,
    )

    assert metrics["num_frames_compared"] == 0
    assert np.isnan(metrics["xyz_ADE"])
