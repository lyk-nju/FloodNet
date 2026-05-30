"""Unit tests for datasets/refiner_dataset.py (T_A_04).

Covers T01-T15 per docs/TODO.md §T_A_04 Unit tests (T14 stats-side is deferred
to T_A_06 compute_5d_stats).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from datasets.refiner_dataset import RefinerDataset
from utils.motion_process import recover_root_rot_pos, root_to_traj_feats_7d
from utils.token_frame import num_frames_for_tokens

ATOL = 1e-4
PI = math.pi


# ---------------------------------------------------------------------------
# Synthetic clip fixtures
# ---------------------------------------------------------------------------


def test_P0_2_normalize_false_ignores_stats_dir():
    """P0-2: with normalize=False the dataset must NOT touch stats_dir, even a
    nonexistent one — the benchmark now passes stats_dir=None when normalize is
    off (mirrors train_refiner), so constructing here must not raise."""
    ds = RefinerDataset(
        [_make_clip(T=50)], full_plan_ratio=1.0, seed=0,
        normalize=False, stats_dir="/does/not/exist/refiner_stats",
    )
    _ = ds[0]   # __getitem__ works without loading any stats


def test_P0_2_normalize_true_requires_stats_dir():
    import pytest
    with pytest.raises(ValueError):
        RefinerDataset([_make_clip(T=50)], normalize=True, stats_dir=None)


def _make_clip(T: int, *, text: str = "walk forward",
                rot_vel_t0: float = 0.0,
                local_vel_xz=(0.0, 0.1)) -> dict:
    """Build a synthetic 263D clip with a small forward velocity.

    Layout (per `utils.motion_process.recover_root_rot_pos`):
      [..., 0] = rot vel
      [..., 1:3] = local linear vel xz
      [..., 3] = root y
      others = 0 (joints unused by RefinerDataset)
    """
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[0, 0] = rot_vel_t0
    vx, vz = local_vel_xz
    motion[:, 1] = vx
    motion[:, 2] = vz
    motion[:, 3] = 1.0   # constant root y
    return {"motion_263": motion, "text": text}


def _expected_anchor_world(motion_263: torch.Tensor, anchor_frame: int):
    """Reconstruct expected world anchor (xz, yaw) given motion + anchor index."""
    from utils.local_frame import root_quat_to_physical_yaw

    quat, xyz = recover_root_rot_pos(motion_263.unsqueeze(0))
    yaw = root_quat_to_physical_yaw(quat[0])
    return xyz[0, anchor_frame, [0, 2]], yaw[anchor_frame], xyz[0, anchor_frame, 1]


# ---------------------------------------------------------------------------
# T01-T04: anchor duplicate convention
# ---------------------------------------------------------------------------


def test_T01_full_plan_anchor_is_frame_zero_and_history_mask_only_last_slot():
    """full mode: anchor_frame=0, valid_history_frames=1, history_mask only [-1]=True."""
    ds = RefinerDataset([_make_clip(T=50)], full_plan_ratio=1.0, seed=0)
    s = ds.get_sample(0, force_mode="full", force_no_path_aug=True)
    assert s["mode"] == "full"
    assert s["anchor_frame"] == 0
    # history_mask: only last position True, rest False.
    assert s["history_mask"].dtype == torch.bool
    assert s["history_mask"][-1].item() is True
    assert int(s["history_mask"].sum().item()) == 1


def test_T02_full_plan_anchor_duplicate_in_current_motion_and_target():
    """current_motion[-1] ≈ (0, y_anchor, 0, 1, 0) AND target_waypoints[0] ≈
    (0, y_anchor, 0, 1, 0, *, *). y values match.
    """
    clip = _make_clip(T=50)
    ds = RefinerDataset([clip], full_plan_ratio=1.0, seed=0)
    s = ds.get_sample(0, force_mode="full", force_no_path_aug=True)
    cm_last = s["current_motion"][-1]   # (x, y, z, cos, sin)
    tw_first = s["target_waypoints"][0]   # (x, y, z, cos, sin, fwd, yaw_delta)

    # anchor world y
    _, _, y_world = _expected_anchor_world(clip["motion_263"], anchor_frame=0)
    expected_y = y_world.item()

    assert abs(cm_last[0].item()) < ATOL
    assert abs(cm_last[1].item() - expected_y) < ATOL
    assert abs(cm_last[2].item()) < ATOL
    assert abs(cm_last[3].item() - 1.0) < ATOL
    assert abs(cm_last[4].item()) < ATOL

    assert abs(tw_first[0].item()) < ATOL
    assert abs(tw_first[1].item() - expected_y) < ATOL
    assert abs(tw_first[2].item()) < ATOL
    assert abs(tw_first[3].item() - 1.0) < ATOL
    assert abs(tw_first[4].item()) < ATOL

    # y consistent across the two
    assert abs(cm_last[1].item() - tw_first[1].item()) < ATOL


def test_T03_sliding_window_history_mask_all_true_and_anchor_duplicate():
    """Sliding: history_mask all True, anchor duplicated at history[-1] and target[0]."""
    T = 60
    clip = _make_clip(T=T)
    ds = RefinerDataset([clip], full_plan_ratio=0.0, seed=0)
    s = ds.get_sample(0, force_mode="sliding",
                       force_anchor_frame=30, force_no_path_aug=True)
    assert s["mode"] == "sliding"
    assert s["history_mask"].all()
    # anchor in local frame: (0, y, 0, 1, 0)
    cm_last = s["current_motion"][-1]
    tw_first = s["target_waypoints"][0]
    assert abs(cm_last[0].item()) < ATOL and abs(cm_last[2].item()) < ATOL
    assert abs(cm_last[3].item() - 1.0) < ATOL and abs(cm_last[4].item()) < ATOL
    assert abs(tw_first[0].item()) < ATOL and abs(tw_first[2].item()) < ATOL
    assert abs(tw_first[3].item() - 1.0) < ATOL and abs(tw_first[4].item()) < ATOL


def test_T04_root_y_preserved_across_canonicalize():
    """canonicalize_5d / _7d do NOT zero y. current_motion / target_waypoints
    y channel matches world root y in valid frames.
    """
    clip = _make_clip(T=50)
    ds = RefinerDataset([clip], full_plan_ratio=1.0, seed=0)
    s = ds.get_sample(0, force_mode="full", force_no_path_aug=True)

    # Compute expected world y for the target frames.
    quat, xyz = recover_root_rot_pos(clip["motion_263"].unsqueeze(0))
    target_frame_count = int(s["target_mask"].sum().item())
    expected_y = xyz[0, :target_frame_count, 1]
    actual_y = s["target_waypoints"][:target_frame_count, 1]
    assert torch.allclose(actual_y, expected_y, atol=ATOL)


# ---------------------------------------------------------------------------
# T05-T06: num_tokens / target_frame_count strict alignment
# ---------------------------------------------------------------------------


def test_T05_target_frame_count_equals_num_frames_for_tokens():
    """num_tokens=1 → target_frame_count=1; =2 → 5; =3 → 9 (and target_mask matches)."""
    clip = _make_clip(T=100)
    ds = RefinerDataset([clip], full_plan_ratio=1.0, max_tokens=49, min_tokens=1, seed=0)
    cases = {1: 1, 2: 5, 3: 9, 5: 17}
    for nt, expected_frames in cases.items():
        s = ds.get_sample(0, force_mode="full", force_num_tokens=nt,
                           force_no_path_aug=True)
        assert s["num_tokens"].item() == nt
        assert int(s["target_mask"].sum().item()) == expected_frames, (
            f"num_tokens={nt}: target_mask.sum()={int(s['target_mask'].sum())}, "
            f"expected {expected_frames}"
        )


def test_num_token_policy_max_disables_random_token_sampling():
    """Diagnostic training can disable random num_tokens by taking the maximum
    valid horizon for the selected anchor."""
    clip = _make_clip(T=80)
    ds = RefinerDataset(
        [clip],
        full_plan_ratio=1.0,
        max_tokens=49,
        min_tokens=4,
        seed=0,
        num_token_policy="max",
    )
    s0 = ds.get_sample(0, force_mode="full", force_no_path_aug=True)
    s1 = ds.get_sample(0, force_mode="full", force_no_path_aug=True)

    assert s0["num_tokens"].item() == 20
    assert s1["num_tokens"].item() == 20
    assert int(s0["target_mask"].sum().item()) == num_frames_for_tokens(20)


def test_T06_target_mask_sum_equals_num_frames_for_tokens_strict():
    """Strict equality (round 8 P0-1 lock-in): target_mask.sum() must equal
    num_frames_for_tokens(num_tokens), NOT just <=.
    """
    # Use min_tokens=1 so we can test num_tokens=2 without clamping.
    clip = _make_clip(T=200)
    ds = RefinerDataset([clip], full_plan_ratio=1.0, min_tokens=1, seed=0)
    for nt in (2, 5, 10, 20, 30):
        s = ds.get_sample(0, force_mode="full", force_num_tokens=nt,
                           force_no_path_aug=True)
        assert int(s["target_mask"].sum().item()) == num_frames_for_tokens(nt)


# ---------------------------------------------------------------------------
# T07: short sample filtering
# ---------------------------------------------------------------------------


def test_T07_short_clip_filtered_from_valid_indices():
    """Clip too short to support min_tokens never appears in valid_indices."""
    min_tokens = 4
    too_short = num_frames_for_tokens(min_tokens) - 1   # =12 frames
    long_enough = num_frames_for_tokens(min_tokens)    # =13 frames

    short_clip = _make_clip(T=too_short)
    long_clip = _make_clip(T=long_enough + 10)
    ds = RefinerDataset(
        [short_clip, long_clip],
        min_tokens=min_tokens,
        full_plan_ratio=1.0,
        seed=0,
    )
    assert 0 not in ds.valid_indices   # short clip filtered
    assert 1 in ds.valid_indices       # long clip kept
    assert len(ds) == 1


def test_sliding_eligibility_split():
    """Sliding requires T >= (n_hist - 1) + num_frames_for_tokens(min_tokens).
    Short-but-full-eligible clip forced to full mode even when sliding drawn.
    """
    min_tokens = 4
    n_hist = 20
    full_only_T = num_frames_for_tokens(min_tokens) + 2   # >= min_full but < sliding
    full_and_sliding_T = (n_hist - 1) + num_frames_for_tokens(min_tokens) + 5

    full_only_clip = _make_clip(T=full_only_T)
    long_clip = _make_clip(T=full_and_sliding_T)
    ds = RefinerDataset(
        [full_only_clip, long_clip],
        n_hist=n_hist, min_tokens=min_tokens,
        full_plan_ratio=0.0,   # always draw sliding
        seed=0,
    )
    # full_only_clip is index 0 → should fall back to full.
    s = ds.get_sample(0, force_no_path_aug=True)
    assert s["mode"] == "full", (
        f"short clip should fall back to full mode under sliding-only draw; "
        f"got mode={s['mode']}"
    )
    # long_clip is index 1 → sliding ok.
    s = ds.get_sample(1, force_no_path_aug=True)
    assert s["mode"] == "sliding"


# ---------------------------------------------------------------------------
# T08-T12: path input construction
# ---------------------------------------------------------------------------


def test_T08_xz_path_starts_at_origin():
    """xz_path[0] = (0, 0) due to synthetic anchor prepend."""
    clip = _make_clip(T=80, local_vel_xz=(0.0, 0.1))
    ds = RefinerDataset([clip], full_plan_ratio=1.0, seed=0)
    s = ds.get_sample(0, force_mode="full", force_num_tokens=10,
                       force_no_path_aug=True)
    xz0 = s["xz_path"][0]
    assert abs(xz0[0].item()) < 1e-4 and abs(xz0[1].item()) < 1e-4


def test_T09_path_trim_aug_does_NOT_affect_target_waypoints():
    """target_waypoints[0] is always anchor regardless of path trim."""
    clip = _make_clip(T=80)
    ds = RefinerDataset(
        [clip],
        full_plan_ratio=1.0,
        path_trim_prob=1.0, path_trim_max_frames=5,
        path_sparse_prob=0.0,
        seed=42,
    )
    # Get sample with aug on
    s_aug = ds.get_sample(0, force_mode="full", force_num_tokens=10)
    # Get sample with aug off
    s_no = ds.get_sample(0, force_mode="full", force_num_tokens=10,
                          force_no_path_aug=True)
    # target_waypoints[0] should be the same anchor frame either way.
    assert torch.allclose(s_aug["target_waypoints"][0], s_no["target_waypoints"][0],
                            atol=ATOL)


def test_T10_path_sparse_fallback_short_path_does_not_crash():
    """Short path: sparse sampling should clamp K and not crash."""
    clip = _make_clip(T=20)
    ds = RefinerDataset(
        [clip],
        full_plan_ratio=1.0,
        path_trim_prob=1.0, path_trim_max_frames=15,
        path_sparse_prob=1.0, path_sparse_range=(3, 8),
        seed=1,
    )
    # force min num_tokens so target is short, exercising fallback paths
    s = ds.get_sample(0, force_mode="full", force_num_tokens=4)
    assert s["xz_path"].shape == (64, 2)
    assert s["path_mask"].shape == (64,)


def test_T11_path_mask_all_true_on_normal_path():
    clip = _make_clip(T=80, local_vel_xz=(0.0, 0.1))
    ds = RefinerDataset([clip], full_plan_ratio=1.0, seed=0)
    s = ds.get_sample(0, force_mode="full", force_num_tokens=10,
                       force_no_path_aug=True)
    assert s["path_mask"].all()


def test_T12_degenerate_path_zero_velocity_fallback_mask_zero():
    """A clip with zero local velocity → path is all zeros after canonicalize
    → arclength_resample falls back to degenerate (mask all False).
    """
    clip = _make_clip(T=80, local_vel_xz=(0.0, 0.0), rot_vel_t0=0.0)
    ds = RefinerDataset([clip], full_plan_ratio=1.0, seed=0)
    s = ds.get_sample(0, force_mode="full", force_num_tokens=10,
                       force_no_path_aug=True)
    # All target xz at origin → control_points all (0, 0) → degenerate.
    # Path mask should be all False per arclength_resample's degenerate branch.
    assert not s["path_mask"].any()


# ---------------------------------------------------------------------------
# T13: cos / sin channels not z-scored
# ---------------------------------------------------------------------------


def test_T13_cos_sin_invariant_under_selective_zscore(tmp_path):
    """When normalize=True with norm_indices excluding [3, 4], the cos/sin
    channels in current_motion and target_waypoints must equal their
    unnormalized values bit-for-bit.
    """
    # Build dummy stats files.
    cm_mean = np.array([1.0, 2.0, 3.0, 0.5, 0.5], dtype=np.float32)
    cm_std = np.array([0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)
    cm_idx = np.array([0, 1, 2], dtype=np.int64)
    wp_mean = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.1, 0.1], dtype=np.float32)
    wp_std = np.array([0.1, 0.2, 0.3, 1.0, 1.0, 0.05, 0.05], dtype=np.float32)
    wp_idx = np.array([0, 1, 2, 5, 6], dtype=np.int64)
    np.save(tmp_path / "current_motion_mean.npy", cm_mean)
    np.save(tmp_path / "current_motion_std.npy", cm_std)
    np.save(tmp_path / "current_motion_norm_indices.npy", cm_idx)
    np.save(tmp_path / "waypoint_mean.npy", wp_mean)
    np.save(tmp_path / "waypoint_std.npy", wp_std)
    np.save(tmp_path / "waypoint_norm_indices.npy", wp_idx)

    clip = _make_clip(T=80)
    ds_raw = RefinerDataset([clip], full_plan_ratio=1.0, seed=0, normalize=False)
    ds_norm = RefinerDataset([clip], full_plan_ratio=1.0, seed=0,
                              normalize=True, stats_dir=tmp_path)
    s_raw = ds_raw.get_sample(0, force_mode="full", force_num_tokens=10,
                                force_no_path_aug=True)
    s_norm = ds_norm.get_sample(0, force_mode="full", force_num_tokens=10,
                                  force_no_path_aug=True)
    # cos/sin at channels [3], [4] must be bit-equal between raw and norm.
    assert torch.equal(s_raw["current_motion"][..., 3], s_norm["current_motion"][..., 3])
    assert torch.equal(s_raw["current_motion"][..., 4], s_norm["current_motion"][..., 4])
    assert torch.equal(s_raw["target_waypoints"][..., 3], s_norm["target_waypoints"][..., 3])
    assert torch.equal(s_raw["target_waypoints"][..., 4], s_norm["target_waypoints"][..., 4])
    # xyz channels of current_motion DIFFER after normalize.
    assert not torch.equal(s_raw["current_motion"][..., 0], s_norm["current_motion"][..., 0])


def test_waypoint_norm_indices_with_heading_channel_raises(tmp_path):
    """A stats file that lists heading channel 3 or 4 in waypoint_norm_indices
    must fail loudly at load (z-scoring cos/sin would break the unit-norm GT the
    cosine heading loss assumes)."""
    cm_mean = np.zeros(5, dtype=np.float32)
    cm_std = np.ones(5, dtype=np.float32)
    cm_idx = np.array([0, 1, 2], dtype=np.int64)
    wp_mean = np.zeros(7, dtype=np.float32)
    wp_std = np.ones(7, dtype=np.float32)
    wp_idx = np.array([0, 1, 2, 3, 5, 6], dtype=np.int64)   # ⚠ includes heading ch 3
    np.save(tmp_path / "current_motion_mean.npy", cm_mean)
    np.save(tmp_path / "current_motion_std.npy", cm_std)
    np.save(tmp_path / "current_motion_norm_indices.npy", cm_idx)
    np.save(tmp_path / "waypoint_mean.npy", wp_mean)
    np.save(tmp_path / "waypoint_std.npy", wp_std)
    np.save(tmp_path / "waypoint_norm_indices.npy", wp_idx)
    with pytest.raises(ValueError, match="heading channels 3/4"):
        RefinerDataset([_make_clip(T=50)], normalize=True, stats_dir=tmp_path)


def test_current_motion_norm_indices_with_heading_channel_raises(tmp_path):
    """Symmetric to the waypoint check: a stats file that lists heading channel
    3 or 4 in current_motion_norm_indices must also fail loudly at load (cos/sin
    yaw are unit-vector invariant; rule 7)."""
    cm_mean = np.zeros(5, dtype=np.float32)
    cm_std = np.ones(5, dtype=np.float32)
    cm_idx = np.array([0, 1, 2, 4], dtype=np.int64)   # ⚠ includes heading ch 4
    wp_mean = np.zeros(7, dtype=np.float32)
    wp_std = np.ones(7, dtype=np.float32)
    wp_idx = np.array([0, 1, 2, 5, 6], dtype=np.int64)
    np.save(tmp_path / "current_motion_mean.npy", cm_mean)
    np.save(tmp_path / "current_motion_std.npy", cm_std)
    np.save(tmp_path / "current_motion_norm_indices.npy", cm_idx)
    np.save(tmp_path / "waypoint_mean.npy", wp_mean)
    np.save(tmp_path / "waypoint_std.npy", wp_std)
    np.save(tmp_path / "waypoint_norm_indices.npy", wp_idx)
    with pytest.raises(ValueError, match="heading channels 3/4"):
        RefinerDataset([_make_clip(T=50)], normalize=True, stats_dir=tmp_path)


# ---------------------------------------------------------------------------
# T15: rigid-invariant fwd_delta / yaw_delta channels
# ---------------------------------------------------------------------------


def test_T15_fwd_delta_yaw_delta_invariant_under_canonicalize_via_pipeline():
    """target_waypoints[:, 5] (fwd_delta) and [:, 6] (yaw_delta) must equal
    motion_7d_world[anchor:anchor+target_count, 5/6] (i.e. unchanged by
    canonicalize_7d).
    """
    clip = _make_clip(T=80, local_vel_xz=(0.0, 0.1), rot_vel_t0=PI / 8)
    ds = RefinerDataset([clip], full_plan_ratio=1.0, seed=0)
    s = ds.get_sample(0, force_mode="full", force_num_tokens=10,
                       force_no_path_aug=True)
    target_count = int(s["target_mask"].sum().item())

    # Recompute world 7D from the clip directly.
    motion_263 = clip["motion_263"].unsqueeze(0)
    quat, xyz = recover_root_rot_pos(motion_263)
    world_7d = root_to_traj_feats_7d(quat, xyz)[0]   # [T, 7]

    expected_fwd = world_7d[: target_count, 5]
    expected_yaw_d = world_7d[: target_count, 6]
    actual_fwd = s["target_waypoints"][: target_count, 5]
    actual_yaw_d = s["target_waypoints"][: target_count, 6]

    assert torch.allclose(actual_fwd, expected_fwd, atol=ATOL)
    assert torch.allclose(actual_yaw_d, expected_yaw_d, atol=ATOL)


# ---------------------------------------------------------------------------
# Schema sanity: shapes / dtypes / keys
# ---------------------------------------------------------------------------


def test_returned_dict_has_required_keys_and_shapes():
    clip = _make_clip(T=80)
    ds = RefinerDataset([clip], full_plan_ratio=1.0, seed=0)
    s = ds.get_sample(0, force_mode="full", force_num_tokens=10,
                       force_no_path_aug=True)
    expected_keys = {
        "text", "xz_path", "path_mask", "path_stats", "current_motion",
        "history_mask", "target_waypoints", "target_mask", "num_tokens", "mode",
        "anchor_frame", "anchor_xz_world", "anchor_yaw_world",
    }
    assert expected_keys.issubset(s.keys())
    new_keys = {
        "path",
        "path_valid_mask",
        "path_control_mask",
        "path_features",
        "path_mode",
        "history_motion",
        "waypoints",
        "waypoints_mask",
        "path_supervision_mask",
        "offset_start_frames",
    }
    assert new_keys.issubset(s.keys())
    assert s["xz_path"].shape == (64, 2)
    assert s["path_mask"].shape == (64,)
    assert s["current_motion"].shape == (20, 5)
    assert s["history_mask"].shape == (20,)
    assert s["target_waypoints"].shape == (num_frames_for_tokens(49), 7)
    assert s["target_mask"].shape == (num_frames_for_tokens(49),)
    assert s["history_mask"].dtype == torch.bool
    assert s["target_mask"].dtype == torch.bool
    assert s["path_mask"].dtype == torch.bool


def test_uniform_shape_between_full_and_sliding_modes():
    """Network input tensor shapes must be identical regardless of mode."""
    T = 80
    clip = _make_clip(T=T)
    ds = RefinerDataset([clip], full_plan_ratio=1.0, seed=0)
    s_full = ds.get_sample(0, force_mode="full", force_num_tokens=10,
                            force_no_path_aug=True)
    s_slide = ds.get_sample(0, force_mode="sliding", force_anchor_frame=30,
                              force_num_tokens=10, force_no_path_aug=True)
    for key in ("xz_path", "path_mask", "current_motion", "history_mask",
                 "target_waypoints", "target_mask"):
        assert s_full[key].shape == s_slide[key].shape, (
            f"{key} shape mismatch: full={s_full[key].shape} "
            f"slide={s_slide[key].shape}"
        )


def test_len_excludes_short_clips():
    short = _make_clip(T=10)   # < num_frames_for_tokens(4) = 13
    long_a = _make_clip(T=50)
    long_b = _make_clip(T=70)
    ds = RefinerDataset([short, long_a, long_b], full_plan_ratio=1.0, seed=0)
    assert len(ds) == 2


def test_reset_rng_makes_get_sample_sequence_reproducible():
    """reset_rng() restores the base-seed RNG so a repeat pass draws the identical
    mode/anchor/num_tokens sequence (benchmark reproducibility)."""
    clips = [_make_clip(T=50) for _ in range(5)]
    ds = RefinerDataset(clips, n_hist=8, n_path=16, max_tokens=8, min_tokens=2,
                         full_plan_ratio=0.5, seed=0)
    first = [int(ds.get_sample(i)["num_tokens"].item()) for i in range(len(ds))]
    # Without reset, a second pass diverges (RNG advanced).
    second_no_reset = [int(ds.get_sample(i)["num_tokens"].item()) for i in range(len(ds))]
    ds.reset_rng()
    third_after_reset = [int(ds.get_sample(i)["num_tokens"].item()) for i in range(len(ds))]
    assert third_after_reset == first, "reset_rng did not reproduce the first pass"
    # (second_no_reset is allowed to differ; assert reset actually changed something
    #  only when the un-reset pass diverged, which it does for full_plan_ratio<1.)
    assert second_no_reset != first or third_after_reset == first


# ---------------------------------------------------------------------------
# Multi-caption text augmentation (clip carries a `texts` list)
# ---------------------------------------------------------------------------


def _make_multicap_clip(T: int, texts: list[str]) -> dict:
    """A clip carrying multiple captions in `texts` (load_clips_from_dir schema)."""
    clip = _make_clip(T=T, text=texts[0])
    clip["texts"] = list(texts)
    return clip


def test_random_caption_selection_uses_all_captions():
    """With a `texts` list, repeated sampling must surface more than one caption
    (text augmentation), and every drawn caption must come from the list."""
    texts = ["walk forward", "stroll ahead", "march onward", "step forward"]
    ds = RefinerDataset([_make_multicap_clip(T=60, texts=texts)],
                         full_plan_ratio=1.0, seed=0)
    drawn = {ds.get_sample(0)["text"] for _ in range(40)}
    assert drawn.issubset(set(texts))
    assert len(drawn) > 1, f"expected caption diversity, only saw {drawn}"


def test_force_text_idx_pins_specific_caption():
    texts = ["walk forward", "stroll ahead", "march onward"]
    ds = RefinerDataset([_make_multicap_clip(T=60, texts=texts)],
                         full_plan_ratio=1.0, seed=0)
    for i, cap in enumerate(texts):
        s = ds.get_sample(0, force_text_idx=i)
        assert s["text"] == cap


def test_randomize_caption_false_pins_first_and_consumes_no_rng():
    """randomize_caption=False (val / benchmark) must always return texts[0] AND
    consume no caption RNG, so the mode/num_tokens draw order is identical to a
    single-caption clip — keeping val/loss comparable across epochs."""
    texts = ["walk forward", "stroll ahead", "march onward", "step forward"]
    ds_fixed = RefinerDataset([_make_multicap_clip(T=60, texts=texts)],
                               full_plan_ratio=0.5, seed=0, randomize_caption=False)
    # Always the first caption, never a random one.
    assert {ds_fixed.get_sample(0)["text"] for _ in range(20)} == {"walk forward"}

    # No caption RNG consumed: the num_tokens sequence matches a clip that has
    # only `text` (the legacy, no-`texts` path) under the same seed.
    multicap_seq = [int(RefinerDataset([_make_multicap_clip(T=60, texts=texts)],
                                       full_plan_ratio=0.5, seed=0,
                                       randomize_caption=False)
                        .get_sample(0, force_no_path_aug=True)["num_tokens"])
                    for _ in range(5)]
    legacy_seq = [int(RefinerDataset([_make_clip(T=60, text="walk forward")],
                                     full_plan_ratio=0.5, seed=0)
                      .get_sample(0, force_no_path_aug=True)["num_tokens"])
                  for _ in range(5)]
    assert multicap_seq == legacy_seq


def test_clip_without_texts_falls_back_to_single_text():
    """Legacy clips (only `text`, no `texts`) keep working: the single caption is
    returned, and the fallback path does NOT call self._rng.choice — so two fresh
    identical legacy datasets draw an identical mode/num_tokens sequence (the
    existing T01/T02/reset_rng tests separately lock in that the legacy draw order
    is unperturbed by the multi-caption change)."""
    def fresh():
        return RefinerDataset([_make_clip(T=60, text="walk forward")],
                              full_plan_ratio=0.5, seed=0)

    ds_a = fresh()
    assert ds_a.get_sample(0)["text"] == "walk forward"   # single-caption fallback

    seq_a = [int(fresh().get_sample(0, force_no_path_aug=True)["num_tokens"])
             for _ in range(5)]
    seq_b = [int(fresh().get_sample(0, force_no_path_aug=True)["num_tokens"])
             for _ in range(5)]
    assert seq_a == seq_b   # deterministic; fallback consumes no caption RNG
