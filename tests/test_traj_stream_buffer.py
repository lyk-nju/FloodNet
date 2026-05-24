"""Unit tests for TrajStreamBuffer.get_body_traj_cond (T_C_01, B01-B09).

Stateless RootPlan → body-window-local 7D transform; no model/data needed.
"""

from __future__ import annotations

import torch

from utils.inference_glue import InferenceGlueState
from utils.root_plan import RootPlan
from utils.token_frame import num_frames_for_tokens, token_start_frame
from utils.traj_stream_buffer import TrajStreamBuffer


def _state(commit_idx, xz=(0.0, 0.0), yaw=0.0):
    return InferenceGlueState(
        commit_idx=commit_idx,
        world_xz=torch.tensor(xz, dtype=torch.float32),
        world_yaw=torch.tensor(float(yaw)),
    )


def _plan(*, valid_frames=200, anchor_commit_idx=0, anchor_xz=(0.0, 0.0),
          anchor_yaw=0.0, wp=None):
    if wp is None:
        wp = torch.zeros(valid_frames, 7)
        wp[:, 3] = 1.0   # identity heading (cos=1, sin=0)
    return RootPlan(
        num_tokens_pred=(valid_frames + 3) // 4,
        valid_frames=valid_frames,
        waypoints_local_7d=wp,
        frame_dt=1.0 / 20.0,
        frames_per_token=4,
        anchor_commit_idx=anchor_commit_idx,
        anchor_world_xz=torch.tensor(anchor_xz, dtype=torch.float32),
        anchor_world_yaw=torch.tensor(float(anchor_yaw)),
    )


def _buf():
    return TrajStreamBuffer(device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# B01: no active plan
# ---------------------------------------------------------------------------


def test_B01_no_plan_requires_slice_and_returns_zeros():
    import pytest
    buf = _buf()
    with pytest.raises(ValueError):
        buf.get_body_traj_cond(head_state=_state(10), body_anchor_state=_state(5),
                               horizon_tokens=20)   # no expected_horizon_frame_slice
    cond, mask = buf.get_body_traj_cond(
        head_state=_state(10), body_anchor_state=_state(5), horizon_tokens=20,
        expected_horizon_frame_slice=slice(0, 77),
    )
    assert cond.shape == (77, 7) and mask.shape == (77,)
    assert torch.count_nonzero(cond) == 0 and not mask.any()


# ---------------------------------------------------------------------------
# B02: current_plan_token uses head_state (not body_anchor_state)
# ---------------------------------------------------------------------------


def test_B02_current_plan_token_from_head():
    wp = torch.zeros(200, 7)
    wp[:, 0] = torch.arange(200, dtype=torch.float32)   # x channel = frame index
    wp[:, 3] = 1.0
    buf = _buf()
    buf.set_root_plan(_plan(valid_frames=200, anchor_commit_idx=20, wp=wp))
    # head=35, body_anchor=20, plan_anchor=20 → current_plan_token = 35-20 = 15
    cond, mask = buf.get_body_traj_cond(
        head_state=_state(35), body_anchor_state=_state(20), horizon_tokens=20)
    # zero anchors → identity transform → cond[0] == wp[token_start_frame(15)]
    assert token_start_frame(15) == 57
    assert abs(float(cond[0, 0]) - 57.0) < 1e-4   # NOT 0 (body_anchor.commit_idx misuse)


# ---------------------------------------------------------------------------
# B03: dynamic H_frame (off-by-3 regression)
# ---------------------------------------------------------------------------


def test_B03_dynamic_H_frame_off_by_3():
    buf = _buf()
    buf.set_root_plan(_plan(valid_frames=400, anchor_commit_idx=0))
    # token 0 → prefix length 77
    cond0, _ = buf.get_body_traj_cond(
        head_state=_state(0), body_anchor_state=_state(0), horizon_tokens=20)
    assert cond0.shape[0] == num_frames_for_tokens(20) == 77
    # token 5 → arbitrary range length 80 (NOT 77)
    cond5, _ = buf.get_body_traj_cond(
        head_state=_state(5), body_anchor_state=_state(0), horizon_tokens=20)
    assert cond5.shape[0] == 80
    assert cond5.shape[0] != num_frames_for_tokens(20)


# ---------------------------------------------------------------------------
# B04: RootPlan overflow (hold-last + mask)
# ---------------------------------------------------------------------------


def test_B04_overflow_mask():
    buf = _buf()
    # B04a: slice fully inside valid_frames → mask all 1
    buf.set_root_plan(_plan(valid_frames=200, anchor_commit_idx=0))
    _, mask_in = buf.get_body_traj_cond(
        head_state=_state(0), body_anchor_state=_state(0), horizon_tokens=20)
    assert mask_in.all()
    # B04c: tiny plan, slice fully beyond valid_frames → mask all 0
    buf.set_root_plan(_plan(valid_frames=5, anchor_commit_idx=0))
    _, mask_out = buf.get_body_traj_cond(
        head_state=_state(10), body_anchor_state=_state(0), horizon_tokens=20,
        expected_horizon_frame_slice=slice(0, 80))
    assert not mask_out.any()
    # B04b: slice straddling the boundary → some 1, some 0
    buf.set_root_plan(_plan(valid_frames=30, anchor_commit_idx=0))
    _, mask_mix = buf.get_body_traj_cond(
        head_state=_state(0), body_anchor_state=_state(0), horizon_tokens=20)
    assert mask_mix.any() and not mask_mix.all()


# ---------------------------------------------------------------------------
# B05: dual anchor canonicalize (core)
# ---------------------------------------------------------------------------


def test_B05_dual_anchor_canonicalize():
    # body_anchor at world (0,0); head/plan anchored at world (0,5); plan target
    # +1m forward (+z) in plan-local. yaw=0 everywhere → pure xz translation.
    wp = torch.zeros(200, 7)
    wp[:, 3] = 1.0
    wp[:, 2] = 1.0   # z = 1m forward (plan-local) for every frame
    buf = _buf()
    buf.set_root_plan(_plan(valid_frames=200, anchor_commit_idx=20,
                            anchor_xz=(0.0, 5.0), wp=wp))
    cond, _ = buf.get_body_traj_cond(
        head_state=_state(20, xz=(0.0, 5.0)),
        body_anchor_state=_state(20, xz=(0.0, 0.0)),   # body anchor at origin
        horizon_tokens=20)
    # world z = plan-local 1 + plan anchor 5 = 6; body-local z = 6 - 0 = 6
    assert abs(float(cond[0, 2]) - 6.0) < 1e-4
    # the WRONG (head-anchor) result would be 6 - 5 = 1
    assert abs(float(cond[0, 2]) - 1.0) > 1.0


# ---------------------------------------------------------------------------
# B06: expected_horizon_frame_slice decoupled from plan-local slice
# ---------------------------------------------------------------------------


def test_B06_slice_decoupled(monkeypatch):
    import utils.root_plan as rp
    real = rp.slice_plan_with_mask
    captured = {}

    def spy(plan, **kw):
        captured["frame_slice"] = kw.get("frame_slice")
        return real(plan, **kw)

    monkeypatch.setattr(rp, "slice_plan_with_mask", spy)
    buf = _buf()
    buf.set_root_plan(_plan(valid_frames=400, anchor_commit_idx=0))
    buf.get_body_traj_cond(
        head_state=_state(5), body_anchor_state=_state(0), horizon_tokens=20,
        expected_horizon_frame_slice=slice(0, 80))
    # plan-local slice starts at token_start_frame(current_plan_token=5)=17,
    # NOT expected_horizon_frame_slice.start (0).
    fs = captured["frame_slice"]
    assert fs.start == token_start_frame(5) == 17
    assert fs.start != 0


# ---------------------------------------------------------------------------
# B07: clear → no-traj
# ---------------------------------------------------------------------------


def test_B07_clear_to_no_traj():
    buf = _buf()
    buf.set_root_plan(_plan(valid_frames=200, anchor_commit_idx=0))
    assert buf.has_active_plan()
    buf.clear()
    assert not buf.has_active_plan()
    cond, mask = buf.get_body_traj_cond(
        head_state=_state(10), body_anchor_state=_state(5), horizon_tokens=20,
        expected_horizon_frame_slice=slice(0, 77))
    assert torch.count_nonzero(cond) == 0 and not mask.any()


# ---------------------------------------------------------------------------
# B08: inactive plan (head before plan anchor)
# ---------------------------------------------------------------------------


def test_B08_inactive_plan_no_traj():
    import pytest
    buf = _buf()
    buf.set_root_plan(_plan(valid_frames=200, anchor_commit_idx=30))
    # head=20 < plan_anchor=30 → current_plan_token=-10 < 0
    with pytest.raises(ValueError):
        buf.get_body_traj_cond(head_state=_state(20), body_anchor_state=_state(10),
                               horizon_tokens=20)   # needs expected slice
    cond, mask = buf.get_body_traj_cond(
        head_state=_state(20), body_anchor_state=_state(10), horizon_tokens=20,
        expected_horizon_frame_slice=slice(0, 77))
    assert not mask.any()


# ---------------------------------------------------------------------------
# B09: expected_horizon_frame_slice overrides output H but not plan-local slice
# ---------------------------------------------------------------------------


def test_B09_expected_slice_overrides_H_only(monkeypatch):
    import utils.root_plan as rp
    real = rp.slice_plan_with_mask
    captured = {}

    def spy(plan, **kw):
        captured["frame_slice"] = kw.get("frame_slice")
        return real(plan, **kw)

    monkeypatch.setattr(rp, "slice_plan_with_mask", spy)
    buf = _buf()
    buf.set_root_plan(_plan(valid_frames=400, anchor_commit_idx=0))
    cond, mask = buf.get_body_traj_cond(
        head_state=_state(0), body_anchor_state=_state(0), horizon_tokens=20,
        expected_horizon_frame_slice=slice(0, 80))   # 80, not the 77 prefix
    assert cond.shape == (80, 7) and mask.shape == (80,)
    # plan-local slice derived from current_plan_token(0)+H_frame(80), start=0
    assert captured["frame_slice"].start == token_start_frame(0) == 0
    assert captured["frame_slice"].stop == 80
