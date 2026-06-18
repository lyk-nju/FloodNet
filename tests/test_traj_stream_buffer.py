"""RootPlan body-condition payload tests."""

from __future__ import annotations

import torch

from utils.inference.root_plan import RootPlan, root_plan_to_body_condition
from utils.inference.timeline import RootFrameState
from utils.token_frame import token_range_to_frame_slice


def _state(commit_idx: int, xz=(0.0, 0.0), yaw=0.0) -> RootFrameState:
    return RootFrameState(
        commit_idx=commit_idx,
        world_xz=torch.tensor(xz, dtype=torch.float32),
        world_yaw=torch.tensor(yaw, dtype=torch.float32),
    )


def _plan(anchor_commit_idx: int = 0) -> RootPlan:
    waypoints = torch.zeros(9, 7, dtype=torch.float32)
    waypoints[:, 0] = torch.arange(9, dtype=torch.float32)
    waypoints[:, 3] = 1.0
    return RootPlan(
        num_tokens_pred=3,
        valid_frames=9,
        waypoints_local_7d=waypoints,
        frame_dt=0.05,
        frames_per_token=4,
        anchor_commit_idx=anchor_commit_idx,
        anchor_world_xz=torch.zeros(2, dtype=torch.float32),
        anchor_world_yaw=torch.tensor(0.0),
    )


def test_root_plan_condition_masks_overflow_hold_last():
    plan = _plan(anchor_commit_idx=0)
    traj, mask = root_plan_to_body_condition(
        plan,
        head_state=_state(2),
        body_anchor_state=_state(2),
        horizon_tokens=2,
        expected_horizon_frame_slice=token_range_to_frame_slice(2, 2),
    )

    assert traj.shape == (8, 7)
    assert mask.tolist() == [True, True, True, True, False, False, False, False]
    assert torch.allclose(traj[:4, 0], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    assert torch.allclose(traj[4:, 0], torch.full((4,), 8.0))


def test_root_plan_condition_pending_prefix_is_unmasked_false():
    plan = _plan(anchor_commit_idx=2)
    traj, mask = root_plan_to_body_condition(
        plan,
        head_state=_state(1),
        body_anchor_state=_state(1),
        horizon_tokens=2,
        expected_horizon_frame_slice=token_range_to_frame_slice(1, 2),
    )

    assert traj.shape == (8, 7)
    assert mask[:4].tolist() == [False, False, False, False]
    assert mask[4:].tolist() == [True, True, True, True]


def test_root_plan_condition_none_returns_zero_payload():
    traj, mask = root_plan_to_body_condition(
        None,
        head_state=_state(0),
        body_anchor_state=_state(0),
        horizon_tokens=1,
        expected_horizon_frame_slice=token_range_to_frame_slice(0, 1),
    )

    assert traj.shape == (1, 7)
    assert not bool(mask.any())
    assert not bool(traj.any())
