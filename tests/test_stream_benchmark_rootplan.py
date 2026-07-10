"""RootPlan streaming payload regression tests."""

from __future__ import annotations

import torch
from torch import nn

from utils.inference.root_plan import RootPlan, build_root_plan_stream_payload
from utils.inference.runtime_update import RootSourceProposal
from utils.inference.stream_generator import StreamGenerator
from utils.inference.timeline import RootFrameState, RootTimeline
from utils.token_frame import commit_boundary_frame, token_start_frame


class _DummyLdf(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.commit_index = 3
        self.chunk_size = 5


def _state(commit_idx: int) -> RootFrameState:
    return RootFrameState(
        commit_idx=commit_idx,
        world_xz=torch.zeros(2),
        world_yaw=torch.tensor(0.0),
    )


def _timeline(num_commits: int) -> RootTimeline:
    timeline = RootTimeline(_state(0))
    for commit_idx in range(1, num_commits + 1):
        timeline.append(_state(commit_idx))
    return timeline


def _root_plan(valid_frames: int = 80) -> RootPlan:
    waypoints = torch.zeros(valid_frames, 7)
    waypoints[:, 0] = torch.arange(valid_frames, dtype=torch.float32)
    waypoints[:, 3] = 1.0
    return RootPlan(
        num_tokens_pred=20,
        valid_frames=valid_frames,
        waypoints_local_7d=waypoints,
        frame_dt=0.05,
        frames_per_token=4,
        anchor_commit_idx=0,
        anchor_world_xz=torch.zeros(2),
        anchor_world_yaw=torch.tensor(0.0),
    )


def _root_source(valid_frames: int = 80) -> RootSourceProposal:
    proposal = torch.zeros(valid_frames, 7)
    proposal[:, 0] = torch.arange(valid_frames, dtype=torch.float32) * 0.1
    proposal[:, 3] = 1.0
    return RootSourceProposal(
        future_traj7=proposal,
        future_frame_mask=torch.ones(valid_frames, dtype=torch.bool),
        source_id="unit_world_route",
        version=1,
        metadata={"source_kind": "synthetic", "update_frames": (12, 24)},
    )


def test_build_root_plan_stream_payload_builds_substep_payloads():
    payload = build_root_plan_stream_payload(
        _root_plan(),
        _timeline(16),
        local_commit_index=3,
        absolute_commit_index=3,
        chunk_size=5,
        history_length=9,
        traj_horizon_tokens=4,
    )

    assert payload is not None
    assert payload["traj_cond_7d_frame"].shape[-1] == 7
    assert payload["traj_substep_payloads"]
    assert {
        subpayload["traj_start_token"]
        for subpayload in payload["traj_substep_payloads"]
    } == {0}
