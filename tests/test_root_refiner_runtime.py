from __future__ import annotations

import torch
from torch import nn

from utils.inference_glue import InferenceGlueState
from utils.refiner.runtime import RootRefinerRuntime, _state_dict_has_pace_duration
from utils.stream_traj import StreamTrajectoryPlan
from utils.token_frame import num_frames_for_tokens


class _FakeTextEncoder(nn.Module):
    def encode(self, texts, device=None):
        return torch.ones(len(texts), 8, device=device)


class _FakeRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_hist = 4
        self.n_path = 4
        self.frames_per_token = 4
        self.max_tokens = 5
        self.max_frames = num_frames_for_tokens(self.max_tokens, self.frames_per_token)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        used_num_tokens = (
            kwargs["num_tokens"].to(device=kwargs["path"].device, dtype=torch.long)
            if kwargs.get("num_tokens") is not None
            else torch.tensor([3], device=kwargs["path"].device)
        )
        waypoints = torch.zeros(1, self.max_frames, 5, device=kwargs["path"].device)
        waypoints[0, :, 2] = torch.arange(
            self.max_frames, device=kwargs["path"].device, dtype=torch.float32
        )
        waypoints[0, :, 3] = 1.0
        return {
            "used_num_tokens": used_num_tokens,
            "waypoints": waypoints,
        }


def test_state_dict_has_pace_duration_detects_legacy_and_new_checkpoints():
    legacy = {
        "refiner.num_token_head.0.weight": torch.zeros(1),
        "refiner.num_token_emb.weight": torch.zeros(1),
    }
    current = {
        **legacy,
        "refiner.pace_head.0.weight": torch.zeros(1),
    }

    assert _state_dict_has_pace_duration(legacy) is False
    assert _state_dict_has_pace_duration(current) is True


def test_root_refiner_runtime_builds_anchor_local_7d_root_plan():
    refiner = _FakeRefiner()
    runtime = RootRefinerRuntime(
        refiner,
        _FakeTextEncoder(),
        device="cpu",
        path_mode="dense_path",
    )
    plan = StreamTrajectoryPlan(
        times=torch.tensor([0.0, 1.0]).numpy(),
        points_xyz=torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 2.0]]).numpy(),
        start_commit_index=7,
        version=1,
        source="manual",
    )
    anchor = InferenceGlueState(
        commit_idx=7,
        world_xz=torch.tensor([10.0, 0.0]),
        world_yaw=torch.tensor(0.0),
    )

    root_plan = runtime.build_root_plan(
        text="walk forward",
        plan=plan,
        anchor_state=anchor,
        token_dt=0.20,
    )

    assert root_plan.source == "root_refiner"
    assert root_plan.anchor_commit_idx == 7
    assert root_plan.num_tokens_pred == 3
    assert root_plan.valid_frames == num_frames_for_tokens(3, 4)
    assert root_plan.waypoints_local_7d.shape == (root_plan.valid_frames, 7)
    assert torch.allclose(root_plan.waypoints_local_7d[0, :5], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
    assert len(refiner.calls) == 1
    call = refiner.calls[0]
    assert call["path_mode"] == ["dense_path"]
    assert call["sample_mode"] == ["full"]
    assert "path_features_raw" in call
    assert torch.allclose(call["path_features"], call["path_features_raw"])
    assert torch.allclose(call["path"][0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(call["path"][0, -1], torch.tensor([0.0, 2.0]))
    assert torch.allclose(call["history_motion"][0, -1], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
    assert call["history_mask"].tolist() == [[False, False, False, True]]


def test_root_refiner_runtime_uses_world_history_when_available():
    refiner = _FakeRefiner()
    runtime = RootRefinerRuntime(
        refiner,
        _FakeTextEncoder(),
        device="cpu",
        path_mode="dense_path",
    )
    plan = StreamTrajectoryPlan(
        times=torch.tensor([0.0, 1.0]).numpy(),
        points_xyz=torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 2.0]]).numpy(),
        start_commit_index=7,
        version=1,
        source="manual",
    )
    anchor = InferenceGlueState(
        commit_idx=7,
        world_xz=torch.tensor([10.0, 0.0]),
        world_yaw=torch.tensor(0.0),
    )
    history_world_5d = torch.tensor(
        [
            [10.0, 0.0, 0.0, 1.0, 0.0],
            [10.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    runtime.build_root_plan(
        text="walk forward",
        plan=plan,
        anchor_state=anchor,
        token_dt=0.20,
        history_motion_world_5d=history_world_5d,
    )

    call = refiner.calls[0]
    assert call["sample_mode"] == ["sliding"]
    assert call["history_mask"].tolist() == [[False, False, True, True]]
    assert torch.allclose(call["history_motion"][0, -2], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(call["history_motion"][0, -1], torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]))


def test_root_refiner_runtime_anchor_only_history_uses_stats_root_height():
    refiner = _FakeRefiner()
    runtime = RootRefinerRuntime(
        refiner,
        _FakeTextEncoder(),
        device="cpu",
        cm_mean=torch.tensor([0.0, 0.9, 0.0, 0.0, 0.0]),
        cm_std=torch.tensor([1.0, 0.1, 1.0, 1.0, 1.0]),
        cm_norm_idx=torch.tensor([0, 1, 2]),
        path_mode="dense_path",
    )
    plan = StreamTrajectoryPlan(
        times=torch.tensor([0.0, 1.0]).numpy(),
        points_xyz=torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 2.0]]).numpy(),
        start_commit_index=7,
        version=1,
        source="manual",
    )
    anchor = InferenceGlueState(
        commit_idx=7,
        world_xz=torch.tensor([10.0, 0.0]),
        world_yaw=torch.tensor(0.0),
    )

    runtime.build_root_plan(
        text="walk forward",
        plan=plan,
        anchor_state=anchor,
        token_dt=0.20,
    )

    call = refiner.calls[0]
    assert torch.allclose(
        call["history_motion"][0, -1],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
    )


def test_root_refiner_runtime_can_force_gt_num_tokens():
    refiner = _FakeRefiner()
    runtime = RootRefinerRuntime(
        refiner,
        _FakeTextEncoder(),
        device="cpu",
        path_mode="dense_path",
    )
    plan = StreamTrajectoryPlan(
        times=torch.tensor([0.0, 1.0]).numpy(),
        points_xyz=torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 2.0]]).numpy(),
        start_commit_index=7,
        version=1,
        source="manual",
    )
    anchor = InferenceGlueState(
        commit_idx=7,
        world_xz=torch.tensor([10.0, 0.0]),
        world_yaw=torch.tensor(0.0),
    )

    root_plan = runtime.build_root_plan(
        text="walk forward",
        plan=plan,
        anchor_state=anchor,
        token_dt=0.20,
        forced_num_tokens=5,
    )

    assert root_plan.source == "root_refiner_gtnum"
    assert root_plan.num_tokens_pred == 5
    assert root_plan.valid_frames == num_frames_for_tokens(5, 4)
    assert torch.equal(refiner.calls[0]["num_tokens"], torch.tensor([5]))
