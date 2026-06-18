from __future__ import annotations

import torch
from torch import nn

from utils.inference.route_condition import RoutePlan
from utils.inference.stream_generator import StreamGenerator
from utils.inference.timeline import RootFrameState
from utils.token_frame import num_tokens_for_frame_len


class _FakeTextEncoder(nn.Module):
    def encode(self, texts, device=None):
        return torch.ones(len(texts), 8, device=device)


class _FakeLdf(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.chunk_size = 5
        self.noise_steps = 10


class _FakeRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_hist = 4
        self.n_path = 4
        self.max_frames = 16
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        used_frames = (
            kwargs["num_frames"].to(device=kwargs["path"].device, dtype=torch.long)
            if kwargs.get("num_frames") is not None
            else torch.tensor([8], device=kwargs["path"].device)
        )
        waypoints = torch.zeros(1, self.max_frames, 5, device=kwargs["path"].device)
        waypoints[0, :, 2] = torch.arange(
            self.max_frames,
            device=kwargs["path"].device,
            dtype=torch.float32,
        )
        waypoints[0, :, 3] = 1.0
        return {"used_frames": used_frames, "waypoints": waypoints}


def _route() -> RoutePlan:
    return RoutePlan(
        times=torch.tensor([0.0, 1.0]).numpy(),
        points_xyz=torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 2.0]]).numpy(),
        start_commit_index=7,
        version=1,
        source="manual",
    )


def _crossing_route() -> RoutePlan:
    return RoutePlan(
        times=torch.tensor([0.0, 1.0]).numpy(),
        points_xyz=torch.tensor([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]).numpy(),
        start_commit_index=0,
        version=1,
        source="manual",
    )


def _anchor() -> RootFrameState:
    return RootFrameState(
        commit_idx=7,
        world_xz=torch.tensor([10.0, 0.0]),
        world_yaw=torch.tensor(0.0),
    )


def _generator(refiner: _FakeRefiner) -> StreamGenerator:
    return StreamGenerator(
        ldf_model=_FakeLdf(),
        root_refiner=refiner,
        root_text_encoder=_FakeTextEncoder(),
        device="cpu",
        token_dt=0.20,
    )


def test_stream_generator_builds_anchor_local_7d_root_plan():
    refiner = _FakeRefiner()
    runtime = _generator(refiner)

    root_plan = runtime.build_root_plan(
        text="walk forward",
        route=_route(),
        anchor_state=_anchor(),
    )

    assert root_plan.source == "root_refiner"
    assert root_plan.anchor_commit_idx == 7
    assert root_plan.num_tokens_pred == num_tokens_for_frame_len(9, 4)
    assert root_plan.valid_frames == 9
    assert root_plan.waypoints_local_7d.shape == (9, 7)
    assert torch.allclose(
        root_plan.waypoints_local_7d[0, :5],
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]),
    )
    call = refiner.calls[0]
    assert "path_mode" not in call
    assert "sample_mode" not in call
    assert "path_features_raw" in call
    assert torch.allclose(call["path"][0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(call["path"][0, -1], torch.tensor([0.0, 2.0]))
    assert torch.allclose(
        call["history_motion"][0, -1],
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]),
    )
    assert call["history_mask"].tolist() == [[False, False, False, True]]


def test_stream_generator_relative_route_uses_projected_suffix_condition():
    refiner = _FakeRefiner()
    runtime = _generator(refiner)
    runtime.condition_manager.set_route_mode("relative_to_actor")
    anchor = RootFrameState(
        commit_idx=0,
        world_xz=torch.tensor([0.0, 0.0]),
        world_yaw=torch.tensor(0.0),
    )

    runtime.build_root_plan(
        text="walk forward",
        route=_crossing_route(),
        anchor_state=anchor,
    )

    call = refiner.calls[0]
    assert torch.allclose(call["path"][0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(call["path"][0, -1], torch.tensor([5.0, 0.0]))


def test_stream_generator_absolute_route_preserves_world_offset_condition():
    refiner = _FakeRefiner()
    runtime = _generator(refiner)
    runtime.condition_manager.set_route_mode("absolute")
    anchor = RootFrameState(
        commit_idx=0,
        world_xz=torch.tensor([0.0, 0.0]),
        world_yaw=torch.tensor(0.0),
    )

    runtime.build_root_plan(
        text="walk forward",
        route=_crossing_route(),
        anchor_state=anchor,
    )

    call = refiner.calls[0]
    assert torch.allclose(call["path"][0, 0], torch.tensor([-5.0, 0.0]))
    assert torch.allclose(call["path"][0, -1], torch.tensor([5.0, 0.0]))


def test_stream_generator_uses_explicit_anchor_y_without_history():
    refiner = _FakeRefiner()
    runtime = _generator(refiner)

    root_plan = runtime.build_root_plan(
        text="walk forward",
        route=_route(),
        anchor_state=_anchor(),
        anchor_world_y=0.875,
    )

    call = refiner.calls[0]
    assert torch.allclose(
        call["history_motion"][0, -1],
        torch.tensor([0.0, 0.875, 0.0, 1.0, 0.0]),
    )
    assert torch.allclose(
        root_plan.waypoints_local_7d[0, :5],
        torch.tensor([0.0, 0.875, 0.0, 1.0, 0.0]),
    )


def test_stream_generator_uses_world_history_when_available():
    refiner = _FakeRefiner()
    runtime = _generator(refiner)
    history_world_5d = torch.tensor(
        [
            [10.0, 0.0, 0.0, 1.0, 0.0],
            [10.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    runtime.build_root_plan(
        text="walk forward",
        route=_route(),
        anchor_state=_anchor(),
        history_motion_world_5d=history_world_5d,
    )

    call = refiner.calls[0]
    assert call["history_mask"].tolist() == [[False, False, True, True]]
    assert torch.allclose(
        call["history_motion"][0, -2],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    assert torch.allclose(
        call["history_motion"][0, -1],
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
    )


def test_stream_generator_can_force_num_frames():
    refiner = _FakeRefiner()
    runtime = _generator(refiner)

    root_plan = runtime.build_root_plan(
        text="walk forward",
        route=_route(),
        anchor_state=_anchor(),
        forced_num_frames=16,
    )

    assert root_plan.source == "root_refiner_forced"
    assert root_plan.num_tokens_pred == 5
    assert root_plan.valid_frames == 17
    assert torch.equal(refiner.calls[0]["num_frames"], torch.tensor([16]))
