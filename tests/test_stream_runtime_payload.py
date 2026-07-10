import torch

from utils.inference.stream_runtime.contracts import (
    ComposeResult,
    RouteProgressState,
    RouteStatus,
    SegmentLabel,
)
from utils.inference.stream_runtime.payload_builder import PayloadBuilder
from utils.inference.timeline import RootFrameState, RootTimeline
from utils.local_frame import uncanonicalize_7d
from utils.motion_process import build_physical_7d_from_5d
from utils.token_frame import token_end_frame, token_range_to_frame_slice


def _timeline() -> RootTimeline:
    timeline = RootTimeline(
        RootFrameState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )
    for commit in range(1, 80):
        timeline.append(
            RootFrameState(
                commit_idx=commit,
                world_xz=torch.tensor([0.0, 0.0]),
                world_yaw=torch.tensor(0.0),
                source="test",
            )
        )
    return timeline


def _composed_condition() -> ComposeResult:
    frame_start_abs = 117
    frames = torch.arange(80, dtype=torch.float32)
    world = torch.stack(
        [
            frames,
            torch.zeros_like(frames),
            frames * 2.0,
            torch.ones_like(frames),
            torch.zeros_like(frames),
        ],
        dim=-1,
    )
    world = build_physical_7d_from_5d(world)
    world[:, 5:7] = -99.0
    frame_mask = torch.zeros(80, dtype=torch.bool)
    frame_mask[8:16] = True
    labels = torch.full((80,), SegmentLabel.PADDING.value)
    labels[8:12] = SegmentLabel.GENERATED_HISTORY.value
    labels[12:16] = SegmentLabel.ROUTE.value
    return ComposeResult(
        frame_start_abs=frame_start_abs,
        world_condition_7d=world,
        frame_mask=frame_mask,
        segment_labels=labels,
        proposed_route_progress=RouteProgressState.initial(),
        route_status=RouteStatus.EXHAUSTED,
        diagnostics={},
    )


def test_payload_slices_by_compose_frame_start_not_tensor_index():
    builder = PayloadBuilder()
    composed = _composed_condition()
    payload = builder.build(
        composed,
        _timeline(),
        local_commit_before=31,
        absolute_commit_before=61,
        chunk_size=5,
        history_tokens=30,
        horizon_tokens=20,
    )

    assert payload is not None
    assert payload["traj_abs_start_token"] == 32
    assert payload["debug_world_frame_start_abs"] == 117
    assert torch.allclose(
        payload["traj_cond_7d_frame"][0, 0, [0, 2]],
        torch.tensor([8.0, 16.0]),
    )


def test_payload_recomputes_delta_after_history_and_padding_overlay():
    builder = PayloadBuilder()
    timeline = _timeline()
    payload = builder.build(
        _composed_condition(),
        timeline,
        local_commit_before=31,
        absolute_commit_before=61,
        chunk_size=5,
        history_tokens=30,
        horizon_tokens=20,
    )

    assert payload is not None
    anchor = timeline.at_commit(payload["body_anchor_abs_token"])
    world = uncanonicalize_7d(
        payload["traj_cond_7d_frame"],
        anchor.world_xz.unsqueeze(0),
        anchor.world_yaw.reshape(1),
    )[0]
    expected = build_physical_7d_from_5d(world[:, :5])
    assert torch.allclose(world[:, 5:7], expected[:, 5:7])


def test_payload_substep_masks_follow_composed_history_route_and_padding():
    builder = PayloadBuilder()
    composed = _composed_condition()
    payload = builder.build(
        composed,
        _timeline(),
        local_commit_before=31,
        absolute_commit_before=61,
        chunk_size=5,
        history_tokens=30,
        horizon_tokens=20,
    )

    assert payload is not None
    subpayloads = payload["traj_substep_payloads"]
    assert subpayloads
    assert [item["traj_start_token"] for item in subpayloads] == [2, 3, 4, 5, 6]
    saw_generated_history = False
    saw_valid_route = False
    saw_terminal_hold_padding = False
    for subpayload in subpayloads:
        frame_slice = token_range_to_frame_slice(
            subpayload["traj_abs_start_token"],
            subpayload["traj_num_tokens"],
        )
        absolute_frames = torch.arange(frame_slice.start, frame_slice.stop)
        indices = absolute_frames - composed.frame_start_abs
        expected = torch.zeros_like(indices, dtype=torch.bool)
        in_condition = (indices >= 0) & (indices < composed.frame_mask.shape[0])
        expected[in_condition] = composed.frame_mask[indices[in_condition]]
        actual = subpayload["traj_cond_frame_mask"][0].bool()
        assert torch.equal(actual, expected)

        labels = torch.full_like(indices, SegmentLabel.PADDING.value)
        labels[in_condition] = composed.segment_labels[indices[in_condition]]
        saw_generated_history |= bool(
            (actual & labels.eq(SegmentLabel.GENERATED_HISTORY.value)).any()
        )
        saw_valid_route |= bool(
            (actual & labels.eq(SegmentLabel.ROUTE.value)).any()
        )
        saw_terminal_hold_padding |= bool(
            ((~actual) & labels.eq(SegmentLabel.PADDING.value)).any()
        )

    assert saw_generated_history
    assert saw_valid_route
    assert saw_terminal_hold_padding
