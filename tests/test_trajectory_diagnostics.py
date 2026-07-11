from types import SimpleNamespace
from types import MappingProxyType

import torch

from utils.inference.stream_runtime import RootSourceProposal
from utils.inference.timeline import RootFrameState, RootTimeline
from utils.motion_process import build_physical_7d_from_5d
from web_demo.runtime.trajectory_diagnostics import TrajectoryDiagnosticsStore


def _proposal(version=1):
    traj = torch.tensor(
        [
            [10.0, 1.0, 20.0, 1.0, 0.0],
            [11.0, 1.0, 20.0, 1.0, 0.0],
            [12.0, 1.0, 20.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return RootSourceProposal(
        future_traj7=build_physical_7d_from_5d(traj),
        future_frame_mask=torch.tensor([True, True, False]),
        source_id=f"source-{version}",
        version=version,
        metadata={},
    )


def _timeline():
    return RootTimeline(
        RootFrameState(
            commit_idx=0,
            world_xz=torch.tensor([10.0, 20.0]),
            world_yaw=torch.tensor(0.0),
        )
    )


def _payload():
    local_5d = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return {
        "traj_cond_7d_frame": build_physical_7d_from_5d(local_5d).unsqueeze(0),
        "traj_cond_frame_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "body_anchor_abs_token": 0,
    }


def _event(version=1, *, active=True, payload=None):
    return SimpleNamespace(
        absolute_commit_before=version - 1,
        source_id=f"source-{version}",
        source_version=version,
        actual_activation_commit=version - 1,
        lifecycle_events=("route_active",) if active else (),
        route_status=SimpleNamespace(value="active"),
        actual_payload=_payload() if payload is None else payload,
    )


def _manager(version=1):
    return SimpleNamespace(active=SimpleNamespace(proposal=_proposal(version)))


def test_store_masks_proposal_and_converts_payload_to_world():
    store = TrajectoryDiagnosticsStore()

    store.update_from_commit(_event(), _manager(), _timeline())
    payload = store.to_payload()

    assert payload["current"]["root_source_proposal"] == [
        [10.0, 1.0, 20.0],
        [11.0, 1.0, 20.0],
    ]
    assert payload["current"]["actual_payload"] == [
        [10.0, 1.0, 20.0],
        [11.0, 1.0, 20.0],
    ]
    assert payload["current"]["source_version"] == 1
    assert len(payload["snapshots"]) == 1


def test_store_accepts_immutable_runtime_payload_mapping():
    store = TrajectoryDiagnosticsStore()

    store.update_from_commit(
        _event(payload=MappingProxyType(_payload())),
        _manager(),
        _timeline(),
    )

    assert len(store.to_payload()["current"]["actual_payload"]) == 2


def test_store_exposes_only_uncommitted_payload_frames_as_future_geometry():
    store = TrajectoryDiagnosticsStore()
    event = _event(active=False)
    event.absolute_commit_before = 1

    store.update_from_commit(event, _manager(), _timeline())

    current = store.to_payload()["current"]
    assert current["actual_payload"] == [
        [10.0, 1.0, 20.0],
        [11.0, 1.0, 20.0],
    ]
    assert current["actual_payload_future"] == [[11.0, 1.0, 20.0]]


def test_store_adds_one_snapshot_per_version_and_caps_history():
    store = TrajectoryDiagnosticsStore(max_snapshots=32)
    timeline = _timeline()

    for version in range(1, 36):
        store.update_from_commit(_event(version), _manager(version), timeline)
        store.update_from_commit(
            _event(version, active=False),
            _manager(version),
            timeline,
        )

    payload = store.to_payload()
    assert len(payload["snapshots"]) == 32
    assert payload["snapshots"][0]["source_version"] == 4
    assert payload["snapshots"][-1]["source_version"] == 35


def test_store_omits_unchanged_snapshots_when_client_has_current_revision():
    store = TrajectoryDiagnosticsStore()
    store.update_from_commit(_event(), _manager(), _timeline())

    first = store.to_payload(client_snapshot_revision=-1)
    revision = first["snapshot_revision"]
    assert first["snapshots"]

    unchanged = store.to_payload(client_snapshot_revision=revision)
    assert unchanged["snapshot_revision"] == revision
    assert "snapshots" not in unchanged
    assert "authored_route" not in unchanged["current"]
    assert "root_source_proposal" not in unchanged["current"]

    store.update_from_commit(_event(2), _manager(2), _timeline())
    changed = store.to_payload(client_snapshot_revision=revision)
    assert changed["snapshot_revision"] > revision
    assert changed["snapshots"][-1]["source_version"] == 2


def test_store_retains_last_valid_geometry_when_payload_is_malformed():
    store = TrajectoryDiagnosticsStore()
    store.update_from_commit(_event(), _manager(), _timeline())

    bad = _event(active=False, payload={"traj_cond_7d_frame": "bad"})
    store.update_from_commit(bad, _manager(), _timeline())
    payload = store.to_payload()

    assert payload["current"]["actual_payload"] == [
        [10.0, 1.0, 20.0],
        [11.0, 1.0, 20.0],
    ]
    assert payload["last_error"]


def test_store_tracks_authored_route_and_clear_resets_everything():
    store = TrajectoryDiagnosticsStore()
    store.set_authored_route([[1.0, 0.0, 2.0], [2.0, 0.0, 3.0]])
    store.update_from_commit(_event(), _manager(), _timeline())

    store.clear()

    payload = store.to_payload()
    assert payload["current"]["authored_route"] == []
    assert payload["current"]["root_source_proposal"] == []
    assert payload["current"]["actual_payload"] == []
    assert payload["snapshots"] == []
