from types import SimpleNamespace

import torch

from tools.run_ldf_condition_update_eval import (
    _build_condition_scenario_from_args,
    _parse_args,
    resolve_effective_update_commit,
)
from eval.ldf.runtime_update.root_source import RootSourceProposal
from utils.motion_process import build_physical_7d_from_5d


def _line_traj7(num_frames: int) -> torch.Tensor:
    x = torch.zeros(num_frames, dtype=torch.float32)
    y = torch.zeros(num_frames, dtype=torch.float32)
    z = torch.arange(num_frames, dtype=torch.float32) * 0.1
    yaw = torch.zeros(num_frames, dtype=torch.float32)
    traj5 = torch.stack([x, y, z, torch.cos(yaw), torch.sin(yaw)], dim=-1)
    return build_physical_7d_from_5d(traj5)


def _base_args(**overrides):
    values = {
        "condition_source": "repeat_splice",
        "repeat_mode": "center_symmetric",
        "update_frame": 4,
        "update_lead_tokens": 6,
        "frames_per_token": 4,
        "suffix_frames": 5,
        "source_start_frame": 0,
        "trim_start_frames": 20,
        "trim_end_frames": 20,
        "derive_heading_from_path": True,
        "transition_frames": 0,
        "transition_output_frames": 0,
        "suffix_rotation_deg": 0.0,
        "turn_transition_frames": 0,
        "arc_speed_window": 8,
        "arc_speed_scale": 1.0,
        "suffix_blend_frames": 8,
        "suffix_min_speed_factor": 0.25,
        "suffix_lateral_scale": 0.0,
        "arc_y_mode": "source",
        "repeat_heading_blend_frames": 0,
        "arc_smooth_profile": "geometric",
        "synthetic_preset": "four_segment_curve",
        "synthetic_frames": 120,
        "synthetic_update_frames": "30,60,90",
        "synthetic_forward_step_length": 0.015,
        "synthetic_total_forward": 4.2,
        "synthetic_arc_turn_deg": 20.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_condition_update_cli_displays_full_condition_by_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ldf_condition_update_eval.py",
            "--ckpt",
            "dummy.ckpt",
            "--meta_path",
            "dummy.txt",
            "--out_dir",
            "dummy_out",
        ],
    )

    args = _parse_args()

    assert args.condition_traj_display == "full"


def test_condition_update_cli_accepts_absolute_route_contract(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ldf_condition_update_eval.py",
            "--ckpt",
            "dummy.ckpt",
            "--meta_path",
            "dummy.txt",
            "--out_dir",
            "dummy_out",
            "--runtime_update_contract",
            "absolute_route",
        ],
    )

    args = _parse_args()

    assert args.runtime_update_contract == "absolute_route"


def test_condition_update_cli_accepts_root_source_refiner_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ldf_condition_update_eval.py",
            "--ckpt",
            "dummy.ckpt",
            "--meta_path",
            "dummy.txt",
            "--out_dir",
            "dummy_out",
            "--root_source_refiner_ckpt",
            "refiner.ckpt",
            "--root_source_refiner_heading_override",
            "path_tangent",
        ],
    )

    args = _parse_args()

    assert args.root_source_refiner_ckpt == "refiner.ckpt"
    assert args.root_source_refiner_heading_override == "path_tangent"


def test_effective_update_commit_waits_for_next_uncommitted_token_boundary():
    assert resolve_effective_update_commit(
        raw_update_frame=123,
        first_uncommitted_token=31,
        frames_per_token=4,
    ) == 32
    assert resolve_effective_update_commit(
        raw_update_frame=121,
        first_uncommitted_token=31,
        frames_per_token=4,
    ) == 31
    assert resolve_effective_update_commit(
        raw_update_frame=123,
        first_uncommitted_token=33,
        frames_per_token=4,
    ) == 33


def test_cli_dispatches_repeat_splice_source():
    scenario = _build_condition_scenario_from_args(
        _base_args(condition_source="repeat_splice"),
        _line_traj7(10),
        original_frames=10,
        sample_name="000021",
        caption_index=0,
    )

    assert scenario.name == "repeat_splice:center_symmetric"
    assert scenario.update_frames == [4]
    assert scenario.base_sample_name == "000021"


def test_cli_repeat_splice_default_uses_trimmed_middle_window():
    scenario = _build_condition_scenario_from_args(
        _base_args(
            condition_source="repeat_splice",
            update_frame=None,
            source_start_frame=None,
            suffix_frames=None,
        ),
        _line_traj7(100),
        original_frames=100,
        sample_name="000021",
        caption_index=0,
    )

    assert scenario.update_frames == [79]
    assert scenario.metadata["source_start_frame"] == 20
    assert scenario.metadata["trim_start_frames"] == 20
    assert scenario.metadata["trim_end_frames"] == 20


def test_cli_dispatches_rotated_suffix_repeat_mode():
    scenario = _build_condition_scenario_from_args(
        _base_args(
            condition_source="repeat_splice",
            repeat_mode="rotated_suffix",
            suffix_rotation_deg=45.0,
            turn_transition_frames=6,
        ),
        _line_traj7(10),
        original_frames=10,
        sample_name="001168",
        caption_index=0,
    )

    assert scenario.name == "repeat_splice:rotated_suffix"
    assert scenario.metadata["suffix_rotation_deg"] == 45.0
    assert scenario.metadata["turn_transition_frames"] == 6


def test_cli_dispatches_rotated_suffix_arc_repeat_mode():
    scenario = _build_condition_scenario_from_args(
        _base_args(
            condition_source="repeat_splice",
            repeat_mode="rotated_suffix_arc",
            suffix_rotation_deg=45.0,
            turn_transition_frames=6,
            arc_speed_window=5,
            suffix_blend_frames=7,
            suffix_min_speed_factor=0.4,
            suffix_lateral_scale=0.2,
        ),
        _line_traj7(10),
        original_frames=10,
        sample_name="001168",
        caption_index=0,
    )

    assert scenario.name == "repeat_splice:rotated_suffix_arc"
    assert scenario.metadata["suffix_rotation_deg"] == 45.0
    assert scenario.metadata["turn_transition_frames"] == 6
    assert scenario.metadata["arc_speed_window"] == 5
    assert scenario.metadata["suffix_blend_frames"] == 7
    assert scenario.metadata["suffix_min_speed_factor"] == 0.4
    assert scenario.metadata["suffix_lateral_scale"] == 0.2


def test_cli_dispatches_synthetic_source_with_multiple_updates():
    scenario = _build_condition_scenario_from_args(
        _base_args(condition_source="synthetic"),
        _line_traj7(10),
        original_frames=10,
        sample_name="001168",
        caption_index=2,
    )

    assert scenario.name == "synthetic:four_segment_curve"
    assert scenario.update_frames == [30, 60, 90]
    assert scenario.caption_index == 2


def test_cli_dispatches_synthetic_constant_arc_source():
    scenario = _build_condition_scenario_from_args(
        _base_args(
            condition_source="synthetic",
            synthetic_preset="constant_arc",
            synthetic_arc_turn_deg=12.5,
        ),
        _line_traj7(10),
        original_frames=10,
        sample_name="001168",
        caption_index=2,
    )

    assert scenario.name == "synthetic:constant_arc"
    assert scenario.metadata["arc_turn_degrees"] == 12.5


def test_condition_scenario_is_normalized_to_root_source_proposal():
    scenario = _build_condition_scenario_from_args(
        _base_args(
            condition_source="repeat_splice",
            repeat_mode="rotated_suffix_arc_chain",
            suffix_rotation_deg=20.0,
            repeat_count=4,
        ),
        _line_traj7(120),
        original_frames=120,
        sample_name="001168",
        caption_index=0,
    )

    proposal = RootSourceProposal.from_condition_scenario(
        scenario,
        source_kind="repeat_splice",
    )

    assert proposal.source_kind == "repeat_splice"
    assert proposal.name == scenario.name
    assert torch.allclose(proposal.proposal_traj7, scenario.condition_traj7)
    assert proposal.update_frames == scenario.update_frames
    assert proposal.base_sample_name == "001168"
    assert proposal.metadata["runtime_role"] == "root_source_proposal"
