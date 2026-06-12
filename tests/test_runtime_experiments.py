from eval.runtime.experiments import (
    build_default_runtime_experiments,
    filter_runtime_experiments,
    parse_csv_ints,
    parse_csv_strings,
    runtime_debug_root_refiner_history_policy,
    runtime_debug_mode_for_source,
    runtime_debug_turn_plan_policy,
    summarize_numeric_records,
)
from eval.runtime.root_sources import (
    normalize_root_source,
    root_source_metadata,
    runtime_debug_condition_source,
)
from eval.runtime.runners import RuntimeGenerationResult


def test_default_runtime_experiments_include_expected_sources_and_families():
    specs = build_default_runtime_experiments(sample_id="001168")
    keys = {(spec.root_source, spec.family, spec.name) for spec in specs}

    assert ("gtroot", "web_stream", "web_stream") in keys
    assert ("rootrefiner", "web_stream", "web_stream") in keys
    assert ("notraj", "web_stream", "web_stream") in keys
    assert ("gtroot", "rotation", "rot_090") in keys
    assert ("rootrefiner", "turn", "delay_020") in keys
    assert all(spec.root_source != "notraj" or spec.family == "web_stream" for spec in specs)


def test_rotation_specs_carry_angle_degrees():
    specs = build_default_runtime_experiments(sample_id="001168")
    rot90 = next(spec for spec in specs if spec.root_source == "gtroot" and spec.name == "rot_090")

    assert rot90.params["rotate_plan_deg"] == 90.0
    assert rot90.parts == ("rot_090",)


def test_turn_specs_carry_delay_and_blend_parameters():
    specs = build_default_runtime_experiments(sample_id="001168")
    delay20 = next(
        spec
        for spec in specs
        if spec.root_source == "rootrefiner" and spec.family == "turn" and spec.name == "delay_020"
    )
    blend4 = next(
        spec
        for spec in specs
        if spec.root_source == "gtroot" and spec.family == "turn" and spec.name == "blend_004"
    )

    assert delay20.parts == ("delay", "delay_020")
    assert delay20.params["mid_update_delay_tokens"] == 20
    assert delay20.params["mid_update_blend_tokens"] == 0
    assert blend4.parts == ("blend", "blend_004")
    assert blend4.params["mid_update_delay_tokens"] == 20
    assert blend4.params["mid_update_blend_tokens"] == 4


def test_filter_runtime_experiments_selects_requested_families():
    specs = build_default_runtime_experiments(sample_id="001168")

    filtered = filter_runtime_experiments(specs, families=("web_stream", "rotation"))

    assert filtered
    assert {spec.family for spec in filtered} == {"web_stream", "rotation"}
    assert all(spec.family != "turn" for spec in filtered)


def test_runtime_debug_mode_for_source_maps_root_source_behavior():
    assert runtime_debug_mode_for_source("gtroot") == ("real_gtroot", True, False)
    assert runtime_debug_mode_for_source("rootrefiner") == ("real_route", False, False)
    assert runtime_debug_mode_for_source("notraj") == ("real_no_traj", False, True)


def test_parse_csv_helpers_use_defaults_and_strip_empty_items():
    assert parse_csv_strings(None, default=("a", "b")) == ("a", "b")
    assert parse_csv_strings(" a, ,b ", default=("x",)) == ("a", "b")
    assert parse_csv_ints(None, default=(1, 2)) == (1, 2)
    assert parse_csv_ints("1, 3,5", default=(2,)) == (1, 3, 5)


def test_summarize_numeric_records_aggregates_only_finite_numbers():
    summary = summarize_numeric_records(
        [
            {"family": "web_stream", "ADE": 1.0, "bad": float("nan")},
            {"family": "rotation", "ADE": 3.0, "flag": True},
        ]
    )

    assert summary["num_records"] == 2
    assert summary["ADE_mean"] == 2.0
    assert summary["ADE_count"] == 2
    assert "bad_mean" not in summary
    assert "flag_mean" not in summary


def test_normalize_root_source_aliases():
    assert normalize_root_source("gtroot") == "gtroot"
    assert normalize_root_source("gt_7d_ldf") == "gtroot"
    assert normalize_root_source("rootrefiner_7d_ldf") == "rootrefiner"
    assert normalize_root_source("root_refiner_7d_ldf") == "rootrefiner"
    assert normalize_root_source("no_traj_ldf") == "notraj"


def test_root_source_metadata_describes_condition_source():
    assert root_source_metadata("gtroot")["condition_source"] == "gt_motion_7d"
    assert root_source_metadata("rootrefiner")["condition_source"] == "rootrefiner_7d"
    assert root_source_metadata("notraj")["condition_source"] == "none"


def test_runtime_debug_condition_source_disambiguates_turn_gtroot():
    assert runtime_debug_condition_source("gtroot", family="web_stream") == "gt_motion_7d"
    assert runtime_debug_condition_source("gtroot", family="rotation") == "gt_motion_7d"
    assert runtime_debug_condition_source("gtroot", family="turn") == "route_derived_7d"
    assert runtime_debug_condition_source("rootrefiner", family="turn") == "rootrefiner_7d"


def test_runtime_debug_metadata_policies_are_explicit():
    assert runtime_debug_turn_plan_policy("web_stream", root_source="gtroot") == "none"
    assert runtime_debug_turn_plan_policy("turn", root_source="gtroot") == "composed_rootplan"
    assert runtime_debug_turn_plan_policy("turn", root_source="notraj") == "none"
    assert runtime_debug_root_refiner_history_policy("gtroot", family="web_stream") == "none"
    assert (
        runtime_debug_root_refiner_history_policy("rootrefiner", family="web_stream")
        == "anchor_only_initial"
    )
    assert (
        runtime_debug_root_refiner_history_policy("rootrefiner", family="turn")
        == "generated_history"
    )


def test_root_source_metadata_documents_family_condition_sources():
    metadata = root_source_metadata("gtroot")

    assert metadata["family_condition_sources"]["web_stream"] == "gt_motion_7d"
    assert metadata["family_condition_sources"]["rotation"] == "gt_motion_7d"
    assert metadata["family_condition_sources"]["turn"] == "route_derived_7d"


def test_runtime_generation_result_keeps_generation_boundary_fields():
    result = RuntimeGenerationResult(
        motion_263=None,
        pred_root_world=None,
        target_root_world=None,
        metrics={"ADE": 0.1},
        root_plan=None,
        metadata={"root_source": "gtroot"},
    )

    assert result.metrics["ADE"] == 0.1
    assert result.metadata["root_source"] == "gtroot"
