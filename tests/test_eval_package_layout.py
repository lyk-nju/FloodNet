from pathlib import Path
import importlib
import subprocess
import sys

import pytest


def test_eval_common_types_have_stable_payloads(tmp_path):
    from eval.common import (
        EvalContext,
        EvalPrediction,
        EvalSample,
        MetricResult,
        ModelBundle,
    )

    ctx = EvalContext(
        config={"name": "smoke"},
        device="cpu",
        seed=7,
        output_dir=tmp_path,
    )
    assert ctx.output_path == tmp_path
    assert ctx.to_metadata() == {
        "device": "cpu",
        "seed": 7,
        "output_dir": str(tmp_path),
    }

    bundle = ModelBundle(model="ldf", vae="vae", stats={"root": 1.0})
    assert bundle.require("model", "vae") == bundle
    with pytest.raises(
        ValueError,
        match="missing required model components: root_refiner",
    ):
        bundle.require("root_refiner")

    sample = EvalSample(text="walk", metadata={"id": "000001"})
    pred = EvalPrediction(root_plan={"tokens": 3}, debug_state={"stage": "runtime"})
    result = MetricResult(
        summary={"ADE": 0.1},
        per_sample=[{"id": sample.metadata["id"]}],
        artifacts={"plan": Path("plan.json")},
    )

    assert sample.metadata["id"] == "000001"
    assert pred.debug_state["stage"] == "runtime"
    assert result.to_json_dict() == {
        "summary": {"ADE": 0.1},
        "per_sample": [{"id": "000001"}],
        "artifacts": {"plan": "plan.json"},
    }


def test_layered_eval_packages_import():
    module_names = [
        "eval.common",
        "eval.common.artifacts",
        "eval.ldf.generation_metrics",
        "eval.ldf.cases",
        "eval.ldf.conditioning",
        "eval.ldf.stream_metrics",
        "eval.ldf.sweep_cfg",
        "eval.ldf.t2m_metrics",
        "eval.root_refiner.adapters",
        "eval.root_refiner.benchmark",
        "eval.root_refiner.metrics",
        "eval.runtime.benchmark",
        "eval.runtime.cases",
        "eval.runtime.events",
        "eval.runtime.metrics",
        "eval.runtime.state_machine",
        "eval.runtime.diagnose_control",
        "eval.runtime.suites",
        "eval.runtime.suites.humanml3d_control",
        "eval.runtime.suites.route_edit",
        "eval.runtime.suites.text_update",
        "eval.runtime.suites.babel_long_session",
    ]
    for module_name in module_names:
        assert importlib.import_module(module_name)


def test_new_root_refiner_metrics_match_legacy_helpers():
    from eval import root_refiner_benchmark as legacy
    from eval.root_refiner import metrics as new_metrics

    assert new_metrics.compute_sample_metrics is legacy.compute_sample_metrics
    assert new_metrics._heading_to_yaw is legacy._heading_to_yaw


def test_layered_eval_modules_are_real_implementations_not_import_star_wrappers():
    module_paths = [
        Path("eval/root_refiner/benchmark.py"),
        Path("eval/root_refiner/metrics.py"),
        Path("eval/runtime/benchmark.py"),
        Path("eval/runtime/cases.py"),
        Path("eval/runtime/events.py"),
        Path("eval/runtime/metrics.py"),
        Path("eval/runtime/state_machine.py"),
        Path("eval/runtime/suites/__init__.py"),
        Path("eval/runtime/suites/humanml3d_control.py"),
        Path("eval/runtime/suites/route_edit.py"),
        Path("eval/runtime/suites/text_update.py"),
        Path("eval/runtime/suites/babel_long_session.py"),
        Path("eval/ldf/generation_metrics.py"),
        Path("eval/ldf/cases.py"),
        Path("eval/ldf/conditioning.py"),
        Path("eval/ldf/stream_metrics.py"),
        Path("eval/ldf/sweep_cfg.py"),
        Path("eval/ldf/t2m_metrics.py"),
        Path("eval/root_refiner/adapters.py"),
    ]
    for path in module_paths:
        text = path.read_text()
        assert "import *" not in text, f"{path} is still a wildcard wrapper"
        assert "from eval.stream_benchmark import" not in text
        assert "from eval.stream_benchmarks import" not in text
        assert "from eval.stream_metrics import" not in text
        assert "from eval.root_refiner_benchmark import" not in text
        assert "from eval.eval_generation_metrics import" not in text
        assert "from eval.eval_stream_metrics import" not in text


def test_legacy_eval_entrypoints_are_thin_compatibility_wrappers():
    cli_wrappers = {
        Path("eval/root_refiner_benchmark.py"): "eval.root_refiner.benchmark",
        Path("eval/stream_benchmark.py"): "eval.runtime.benchmark",
        Path("eval/eval_generation_metrics.py"): "eval.ldf.generation_metrics",
        Path("eval/eval_stream_metrics.py"): "eval.ldf.stream_metrics",
        Path("eval/run_t2m_metrics.py"): "eval.ldf.t2m_metrics",
        Path("eval/sweep_cfg.py"): "eval.ldf.sweep_cfg",
    }
    module_wrappers = {
        Path("eval/stream_benchmarks.py"): "eval.runtime.cases",
        Path("eval/stream_metrics.py"): "eval.runtime.metrics",
    }
    for path, target in cli_wrappers.items():
        text = path.read_text()
        assert f"from {target} import *" in text
        assert f"from {target} import main" in text
    for path, target in module_wrappers.items():
        text = path.read_text()
        assert f"from {target} import *" in text
        assert " import main" not in text


def test_legacy_import_compatibility_exports_required_symbols():
    from eval.root_refiner_benchmark import run_suite_benchmark
    from eval.stream_benchmark import (
        _build_turn_metric_target,
        _csv_safe_record,
        _run_turn,
    )
    from eval.stream_metrics import build_plan_metrics

    assert callable(_build_turn_metric_target)
    assert callable(_csv_safe_record)
    assert callable(_run_turn)
    assert callable(build_plan_metrics)
    assert callable(run_suite_benchmark)


def test_layered_ldf_defaults_preserve_legacy_eval_artifact_dirs():
    from eval.ldf import generation_metrics, t2m_metrics

    eval_root = Path(generation_metrics.__file__).resolve().parents[1]
    assert generation_metrics._default_output_dir() == eval_root
    assert t2m_metrics._default_shards_dir() == eval_root / "t2m_parallel_shards"


def test_new_cli_wrappers_expose_main_functions():
    from eval.ldf import generation_metrics, stream_metrics, t2m_metrics
    from eval.root_refiner import benchmark as root_refiner_benchmark
    from eval.runtime import benchmark, diagnose_control

    assert callable(generation_metrics.main)
    assert callable(stream_metrics.main)
    assert callable(t2m_metrics.main)
    assert callable(root_refiner_benchmark.main)
    assert callable(benchmark.main)
    assert callable(diagnose_control.main)


@pytest.mark.parametrize(
    "script_path",
    [
        "eval/ldf/generation_metrics.py",
        "eval/ldf/stream_metrics.py",
        "eval/ldf/sweep_cfg.py",
        "eval/ldf/t2m_metrics.py",
        "eval/root_refiner/benchmark.py",
        "eval/runtime/benchmark.py",
        "eval/runtime/diagnose_control.py",
    ],
)
def test_new_cli_wrappers_can_run_as_scripts(script_path):
    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "script_path",
    [
        "eval/root_refiner_benchmark.py",
        "eval/stream_benchmark.py",
        "eval/eval_generation_metrics.py",
        "eval/eval_stream_metrics.py",
        "eval/run_t2m_metrics.py",
        "eval/sweep_cfg.py",
    ],
)
def test_legacy_cli_wrappers_still_run_as_scripts(script_path):
    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
