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
        "eval.ldf.generation_metrics",
        "eval.ldf.stream_metrics",
        "eval.ldf.t2m_metrics",
        "eval.root_refiner.benchmark",
        "eval.root_refiner.metrics",
        "eval.runtime.benchmark",
        "eval.runtime.cases",
        "eval.runtime.metrics",
        "eval.runtime.diagnose_control",
    ]
    for module_name in module_names:
        assert importlib.import_module(module_name)


def test_new_root_refiner_metrics_match_legacy_helpers():
    from eval import root_refiner_benchmark as legacy
    from eval.root_refiner import metrics as new_metrics

    assert new_metrics.compute_sample_metrics is legacy.compute_sample_metrics
    assert new_metrics._heading_to_yaw is legacy._heading_to_yaw


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
