from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_web_demo_exposes_root_feedback_controls():
    html = (ROOT / "web_demo" / "templates" / "index.html").read_text()
    js = (ROOT / "web_demo" / "static" / "js" / "main.js").read_text()

    for element_id in (
        "rootFeedbackEnabled",
        "rootFeedbackAlpha",
        "rootFeedbackValue",
        "currentRootFeedback",
    ):
        assert f'id="{element_id}"' in html
        assert f"getElementById('{element_id}')" in js


def test_web_demo_sends_root_feedback_controls_to_start_and_reset():
    js = (ROOT / "web_demo" / "static" / "js" / "main.js").read_text()

    assert "root_feedback_enabled: rootFeedbackEnabled" in js
    assert "root_feedback_xz_blend_alpha: rootFeedbackAlpha" in js


def test_web_demo_defaults_root_feedback_to_hard_replacement():
    html = (ROOT / "web_demo" / "templates" / "index.html").read_text()

    assert 'id="rootFeedbackEnabled" checked' in html
    assert 'id="rootFeedbackAlpha" min="0" max="1" step="0.05" value="1.0"' in html
    assert 'id="rootFeedbackValue" class="slider-value">1.00<' in html
    assert 'id="currentRootFeedback" class="status-value">On · 1.00<' in html


def test_web_demo_exposes_committed_trajectory_diagnostic_layers():
    html = (ROOT / "web_demo" / "templates" / "index.html").read_text()
    js = (ROOT / "web_demo" / "static" / "js" / "main.js").read_text()

    for element_id in (
        "showAuthoredTrajectory",
        "showProposalTrajectory",
        "showPayloadTrajectory",
        "showTrajectoryHistory",
    ):
        assert f'id="{element_id}"' in html
        assert f"getElementById('{element_id}')" in js
    for symbol in (
        "trajAuthoredLine",
        "trajProposalLine",
        "trajPayloadLine",
        "trajHistoryGroup",
        "updateTrajectoryDiagnostics",
    ):
        assert symbol in js


def test_web_demo_requests_snapshot_deltas_and_renders_future_payload_only():
    js = (ROOT / "web_demo" / "static" / "js" / "main.js").read_text()

    assert "trajectory_snapshot_revision=" in js
    assert "current.actual_payload_future" in js
    assert "if (!Object.prototype.hasOwnProperty.call(debug, 'snapshots')) return" in js


def test_web_demo_reuses_line_geometry_buffers_between_frames():
    js = (ROOT / "web_demo" / "static" / "js" / "main.js").read_text()

    assert "line.geometry.dispose()" not in js[js.index("setTrajectoryLinePoints"):js.index("clearTrajectoryHistory")]
    assert "line.geometry.setDrawRange(0, count)" in js
    assert "line.userData.trajectoryPointCount = count" in js
