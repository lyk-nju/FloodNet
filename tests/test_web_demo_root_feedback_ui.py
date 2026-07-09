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
