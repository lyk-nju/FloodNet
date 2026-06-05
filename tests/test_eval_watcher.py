from __future__ import annotations

from eval.eval_watcher import _mark_completed_without_summary


def test_mark_completed_without_summary_is_terminal_and_deduped():
    state = {"completed": [], "failed": {"request.json": 2}}

    _mark_completed_without_summary(state, "request.json")
    _mark_completed_without_summary(state, "request.json")

    assert state["completed"] == ["request.json"]
    assert state["completed_without_summary"] == ["request.json"]
    assert state["failed"] == {}
