from __future__ import annotations

import time

from web_demo.services.session_service import SessionService


def test_session_service_claim_conflict_and_force_takeover():
    service = SessionService()

    first = service.claim_generation("a")
    conflict = service.claim_generation("b", force=False)
    forced = service.claim_generation("b", force=True)

    assert first.ok is True
    assert conflict.ok is False
    assert conflict.status_code == 409
    assert conflict.conflict is True
    assert conflict.active_session_id == "a"
    assert forced.ok is True
    assert forced.need_force_takeover is True
    assert forced.previous_session_id == "a"
    assert service.active_session_id == "b"


def test_session_service_rejects_duplicate_running_session():
    service = SessionService()
    service.claim_generation("a")

    claim = service.claim_generation("a", is_generating=True)

    assert claim.ok is False
    assert claim.status_code == 400
    assert "already running" in claim.message


def test_session_service_release_and_reset_permissions():
    service = SessionService()
    service.claim_generation("a")

    assert service.is_active("a") is True
    assert service.is_active("b") is False
    assert service.can_reset("b") is False

    service.release("b")
    assert service.active_session_id == "a"

    service.release("a")
    assert service.active_session_id is None
    assert service.can_reset("b") is True


def test_session_service_consumption_timeout_candidate():
    service = SessionService(consumption_timeout=0.1)
    service.claim_generation("a")
    service.touch_consumption()

    assert service._timeout_candidate() is None

    with service._consumption_lock:
        service._last_frame_consumed_time = time.time() - 1.0
    candidate = service._timeout_candidate()

    assert candidate is not None
    assert candidate.session_id == "a"
    assert candidate.elapsed >= 0.1


def test_session_service_timeout_callback_releases_session_only_on_success():
    service = SessionService(consumption_timeout=0.1)
    service.claim_generation("a")
    with service._consumption_lock:
        service._last_frame_consumed_time = time.time() - 1.0

    assert service.handle_timeout_once(lambda session_id, elapsed: False) is False
    assert service.active_session_id == "a"

    assert service.handle_timeout_once(lambda session_id, elapsed: True) is True
    assert service.active_session_id is None
    assert service._timeout_candidate() is None


def test_session_service_timeout_candidate_is_ignored_after_new_consumption():
    service = SessionService(consumption_timeout=0.1)
    service.claim_generation("a")
    with service._consumption_lock:
        service._last_frame_consumed_time = time.time() - 1.0
    stale_candidate = service._timeout_candidate()
    assert stale_candidate is not None
    service.touch_consumption()

    called = False

    def callback(session_id, elapsed):
        nonlocal called
        called = True
        return True

    service._timeout_candidate = lambda: stale_candidate

    assert service.handle_timeout_once(callback) is False
    assert called is False
    assert service.active_session_id == "a"
