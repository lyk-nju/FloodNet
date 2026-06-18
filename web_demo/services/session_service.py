"""Session ownership and frame-consumption tracking for the web demo."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SessionClaim:
    ok: bool
    status_code: int = 200
    message: str = ""
    conflict: bool = False
    active_session_id: str | None = None
    need_force_takeover: bool = False
    previous_session_id: str | None = None


@dataclass(frozen=True)
class TimeoutCandidate:
    session_id: str
    elapsed: float
    last_frame_consumed_time: float


class SessionService:
    """Owns active session state and frame-consumption timeout tracking."""

    def __init__(self, *, consumption_timeout: float = 5.0, poll_interval: float = 2.0):
        self.consumption_timeout = float(consumption_timeout)
        self.poll_interval = float(poll_interval)
        self._active_session_id: str | None = None
        self._session_lock = threading.Lock()
        self._last_frame_consumed_time: float | None = None
        self._consumption_lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None

    @property
    def active_session_id(self) -> str | None:
        with self._session_lock:
            return self._active_session_id

    def claim_generation(
        self,
        session_id: str,
        *,
        force: bool = False,
        is_generating: bool = False,
    ) -> SessionClaim:
        with self._session_lock:
            if self._active_session_id and self._active_session_id != session_id:
                if not force:
                    return SessionClaim(
                        ok=False,
                        status_code=409,
                        message="Another session is already generating.",
                        conflict=True,
                        active_session_id=self._active_session_id,
                    )
                previous_session = self._active_session_id
                self._active_session_id = session_id
                return SessionClaim(
                    ok=True,
                    active_session_id=session_id,
                    need_force_takeover=True,
                    previous_session_id=previous_session,
                )

            if is_generating and self._active_session_id == session_id:
                return SessionClaim(
                    ok=False,
                    status_code=400,
                    message="Generation is already running for this session.",
                    active_session_id=self._active_session_id,
                )

            self._active_session_id = session_id
            return SessionClaim(ok=True, active_session_id=session_id)

    def is_active(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        with self._session_lock:
            return self._active_session_id == session_id

    def can_reset(self, session_id: str | None) -> bool:
        with self._session_lock:
            return (
                not session_id
                or not self._active_session_id
                or self._active_session_id == session_id
            )

    def release(self, session_id: str | None = None) -> None:
        with self._session_lock:
            if session_id is None or self._active_session_id == session_id:
                self._active_session_id = None

    def active_status(self, session_id: str | None) -> tuple[bool, str | None]:
        with self._session_lock:
            active_session_id = self._active_session_id
        return bool(session_id and active_session_id == session_id), active_session_id

    def touch_consumption(self) -> None:
        with self._consumption_lock:
            self._last_frame_consumed_time = time.time()

    def clear_consumption(self) -> None:
        with self._consumption_lock:
            self._last_frame_consumed_time = None

    def _timeout_candidate(self) -> TimeoutCandidate | None:
        with self._consumption_lock:
            last_frame_consumed_time = self._last_frame_consumed_time
            if last_frame_consumed_time is None:
                return None
            elapsed = time.time() - last_frame_consumed_time
            if elapsed <= self.consumption_timeout:
                return None
        with self._session_lock:
            if self._active_session_id is None:
                return None
            return TimeoutCandidate(
                session_id=self._active_session_id,
                elapsed=elapsed,
                last_frame_consumed_time=last_frame_consumed_time,
            )

    def _timeout_candidate_still_current(self, candidate: TimeoutCandidate) -> bool:
        with self._consumption_lock:
            if self._last_frame_consumed_time != candidate.last_frame_consumed_time:
                return False
            if time.time() - candidate.last_frame_consumed_time <= self.consumption_timeout:
                return False
        with self._session_lock:
            return self._active_session_id == candidate.session_id

    def start_consumption_monitor(
        self,
        reset_callback: Callable[[str, float], bool],
    ) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(reset_callback,),
            daemon=True,
        )
        self._monitor_thread.start()

    def handle_timeout_once(self, reset_callback: Callable[[str, float], bool]) -> bool:
        candidate = self._timeout_candidate()
        if candidate is None:
            return False
        if not self._timeout_candidate_still_current(candidate):
            return False
        if not reset_callback(candidate.session_id, candidate.elapsed):
            return False
        self.release(candidate.session_id)
        self.clear_consumption()
        return True

    def _monitor_loop(self, reset_callback: Callable[[str, float], bool]) -> None:
        while True:
            time.sleep(self.poll_interval)
            self.handle_timeout_once(reset_callback)


__all__ = ["SessionClaim", "SessionService", "TimeoutCandidate"]
