"""Background generation worker for the web runtime."""

from __future__ import annotations

import threading


class GenerationWorker:
    """Owns generation thread lifecycle for a target loop."""

    def __init__(self, target):
        self.target = target
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self.is_running:
                return self.thread
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return self.thread

    def _run(self):
        self.target(self.stop_event)

    def stop(self, timeout: float = 5.0) -> bool:
        self.stop_event.set()
        thread = self.thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


__all__ = ["GenerationWorker"]
