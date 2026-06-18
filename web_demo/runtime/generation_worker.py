"""Background generation worker for the web runtime."""

from __future__ import annotations

import threading


class GenerationWorker:
    """Owns generation thread lifecycle for a target loop."""

    def __init__(self, target):
        self.target = target
        self.thread: threading.Thread | None = None

    def start(self):
        if self.is_running:
            return self.thread
        self.thread = threading.Thread(target=self.target)
        self.thread.daemon = True
        self.thread.start()
        return self.thread

    def stop(self, timeout: float = 5.0) -> bool:
        if self.thread is None:
            return True
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


__all__ = ["GenerationWorker"]
