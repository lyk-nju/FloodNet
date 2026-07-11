"""Thread-safe frame buffer for generated web demo frames."""

from __future__ import annotations

import threading
from collections import deque


class FrameBuffer:
    """Small queue used by the web server to stream generated frames."""

    def __init__(self, target_buffer_size=4):
        self.buffer = deque(maxlen=100)
        self.target_size = target_buffer_size
        self.lock = threading.Lock()

    def add_frame(self, joints):
        with self.lock:
            self.buffer.append(joints)

    def add_frames_atomic(self, frames):
        batch = list(frames)
        with self.lock:
            self.buffer.extend(batch)

    def get_frame(self):
        with self.lock:
            if len(self.buffer) > 0:
                return self.buffer.popleft()
            return None

    def size(self):
        with self.lock:
            return len(self.buffer)

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def needs_generation(self):
        return self.size() < self.target_size


__all__ = ["FrameBuffer"]
