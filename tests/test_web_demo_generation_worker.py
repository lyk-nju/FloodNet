from __future__ import annotations

import threading
import time

from web_demo.runtime.generation_worker import GenerationWorker


def test_generation_worker_start_twice_reuses_running_thread():
    started = threading.Event()
    stop_seen = threading.Event()
    calls = []

    def loop(stop_event):
        calls.append(stop_event)
        started.set()
        while not stop_event.is_set():
            time.sleep(0.005)
        stop_seen.set()

    worker = GenerationWorker(loop)
    first_thread = worker.start()
    assert started.wait(timeout=1.0)

    second_thread = worker.start()

    assert second_thread is first_thread
    assert len(calls) == 1
    assert worker.stop(timeout=1.0) is True
    assert stop_seen.is_set()


def test_generation_worker_stop_timeout_returns_false_until_thread_exits():
    release = threading.Event()
    started = threading.Event()

    def loop(stop_event):
        started.set()
        release.wait(timeout=1.0)

    worker = GenerationWorker(loop)
    worker.start()
    assert started.wait(timeout=1.0)

    assert worker.stop(timeout=0.01) is False
    assert worker.stop_event.is_set()

    release.set()
    assert worker.stop(timeout=1.0) is True


def test_generation_worker_restart_clears_stop_event():
    starts = []

    def loop(stop_event):
        starts.append(stop_event.is_set())

    worker = GenerationWorker(loop)
    worker.start().join(timeout=1.0)
    assert starts == [False]

    worker.stop_event.set()
    worker.start().join(timeout=1.0)

    assert starts == [False, False]
