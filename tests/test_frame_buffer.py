import pytest

from web_demo.runtime.frame_buffer import FrameBuffer


def test_add_frames_atomic_does_not_publish_partial_batch():
    buffer = FrameBuffer(target_buffer_size=4)

    def broken_frames():
        yield "frame-0"
        raise RuntimeError("invalid frame batch")

    with pytest.raises(RuntimeError, match="invalid frame batch"):
        buffer.add_frames_atomic(broken_frames())

    assert buffer.size() == 0


def test_add_frames_atomic_publishes_whole_batch_in_order():
    buffer = FrameBuffer(target_buffer_size=4)

    buffer.add_frames_atomic(["frame-0", "frame-1", "frame-2"])

    assert [buffer.get_frame(), buffer.get_frame(), buffer.get_frame()] == [
        "frame-0",
        "frame-1",
        "frame-2",
    ]
