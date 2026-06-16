from .buffer import TrajStreamBuffer


def build_stream_traj_buffer(model, seq_len: int, batch_size: int = 1):
    return TrajStreamBuffer(
        batch_size=batch_size,
        buf_len=int(seq_len) * 2 + int(model.chunk_size),
        traj_encoder=model.traj_encoder,
        use_emb_cache=getattr(model, "use_traj_emb_cache", False),
    )


def init_stream_generation(
    model,
    seq_len: int,
    *,
    batch_size: int = 1,
    num_denoise_steps=None,
):
    if hasattr(model, "traj_encoder"):
        traj_buffer = build_stream_traj_buffer(model, seq_len, batch_size=batch_size)
        model.init_generated(
            seq_len,
            batch_size=batch_size,
            num_denoise_steps=num_denoise_steps,
            traj_buffer=traj_buffer,
        )
    else:
        traj_buffer = None
        model.init_generated(
            seq_len,
            batch_size=batch_size,
            num_denoise_steps=num_denoise_steps,
        )
    return traj_buffer
