"""Causal VAE token/frame mapping.

The causal VAE layout is not simply ``N tokens == 4N frames``. Token 0 covers
one effective frame, while each later token covers ``frames_per_token`` frames.
Keep all token/frame arithmetic in this module instead of duplicating formulas.

Layout:
    token 0: frame [0, 0]
    token k >= 1: frame [4k - 3, 4k] for the default frames_per_token=4

Prefix helpers and arbitrary-range helpers are different:
    num_frames_for_tokens(N) gives prefix [0, N) length: 4N - 3.
    token_range_to_frame_slice(start, N) gives length 4N when start >= 1.

This module uses pure integer arithmetic and has no project-level imports.
"""

from __future__ import annotations

FRAMES_PER_TOKEN_DEFAULT = 4


def token_start_frame(
    token_idx: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    """First frame index covered by `token_idx`.

    ``token 0 -> 0`` and
    ``token k >= 1 -> frames_per_token * k - (frames_per_token - 1)``.
    """
    if token_idx <= 0:
        return 0
    return frames_per_token * token_idx - (frames_per_token - 1)


def token_end_frame(
    token_idx: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    """Last frame index, inclusive, covered by `token_idx`."""
    if token_idx <= 0:
        return 0
    return frames_per_token * token_idx


def commit_boundary_frame(
    commit_idx: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    """Latest generated frame available before producing ``commit_idx``.

    Commit zero starts from the initial frame. For later commits, the observable
    actor state is the final frame covered by the previously committed token.
    """

    commit = int(commit_idx)
    if commit <= 0:
        return 0
    return token_end_frame(commit - 1, frames_per_token)


def first_future_frame_abs(
    commit_idx: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    """Return the absolute frame index immediately after committed tokens."""
    return num_frames_for_tokens(max(0, int(commit_idx)), frames_per_token)


def last_generated_frame_abs(
    commit_idx: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int | None:
    """Return the last generated absolute frame, or ``None`` at cold start."""
    future = first_future_frame_abs(commit_idx, frames_per_token)
    return None if future == 0 else future - 1


def num_frames_for_tokens(
    num_tokens: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    """Effective frame count for the **prefix** `[0, num_tokens)`.

    Returns `4N - 3` for N >= 1, and `0` for N <= 0. Only valid for prefix
    windows starting at token 0. For arbitrary sub-windows, use
    `token_range_to_frame_slice(start, N).stop - .start`.
    """
    if num_tokens <= 0:
        return 0
    return frames_per_token * num_tokens - (frames_per_token - 1)


def frame_idx_to_token_idx(
    frame_idx: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    """Inverse of `token_start_frame`: which token covers `frame_idx`.

    Layout (`frames_per_token=4`):
        frame 0      -> token 0
        frame 1..4   -> token 1
        frame 5..8   -> token 2
        frame 9..12  -> token 3
        ...

    The correct formula is `(frame_idx - 1) // frames_per_token + 1`.
    """
    if frame_idx <= 0:
        return 0
    return (frame_idx - 1) // frames_per_token + 1


def num_tokens_for_frame_len(
    frame_len: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    """Number of tokens whose prefix covers a `frame_len`-frame prefix.

    This is the inverse of `num_frames_for_tokens`: for `frame_len > 0`, return
    the token covering the last frame plus one.
    """
    if frame_len <= 0:
        return 0
    return frame_idx_to_token_idx(frame_len - 1, frames_per_token) + 1


def token_range_to_frame_slice(
    start_token_idx: int,
    num_tokens: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> slice:
    """Map an arbitrary token range `[start, start + num_tokens)` to a frame slice.

    Length of the resulting slice differs from `num_frames_for_tokens(N)` when
    `start_token_idx >= 1`:
        start_token_idx == 0: length = 4N - 3
        start_token_idx >= 1: length = 4N
    """
    start_frame = token_start_frame(start_token_idx, frames_per_token)
    if num_tokens <= 0:
        return slice(start_frame, start_frame)
    end_token_idx = start_token_idx + num_tokens - 1
    end_frame_exclusive = token_end_frame(end_token_idx, frames_per_token) + 1
    return slice(start_frame, end_frame_exclusive)


def token_active_window_left_frame(
    end_token_idx: int,
    chunk_size_tokens: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    """Frame index at the left edge of the active token window.

    This is for active loss/control ranges. Body-window canonicalization uses
    `token_body_window_left_frame`.
    """
    left_token_idx = max(0, end_token_idx - chunk_size_tokens)
    return token_start_frame(left_token_idx, frames_per_token)


def token_body_window_left_frame(
    end_token_idx: int,
    body_window_tokens: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    """Frame index at the left edge of the body window.

    `end_token_idx`: right edge of the body window, also the active-window right edge.
    `body_window_tokens`: total body window size in tokens (history + active).
    """
    left_token_idx = max(0, end_token_idx - body_window_tokens)
    return token_start_frame(left_token_idx, frames_per_token)


def frames_to_token_mask(
    mask_frame,
    num_tokens: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
):
    """Aggregate a frame-level mask to token level by OR.

    A token is valid (1) iff ANY frame it covers is valid; token 0 covers the
    single frame [0,0], token k >= 1 covers [4k-3, 4k]. Tokens whose frame span
    falls entirely beyond `mask_frame`'s length stay 0.

    `mask_frame`: tensor [..., T_frame] (any leading dims). Returns
    [..., num_tokens] in the same dtype/device.
    """
    num_frames = mask_frame.shape[-1]
    leading_shape = mask_frame.shape[:-1]
    out = mask_frame.new_zeros(*leading_shape, num_tokens)
    for token_idx in range(num_tokens):
        start = token_start_frame(token_idx, frames_per_token)
        stop = min(token_end_frame(token_idx, frames_per_token) + 1, num_frames)
        if start >= num_frames or start >= stop:
            continue
        out[..., token_idx] = (
            (mask_frame[..., start:stop] > 0).any(dim=-1).to(mask_frame.dtype)
        )
    return out


def prefix_len_from_tail_invalid(token_mask):
    """Per-sample valid-token prefix length for pure suffix-invalid masks.

    `token_mask`: [B, T] with 1 = valid. A middle hole cannot be represented as
    a single prefix length, so those rows return full T.
    """
    import torch

    valid = token_mask > 0
    _, num_tokens = valid.shape
    invalid = ~valid
    has_invalid = invalid.any(dim=1)
    first_invalid = torch.argmax(invalid.to(torch.int8), dim=1)
    prefix_len = torch.where(
        has_invalid, first_invalid, torch.full_like(first_invalid, num_tokens)
    )
    num_valid = valid.sum(dim=1)
    pure_prefix = num_valid == prefix_len
    return torch.where(
        pure_prefix, prefix_len, torch.full_like(prefix_len, num_tokens)
    ).to(torch.long)


__all__ = [
    "FRAMES_PER_TOKEN_DEFAULT",
    "token_start_frame",
    "token_end_frame",
    "commit_boundary_frame",
    "first_future_frame_abs",
    "last_generated_frame_abs",
    "num_frames_for_tokens",
    "num_tokens_for_frame_len",
    "frame_idx_to_token_idx",
    "token_range_to_frame_slice",
    "token_active_window_left_frame",
    "token_body_window_left_frame",
    "frames_to_token_mask",
    "prefix_len_from_tail_invalid",
]
