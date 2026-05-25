"""Causal VAE token ↔ frame mapping (sole project-wide implementation).

References:
- docs/TODO.md §T_A_01b — full spec, unit tests T01-T18.
- docs/design.md §0.x — causal VAE convention notes.

Why this module: causal VAE token-frame is **not** `N tokens = 4N frames`.
Token 0 covers 1 frame; token k≥1 covers 4 frames each. Multiple hand-written
formulas already exist in `utils/traj_batch.py:149-150`, `utils/training/
control_loss.py:46`, etc. (known bug sources); this module is the canonical
replacement.

Layout:
    token 0     → frame [0, 0]      (1 effective frame, VAE pads to 4 copies)
    token k≥1   → frame [4k-3, 4k]  (4 effective frames)

    1 token   → 1  effective frame
    2 tokens  → 5  effective frames
    20 tokens → 77 effective frames   (only valid for prefix [0, 20))
    49 tokens → 193 effective frames

⚠ Prefix vs arbitrary range:
    num_frames_for_tokens(N) gives prefix [0, N) length, = 4N - 3.
    token_range_to_frame_slice(start, N) length is:
        4N - 3   if start == 0  (includes token 0, which counts as 1 frame)
        4N       if start ≥ 1   (excludes token 0)
Always use token_range_to_frame_slice for non-prefix windows
(horizon mask cutoff, plan sub-window, etc.).

Pure-integer arithmetic. No torch / numpy / FloodNet-module imports.
"""

from __future__ import annotations

FRAMES_PER_TOKEN_DEFAULT = 4


def token_start_frame(token_idx: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    """First frame index covered by `token_idx`.

    `token 0 → 0; token k≥1 → frames_per_token * k - (frames_per_token - 1)`
    (i.e. 1, 5, 9, ... for the default `frames_per_token=4`).
    """
    if token_idx <= 0:
        return 0
    return frames_per_token * token_idx - (frames_per_token - 1)


def token_end_frame(token_idx: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    """Last frame index (inclusive) covered by `token_idx`.

    `token 0 → 0; token k≥1 → frames_per_token * k`
    (i.e. 4, 8, 12, ... for the default `frames_per_token=4`).
    """
    if token_idx <= 0:
        return 0
    return frames_per_token * token_idx


def num_frames_for_tokens(num_tokens: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    """Effective frame count for the **prefix** `[0, num_tokens)`.

    Returns `4N - 3` for N ≥ 1, `0` for N ≤ 0. ⚠ Only valid for prefix windows
    starting at token 0. For arbitrary sub-windows, use
    `token_range_to_frame_slice(start, N).stop - .start`.
    """
    if num_tokens <= 0:
        return 0
    return frames_per_token * num_tokens - (frames_per_token - 1)


def frame_idx_to_token_idx(frame_idx: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    """Inverse of `token_start_frame`: which token covers `frame_idx`.

    Layout (`frames_per_token=4`):
        frame 0       → token 0
        frame 1..4    → token 1
        frame 5..8    → token 2
        frame 9..12   → token 3
        ...

    Correct formula: `(frame_idx - 1) // frames_per_token + 1` (T11 regression).
    Wrong formula `(frame_idx + frames_per_token - 2) // frames_per_token + 1`
    would return 2 at frame_idx=4 — that's the bug this regression test guards.
    """
    if frame_idx <= 0:
        return 0
    return (frame_idx - 1) // frames_per_token + 1


def num_tokens_for_frame_len(frame_len: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    """Number of causal-VAE tokens whose PREFIX `[0, num_tokens)` covers a
    `frame_len`-frame prefix — the inverse of `num_frames_for_tokens`.

    `frame_len <= 0 → 0`; otherwise `frame_idx_to_token_idx(frame_len - 1) + 1`
    (the token covering the last frame, plus one). Use this instead of hand-rolled
    `(L + 2) // 4 + 1`-style formulas. The tensor path in
    DiffForcingWanModel._get_traj_seq_lens mirrors this elementwise.
    """
    if frame_len <= 0:
        return 0
    return frame_idx_to_token_idx(frame_len - 1, frames_per_token) + 1


def token_range_to_frame_slice(start_token_idx: int, num_tokens: int,
                                frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> slice:
    """Map an arbitrary token range `[start, start+num_tokens)` → frame slice.

    Length of the resulting slice differs from `num_frames_for_tokens(N)` when
    `start_token_idx >= 1`:
        start_token_idx == 0: length = 4N - 3
        start_token_idx >= 1: length = 4N
    See T13 unit test for the lock-in regression.
    """
    start_frame = token_start_frame(start_token_idx, frames_per_token)
    if num_tokens <= 0:
        return slice(start_frame, start_frame)
    end_token_idx = start_token_idx + num_tokens - 1
    end_frame_exclusive = token_end_frame(end_token_idx, frames_per_token) + 1
    return slice(start_frame, end_frame_exclusive)


def token_active_window_left_frame(end_token_idx: int, chunk_size_tokens: int,
                                    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    """Frame index at the LEFT edge of the active window (β>0 region).

    `end_token_idx`: right edge (exclusive in some downstream conventions; here
    we treat it as the rightmost token's index, with the left edge being
    `end_token_idx - chunk_size_tokens` clamped to ≥ 0).
    `chunk_size_tokens`: width of the active window in tokens.

    Used for: heading / control loss active frame range. ⚠ NOT for body anchor
    canonicalize (use `token_body_window_left_frame` for that — the two
    differ by tens of tokens since body window is much wider than chunk size).
    """
    left_token_idx = max(0, end_token_idx - chunk_size_tokens)
    return token_start_frame(left_token_idx, frames_per_token)


def token_body_window_left_frame(end_token_idx: int, body_window_tokens: int,
                                  frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    """Frame index at the LEFT edge of the body window (= history0 anchor).

    `end_token_idx`: right edge of the body window (= active window right edge).
    `body_window_tokens`: total body window size in tokens (history + active).

    Used by body fine-tune anchor canonicalization; matches inference
    `TrajStreamBuffer.start_t = max(0, end_index - seq_len)` semantics. ⚠ NOT
    interchangeable with `token_active_window_left_frame(end_token_idx,
    chunk_size_tokens)`.
    """
    left_token_idx = max(0, end_token_idx - body_window_tokens)
    return token_start_frame(left_token_idx, frames_per_token)


def frames_to_token_mask(mask_frame, num_tokens: int,
                         frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT):
    """Aggregate a frame-level mask to token level by OR (design §2.2.2).

    A token is valid (1) iff ANY frame it covers is valid; token 0 covers the
    single frame [0,0], token k≥1 covers [4k-3, 4k]. Tokens whose frame span
    falls entirely beyond `mask_frame`'s length stay 0.

    `mask_frame`: tensor [..., T_frame] (any leading dims). Returns
    [..., num_tokens] in the same dtype/device.
    """
    T_frame = mask_frame.shape[-1]
    lead = mask_frame.shape[:-1]
    out = mask_frame.new_zeros(*lead, num_tokens)
    for k in range(num_tokens):
        s = token_start_frame(k, frames_per_token)
        e = min(token_end_frame(k, frames_per_token) + 1, T_frame)  # inclusive→exclusive
        if s >= T_frame or s >= e:
            continue  # token beyond available frames → stays 0
        out[..., k] = (mask_frame[..., s:e] > 0).any(dim=-1).to(mask_frame.dtype)
    return out


def prefix_len_from_tail_invalid(token_mask):
    """Per-sample valid-token PREFIX length, but ONLY when the invalid region is
    a pure suffix. Wired into DiffForcingWanModel._get_traj_seq_lens (B-P0-1) to
    truncate ControlNet attention past an out-of-horizon / overflow tail.

    `token_mask`: [B, T] (1 = valid). Returns LongTensor [B]:
      - tail-invalid (e.g. [1,1,1,1,0,0,0]) → prefix length 4 (safe to truncate
        traj_seq_lens; attention then ignores the out-of-horizon/overflow tail);
      - middle hole (e.g. [1,1,0,1,1,0,0]) → full T (a hole is NOT expressible as
        a single prefix length; truncating would wrongly drop later valid tokens —
        these stay handled by the per-token embedding zeroing, not by seq_lens);
      - all-invalid → 0; all-valid → T.

    Fully vectorized (no per-sample Python loop / host sync): a row is a pure
    valid-prefix iff its valid count equals the index of its first invalid token.
    """
    import torch

    valid = token_mask > 0
    B, T = valid.shape
    invalid = ~valid
    has_invalid = invalid.any(dim=1)
    # Index of the first invalid token (0 when the row is all-valid; the
    # has_invalid mask below replaces those with T).
    first_invalid = torch.argmax(invalid.to(torch.int8), dim=1)
    prefix_len = torch.where(
        has_invalid, first_invalid, torch.full_like(first_invalid, T)
    )
    num_valid = valid.sum(dim=1)
    # Pure valid-prefix ⇔ every valid token is before the first invalid one
    # (num_valid == first_invalid); a middle hole has num_valid > first_invalid.
    pure_prefix = num_valid == prefix_len
    return torch.where(
        pure_prefix, prefix_len, torch.full_like(prefix_len, T)
    ).to(torch.long)


__all__ = [
    "FRAMES_PER_TOKEN_DEFAULT",
    "token_start_frame",
    "token_end_frame",
    "num_frames_for_tokens",
    "num_tokens_for_frame_len",
    "frame_idx_to_token_idx",
    "token_range_to_frame_slice",
    "token_active_window_left_frame",
    "token_body_window_left_frame",
    "frames_to_token_mask",
    "prefix_len_from_tail_invalid",
]
