"""Pure runtime-glue helpers for the streaming dual-anchor commit path (T_C_02).

These are the correctness-critical, off-by-error-prone index computations the
ModelManager / stream_benchmark integration needs. Extracted here so they are
unit-tested independently of the (model+VAE-dependent) live streaming loop.

The full ModelManager rewrite (Refiner anchor → RootPlan → body → un-canonicalize
→ commit + timeline advance) requires the trained Refiner + body model and is
verified end-to-end on the runtime box; these helpers are its tested building
blocks.
"""

from __future__ import annotations

from utils.token_frame import token_range_to_frame_slice


def body_window_start_commit_idx(head_commit_idx: int,
                                 body_history_tokens: int,
                                 explicit_start: int | None = None) -> int:
    """Single source for the body window start commit_idx (= history0 anchor).

    If the body-window builder provides an explicit start, use it; otherwise
    `max(0, head_commit_idx - body_history_tokens)`. Used by _step_body_window,
    get_debug_panel and _commit so the body anchor is computed consistently.
    """
    if explicit_start is not None:
        return int(explicit_start)
    return max(0, int(head_commit_idx) - int(body_history_tokens))


def committed_frame_slice(head_commit_idx: int,
                          body_anchor_commit_idx: int,
                          committed_tokens: int,
                          frames_per_token: int = 4) -> slice:
    """Frame slice of the committed region WITHIN a body output that starts at
    the body-window anchor (history0).

    The body output is body-window-local: frame 0 == history0 ==
    body_anchor_commit_idx. So the committed region must be sliced with the
    body-window-local token offset:

        relative_start_token = head_commit_idx - body_anchor_commit_idx

    NOT the global commit_idx (that would put frame 0 somewhere other than
    history0). Done-criterion #5 example: body_anchor=20, head=35, committed=5
    → relative_start_token=15 → frames [token_start_frame(15), ...).
    """
    relative_start_token = int(head_commit_idx) - int(body_anchor_commit_idx)
    if relative_start_token < 0:
        raise ValueError(
            f"head_commit_idx ({head_commit_idx}) < body_anchor_commit_idx "
            f"({body_anchor_commit_idx}); head must be at/after the body window start"
        )
    if committed_tokens <= 0:
        raise ValueError(f"committed_tokens must be > 0, got {committed_tokens}")
    return token_range_to_frame_slice(relative_start_token, committed_tokens, frames_per_token)


__all__ = ["body_window_start_commit_idx", "committed_frame_slice"]
