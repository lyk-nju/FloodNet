"""Runtime text condition state for streaming generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TextSegment:
    """Text active until `commit_end`, or indefinitely when `commit_end` is None."""

    text: str
    commit_start: int = 0
    commit_end: int | None = None


@dataclass(frozen=True)
class TextConditionBundle:
    """Resolved text condition for a single streaming step."""

    text: str
    commit_idx: int


class TextConditionState:
    """Tracks text edits on the commit timeline."""

    def __init__(self, initial_text: str = ""):
        self._segments: list[TextSegment] = [
            TextSegment(text=str(initial_text), commit_start=0, commit_end=None)
        ]

    @property
    def segments(self) -> tuple[TextSegment, ...]:
        return tuple(self._segments)

    def reset(self, text: str = "") -> None:
        self._segments = [
            TextSegment(text=str(text), commit_start=0, commit_end=None)
        ]

    def update_text(self, text: str, commit_idx: int | None = None) -> None:
        commit = 0 if commit_idx is None else int(commit_idx)
        if self._segments:
            last = self._segments[-1]
            if last.commit_start == commit:
                self._segments[-1] = TextSegment(
                    text=str(text),
                    commit_start=commit,
                    commit_end=None,
                )
                return
            self._segments[-1] = TextSegment(
                text=last.text,
                commit_start=last.commit_start,
                commit_end=commit,
            )
        self._segments.append(
            TextSegment(text=str(text), commit_start=commit, commit_end=None)
        )

    def text_at(self, commit_idx: int) -> str:
        commit = int(commit_idx)
        for segment in reversed(self._segments):
            if commit < segment.commit_start:
                continue
            if segment.commit_end is None or commit < segment.commit_end:
                return segment.text
        return self._segments[0].text if self._segments else ""

    def build_bundle(self, commit_idx: int) -> TextConditionBundle:
        commit = int(commit_idx)
        return TextConditionBundle(text=self.text_at(commit), commit_idx=commit)

    @classmethod
    def from_timed_segments(cls, segments: Sequence[TextSegment]) -> "TextConditionState":
        state = cls("")
        if not segments:
            return state
        cleaned = sorted(segments, key=lambda item: int(item.commit_start))
        state._segments = [
            TextSegment(
                text=str(segment.text),
                commit_start=int(segment.commit_start),
                commit_end=None
                if segment.commit_end is None
                else int(segment.commit_end),
            )
            for segment in cleaned
        ]
        return state


__all__ = ["TextConditionBundle", "TextConditionState", "TextSegment"]
