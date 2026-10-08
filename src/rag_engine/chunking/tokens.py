"""Whitespace token accounting with O(log n) span counts (see ADR-0005)."""

from __future__ import annotations

import re
from bisect import bisect_left

_TOKEN = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN.finditer(text))


class TokenIndex:
    """Positions of whitespace-delimited tokens in a text.

    A token belongs to a span if the token *starts* inside it, so counts of
    adjacent spans always add up to the count of their union.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.starts: list[int] = []
        self.ends: list[int] = []
        for match in _TOKEN.finditer(text):
            self.starts.append(match.start())
            self.ends.append(match.end())

    def __len__(self) -> int:
        return len(self.starts)

    def first_at_or_after(self, char_offset: int) -> int:
        """Index of the first token starting at or after ``char_offset``."""
        return bisect_left(self.starts, char_offset)

    def count(self, start: int, end: int) -> int:
        """Number of tokens starting in ``[start, end)``."""
        if end <= start:
            return 0
        return bisect_left(self.starts, end) - bisect_left(self.starts, start)
