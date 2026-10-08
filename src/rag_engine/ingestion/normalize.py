"""Text normalization applied to every loaded document."""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")
# Collapse runs of spaces *inside* a line; leading indentation (code) is kept.
_INLINE_WS = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")
# "infor-\nmation" -> "information" (only lowercase continuation, to keep
# real hyphenated compounds such as "COVID-\n19" intact).
_LINEBREAK_HYPHEN = re.compile(r"(\w)-\n([a-z])")


def normalize_text(text: str, *, dehyphenate: bool = False) -> str:
    """Return canonical text suitable for chunking and indexing.

    * Unicode NFKC (ligatures, full-width forms, non-breaking spaces).
    * ``\\r\\n``/``\\r`` -> ``\\n``; control characters removed (tabs kept).
    * Internal runs of spaces collapsed (indentation kept); trailing
      whitespace stripped; runs of blank lines collapsed to one.
    * Optional de-hyphenation of words split across lines (PDF output).
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    if dehyphenate:
        text = _LINEBREAK_HYPHEN.sub(r"\1\2", text)
    text = _INLINE_WS.sub(" ", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
