import pytest
from pydantic import ValidationError

from rag_engine.models import Document, Section


def test_overlapping_sections_are_rejected() -> None:
    with pytest.raises(ValidationError, match="non-overlapping"):
        Document.create(
            source="x",
            title="x",
            format="text",
            text="abcdef",
            sections=(Section(start_char=0, end_char=4), Section(start_char=3, end_char=6)),
        )


def test_section_past_end_of_text_is_rejected() -> None:
    with pytest.raises(ValidationError, match="past end"):
        Document.create(
            source="x",
            title="x",
            format="text",
            text="abc",
            sections=(Section(start_char=0, end_char=10),),
        )


def test_section_at_finds_containing_section() -> None:
    doc = Document.create(
        source="x",
        title="x",
        format="markdown",
        text="aaaa bbbb",
        sections=(
            Section(start_char=0, end_char=5, heading_path=("A",)),
            Section(start_char=5, end_char=9, heading_path=("B",)),
        ),
    )
    assert doc.section_at(0) is not None and doc.section_at(0).heading_path == ("A",)  # type: ignore[union-attr]
    assert doc.section_at(6).heading_path == ("B",)  # type: ignore[union-attr]
    assert doc.section_at(99) is None
