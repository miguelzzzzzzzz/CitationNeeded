from rag_engine.ingestion.normalize import normalize_text


def test_unicode_and_line_endings_are_canonicalized() -> None:
    raw = "\ufb01nancial\u00a0report\r\nline two\rline three"
    assert normalize_text(raw) == "financial report\nline two\nline three"


def test_control_characters_removed_but_tabs_and_newlines_kept() -> None:
    assert normalize_text("a\x00b\x07c\td\ne") == "abc\td\ne"


def test_blank_line_runs_collapse_and_trailing_space_is_stripped() -> None:
    assert normalize_text("  para one   \n\n\n\n\npara two\t \n") == "para one\n\npara two"


def test_internal_space_runs_collapse_but_indentation_is_kept() -> None:
    text = "def f():\n    return  1"
    assert normalize_text(text) == "def f():\n    return 1"


def test_dehyphenation_is_opt_in_and_only_joins_lowercase_continuations() -> None:
    raw = "infor-\nmation about COVID-\n19"
    assert normalize_text(raw) == raw
    assert normalize_text(raw, dehyphenate=True) == "information about COVID-\n19"
