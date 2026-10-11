"""CLI tests for ``rag-engine answer``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_engine.answering import AnswerResponse, OpenAICompatibleAnswerGenerator
from rag_engine.cli import main
from rag_engine.config import Settings

DOC_TEXT = (
    "# Retrieval\n\n"
    "Retrieval augmented generation pairs a retriever with a generator.\n"
    "Dense retrieval embeds queries and documents into a shared vector space.\n"
)

LLM_ENV_VARS = ("RAG_LLM_BASE_URL", "RAG_LLM_MODEL", "RAG_LLM_API_KEY")


def _write_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "doc.md"
    doc.write_text(DOC_TEXT, encoding="utf-8")
    return doc


def _build_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    index_dir = tmp_path / "index"
    exit_code = main(
        [
            "index",
            str(_write_doc(tmp_path)),
            "--out",
            str(index_dir),
            "--embedder",
            "hashing",
            "--corpus-id",
            "demo",
        ]
    )
    assert exit_code == 0
    capsys.readouterr()
    return index_dir


def _answer_args(index_dir: Path, query: str) -> list[str]:
    return ["answer", str(index_dir), query, "--mode", "dense", "-k", "3"]


def test_answer_extractive_prints_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = _build_index(tmp_path, capsys)
    exit_code = main(
        [*_answer_args(index_dir, "dense retrieval vector space"), "--generator", "extractive"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "retrieval" in out.lower() or "vector" in out.lower()


def test_answer_extractive_json_payload(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index_dir = _build_index(tmp_path, capsys)
    exit_code = main(
        [
            *_answer_args(index_dir, "dense retrieval vector space"),
            "--generator",
            "extractive",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    payload = json.loads(out)
    assert payload["generator"] == "extractive"
    assert payload["schema_version"] == 1
    assert payload["answer"].strip()


def test_answer_missing_index_dir_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does-not-exist"
    exit_code = main(
        ["answer", str(missing), "anything", "--generator", "extractive", "--mode", "dense"]
    )
    capsys.readouterr()
    assert exit_code == 2


def test_answer_openai_without_env_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    index_dir = _build_index(tmp_path, capsys)
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    exit_code = main([*_answer_args(index_dir, "dense retrieval"), "--generator", "openai"])
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "RAG_LLM" in err


def test_answer_openai_generator_uses_llm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    index_dir = _build_index(tmp_path, capsys)
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("RAG_LLM_MODEL", "unit-test-model")
    monkeypatch.delenv("RAG_LLM_API_KEY", raising=False)

    fixed = AnswerResponse(
        query="q",
        answer="Stubbed LLM answer.",
        citations=(),
        generator="openai",
        model_name="unit-test-model",
        abstained=False,
    )

    def fake_generate(
        self: OpenAICompatibleAnswerGenerator,
        query: str,
        hits: object,
        *,
        max_sentences: int = 5,
    ) -> AnswerResponse:
        return fixed

    monkeypatch.setattr(OpenAICompatibleAnswerGenerator, "generate", fake_generate)
    exit_code = main(
        [*_answer_args(index_dir, "retrieval augmented generation"), "--generator", "openai"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Stubbed LLM answer." in out


def test_answer_auto_picks_openai_when_configured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    index_dir = _build_index(tmp_path, capsys)
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("RAG_LLM_MODEL", "unit-test-model")
    monkeypatch.setenv("RAG_LLM_API_KEY", "sk-test")

    fixed = AnswerResponse(
        query="q",
        answer="Auto picked the LLM.",
        citations=(),
        generator="openai",
        model_name="unit-test-model",
        abstained=False,
    )

    def fake_generate(
        self: OpenAICompatibleAnswerGenerator,
        query: str,
        hits: object,
        *,
        max_sentences: int = 5,
    ) -> AnswerResponse:
        return fixed

    monkeypatch.setattr(OpenAICompatibleAnswerGenerator, "generate", fake_generate)
    exit_code = main(
        [*_answer_args(index_dir, "retrieval augmented generation"), "--generator", "auto"]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Auto picked the LLM." in out


def test_llm_env_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("RAG_LLM_MODEL", "example-model")
    monkeypatch.setenv("RAG_LLM_API_KEY", "sk-test")
    settings = Settings.from_env()
    assert settings.llm.base_url == "https://llm.example.test/v1"
    assert settings.llm.model == "example-model"
    assert settings.llm.api_key == "sk-test"


def test_answer_rejects_bad_k_and_max_sentences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = _build_index(tmp_path, capsys)
    assert main([*_answer_args(index_dir, "q"), "--generator", "extractive", "-k", "0"]) == 2
    assert (
        main(
            [
                *_answer_args(index_dir, "q"),
                "--generator",
                "extractive",
                "--max-sentences",
                "0",
            ]
        )
        == 2
    )
    capsys.readouterr()


def test_answer_abstained_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index_dir = _build_index(tmp_path, capsys)
    exit_code = main(
        [
            *_answer_args(index_dir, "quantum entanglement unicorn"),
            "--generator",
            "extractive",
        ]
    )
    err = capsys.readouterr().err
    assert exit_code == 0
    assert "abstained" in err.lower()


def test_answer_prints_citations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index_dir = _build_index(tmp_path, capsys)
    exit_code = main(
        [
            *_answer_args(index_dir, "dense retrieval vector space"),
            "--generator",
            "extractive",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "[1]" in out
