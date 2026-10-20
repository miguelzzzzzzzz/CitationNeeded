"""Unit tests for the pieces the CLI and the HTTP service share.

``compose_retriever`` and ``select_generator`` back both ``rag-engine
search``/``answer`` and ``POST /search``/``/answer``; ``ServiceConfig`` is read
from ``RAG_*`` like every other setting. Nothing here loads a model or opens a
connection: the OpenAI-compatible client is only constructed.
"""

from __future__ import annotations

import pytest

from rag_engine.answering import (
    ExtractiveAnswerGenerator,
    OpenAICompatibleAnswerGenerator,
    select_generator,
)
from rag_engine.api import RagService
from rag_engine.api.schemas import SearchRequest
from rag_engine.config import DEFAULT_RERANK_MODEL, LLMConfig, Settings
from rag_engine.retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    LexicalRetriever,
    RerankingRetriever,
    compose_retriever,
)


class _NoopReranker:
    name = "noop"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)


def _pair() -> tuple[DenseRetriever, LexicalRetriever]:
    return DenseRetriever(HashingEmbedder()), LexicalRetriever()


def test_compose_returns_each_mode() -> None:
    dense, lexical = _pair()
    assert compose_retriever(dense, lexical, mode="dense") is dense
    assert compose_retriever(dense, lexical, mode="lexical") is lexical
    hybrid = compose_retriever(dense, lexical, mode="hybrid", fusion="weighted")
    assert isinstance(hybrid, HybridRetriever)
    reranked = compose_retriever(dense, lexical, mode="hybrid", reranker=_NoopReranker())
    assert isinstance(reranked, RerankingRetriever)
    assert reranked.name == "hybrid+rerank"
    assert isinstance(reranked.base, HybridRetriever)


def test_compose_rejects_unknown_mode_and_bad_options() -> None:
    dense, lexical = _pair()
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        compose_retriever(dense, lexical, mode="semantic")
    with pytest.raises(ValueError, match="rrf_k"):
        compose_retriever(dense, lexical, mode="hybrid", rrf_k=-1)
    with pytest.raises(ValueError, match="candidates"):
        compose_retriever(dense, lexical, mode="dense", reranker=_NoopReranker(), candidates=0)


def test_select_generator_auto_depends_on_llm_settings() -> None:
    assert isinstance(select_generator("auto", LLMConfig()), ExtractiveAnswerGenerator)
    configured = LLMConfig(base_url="http://127.0.0.1:9/v1", model="tiny", max_tokens=64)
    generator = select_generator("auto", configured)
    assert isinstance(generator, OpenAICompatibleAnswerGenerator)
    assert generator.name == select_generator("openai", configured).name
    assert isinstance(select_generator("extractive", configured), ExtractiveAnswerGenerator)


def test_select_generator_errors() -> None:
    with pytest.raises(ValueError, match="unknown generator"):
        select_generator("gpt", LLMConfig())
    with pytest.raises(ValueError, match="RAG_LLM_BASE_URL and RAG_LLM_MODEL"):
        select_generator("openai", LLMConfig(base_url="http://127.0.0.1:9/v1"))


def test_service_settings_from_env() -> None:
    defaults = Settings.from_env({})
    assert defaults.service.index_dir is None
    assert defaults.service.rerank_model == DEFAULT_RERANK_MODEL
    settings = Settings.from_env(
        {"RAG_INDEX_DIR": " /data/index ", "RAG_RERANK_MODEL": "org/other-cross-encoder"}
    )
    assert settings.service.index_dir == "/data/index"
    assert settings.service.rerank_model == "org/other-cross-encoder"
    assert Settings.from_env({"RAG_RERANK_MODEL": "  "}).service.rerank_model == (
        DEFAULT_RERANK_MODEL
    )


def test_default_reranker_uses_the_configured_model_lazily() -> None:
    settings = Settings.from_env(
        {
            "RAG_RERANK_MODEL": "org/other-cross-encoder",
            "RAG_EMBEDDING_CACHE_DIR": "/tmp/models",
            "RAG_EMBEDDING_THREADS": "2",
        }
    )
    service = RagService(None, settings)
    reranker = service._shared_reranker()
    assert isinstance(reranker, CrossEncoderReranker)
    assert (reranker.model_name, reranker.cache_dir, reranker.threads) == (
        "org/other-cross-encoder",
        "/tmp/models",
        2,
    )
    assert service._shared_reranker() is reranker
    # Constructing it loads nothing; scoring an empty list never triggers the load.
    assert reranker.score("query", []) == []


def test_explicit_null_filters_mean_no_filters() -> None:
    assert SearchRequest.model_validate({"query": "fusion", "filters": None}).filters is None
