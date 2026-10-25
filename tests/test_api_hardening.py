"""M7 service hardening: locked first model loads, opt-in warm-up, request logs.

Models are stand-ins (a fake ``fastembed`` module, counting factories); no test
downloads a model or opens a network connection.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_engine.api import RagService, ServedIndex
from rag_engine.api.app import REQUEST_ID_HEADER, configure_logging, create_app
from rag_engine.api.schemas import AnswerRequest, SearchRequest
from rag_engine.api.service import WARMUP_PASSAGE, WARMUP_QUERY, GenerationError
from rag_engine.cli import main
from rag_engine.config import EmbeddingConfig, Settings
from rag_engine.retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    FastEmbedEmbedder,
    HashingEmbedder,
    LexicalRetriever,
    load_index,
)
from rag_engine.retrieval.embedders import Matrix, Vector

REQUESTS_LOGGER = "rag_engine.api.requests"
QUERY = "reciprocal rank fusion"


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("hardening")
    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "fusion.md").write_text(
        "# Fusion\n\nHybrid search fuses dense and lexical rankings with reciprocal rank fusion.\n",
        encoding="utf-8",
    )
    (corpus / "cooking.md").write_text(
        "# Cooking\n\nSimmer the tomato sauce slowly and season it with basil.\n", encoding="utf-8"
    )
    out = root / "index"
    args = ["index", str(corpus), "--out", str(out), "--embedder", "hashing"]
    assert main([*args, "--corpus-id", "hardening"]) == 0
    return out


def _settings(index_dir: Path | None = None, **env: str) -> Settings:
    values = {f"RAG_{key.upper()}": value for key, value in env.items()}
    if index_dir is not None:
        values["RAG_INDEX_DIR"] = str(index_dir)
    return Settings.from_env(values)


class CountingReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def name(self) -> str:
        return f"counting:{self.model_name}"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return [float(len(text)) for text in texts]


class SpyEmbedder:
    """Hashing embedder that records the queries it embeds."""

    def __init__(self) -> None:
        self._inner = HashingEmbedder()
        self.queries: list[str] = []

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> Vector:
        self.queries.append(text)
        return self._inner.embed_query(text)


def _spy_index(index_dir: Path) -> tuple[ServedIndex, SpyEmbedder]:
    _, lexical = load_index(index_dir, EmbeddingConfig())
    spy = SpyEmbedder()
    return ServedIndex(DenseRetriever(spy), lexical, documents=2), spy


def _request_records(caplog: pytest.LogCaptureFixture) -> list[tuple[int, dict[str, Any]]]:
    return [
        (record.levelno, json.loads(record.getMessage()))
        for record in caplog.records
        if record.name == REQUESTS_LOGGER
    ]


# --- settings ----------------------------------------------------------------


def test_service_hardening_settings_from_env() -> None:
    default = Settings.from_env({}).service
    assert (default.warmup, default.log_level) == ("off", "INFO")
    custom = _settings(warmup=" All ", log_level="debug").service
    assert (custom.warmup, custom.log_level) == ("all", "DEBUG")
    assert _settings(warmup="embedder").service.warmup == "embedder"


@pytest.mark.parametrize(("key", "value"), [("warmup", "reranker"), ("log_level", "verbose")])
def test_invalid_hardening_settings_are_rejected(key: str, value: str) -> None:
    with pytest.raises(ValueError, match="invalid RAG_"):
        _settings(**{key: value})


# --- locked first loads ------------------------------------------------------


def _fake_fastembed(monkeypatch: pytest.MonkeyPatch, constructed: list[dict[str, Any]]) -> None:
    """Install a fake ``fastembed`` whose model constructors are slow and counted."""

    def slow_model(**kwargs: Any) -> object:
        time.sleep(0.05)  # widen the window in which an unlocked load would race
        constructed.append(kwargs)
        return types.SimpleNamespace(**kwargs)

    package = types.ModuleType("fastembed")
    package.TextEmbedding = slow_model  # type: ignore[attr-defined]
    rerank = types.ModuleType("fastembed.rerank")
    cross_encoder = types.ModuleType("fastembed.rerank.cross_encoder")
    cross_encoder.TextCrossEncoder = slow_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", package)
    monkeypatch.setitem(sys.modules, "fastembed.rerank", rerank)
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", cross_encoder)


def _load_concurrently(load: Any, threads: int = 8) -> list[object]:
    barrier = threading.Barrier(threads)
    results: list[object] = []

    def worker() -> None:
        barrier.wait()
        results.append(load())

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=10)
    return results


def test_concurrent_first_embedder_loads_construct_one_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, Any]] = []
    _fake_fastembed(monkeypatch, constructed)
    embedder = FastEmbedEmbedder(EmbeddingConfig(cache_dir="/tmp/models", threads=2))
    results = _load_concurrently(embedder._load)
    assert len(results) == 8
    assert constructed == [
        {"model_name": "BAAI/bge-small-en-v1.5", "cache_dir": "/tmp/models", "threads": 2}
    ]
    assert all(result is results[0] for result in results)
    assert embedder._load() is results[0]  # later calls reuse it without the lock


def test_concurrent_first_reranker_loads_construct_one_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, Any]] = []
    _fake_fastembed(monkeypatch, constructed)
    reranker = CrossEncoderReranker("org/cross", cache_dir="/tmp/models", threads=1)
    results = _load_concurrently(reranker._load)
    assert len(results) == 8
    assert constructed == [{"model_name": "org/cross", "cache_dir": "/tmp/models", "threads": 1}]
    assert all(result is results[0] for result in results)


def test_loads_without_fastembed_raise_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("fastembed", "fastembed.rerank", "fastembed.rerank.cross_encoder"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"pip install -e '\.\[embeddings\]'"):
        FastEmbedEmbedder()._load()
    with pytest.raises(ImportError, match=r"pip install -e '\.\[embeddings\]'"):
        CrossEncoderReranker()._load()


# --- warm-up -----------------------------------------------------------------


def test_warm_up_is_off_by_default(index_dir: Path) -> None:
    index, spy = _spy_index(index_dir)
    service = RagService(index, _settings(), reranker_factory=CountingReranker)
    assert service.warm_up() == {}
    assert spy.queries == []
    assert service._reranker is None


def test_warm_up_embedder_embeds_one_probe_query(index_dir: Path) -> None:
    index, spy = _spy_index(index_dir)
    made: list[CountingReranker] = []

    def factory(name: str) -> CountingReranker:
        made.append(CountingReranker(name))
        return made[-1]

    service = RagService(index, _settings(warmup="embedder"), reranker_factory=factory)
    timings = service.warm_up()
    assert set(timings) == {"embedder"} and timings["embedder"] >= 0
    assert spy.queries == [WARMUP_QUERY]
    assert made == []  # the reranker is only warmed with "all"


def test_warm_up_all_loads_the_shared_reranker(
    index_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    index, spy = _spy_index(index_dir)
    made: list[CountingReranker] = []

    def factory(name: str) -> CountingReranker:
        made.append(CountingReranker(name))
        return made[-1]

    service = RagService(index, _settings(warmup="all"), reranker_factory=factory)
    with caplog.at_level(logging.INFO, logger="rag_engine.api"):
        assert set(service.warm_up()) == {"embedder", "reranker"}
    assert spy.queries == [WARMUP_QUERY]
    assert len(made) == 1 and made[0].calls == [(WARMUP_QUERY, [WARMUP_PASSAGE])]
    assert "warm-up: reranker ready" in caplog.text
    # The warmed instance is the one requests use.
    assert service._shared_reranker() is made[0]


def test_warm_up_without_an_index_loads_nothing() -> None:
    service = RagService(None, _settings(warmup="all"), reranker_factory=CountingReranker)
    assert service.warm_up() == {}
    assert service._reranker is None


def test_failed_warm_up_is_logged_and_retried_on_first_use(
    index_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    attempts: list[str] = []

    def flaky(name: str) -> CountingReranker:
        attempts.append(name)
        if len(attempts) == 1:
            raise ImportError("fastembed is not installed")
        return CountingReranker(name)

    service = RagService.from_settings(_settings(index_dir, warmup="all"), reranker_factory=flaky)
    with caplog.at_level(logging.WARNING, logger="rag_engine.api"):
        assert set(service.warm_up()) == {"embedder"}
    assert "warm-up of the reranker failed" in caplog.text
    assert "fastembed is not installed" in caplog.text
    with TestClient(create_app(service)) as client:  # startup warms again and succeeds
        response = client.post("/search", json={"query": QUERY, "k": 1, "rerank": True})
    assert response.status_code == 200, response.text
    assert len(attempts) == 2


def test_app_startup_warms_up_before_serving(index_dir: Path) -> None:
    made: list[CountingReranker] = []

    def factory(name: str) -> CountingReranker:
        made.append(CountingReranker(name))
        return made[-1]

    service = RagService.from_settings(_settings(index_dir, warmup="all"), reranker_factory=factory)
    with TestClient(create_app(service)) as client:
        assert len(made) == 1 and made[0].calls == [(WARMUP_QUERY, [WARMUP_PASSAGE])]
        assert client.post("/search", json={"query": QUERY, "rerank": True}).status_code == 200
    assert len(made) == 1  # the request reused the warmed reranker


# --- request logging -----------------------------------------------------------


@pytest.fixture()
def client(index_dir: Path) -> Iterator[TestClient]:
    service = RagService.from_settings(_settings(index_dir))
    with TestClient(create_app(service), raise_server_exceptions=False) as test_client:
        yield test_client


def test_search_logs_one_structured_line_with_retrieval_latency(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=REQUESTS_LOGGER):
        response = client.post("/search", json={"query": QUERY, "k": 1})
    assert response.status_code == 200
    [(level, record)] = _request_records(caplog)
    assert level == logging.INFO
    assert record["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert len(record["request_id"]) == 32
    assert {key: record[key] for key in ("event", "method", "path", "status")} == {
        "event": "request",
        "method": "POST",
        "path": "/search",
        "status": 200,
    }
    assert 0 <= record["retrieve_ms"] <= record["duration_ms"]
    assert "generate_ms" not in record
    assert QUERY not in caplog.text  # query text is never logged


def test_answer_logs_retrieval_and_generation_latency(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=REQUESTS_LOGGER):
        response = client.post("/answer", json={"query": QUERY, "generator": "extractive"})
    assert response.status_code == 200
    [(_, record)] = _request_records(caplog)
    assert record["path"] == "/answer" and record["status"] == 200
    assert record["retrieve_ms"] >= 0 and record["generate_ms"] >= 0
    assert record["retrieve_ms"] + record["generate_ms"] <= record["duration_ms"]


def test_request_ids_are_unique(client: TestClient) -> None:
    ids = {client.get("/health").headers[REQUEST_ID_HEADER] for _ in range(5)}
    assert len(ids) == 5


def test_health_requests_log_at_debug_only(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=REQUESTS_LOGGER):
        client.get("/health")
    assert _request_records(caplog) == []
    with caplog.at_level(logging.DEBUG, logger=REQUESTS_LOGGER):
        client.get("/health")
    [(level, record)] = _request_records(caplog)
    assert (level, record["path"], record["status"]) == (logging.DEBUG, "/health", 200)


def test_error_responses_are_logged_with_their_status(caplog: pytest.LogCaptureFixture) -> None:
    unavailable_app = create_app(RagService.from_settings(_settings()))
    with (
        TestClient(unavailable_app) as unavailable,
        caplog.at_level(logging.INFO, logger=REQUESTS_LOGGER),
    ):
        assert unavailable.post("/search", json={"query": QUERY}).status_code == 503
        assert unavailable.post("/search", json={"query": QUERY, "k": 0}).status_code == 422
    statuses = [
        (record["status"], "retrieve_ms" in record) for _, record in _request_records(caplog)
    ]
    # Neither request reached retrieval, so neither has a retrieval timing.
    assert statuses == [(503, False), (422, False)]


def test_generation_failure_still_records_both_stages(
    index_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class Failing:
        name = "failing"

        def generate(self, query: str, hits: Sequence[Any], *, max_sentences: int = 5) -> Any:
            raise RuntimeError("endpoint unreachable")

    service = RagService.from_settings(
        _settings(index_dir), generator_factory=lambda choice: Failing()
    )
    timings: dict[str, float] = {}
    with TestClient(create_app(service)) as client:
        with caplog.at_level(logging.INFO, logger=REQUESTS_LOGGER):
            assert client.post("/answer", json={"query": QUERY}).status_code == 502
        with pytest.raises(GenerationError, match="answer generation failed"):
            service.answer(AnswerRequest(query=QUERY), timings=timings)
    [(_, record)] = _request_records(caplog)
    assert record["status"] == 502 and "retrieve_ms" in record and "generate_ms" in record
    assert set(timings) == {"retrieve_ms", "generate_ms"}


def test_unhandled_errors_are_logged_as_500(
    client: TestClient, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: LexicalRetriever, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("corrupt posting list")

    monkeypatch.setattr(LexicalRetriever, "retrieve", broken)
    with caplog.at_level(logging.INFO, logger=REQUESTS_LOGGER):
        response = client.post("/search", json={"query": QUERY, "mode": "lexical"})
    assert response.status_code == 500
    [(_, record)] = _request_records(caplog)
    assert record["status"] == 500 and "retrieve_ms" in record


# --- logging setup -------------------------------------------------------------


@pytest.fixture()
def package_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger("rag_engine")
    level, handlers = logger.level, list(logger.handlers)
    yield logger
    logger.setLevel(level)
    logger.handlers[:] = handlers


def test_configure_logging_adds_one_handler_when_nothing_logs(
    package_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    package_logger.handlers[:] = []
    configure_logging("WARNING")
    configure_logging("DEBUG")  # idempotent: still one handler
    assert package_logger.level == logging.DEBUG
    assert len(package_logger.handlers) == 1
    assert isinstance(package_logger.handlers[0], logging.StreamHandler)


def test_configure_logging_defers_to_existing_handlers(package_logger: logging.Logger) -> None:
    package_logger.handlers[:] = []
    assert logging.getLogger().handlers  # pytest's capture handler
    configure_logging("ERROR")
    assert package_logger.level == logging.ERROR
    assert package_logger.handlers == []


def test_app_from_environment_applies_log_level_and_warm_up(
    index_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    package_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RAG_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("RAG_LOG_LEVEL", "info")
    monkeypatch.setenv("RAG_WARMUP", "embedder")
    with caplog.at_level(logging.INFO, logger="rag_engine.api"), TestClient(create_app()) as client:
        assert package_logger.level == logging.INFO
        assert client.get("/health").status_code == 200
    assert "warm-up: embedder ready" in caplog.text


def test_service_calls_without_a_timings_dict_record_nothing(index_dir: Path) -> None:
    service = RagService.from_settings(_settings(index_dir))
    assert service.search(SearchRequest(query=QUERY, k=1)).hits
    timings: dict[str, float] = {}
    assert service.search(SearchRequest(query=QUERY, k=1), timings=timings).hits
    assert set(timings) == {"retrieve_ms"}
