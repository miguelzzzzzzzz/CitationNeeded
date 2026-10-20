"""HTTP API tests (``rag_engine.api``) over a real hashing-embedder index.

The index is built from a small Markdown corpus through the same ingest ->
chunk -> index -> save path as the CLI, so every hit and citation checked here
comes from real retrieval. Rerankers and LLM generators are injected fakes; no
test needs a model download or a network call.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_engine import __version__
from rag_engine.answering import AnswerResponse, ExtractiveAnswerGenerator
from rag_engine.api import (
    DependencyMissingError,
    IndexNotLoadedError,
    InvalidRequestError,
    RagService,
)
from rag_engine.api.app import create_app
from rag_engine.api.schemas import SearchRequest
from rag_engine.cli import main
from rag_engine.config import Settings
from rag_engine.retrieval import HashingEmbedder, LexicalRetriever, RetrievedChunk
from rag_engine.retrieval.index_store import load_documents

DOCS = {
    "retrieval.md": (
        "# Retrieval\n\n"
        "Dense retrieval embeds queries and passages into one vector space.\n"
        "Lexical retrieval scores passages with BM25 term statistics.\n\n"
        "## Fusion\n\n"
        "Hybrid search fuses the dense and lexical rankings with reciprocal rank fusion.\n"
    ),
    "reranking.md": (
        "# Reranking\n\n"
        "A cross-encoder reranker reads the query and each candidate passage together.\n"
        "Reranking trades extra latency for better ordering of the top candidates.\n"
    ),
    "cooking.md": (
        "# Cooking\n\nSimmer the tomato sauce slowly and season it with basil before serving.\n"
    ),
}


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("api")
    corpus = root / "corpus"
    corpus.mkdir()
    for name, text in DOCS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    out = root / "index"
    code = main(
        [
            "index",
            str(corpus),
            "--out",
            str(out),
            "--embedder",
            "hashing",
            "--corpus-id",
            "api-test",
            "--strategy",
            "structure",
        ]
    )
    assert code == 0
    return out


def _settings(index_dir: Path | None, **llm: str) -> Settings:
    env = {} if index_dir is None else {"RAG_INDEX_DIR": str(index_dir)}
    env.update({f"RAG_LLM_{key.upper()}": value for key, value in llm.items()})
    return Settings.from_env(env)


@pytest.fixture()
def client(index_dir: Path) -> Iterator[TestClient]:
    service = RagService.from_settings(_settings(index_dir))
    with TestClient(create_app(service)) as test_client:
        yield test_client


def _search(client: TestClient, **body: Any) -> Any:
    response = client.post("/search", json=body)
    assert response.status_code == 200, response.text
    return response.json()


class OverlapReranker:
    """Scores by word overlap with the query, so a reranked order is predictable."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls = 0

    @property
    def name(self) -> str:
        return f"fake:{self.model_name}"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls += 1
        terms = set(query.lower().split())
        return [float(len(terms & set(text.lower().split()))) for text in texts]


class RecordingGenerator:
    """Records what the service hands to a generator and returns a fixed answer."""

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[str, list[RetrievedChunk], int]] = []

    @property
    def name(self) -> str:
        return "recording"

    def generate(
        self, query: str, hits: Sequence[RetrievedChunk], *, max_sentences: int = 5
    ) -> AnswerResponse:
        self.calls.append((query, list(hits), max_sentences))
        if self.fail is not None:
            raise self.fail
        return AnswerResponse(
            query=query, answer="stub answer", generator=self.name, model_name="stub-model"
        )


# --- /health -----------------------------------------------------------------


def test_health_reports_the_served_index(client: TestClient, index_dir: Path) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    manifest = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
    assert body == {
        "status": "ok",
        "version": __version__,
        "index_loaded": True,
        "embedder": HashingEmbedder().name,
        "chunks": manifest["chunks"],
        "documents": 3,
        "detail": None,
    }


def test_health_is_503_without_a_configured_index() -> None:
    with TestClient(create_app(RagService.from_settings(_settings(None)))) as client:
        response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["index_loaded"] is False
    assert "RAG_INDEX_DIR" in body["detail"]


def test_index_dir_without_manifest_starts_unavailable(tmp_path: Path) -> None:
    service = RagService.from_settings(_settings(tmp_path / "not-built-yet"))
    assert service.index is None
    health = service.health()
    assert health.status == "unavailable"
    assert "rag-engine index" in (health.detail or "")


def test_corrupt_index_fails_startup(tmp_path: Path, index_dir: Path) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "index.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        RagService.from_settings(_settings(broken))


def test_corrupt_index_from_the_environment_aborts_app_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "index.json").write_text('{"format_version": 1}', encoding="utf-8")
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="no longer supported"), TestClient(create_app()):
        pass


def test_app_without_a_service_loads_from_the_environment(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_INDEX_DIR", str(index_dir))
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["index_loaded"] is True
        assert _search(client, query="reciprocal rank fusion", k=1)["hits"]


def test_no_index_makes_search_and_answer_503() -> None:
    with TestClient(create_app(RagService.from_settings(_settings(None)))) as client:
        search = client.post("/search", json={"query": "fusion"})
        answer = client.post("/answer", json={"query": "fusion"})
    assert search.status_code == answer.status_code == 503
    assert "RAG_INDEX_DIR" in search.json()["detail"]


# --- /search -----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_search_matches_the_cli_json_output(
    mode: str, client: TestClient, index_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    query = "reciprocal rank fusion of dense and lexical rankings"
    body = _search(client, query=query, k=3, mode=mode)

    assert main(["search", str(index_dir), query, "--mode", mode, "-k", "3", "--json"]) == 0
    cli = json.loads(capsys.readouterr().out)

    assert body["schema_version"] == cli["schema_version"] == 1
    assert (body["query"], body["mode"], body["rerank"]) == (query, mode, False)
    assert len(body["hits"]) == len(cli["hits"]) > 0
    for api_hit, cli_hit in zip(body["hits"], cli["hits"], strict=True):
        # The API always includes components/ranks; the CLI omits them when empty.
        assert {**{"components": {}, "ranks": {}}, **cli_hit} == api_hit
    if mode == "hybrid":
        assert set(body["hits"][0]["components"]) == {"dense", "lexical"}


def test_search_hits_carry_verifiable_offsets(client: TestClient, index_dir: Path) -> None:
    documents = {doc.doc_id: doc for doc in load_documents(index_dir)}
    hits = _search(client, query="cross-encoder reranker latency", k=5, mode="lexical")["hits"]
    assert hits[0]["source"] == "reranking.md"
    for hit in hits:
        document = documents[hit["doc_id"]]
        assert document.text[hit["start_char"] : hit["end_char"]] == hit["text"]
        assert hit["content_hash"] == document.content_hash


def test_search_filters_apply_before_ranking(client: TestClient) -> None:
    query = "candidate passages and lexical retrieval"
    unfiltered = _search(client, query=query, k=10, mode="lexical")["hits"]
    assert {hit["source"] for hit in unfiltered} >= {"retrieval.md", "reranking.md"}

    only = _search(client, query=query, k=10, mode="lexical", filters={"source": "reranking.md"})[
        "hits"
    ]
    assert only and {hit["source"] for hit in only} == {"reranking.md"}

    any_of = _search(
        client,
        query=query,
        k=10,
        mode="hybrid",
        filters={"source": ["reranking.md", "cooking.md"]},
    )["hits"]
    assert any_of and {hit["source"] for hit in any_of} <= {"reranking.md", "cooking.md"}


def test_search_with_no_match_returns_no_hits(client: TestClient) -> None:
    body = _search(client, query="passages", mode="lexical", filters={"source": "missing.md"})
    assert body["hits"] == []


def test_weighted_fusion_options_reach_the_retriever(client: TestClient) -> None:
    lexical_only = _search(
        client,
        query="BM25 term statistics",
        k=3,
        fusion="weighted",
        weights={"dense": 0.0, "lexical": 1.0},
    )["hits"]
    lexical = _search(client, query="BM25 term statistics", k=3, mode="lexical")["hits"]
    # With a zero dense weight the lexical hits lead; dense-only chunks score 0.
    assert lexical
    head, tail = lexical_only[: len(lexical)], lexical_only[len(lexical) :]
    assert [hit["chunk_id"] for hit in head] == [hit["chunk_id"] for hit in lexical]
    assert all(hit["score"] == 0.0 for hit in tail)
    assert all(hit["retriever"] == "hybrid-weighted" for hit in lexical_only)


def test_rerank_uses_one_configured_reranker(index_dir: Path) -> None:
    created: list[OverlapReranker] = []

    def factory(model_name: str) -> OverlapReranker:
        created.append(OverlapReranker(model_name))
        return created[-1]

    settings = Settings.from_env(
        {"RAG_INDEX_DIR": str(index_dir), "RAG_RERANK_MODEL": "org/tiny-cross-encoder"}
    )
    service = RagService.from_settings(settings, reranker_factory=factory)
    with TestClient(create_app(service)) as client:
        query = "tomato sauce basil"
        first = _search(client, query=query, k=2, mode="dense", rerank=True, candidates=10)
        second = _search(client, query=query, k=2, mode="hybrid", rerank=True)

    assert len(created) == 1
    assert created[0].model_name == "org/tiny-cross-encoder"
    assert created[0].calls == 2
    assert first["rerank"] is True
    assert first["hits"][0]["source"] == "cooking.md"
    assert first["hits"][0]["retriever"] == "dense+rerank"
    assert set(first["hits"][0]["ranks"]) == {"dense"}
    assert second["hits"][0]["retriever"] == "hybrid+rerank"
    assert set(second["hits"][0]["components"]) == {"dense", "lexical", "hybrid"}


def test_missing_reranker_dependency_is_503(index_dir: Path) -> None:
    class Unavailable:
        name = "unavailable"

        def score(self, query: str, texts: Sequence[str]) -> list[float]:
            raise ImportError("fastembed is not installed; run: pip install -e '.[embeddings]'")

    service = RagService.from_settings(
        _settings(index_dir), reranker_factory=lambda _name: Unavailable()
    )
    with TestClient(create_app(service)) as client:
        response = client.post("/search", json={"query": "fusion", "rerank": True})
    assert response.status_code == 503
    assert "fastembed is not installed" in response.json()["detail"]


def test_missing_embedder_dependency_is_503_but_lexical_still_works(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_fastembed(self: HashingEmbedder, text: str) -> Any:
        raise ImportError("fastembed is not installed")

    monkeypatch.setattr(HashingEmbedder, "embed_query", no_fastembed)
    response = client.post("/search", json={"query": "fusion", "mode": "dense"})
    assert response.status_code == 503
    assert _search(client, query="fusion", mode="lexical")["hits"]


def test_errors_during_retrieval_are_500(index_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: LexicalRetriever, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("corrupt posting list")

    monkeypatch.setattr(LexicalRetriever, "retrieve", broken)
    service = RagService.from_settings(_settings(index_dir))
    with TestClient(create_app(service), raise_server_exceptions=False) as client:
        response = client.post("/search", json={"query": "fusion", "mode": "lexical"})
    # A ValueError raised while retrieving is a server fault, not a bad request.
    assert response.status_code == 500
    assert "corrupt posting list" not in response.text


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({}, "query"),
        ({"query": "   "}, "non-whitespace"),
        ({"query": "x" * 4097}, "query"),
        ({"query": "fusion", "k": 0}, "k"),
        ({"query": "fusion", "k": 101}, "k"),
        ({"query": "fusion", "unknown": 1}, "unknown"),
        ({"query": "fusion", "mode": "semantic"}, "mode"),
        ({"query": "fusion", "mode": "dense", "fusion": "rrf"}, "require mode 'hybrid'"),
        ({"query": "fusion", "mode": "lexical", "rrf_k": 10}, "require mode 'hybrid'"),
        ({"query": "fusion", "fusion": "weighted", "rrf_k": 10}, "rrf_k requires fusion"),
        ({"query": "fusion", "mode": "dense", "candidates": 5}, "candidates requires"),
        ({"query": "fusion", "candidates": 0}, "candidates"),
        ({"query": "fusion", "weights": {"dense": -1}}, "finite and >= 0"),
        ({"query": "fusion", "weights": {"dense": 0, "lexical": 0}}, "at least one weight"),
        ({"query": "fusion", "weights": {}}, "weights must name"),
        ({"query": "fusion", "weights": {"sparse": 1}}, "weights"),
        ({"query": "fusion", "filters": {"source": []}}, "must not be empty"),
        ({"query": "fusion", "filters": {"": "a.md"}}, "non-empty"),
        ({"query": "fusion", "filters": {"page": [1] * 101}}, "at most 100"),
        ({"query": "fusion", "filters": {f"k{i}": "v" for i in range(17)}}, "at most 16 items"),
        ({"query": "fusion", "filters": {"source": {"nested": 1}}}, "filters"),
    ],
)
def test_invalid_search_requests_are_422(
    client: TestClient, body: dict[str, Any], fragment: str
) -> None:
    response = client.post("/search", json=body)
    assert response.status_code == 422
    assert fragment in response.text


def test_non_finite_numbers_are_rejected(client: TestClient) -> None:
    # Python's JSON parser accepts NaN/Infinity literals; the schema must not.
    for raw in (
        '{"query": "fusion", "weights": {"dense": NaN}}',
        '{"query": "fusion", "filters": {"score": Infinity}}',
    ):
        response = client.post("/search", content=raw, headers={"Content-Type": "application/json"})
        assert response.status_code == 422, raw
        # The error body must stay strict JSON: rejected input is not echoed back.
        assert "NaN" not in response.text
        assert "Infinity" not in response.text


def test_compose_failure_is_reported_as_invalid_request(index_dir: Path) -> None:
    service = RagService.from_settings(_settings(index_dir))
    # model_construct skips validation, standing in for a rule the schema missed.
    request = SearchRequest.model_construct(
        query="fusion", k=3, mode="hybrid", fusion=None, rrf_k=-1, weights=None,
        candidates=None, rerank=False, filters=None,
    )  # fmt: skip
    with pytest.raises(InvalidRequestError, match="rrf_k"):
        service.search(request)


def test_service_errors_without_http(index_dir: Path) -> None:
    empty = RagService(None, Settings(), unavailable_reason="maintenance")
    with pytest.raises(IndexNotLoadedError, match="maintenance"):
        empty.search(SearchRequest(query="fusion"))
    assert empty.health().detail == "maintenance"
    assert RagService(None, Settings()).health().detail == "no index loaded"
    assert issubclass(DependencyMissingError, RuntimeError)


# --- /answer -----------------------------------------------------------------


def test_extractive_answer_cites_spans_of_the_documents(
    client: TestClient, index_dir: Path
) -> None:
    response = client.post(
        "/answer",
        json={"query": "How does a cross-encoder reranker work?", "generator": "extractive"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["generator"] == ExtractiveAnswerGenerator().name
    assert body["abstained"] is False
    assert body["citations"]
    documents = {doc.doc_id: doc for doc in load_documents(index_dir)}
    for citation in body["citations"]:
        document = documents[citation["doc_id"]]
        assert document.text[citation["start_char"] : citation["end_char"]] == citation["quote"]
    assert any(c["source"] == "reranking.md" for c in body["citations"])


def test_extractive_answer_abstains_without_evidence(client: TestClient) -> None:
    body = client.post(
        "/answer",
        json={
            "query": "reranking",
            "mode": "lexical",
            "generator": "extractive",
            "filters": {"source": "missing.md"},
        },
    ).json()
    assert body["abstained"] is True
    assert body["citations"] == []


def test_answer_hands_k_hits_and_options_to_the_generator(index_dir: Path) -> None:
    generator = RecordingGenerator()
    choices: list[str] = []

    def factory(choice: str) -> RecordingGenerator:
        choices.append(choice)
        return generator

    service = RagService.from_settings(_settings(index_dir), generator_factory=factory)
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/answer",
            json={"query": "hybrid fusion", "k": 2, "max_sentences": 3, "generator": "openai"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "stub answer"
    assert response.json()["model_name"] == "stub-model"
    assert choices == ["openai"]
    ((query, hits, max_sentences),) = generator.calls
    assert (query, len(hits), max_sentences) == ("hybrid fusion", 2, 3)
    assert all(hit.retriever == "hybrid-rrf" for hit in hits)


@pytest.mark.parametrize(
    "failure",
    [ValueError("LLM endpoint returned HTTP 500"), RuntimeError("model output unusable")],
)
def test_generator_failure_is_502(index_dir: Path, failure: Exception) -> None:
    generator = RecordingGenerator(fail=failure)
    service = RagService.from_settings(
        _settings(index_dir), generator_factory=lambda _choice: generator
    )
    with TestClient(create_app(service)) as client:
        response = client.post("/answer", json={"query": "hybrid fusion"})
    assert response.status_code == 502
    assert response.json()["detail"] == f"answer generation failed: {failure}"


def test_openai_generator_without_llm_settings_is_400(client: TestClient) -> None:
    response = client.post("/answer", json={"query": "fusion", "generator": "openai"})
    assert response.status_code == 400
    assert "RAG_LLM_BASE_URL" in response.json()["detail"]


def test_auto_generator_is_extractive_without_llm_settings(client: TestClient) -> None:
    body = client.post("/answer", json={"query": "reciprocal rank fusion"}).json()
    assert body["generator"] == ExtractiveAnswerGenerator().name


@pytest.mark.parametrize(
    "body",
    [
        {"query": "fusion", "max_sentences": 0},
        {"query": "fusion", "max_sentences": 21},
        {"query": "fusion", "generator": "gpt"},
    ],
)
def test_invalid_answer_requests_are_422(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/answer", json=body).status_code == 422


def test_openapi_documents_the_three_endpoints(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/health", "/search", "/answer"}
    assert schema["info"]["version"] == __version__
    assert set(schema["paths"]["/answer"]["post"]["responses"]) >= {"200", "400", "502", "503"}
