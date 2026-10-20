"""Framework-free core of the HTTP API: one read-only index, search and answer.

The service holds a single index loaded at startup (``RAG_INDEX_DIR``, built
offline by ``rag-engine index``) and never writes to it, so concurrent requests
only read shared state. Rebuilding the index means restarting the service. The
reranker and the answer generator come from injectable factories, so tests and
alternative deployments supply stand-ins and no test needs a live model.

Error contract (mapped to HTTP status codes in :mod:`rag_engine.api.app`):

* :class:`IndexNotLoadedError` -> 503: no index is being served.
* :class:`DependencyMissingError` -> 503: an optional dependency is missing,
  e.g. fastembed for a fastembed index's dense queries or for reranking.
* :class:`GenerationError` -> 502: the answer generator failed (e.g. the LLM
  endpoint was unreachable or returned unusable output).
* :class:`InvalidRequestError` -> 400: the request passed schema validation
  but cannot be served as asked (e.g. the ``openai`` generator without
  ``RAG_LLM_*`` configured).

Anything else is a server error (500): a ``ValueError`` raised *during*
retrieval, for example, points at the index or the code, not at the request.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rag_engine import __version__
from rag_engine.answering import AnswerGenerator, AnswerResponse, select_generator
from rag_engine.api.schemas import (
    AnswerRequest,
    HealthResponse,
    RetrievalOptions,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from rag_engine.config import Settings
from rag_engine.retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    LexicalRetriever,
    Reranker,
    RetrievedChunk,
    compose_retriever,
    load_index,
)
from rag_engine.retrieval.index_store import load_documents

__all__ = [
    "DependencyMissingError",
    "GenerationError",
    "GeneratorFactory",
    "IndexNotLoadedError",
    "InvalidRequestError",
    "RagService",
    "RerankerFactory",
    "ServedIndex",
]

RerankerFactory = Callable[[str], Reranker]
GeneratorFactory = Callable[[str], AnswerGenerator]


class IndexNotLoadedError(RuntimeError):
    """No index is being served."""


class DependencyMissingError(RuntimeError):
    """An optional dependency needed for the request is not installed."""


class InvalidRequestError(ValueError):
    """A schema-valid request that this service cannot serve as asked."""


class GenerationError(RuntimeError):
    """The answer generator failed after retrieval succeeded."""


@dataclass(frozen=True)
class ServedIndex:
    """The retrievers of one loaded index plus the counts reported by ``/health``."""

    dense: DenseRetriever
    lexical: LexicalRetriever
    documents: int

    @property
    def chunks(self) -> int:
        return len(self.lexical.bm25)

    @property
    def embedder(self) -> str:
        return self.dense.embedder.name


class RagService:
    """Serves search and cited answers over one immutable :class:`ServedIndex`."""

    def __init__(
        self,
        index: ServedIndex | None,
        settings: Settings,
        *,
        unavailable_reason: str | None = None,
        reranker_factory: RerankerFactory | None = None,
        generator_factory: GeneratorFactory | None = None,
    ) -> None:
        self._index = index
        self._settings = settings
        self._unavailable_reason = unavailable_reason or "no index loaded"
        self._reranker_factory = reranker_factory or self._default_reranker
        self._generator_factory = generator_factory or self._default_generator
        self._reranker: Reranker | None = None
        self._reranker_lock = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        reranker_factory: RerankerFactory | None = None,
        generator_factory: GeneratorFactory | None = None,
    ) -> RagService:
        """Load ``settings.service.index_dir`` if it holds an index.

        A missing setting or a directory without ``index.json`` (not built yet,
        e.g. an empty volume) starts the service without an index, reported by
        ``/health`` as unavailable. A directory that has a manifest but fails to
        load raises ``ValueError``: a corrupt index must stop startup rather than
        be masked as "not built yet".
        """
        index_dir = settings.service.index_dir
        index: ServedIndex | None = None
        reason: str | None = None
        if index_dir is None:
            reason = (
                "no index configured: set RAG_INDEX_DIR to a directory built by 'rag-engine index'"
            )
        elif not (Path(index_dir) / "index.json").is_file():
            reason = (
                f"no index found in RAG_INDEX_DIR ({index_dir}): build one with "
                "'rag-engine index' and restart the service"
            )
        else:
            dense, lexical = load_index(index_dir, settings.embedding)
            index = ServedIndex(dense, lexical, documents=len(load_documents(index_dir)))
        return cls(
            index,
            settings,
            unavailable_reason=reason,
            reranker_factory=reranker_factory,
            generator_factory=generator_factory,
        )

    @property
    def index(self) -> ServedIndex | None:
        return self._index

    def health(self) -> HealthResponse:
        index = self._index
        if index is None:
            return HealthResponse(
                status="unavailable",
                version=__version__,
                index_loaded=False,
                detail=self._unavailable_reason,
            )
        return HealthResponse(
            status="ok",
            version=__version__,
            index_loaded=True,
            embedder=index.embedder,
            chunks=index.chunks,
            documents=index.documents,
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        hits = self._retrieve(request)
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            rerank=request.rerank,
            hits=[SearchHit.from_hit(hit) for hit in hits],
        )

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        # Resolve the generator first: a request it cannot serve fails before
        # any retrieval work is spent on it.
        try:
            generator = self._generator_factory(request.generator)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        hits = self._retrieve(request)
        try:
            return generator.generate(request.query, hits, max_sentences=request.max_sentences)
        except (ValueError, RuntimeError) as exc:
            # The request was validated already, so a failure here comes from
            # the backend (unreachable endpoint, unusable model output).
            raise GenerationError(f"answer generation failed: {exc}") from exc

    def _retrieve(self, options: RetrievalOptions) -> list[RetrievedChunk]:
        index = self._index
        if index is None:
            raise IndexNotLoadedError(self._unavailable_reason)
        weights: dict[str, float] | None = None
        if options.weights is not None:
            weights = dict(options.weights.items())
        try:
            retriever = compose_retriever(
                index.dense,
                index.lexical,
                mode=options.mode,
                fusion=options.effective_fusion(),
                weights=weights,
                rrf_k=options.effective_rrf_k(),
                candidates=options.effective_candidates(),
                reranker=self._shared_reranker() if options.rerank else None,
            )
        except ValueError as exc:
            # The schema mirrors the constructors' checks; this is a backstop.
            raise InvalidRequestError(str(exc)) from exc
        try:
            return retriever.retrieve(options.query, options.k, options.filters)
        except ImportError as exc:
            raise DependencyMissingError(str(exc)) from exc

    def _shared_reranker(self) -> Reranker:
        """One reranker instance per service, so the model is not reloaded per request.

        The model itself still loads lazily on the first rerank; that load is not
        synchronized, so concurrent first requests may each load it once (results
        are unaffected). The same holds for a fastembed query embedder.
        """
        with self._reranker_lock:
            if self._reranker is None:
                self._reranker = self._reranker_factory(self._settings.service.rerank_model)
            return self._reranker

    def _default_reranker(self, model_name: str) -> Reranker:
        embedding = self._settings.embedding
        return CrossEncoderReranker(
            model_name, cache_dir=embedding.cache_dir, threads=embedding.threads
        )

    def _default_generator(self, choice: str) -> AnswerGenerator:
        return select_generator(choice, self._settings.llm)
