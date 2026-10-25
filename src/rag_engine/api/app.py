"""FastAPI application: ``GET /health``, ``POST /search``, ``POST /answer``.

Run with ``uvicorn rag_engine.api.app:create_app --factory``. Configuration
comes from ``RAG_*`` variables (``Settings.from_env``); ``RAG_INDEX_DIR`` names
the index directory to serve. Endpoints are plain ``def`` functions, so FastAPI
runs the CPU-bound retrieval in its worker thread pool instead of blocking the
event loop. Errors use FastAPI's ``{"detail": ...}`` body.

Every request gets an ``X-Request-ID`` response header and one JSON log line on
the ``rag_engine.api.requests`` logger: request id, method, path, status, total
milliseconds and the per-stage milliseconds the service recorded
(``retrieve_ms``, ``generate_ms``). Query text is never logged. ``/health``
lines are logged at DEBUG so container health probes do not flood the log.

The service has no authentication: expose it only on a trusted network or
behind a proxy that authenticates (``compose.yaml`` binds it to 127.0.0.1).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_engine import __version__
from rag_engine.answering import AnswerResponse
from rag_engine.api.schemas import AnswerRequest, HealthResponse, SearchRequest, SearchResponse
from rag_engine.api.service import (
    DependencyMissingError,
    GenerationError,
    IndexNotLoadedError,
    InvalidRequestError,
    RagService,
)
from rag_engine.config import Settings

__all__ = ["configure_logging", "create_app"]

logger = logging.getLogger("rag_engine.api")
request_logger = logging.getLogger("rag_engine.api.requests")

REQUEST_ID_HEADER = "X-Request-ID"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str) -> None:
    """Set the ``rag_engine`` log level; add a stderr handler if nothing logs.

    Uvicorn configures only its own loggers, so without a handler here the
    service's INFO lines (startup, warm-up, requests) would be dropped. When the
    root logger or ``rag_engine`` already has handlers (an embedding app, a
    ``--log-config``, pytest), records propagate to those instead.
    """
    package = logging.getLogger("rag_engine")
    package.setLevel(level)
    if not package.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        package.addHandler(handler)


_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"description": "The request cannot be served as asked"},
    502: {"description": "The answer generator failed"},
    503: {"description": "No index is served or an optional dependency is missing"},
}


def create_app(service: RagService | None = None) -> FastAPI:
    """Build the app; without ``service``, load one from ``RAG_*`` at startup.

    Startup fails (the process exits) when the configured index exists but is
    corrupt, so a bad deployment is loud instead of serving 503s forever.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            settings = Settings.from_env()
            configure_logging(settings.service.log_level)
            app.state.service = RagService.from_settings(settings)
        else:
            app.state.service = service
        health = app.state.service.health()
        if health.index_loaded:
            logger.info(
                "serving index: %d chunks, %d documents, embedder %s",
                health.chunks,
                health.documents,
                health.embedder,
            )
        else:
            logger.warning("no index served: %s", health.detail)
        # Before the first request is accepted, so "healthy" also means "warm".
        app.state.service.warm_up()
        yield

    app = FastAPI(
        title="Citation Needed",
        version=__version__,
        description="Hybrid retrieval and span-cited answers over a prebuilt index.",
        lifespan=lifespan,
    )

    def _service(request: Request) -> RagService:
        current: RagService = request.app.state.service
        return current

    def _timings(request: Request) -> dict[str, float] | None:
        timings: dict[str, float] | None = getattr(request.state, "timings", None)
        return timings

    @app.middleware("http")
    async def _log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex
        timings: dict[str, float] = {}
        request.state.timings = timings
        start = time.perf_counter()
        status = 500  # unless a response comes back: an unhandled error
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            record = {
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                **timings,
            }
            level = logging.DEBUG if request.url.path == "/health" else logging.INFO
            request_logger.log(level, "%s", json.dumps(record, sort_keys=True))
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def _invalid_body(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler echoes each rejected input back, which fails
        # to serialize for NaN/Infinity (Python's JSON parser accepts them) and
        # would repeat large query strings; keep only where and why.
        errors = [
            {
                "type": str(error.get("type", "")),
                "loc": list(error.get("loc", ())),
                "msg": str(error.get("msg", "")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(IndexNotLoadedError)
    async def _no_index(_: Request, exc: IndexNotLoadedError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(DependencyMissingError)
    async def _missing_dependency(_: Request, exc: DependencyMissingError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(GenerationError)
    async def _generation_failed(_: Request, exc: GenerationError) -> JSONResponse:
        logger.warning("%s", exc)
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(InvalidRequestError)
    async def _bad_request(_: Request, exc: InvalidRequestError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "No index is served"}},
    )
    def health(request: Request) -> JSONResponse:
        """Liveness plus readiness: 200 when an index is served, 503 otherwise."""
        body = _service(request).health()
        status_code = 200 if body.index_loaded else 503
        return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))

    @app.post("/search", response_model=SearchResponse, responses=_ERROR_RESPONSES)
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        """Ranked chunks with citation provenance (same fields as ``search --json``)."""
        return _service(request).search(body, timings=_timings(request))

    @app.post("/answer", response_model=AnswerResponse, responses=_ERROR_RESPONSES)
    def answer(body: AnswerRequest, request: Request) -> AnswerResponse:
        """Retrieve ``k`` chunks, then generate an answer with verifiable span citations."""
        return _service(request).answer(body, timings=_timings(request))

    return app
