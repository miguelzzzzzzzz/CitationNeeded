# syntax=docker/dockerfile:1
# Citation Needed HTTP service: GET /health, POST /search, POST /answer.
# The image serves a prebuilt index (rag-engine index) mounted at /data/index.

# Base image pinned by the digest of its multi-arch index (resolved 2026-10-27),
# so rebuilds use the same OS and Python layers. The tag stays for readability;
# the digest wins. To refresh, resolve the current digest of python:3.13-slim
# (e.g. `docker buildx imagetools inspect python:3.13-slim`), replace it here and
# let CI rebuild and smoke-test the image.
ARG PYTHON_IMAGE=python:3.13-slim@sha256:d3449c0493345868b09708d7056dce5edda3a75233ea86008fc80c934d355a10

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# "api" is required; "embeddings" adds fastembed for dense queries and reranking
# (models download on first use into /data/models, not at build time).
ARG EXTRAS="api,embeddings"
RUN pip install ".[${EXTRAS}]"

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="Citation Needed" \
      org.opencontainers.image.description="Hybrid retrieval and span-cited answers over a prebuilt index" \
      org.opencontainers.image.source="https://github.com/miguelzzzzzzzz/CitationNeeded" \
      org.opencontainers.image.licenses="MIT"
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app \
       --shell /usr/sbin/nologin app \
    && mkdir -p /data/index /data/models \
    && chown -R app:app /data
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_INDEX_DIR=/data/index \
    RAG_EMBEDDING_CACHE_DIR=/data/models \
    HF_HOME=/data/models/huggingface
WORKDIR /home/app
USER 10001:10001
EXPOSE 8000
# 200 only while an index is served (503 otherwise), so "healthy" means ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
# The app logs one JSON line per request (rag_engine.api.requests), which
# replaces uvicorn's access log. No authentication: publish the port only on a
# trusted interface (compose.yaml binds 127.0.0.1) or behind an auth proxy.
CMD ["uvicorn", "rag_engine.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--no-access-log"]
