#!/usr/bin/env bash
# Smoke-test compose.yaml end to end with a built image: run the `index` tool
# service to build a hashing-embedder index of evals/structured/docs (no model
# downloads), then `docker compose up --wait` the `api` service on it (the index
# must exist first: /health, and so the HEALTHCHECK, is 503 until an index is
# served), check the published port, read-only root fs and read-only index
# mount, run the shared HTTP checks from scripts/smoke_lib.sh, and finally
# `docker compose down -v`.
# Usage: scripts/compose_smoke.sh [IMAGE]   (default: citation-needed:ci)
# Uses port 127.0.0.1:8000, as compose.yaml publishes it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="http://127.0.0.1:8000"
# shellcheck source=scripts/smoke_lib.sh
source "$ROOT/scripts/smoke_lib.sh"

export RAG_IMAGE="${1:-citation-needed:ci}"
export RAG_CORPUS_DIR="$ROOT/evals/structured/docs"
export RAG_INDEX_EMBEDDER=hashing
export RAG_CORPUS_ID=structured-eval
export RAG_WARMUP=embedder
compose() { docker compose --project-directory "$ROOT" -f "$ROOT/compose.yaml" -p "rag-smoke-$$" "$@"; }

cleanup() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "--- compose ps / logs"
        compose ps -a || true
        compose logs --no-color api 2>&1 | tail -n 50 || true
    fi
    compose down -v --remove-orphans >/dev/null 2>&1 || true
    rm -f "$SMOKE_BODY"
    exit "$status"
}
trap cleanup EXIT

docker image inspect "$RAG_IMAGE" >/dev/null || fail "image $RAG_IMAGE not found; build it first"

echo "== docker compose run --rm index (hashing embedder, no downloads)"
compose run --rm index

echo "== docker compose up -d --wait api"
compose up -d --no-build --wait --wait-timeout 120 api

container="$(compose ps -q api)"
[ -n "$container" ] || fail "no api container"
[ "$(docker inspect --format '{{.State.Health.Status}}' "$container")" = healthy ] \
    || fail "api container is not healthy"

echo "== published port, read-only root fs and index mount"
published="$(compose port api 8000)"
[ "$published" = "127.0.0.1:8000" ] || fail "api published on '$published', expected 127.0.0.1:8000"
[ "$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "$container")" = true ] \
    || fail "api root filesystem is writable"
index_rw="$(docker inspect --format \
    '{{range .Mounts}}{{if eq .Destination "/data/index"}}{{.RW}}{{end}}{{end}}' "$container")"
[ "$index_rw" = false ] || fail "index mount RW is '$index_rw', expected false"

check_health
check_endpoints
check_request_logs compose logs --no-color --no-log-prefix api

echo "== docker compose down -v"
compose down -v --remove-orphans
[ -z "$(compose ps -a -q)" ] || fail "containers left after compose down"

echo "compose smoke test: OK"
