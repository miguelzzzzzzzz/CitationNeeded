#!/usr/bin/env bash
# Smoke-test a built image end to end: build an index with the offline hashing
# embedder from evals/structured/docs, serve it read-only, and exercise
# /health, /search (dense, lexical, hybrid), /answer and request validation,
# plus the startup warm-up and the per-request JSON log lines.
# Also checks that a container without an index reports 503 / unhealthy.
# Usage: scripts/docker_smoke.sh [IMAGE]   (default: citation-needed:ci)
set -euo pipefail

IMAGE="${1:-citation-needed:ci}"
PORT="${SMOKE_PORT:-18000}"
EMPTY_PORT="$((PORT + 1))"
SUFFIX="smoke-$$"
VOLUME="rag-index-$SUFFIX"
EMPTY_VOLUME="rag-empty-$SUFFIX"
API="rag-api-$SUFFIX"
EMPTY_API="rag-empty-$SUFFIX"
DOCS="$(cd "$(dirname "$0")/.." && pwd)/evals/structured/docs"
BASE="http://127.0.0.1:$PORT"

cleanup() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "--- logs ($API)"
        docker logs "$API" 2>&1 | tail -n 50 || true
    fi
    docker rm -f "$API" "$EMPTY_API" >/dev/null 2>&1 || true
    docker volume rm "$VOLUME" "$EMPTY_VOLUME" >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
post() { curl -sS -o /tmp/body.json -w '%{http_code}' -H 'Content-Type: application/json' -d "$2" "$BASE$1"; }

wait_health() {  # container, wanted status
    for _ in $(seq 1 60); do
        state="$(docker inspect --format '{{.State.Health.Status}}' "$1")"
        [ "$state" = "$2" ] && return 0
        sleep 1
    done
    fail "$1 health is '$state', expected '$2'"
}

echo "== image runs as non-root"
user="$(docker image inspect --format '{{.Config.User}}' "$IMAGE")"
[ "$user" = "10001:10001" ] || fail "image user is '$user'"

echo "== build an index inside the container (hashing embedder, no downloads)"
docker volume create "$VOLUME" >/dev/null
docker run --rm -v "$DOCS:/corpus:ro" -v "$VOLUME:/data/index" "$IMAGE" \
    rag-engine index /corpus --out /data/index --embedder hashing --corpus-id structured-eval

echo "== serve it read-only (with the embedder warm-up)"
docker run -d --name "$API" --read-only --tmpfs /tmp -p "127.0.0.1:$PORT:8000" \
    -e RAG_WARMUP=embedder -v "$VOLUME:/data/index:ro" "$IMAGE" >/dev/null
wait_health "$API" healthy
health="$(curl -fsS "$BASE/health")"
echo "$health"
echo "$health" | jq -e '.status == "ok" and .index_loaded and .chunks > 0 and .documents == 3' \
    >/dev/null || fail "unexpected /health body"

for mode in dense lexical hybrid; do
    echo "== POST /search mode=$mode"
    code="$(post /search "{\"query\": \"quicksort pivot partition\", \"k\": 3, \"mode\": \"$mode\"}")"
    [ "$code" = 200 ] || fail "/search $mode returned $code: $(cat /tmp/body.json)"
    jq -e '(.hits | length) > 0 and .hits[0].source == "sorting_algorithms.md"' /tmp/body.json \
        >/dev/null || fail "/search $mode top hit: $(jq -c '.hits[0].source' /tmp/body.json)"
done

echo "== POST /answer (extractive)"
code="$(post /answer '{"query": "How does quicksort choose a pivot?", "generator": "extractive"}')"
[ "$code" = 200 ] || fail "/answer returned $code: $(cat /tmp/body.json)"
jq -e '.abstained == false and (.citations | length) > 0' /tmp/body.json >/dev/null \
    || fail "/answer did not cite: $(cat /tmp/body.json)"
jq -c '{answer: .answer[:120], first_citation: .citations[0].source}' /tmp/body.json

echo "== request ids and structured request logs"
request_id="$(curl -sS -o /dev/null -D - -H 'Content-Type: application/json' \
    -d '{"query": "quicksort pivot partition", "mode": "lexical"}' "$BASE/search" \
    | tr -d '\r' | awk 'tolower($1) == "x-request-id:" {print $2}')"
[ "${#request_id}" = 32 ] || fail "missing X-Request-ID header: '$request_id'"
logs="$(docker logs "$API" 2>&1)"
grep -q 'warm-up: embedder ready' <<<"$logs" || fail "no warm-up line in the logs"
line="$(grep -F "\"request_id\": \"$request_id\"" <<<"$logs" | sed 's/^[^{]*//' || true)"
[ -n "$line" ] || fail "no log line for request $request_id"
echo "$line"
echo "$line" | jq -e '.path == "/search" and .status == 200 and .retrieve_ms >= 0' >/dev/null \
    || fail "unexpected request log line"
if grep -q 'quicksort pivot' <<<"$logs"; then fail "query text leaked into the logs"; fi

echo "== invalid requests"
[ "$(post /search '{"query": "x", "k": 0}')" = 422 ] || fail "k=0 not rejected"
[ "$(post /answer '{"query": "x", "generator": "openai"}')" = 400 ] \
    || fail "openai generator without RAG_LLM_* not rejected"

echo "== a container without an index is unhealthy and answers 503"
docker volume create "$EMPTY_VOLUME" >/dev/null
docker run -d --name "$EMPTY_API" -p "127.0.0.1:$EMPTY_PORT:8000" \
    -v "$EMPTY_VOLUME:/data/index" "$IMAGE" >/dev/null
code=000
for _ in $(seq 1 60); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$EMPTY_PORT/health" 2>/dev/null || true)"
    [ "$code" != 000 ] && break
    sleep 1
done
[ "$code" = 503 ] || fail "empty container /health returned $code"
# The image's HEALTHCHECK probe must fail too (during the start period Docker
# keeps the status at "starting", so check the probe result, not the status).
probe=""
for _ in $(seq 1 60); do
    probe="$(docker inspect --format '{{range .State.Health.Log}}{{.ExitCode}} {{end}}' "$EMPTY_API")"
    [ -n "$probe" ] && break
    sleep 1
done
case "$probe" in
    0*|"") fail "HEALTHCHECK probe without an index exited '${probe:-none}', expected failure" ;;
esac
[ "$(docker inspect --format '{{.State.Health.Status}}' "$EMPTY_API")" != healthy ] \
    || fail "container without an index reported healthy"

echo "docker smoke test: OK"
