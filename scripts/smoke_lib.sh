# shellcheck shell=bash
# Shared HTTP checks for scripts/docker_smoke.sh and scripts/compose_smoke.sh.
# Source it (do not run it) after setting BASE to the API's base URL, e.g.
# BASE=http://127.0.0.1:8000. Needs curl and jq. The served index must be a
# build of evals/structured/docs (3 documents).

SMOKE_BODY="$(mktemp)"

fail() { echo "FAIL: $*" >&2; exit 1; }

# post PATH JSON -> prints the status code; the body is left in $SMOKE_BODY.
post() {
    curl -sS -o "$SMOKE_BODY" -w '%{http_code}' -H 'Content-Type: application/json' \
        -d "$2" "$BASE$1"
}

# /health of a served structured-eval index.
check_health() {
    local health
    health="$(curl -fsS "$BASE/health")"
    echo "$health"
    echo "$health" | jq -e '.status == "ok" and .index_loaded and .chunks > 0 and .documents == 3' \
        >/dev/null || fail "unexpected /health body"
}

# Every endpoint: /search in each mode, an extractive /answer, 422/400 errors.
check_endpoints() {
    local mode code
    for mode in dense lexical hybrid; do
        echo "== POST /search mode=$mode"
        code="$(post /search "{\"query\": \"quicksort pivot partition\", \"k\": 3, \"mode\": \"$mode\"}")"
        [ "$code" = 200 ] || fail "/search $mode returned $code: $(cat "$SMOKE_BODY")"
        jq -e '(.hits | length) > 0 and .hits[0].source == "sorting_algorithms.md"' "$SMOKE_BODY" \
            >/dev/null || fail "/search $mode top hit: $(jq -c '.hits[0].source' "$SMOKE_BODY")"
    done

    echo "== POST /answer (extractive)"
    code="$(post /answer '{"query": "How does quicksort choose a pivot?", "generator": "extractive"}')"
    [ "$code" = 200 ] || fail "/answer returned $code: $(cat "$SMOKE_BODY")"
    jq -e '.abstained == false and (.citations | length) > 0' "$SMOKE_BODY" >/dev/null \
        || fail "/answer did not cite: $(cat "$SMOKE_BODY")"
    jq -c '{answer: .answer[:120], first_citation: .citations[0].source}' "$SMOKE_BODY"

    echo "== invalid requests"
    [ "$(post /search '{"query": "x", "k": 0}')" = 422 ] || fail "k=0 not rejected"
    [ "$(post /answer '{"query": "x", "generator": "openai"}')" = 400 ] \
        || fail "openai generator without RAG_LLM_* not rejected"
}

# check_request_logs LOGS_COMMAND...: sends one /search, then checks that the
# response has an X-Request-ID, that the command's output (the service logs)
# has the embedder warm-up line and that request's JSON log line, and that no
# query text was logged. Run it after check_endpoints so every query is in the logs.
check_request_logs() {
    local request_id logs line
    echo "== request ids and structured request logs"
    request_id="$(curl -sS -o /dev/null -D - -H 'Content-Type: application/json' \
        -d '{"query": "quicksort pivot partition", "mode": "lexical"}' "$BASE/search" \
        | tr -d '\r' | awk 'tolower($1) == "x-request-id:" {print $2}')"
    [ "${#request_id}" = 32 ] || fail "missing X-Request-ID header: '$request_id'"
    logs="$("$@" 2>&1)" || fail "could not read the service logs"
    grep -q 'warm-up: embedder ready' <<<"$logs" || fail "no warm-up line in the logs"
    line="$(grep -F "\"request_id\": \"$request_id\"" <<<"$logs" | sed 's/^[^{]*//' || true)"
    [ -n "$line" ] || fail "no log line for request $request_id"
    echo "$line"
    echo "$line" | jq -e '.path == "/search" and .status == 200 and .retrieve_ms >= 0' >/dev/null \
        || fail "unexpected request log line"
    if grep -q 'quicksort pivot' <<<"$logs"; then fail "query text leaked into the logs"; fi
}
