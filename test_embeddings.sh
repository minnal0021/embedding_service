#!/usr/bin/env bash
set -euo pipefail
#
# Test the batch embedding service.
# For each model, tests /embedding/{model}/query and /embedding/{model}/document
# and verifies Matryoshka dimension truncation. Also checks that unknown model
# names and paths without a model return 404.
#
# Usage:
#   ./test_embeddings.sh [MODEL]
#
#   MODEL   gemma | qwen. If omitted, every model the service reports as
#           loaded (via /healthcheck) is tested.
#
# Environment overrides:
#   SERVICE_URL   Embedding service base URL  (default: http://localhost:8001)

SERVICE_URL="${SERVICE_URL:-http://localhost:8001}"
MODEL_ARG="${1:-}"

if [[ "${MODEL_ARG}" == "-h" || "${MODEL_ARG}" == "--help" ]]; then
  sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

echo "════════════════════════════════════════════════════════════════"
echo "  Embedding Service – Test Suite"
echo "════════════════════════════════════════════════════════════════"
echo "  Service    : ${SERVICE_URL}"
echo "  Model(s)   : ${MODEL_ARG:-all loaded}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Health check ─────────────────────────────────────────────────────
echo "── Health Check ────────────────────────────────────────────────"
health="$(curl -sS --connect-timeout 5 --max-time 10 "${SERVICE_URL}/healthcheck" 2>/dev/null || true)"
if echo "${health}" | grep -q '"healthy"'; then
  echo "  ✅ Service is healthy"
else
  echo "  ❌ Service health check failed: ${health}" >&2
  echo "  Make sure the service is running at ${SERVICE_URL}" >&2
  exit 1
fi

# "<model> <native-dim> <default-dim>" per loaded model, one per line.
LOADED="$(echo "${health}" | python3 -c "
import json, sys
for key, m in json.load(sys.stdin)['models'].items():
    print(key, m['dimension'], m['default_dimension'])
")"
echo "${LOADED}" | while read -r m dim def; do echo "  loaded: ${m} (native ${dim} dims, default ${def})"; done
echo ""

if [[ -n "${MODEL_ARG}" ]]; then
  SELECTED="$(echo "${LOADED}" | awk -v m="${MODEL_ARG}" '$1 == m')"
  if [[ -z "${SELECTED}" ]]; then
    echo "  ❌ Model '${MODEL_ARG}' is not loaded (loaded: $(echo "${LOADED}" | awk '{print $1}' | tr '\n' ' '))" >&2
    exit 1
  fi
else
  SELECTED="${LOADED}"
fi

# Validate one endpoint's response: correct count, dimension, and L2 norm.
# The response goes in on stdin: a 4096-dim batch overflows the argv limit.
# (`read -d ''` rather than a heredoc in $(...), which macOS's bash 3.2 misparses.)
read -r -d '' CHECK_EMBEDDINGS_PY <<'EOF' || true
import sys, json, math
data = json.load(sys.stdin)
expected_dim = int(sys.argv[1])
embeddings = data.get("embeddings", [])
print(f"  Embeddings returned: {len(embeddings)}")
all_pass = True
for i, vec in enumerate(embeddings):
    dim = len(vec)
    norm = math.sqrt(sum(x * x for x in vec))
    dim_ok = dim == expected_dim
    norm_ok = abs(norm - 1.0) < 1e-3
    status = "✅" if (dim_ok and norm_ok) else "❌"
    if not (dim_ok and norm_ok):
        all_pass = False
    preview = ", ".join(f"{v:.6f}" for v in vec[:4])
    print(f"    [{i+1}] dim={dim} norm={norm:.6f} {status}  [{preview}, ...]")
if not all_pass:
    print("  ❌ FAIL: dimension or L2 norm check failed", file=sys.stderr)
    sys.exit(1)
print("  ✅ PASS (all L2-normalised)")
EOF
check_embeddings() {
    printf '%s' "$1" | python3 -c "${CHECK_EMBEDDINGS_PY}" "$2"
}

# POST a body to an endpoint; prints the response body, then the HTTP status
# on the last line.
post() {
    curl -sS --max-time 600 -w '\n%{http_code}' \
        -X POST "${SERVICE_URL}$1" \
        -H "Content-Type: application/json" \
        -d "$2"
}

# POST a body to an endpoint, assert HTTP 200, and validate the embeddings.
post_and_check() {
    local endpoint="$1" body="$2" dim="$3"
    local resp http body_resp
    resp="$(post "${endpoint}" "${body}")"
    http="$(echo "${resp}" | tail -n 1)"
    body_resp="$(echo "${resp}" | sed '$d')"
    if [[ "${http}" != "200" ]]; then
        echo "  ❌ FAIL: HTTP ${http}"
        echo "${body_resp}" | python3 -m json.tool 2>/dev/null || echo "${body_resp}"
        exit 1
    fi
    check_embeddings "${body_resp}" "${dim}"
}

# Assert that POSTing to an endpoint returns 404.
expect_404() {
    local http
    http="$(post "$1" '{"payloads": ["hello"]}' | tail -n 1)"
    if [[ "${http}" == "404" ]]; then
        echo "  ✅ $1 → 404"
    else
        echo "  ❌ FAIL: $1 → HTTP ${http} (expected 404)"
        exit 1
    fi
}

# Run the per-model suite.
test_model() {
    local m="$1" native="$2" default="$3"
    local base="/embedding/${m}"

    echo "════════════════════════════════════════════════════════════════"
    echo "  Model: ${m} (native ${native} dims)"
    echo "════════════════════════════════════════════════════════════════"

    echo "── Test 1: Batch Query Embedding (${base}/query) ──"
    post_and_check "${base}/query" \
        "{\"payloads\": [\"hello world\", \"What is machine learning?\"], \"dimensions\": ${native}}" \
        "${native}"
    echo ""

    echo "── Test 2: Batch Document Embedding (${base}/document) ──"
    local docs='["The court filed the complaint on Tuesday.", "Meta used copyrighted works to train Llama.", "Engineers relied on pirated books and articles."]'
    post_and_check "${base}/document" "{\"payloads\": ${docs}, \"dimensions\": ${native}}" "${native}"
    echo ""

    echo "── Test 3: Default dimensions (omitted → ${default}) ──"
    post_and_check "${base}/query" '{"payloads": ["What is machine learning?"]}' "${default}"
    echo ""

    # Matryoshka truncation lets callers request a smaller width.
    echo "── Test 4: Dimension Truncation (${native} → 256) ──"
    post_and_check "${base}/query" '{"payloads": ["What is machine learning?"], "dimensions": 256}' 256
    echo ""
}

while read -r m dim def; do
    test_model "${m}" "${dim}" "${def}"
done <<<"${SELECTED}"

# ── Unknown model / no model ─────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "── Test: Unknown model and paths without a model return 404 ──"
expect_404 "/embedding/bogus/query"
expect_404 "/embedding/bogus/document"
expect_404 "/embedding/query"
expect_404 "/embedding/document"
echo ""

# ── Summary ──────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ All tests passed"
echo "════════════════════════════════════════════════════════════════"
