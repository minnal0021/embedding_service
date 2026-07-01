#!/usr/bin/env bash
set -euo pipefail
#
# Test the EmbeddingGemma-300M batch embedding service.
# Tests both /embedding/query and /embedding/document endpoints,
# and verifies dimension truncation works correctly.
#
# Usage:
#   ./test_embeddings.sh [service-url]
#
# Environment overrides:
#   SERVICE_URL   Embedding service base URL  (default: http://localhost:8000)
#   DIMENSIONS    Embedding dimensions        (default: 768)

SERVICE_URL="${1:-${SERVICE_URL:-http://localhost:8001}}"
DIMENSIONS="${DIMENSIONS:-768}"

echo "════════════════════════════════════════════════════════════════"
echo "  EmbeddingGemma-300M Service – Test Suite"
echo "════════════════════════════════════════════════════════════════"
echo "  Service    : ${SERVICE_URL}"
echo "  Dimensions : ${DIMENSIONS}"
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
echo ""

# Validate one endpoint's response: correct count, dimension, and L2 norm.
check_embeddings() {
    python3 - "$1" "$2" <<'EOF'
import sys, json, math
data = json.loads(sys.argv[1])
expected_dim = int(sys.argv[2])
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
}

# POST a body to an endpoint, assert HTTP 200, and validate the embeddings.
post_and_check() {
    local endpoint="$1" body="$2"
    local resp http body_resp
    resp="$(curl -sS -w '\n__HTTP_STATUS__%{http_code}' \
        -X POST "${SERVICE_URL}${endpoint}" \
        -H "Content-Type: application/json" \
        -d "${body}")"
    http="$(echo "${resp}" | grep '__HTTP_STATUS__' | sed 's/__HTTP_STATUS__//')"
    body_resp="$(echo "${resp}" | grep -v '__HTTP_STATUS__')"
    if [[ "${http}" != "200" ]]; then
        echo "  ❌ FAIL: HTTP ${http}"
        echo "${body_resp}" | python3 -m json.tool 2>/dev/null || echo "${body_resp}"
        exit 1
    fi
    check_embeddings "${body_resp}" "${DIMENSIONS}"
}

# ── Test 1: Batch Query Embedding ────────────────────────────────────
echo "── Test 1: Batch Query Embedding (/embedding/query) ──────────"
QUERY_BODY="{\"payloads\": [\"hello world\", \"What is machine learning?\"], \"dimensions\": ${DIMENSIONS}}"
post_and_check "/embedding/query" "${QUERY_BODY}"
echo ""

# ── Test 2: Batch Document Embedding ─────────────────────────────────
echo "── Test 2: Batch Document Embedding (/embedding/document) ────"
DOC_PAYLOADS='["The court filed the complaint on Tuesday.", "Meta used copyrighted works to train Llama.", "Engineers relied on pirated books and articles."]'
DOC_BODY="{\"payloads\": ${DOC_PAYLOADS}, \"dimensions\": ${DIMENSIONS}}"
post_and_check "/embedding/document" "${DOC_BODY}"
echo ""

# ── Test 3: Dimension Truncation ─────────────────────────────────────
# EmbeddingGemma's native width is 768; Matryoshka truncation lets callers
# request a smaller width (e.g. 256). Requesting >768 is not truncation and
# returns the full 768, so we compare the full width against a truncated one.
echo "── Test 3: Dimension Truncation ──────────────────────────────"
DIM_A=768
DIM_B=256

body_a="{\"payloads\": [\"What is machine learning?\"], \"dimensions\": ${DIM_A}}"
resp_a="$(curl -sS --max-time 120 -X POST "${SERVICE_URL}/embedding/query" \
  -H "Content-Type: application/json" -d "${body_a}" 2>/dev/null)"
len_a="$(echo "${resp_a}" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]))")"

body_b="{\"payloads\": [\"What is machine learning?\"], \"dimensions\": ${DIM_B}}"
resp_b="$(curl -sS --max-time 120 -X POST "${SERVICE_URL}/embedding/query" \
  -H "Content-Type: application/json" -d "${body_b}" 2>/dev/null)"
len_b="$(echo "${resp_b}" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]))")"

echo "  dimensions=${DIM_A}  →  got ${len_a}"
echo "  dimensions=${DIM_B}  →  got ${len_b}"

PASS=1
[[ "${len_a}" -ne "${DIM_A}" ]] && echo "  ❌ Expected ${DIM_A}, got ${len_a}" && PASS=0
[[ "${len_b}" -ne "${DIM_B}" ]] && echo "  ❌ Expected ${DIM_B}, got ${len_b}" && PASS=0
[[ "${len_a}" -eq "${len_b}" ]] && echo "  ❌ Both same dimension — truncation broken" && PASS=0

if [[ "${PASS}" -eq 1 ]]; then
  echo "  ✅ PASS"
else
  exit 1
fi
echo ""

# ── Summary ──────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ All tests passed"
echo "════════════════════════════════════════════════════════════════"
