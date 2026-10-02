#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./generate_sample_embeddings.sh MODEL

Embeds the ELI5 sample QA pairs with MODEL (gemma | qwen) through the running
embedding service, and writes embedding/MODEL/sample_data_embedding.parquet
for ./generate_cluster_centroids.sh MODEL. Start the service first
(./embedding_service.sh start, with MODEL among its --models).

Environment overrides: DIMENSIONS, BATCH_SIZE, MAX_RECORDS,
SERVICE_URL (default: http://localhost:8001)
USAGE
}

MODELS="gemma qwen"
MODEL="${1:-}"
if [[ "${MODEL}" == "-h" || "${MODEL}" == "--help" ]]; then usage; exit 0; fi
if [[ -z "${MODEL}" || " ${MODELS} " != *" ${MODEL} "* ]]; then
  echo "ERROR: expected a model name (one of: ${MODELS}), got '${MODEL}'." >&2
  usage >&2
  exit 1
fi

# ── Configuration ────────────────────────────────────────────────────────────
INPUT="sample_data/eli5_question_answer.jsonl"
OUTPUT="embedding/${MODEL}/sample_data_embedding.parquet"
DIMENSIONS="${DIMENSIONS:-768}"      # MRL-truncated from the native width; 0 = native (gemma 768, qwen 4096)
BATCH_SIZE="${BATCH_SIZE:-32}"       # QA pairs per embedding call
MAX_RECORDS="${MAX_RECORDS:-25000}"  # max QA pairs to embed (0 = all)
SERVICE_URL="${SERVICE_URL:-http://localhost:8001}"
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Sample Data Embedding Generator (ELI5 QA)"
echo "════════════════════════════════════════════════════════════════"
echo "  Model        : ${MODEL}  (via ${SERVICE_URL})"
echo "  Input        : ${INPUT}"
echo "  Output       : ${OUTPUT}"
echo "  Dimensions   : ${DIMENSIONS}  (0 = native)"
echo "  Batch size   : ${BATCH_SIZE}"
echo "  Max records  : ${MAX_RECORDS}  (0 = all)"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Input file not found: ${INPUT}" >&2
  exit 1
fi

uv run python src/sample_embedding_generator.py \
  --model        "${MODEL}" \
  --input        "${INPUT}" \
  --output       "${OUTPUT}" \
  --dimensions   "${DIMENSIONS}" \
  --batch-size   "${BATCH_SIZE}" \
  --max-records  "${MAX_RECORDS}" \
  --service-url  "${SERVICE_URL}"
