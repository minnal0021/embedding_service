#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./generate_sample_embeddings.sh MODEL

Embeds the ELI5 sample QA pairs with MODEL (gemma | qwen), in-process (no
running service needed), and writes embedding/MODEL/sample_data_embedding.parquet
for ./generate_cluster_centroids.sh MODEL.

Environment overrides: DIMENSIONS, BATCH_SIZE, MAX_RECORDS, EMBEDDING_CACHE_PATH,
EMBEDDING_DEVICE, EMBEDDING_GEMMA_GGUF, EMBEDDING_QWEN_GGUF
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
CACHE_DIR="model_cache"              # local model cache
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# EmbeddingService reads EMBEDDING_CACHE_PATH; share the service's model cache.
export EMBEDDING_CACHE_PATH="${EMBEDDING_CACHE_PATH:-${ROOT_DIR}/${CACHE_DIR}}"

# Detect hardware + build the llama.cpp backend on first run (no-op after).
"${ROOT_DIR}/install_backend.sh" --if-needed
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  Sample Data Embedding Generator (ELI5 QA)"
echo "════════════════════════════════════════════════════════════════"
echo "  Model        : ${MODEL}  (llama.cpp, in-process)"
echo "  Input        : ${INPUT}"
echo "  Output       : ${OUTPUT}"
echo "  Dimensions   : ${DIMENSIONS}  (0 = native)"
echo "  Batch size   : ${BATCH_SIZE}"
echo "  Max records  : ${MAX_RECORDS}  (0 = all)"
echo "  Cache dir    : ${EMBEDDING_CACHE_PATH}"
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
  --max-records  "${MAX_RECORDS}"
