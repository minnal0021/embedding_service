#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
INPUT="sample_data/eli5_question_answer.jsonl"
OUTPUT="embedding/sample_data_embedding.parquet"
DIMENSIONS=768            # EmbeddingGemma native width; MRL-truncated below this (0 = native 768)
BATCH_SIZE=32             # QA pairs per embedding call
MAX_RECORDS=25000        # max QA pairs to embed (0 = all)
CACHE_DIR="fastembed_cache"  # local model cache (EmbeddingService defaults to /app, unwritable off-container)
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# EmbeddingService reads FASTEMBED_CACHE_PATH; point it at the repo-local cache
# so model files download to a writable location instead of /app/fastembed_cache.
export FASTEMBED_CACHE_PATH="${FASTEMBED_CACHE_PATH:-${ROOT_DIR}/${CACHE_DIR}}"

echo "════════════════════════════════════════════════════════════════"
echo "  Sample Data Embedding Generator (ELI5 QA)"
echo "════════════════════════════════════════════════════════════════"
echo "  Input        : ${INPUT}"
echo "  Output       : ${OUTPUT}"
echo "  Dimensions   : ${DIMENSIONS}  (FastEmbed + EmbeddingGemma-300M, in-process)"
echo "  Batch size   : ${BATCH_SIZE}"
echo "  Max records  : ${MAX_RECORDS}  (0 = all)"
echo "  Cache dir    : ${FASTEMBED_CACHE_PATH}"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Input file not found: ${INPUT}" >&2
  exit 1
fi

uv run python src/sample_embedding_generator.py \
  --input        "${INPUT}" \
  --output       "${OUTPUT}" \
  --dimensions   "${DIMENSIONS}" \
  --batch-size   "${BATCH_SIZE}" \
  --max-records  "${MAX_RECORDS}"
