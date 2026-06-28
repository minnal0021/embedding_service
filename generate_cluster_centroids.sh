#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
INPUT="embedding/sample_data_embedding.parquet"
OUTPUT="embedding/cluster_centroids.jsonl"
EMBEDDING_COL="qa_embedding"
DIMENSIONS=768
N_CLUSTERS=256
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Cluster Centroid Generator"
echo "════════════════════════════════════════════════════════════════"
echo "  Input         : ${INPUT}"
echo "  Output        : ${OUTPUT}"
echo "  Embedding col : ${EMBEDDING_COL}"
echo "  Dimensions    : ${DIMENSIONS}"
echo "  Clusters      : ${N_CLUSTERS}"
echo "  Seed          : ${SEED}"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Embedding file not found: ${INPUT}" >&2
  echo "Generate embeddings first with:  uv run python src/sample_embedding_generator.py" >&2
  exit 1
fi

uv run python src/cluster_centroid_generator.py \
  --input         "${INPUT}" \
  --output        "${OUTPUT}" \
  --embedding-col "${EMBEDDING_COL}" \
  --dimensions    "${DIMENSIONS}" \
  --n-clusters    "${N_CLUSTERS}" \
  --seed          "${SEED}"
