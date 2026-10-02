#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./generate_cluster_centroids.sh MODEL

Clusters MODEL's (gemma | qwen) sample embeddings with K-means and writes the
centroids to embedding/MODEL/cluster_centroids.jsonl. Generate the embeddings
first with ./generate_sample_embeddings.sh MODEL.

Environment overrides: DIMENSIONS, N_CLUSTERS, SEED
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
INPUT="embedding/${MODEL}/sample_data_embedding.parquet"
OUTPUT="embedding/${MODEL}/cluster_centroids.jsonl"
EMBEDDING_COL="qa_embedding"
DIMENSIONS="${DIMENSIONS:-768}" # 0 = stored width as-is
N_CLUSTERS="${N_CLUSTERS:-256}"
SEED="${SEED:-42}"
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Cluster Centroid Generator"
echo "════════════════════════════════════════════════════════════════"
echo "  Model         : ${MODEL}"
echo "  Input         : ${INPUT}"
echo "  Output        : ${OUTPUT}"
echo "  Embedding col : ${EMBEDDING_COL}"
echo "  Dimensions    : ${DIMENSIONS}  (0 = stored width)"
echo "  Clusters      : ${N_CLUSTERS}"
echo "  Seed          : ${SEED}"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Embedding file not found: ${INPUT}" >&2
  echo "Generate embeddings first with:  ./generate_sample_embeddings.sh ${MODEL}" >&2
  exit 1
fi

uv run python src/cluster_centroid_generator.py \
  --model         "${MODEL}" \
  --input         "${INPUT}" \
  --output        "${OUTPUT}" \
  --embedding-col "${EMBEDDING_COL}" \
  --dimensions    "${DIMENSIONS}" \
  --n-clusters    "${N_CLUSTERS}" \
  --seed          "${SEED}"
