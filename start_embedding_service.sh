#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./start_embedding_service.sh [options]

Syncs dependencies, then starts the EmbeddingGemma-300M FastAPI service
(PyTorch/ROCm + sentence-transformers + uvicorn) in the background and waits
for it to become healthy.

Options:
  --host HOST        Bind address              (default: 0.0.0.0)
  --port PORT        Bind port                 (default: 8000)
  --model REPO       Hugging Face model repo   (default: unsloth/embeddinggemma-300m)
                     e.g. google/embeddinggemma-300m (gated: needs HF_TOKEN)
  --cache-dir DIR    Model cache               (default: ./model_cache)
  --device DEVICE    auto | gpu | cpu          (default: auto)
                     auto uses a supported AMD GPU (ROCm) when found, else the CPU
  --dtype DTYPE      bfloat16 | float32        (default: bfloat16 on GPU, float32 on CPU)
  -f, --foreground   Run in the foreground (Ctrl-C to stop) instead of detaching
  -h, --help         Show this help message

Environment overrides (used as defaults if the matching flag is omitted):
  HOST, PORT, EMBEDDING_MODEL, EMBEDDING_CACHE_PATH, EMBEDDING_DEVICE, EMBEDDING_DTYPE
  Also honoured by the service: EMBEDDING_BATCH_SIZE, EMBEDDING_GPU_INDEX,
  EMBEDDING_MAX_SEQ_LENGTH, HF_TOKEN
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults (env-overridable) ────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
MODEL="${EMBEDDING_MODEL:-unsloth/embeddinggemma-300m}"
CACHE_DIR="${EMBEDDING_CACHE_PATH:-${ROOT_DIR}/model_cache}"
DEVICE="${EMBEDDING_DEVICE:-auto}"
DTYPE="${EMBEDDING_DTYPE:-}"
FOREGROUND=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while (( "$#" )); do
  case "$1" in
    --host)       HOST="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --model)      MODEL="$2"; shift 2 ;;
    --cache-dir)  CACHE_DIR="$2"; shift 2 ;;
    --device)     DEVICE="$2"; shift 2 ;;
    --dtype)      DTYPE="$2"; shift 2 ;;
    -f|--foreground) FOREGROUND=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SERVICE_URL="http://${HOST}:${PORT}"
PID_FILE="${ROOT_DIR}/embedding_service.pid"
LOG_FILE="${ROOT_DIR}/embedding_service.log"

export EMBEDDING_MODEL="${MODEL}"
export EMBEDDING_CACHE_PATH="${CACHE_DIR}"
export EMBEDDING_DEVICE="${DEVICE}"
if [[ -n "${DTYPE}" ]]; then export EMBEDDING_DTYPE="${DTYPE}"; fi

# ── Preflight: uv + dependencies ──────────────────────────────────────────────
echo "Checking uv ..."
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed or not on PATH." >&2
  echo "Install it from https://docs.astral.sh/uv/ and re-run." >&2
  exit 1
fi
echo "  ✅ uv is available."

echo ""
echo "Syncing dependencies (uv sync) ..."
uv sync
echo "  ✅ Dependencies are in sync."

# ── Already running? ──────────────────────────────────────────────────────────
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo ""
  echo "ERROR: A service appears to be running (PID $(cat "${PID_FILE}"), see ${PID_FILE})." >&2
  echo "Stop it first:  kill \$(cat '${PID_FILE}')" >&2
  exit 1
fi

UVICORN_ARGS=(uvicorn server:app --app-dir "${ROOT_DIR}/src" --host "${HOST}" --port "${PORT}")

echo ""
echo "Starting embedding service ..."
echo "  Model      : ${MODEL}"
echo "  Cache dir  : ${CACHE_DIR}"
echo "  Device     : ${DEVICE}${DTYPE:+ (${DTYPE})}"
echo "  URL        : ${SERVICE_URL}"

# ── Foreground mode: hand the terminal to uvicorn ─────────────────────────────
if (( FOREGROUND )); then
  echo "  Mode       : foreground (Ctrl-C to stop)"
  echo ""
  exec uv run "${UVICORN_ARGS[@]}"
fi

# ── Background mode: detach, then wait for health ─────────────────────────────
echo "  Mode       : background (logs → ${LOG_FILE})"
nohup uv run "${UVICORN_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
SERVICE_PID=$!
echo "${SERVICE_PID}" >"${PID_FILE}"
echo "  PID        : ${SERVICE_PID}"

# Loading (and on first run, downloading ~1.2 GB of weights) can take a while.
echo ""
echo "Waiting for ${SERVICE_URL}/healthcheck ..."
retries=60
for ((w=1; w<=retries; w++)); do
  if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
    echo "ERROR: Service process exited during startup. Last log lines:" >&2
    tail -n 50 "${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi
  if curl -fsS "${SERVICE_URL}/healthcheck" >/dev/null 2>&1; then
    echo "  ✅ Service is ready (attempt ${w})."
    break
  fi
  echo "  Not ready yet (${w}/${retries})..."
  sleep 2
done
if (( w > retries )); then
  echo "ERROR: Health check failed after ${retries} attempts. Last log lines:" >&2
  tail -n 50 "${LOG_FILE}" >&2
  exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Embedding service is up!"
echo "  Query endpoint   : ${SERVICE_URL}/embedding/query"
echo "  Document endpoint: ${SERVICE_URL}/embedding/document"
echo "  Health endpoint  : ${SERVICE_URL}/healthcheck"
echo "  Model            : ${MODEL}"
echo "  Device           : $(curl -fsS "${SERVICE_URL}/healthcheck" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("device","?"))')"
echo "  Logs             : ${LOG_FILE}"
echo "  Stop             : kill \$(cat '${PID_FILE}')"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Quick smoke test ──────────────────────────────────────────────────────────
echo "Running quick smoke test..."
SMOKE_RESPONSE="$(curl -sS --connect-timeout 5 --max-time 60 \
  -X POST "${SERVICE_URL}/embedding/query" \
  -H "Content-Type: application/json" \
  -d '{"payloads": ["smoke test"]}' 2>/dev/null || true)"

if echo "${SMOKE_RESPONSE}" | grep -q '"embeddings"'; then
  python3 -c "
import json
data = json.loads('''${SMOKE_RESPONSE}''')
vecs = data['embeddings']
dims = len(vecs[0]) if vecs else 0
print(f'  Embeddings returned : {len(vecs)}')
print(f'  Dimensions          : {dims}')
print(f'  Status              : ✅ OK')
"
else
  echo "  ⚠️  Smoke test did not return embeddings — check ${LOG_FILE}."
  echo "  Response: ${SMOKE_RESPONSE}"
fi
echo ""
