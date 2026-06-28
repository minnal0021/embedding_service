#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./stop_embedding_service.sh [options]

Stops the background EmbeddingGemma-300M service started by
./start_embedding_service.sh (identified via its PID file).

Options:
  --pid-file FILE   PID file to read    (default: ./embedding_service.pid)
  -h, --help        Show this help message
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT_DIR}/embedding_service.pid"

while (( "$#" )); do
  case "$1" in
    --pid-file) PID_FILE="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No PID file at ${PID_FILE} — service does not appear to be running."
  exit 0
fi

PID="$(cat "${PID_FILE}")"

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "Process ${PID} is not running (stale PID file). Cleaning up."
  rm -f "${PID_FILE}"
  exit 0
fi

echo "Stopping embedding service (PID ${PID}) ..."
kill "${PID}"

# Wait for a graceful exit, then force-kill if it overstays.
for ((w=1; w<=15; w++)); do
  if ! kill -0 "${PID}" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "${PID}" 2>/dev/null; then
  echo "  Did not stop gracefully; sending SIGKILL ..."
  kill -9 "${PID}" 2>/dev/null || true
fi

rm -f "${PID_FILE}"
echo "  ✅ Embedding service stopped."
