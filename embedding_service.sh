#!/usr/bin/env bash
set -euo pipefail

usage() {
  local detected
  detected="$(backend_candidates 2>/dev/null || echo cpu)"
  detected="${detected%% *}"
  cat <<EOF
Usage: ./embedding_service.sh <command> [options]

Runs the embedding service with Docker Compose: a FastAPI gateway (the only
published port) in front of one llama.cpp server container per model.

COMMANDS
  start            Find a llama.cpp image that can use this machine's GPU,
                   build the gateway image, download the models (first run
                   only), start the containers, wait until every model is
                   loaded, then smoke test each model
  stop             Stop and remove the containers
  restart          stop, then start again with the last start's settings;
                   start options given here override them
  status           Show the containers, loaded models and their device
  logs [SERVICE]   Follow the logs of all containers, or of one SERVICE:
                   gateway | llama-gemma | llama-qwen | model-fetch
  -h, --help       Show this help

START OPTIONS                        DEFAULT          ALLOWED VALUES
  --models LIST                      gemma,qwen       gemma, qwen, or gemma,qwen
      Models to load. Requests for any other model return 404.

  --qwen-quant QUANT                 Q4_K_M           Q4_K_M | Q5_0 | Q5_K_M |
                                                      Q6_K | Q8_0 | f16
      Qwen3-Embedding-8B build (case-insensitive). Higher = closer to the
      full-precision model, bigger download, more GPU memory. restart keeps
      the last start's build unless this is given.

        QUANT    download   GPU memory (approx.)  notes
        Q4_K_M   4.7 GB     ~10.6 GB              default; measured on Radeon 890M
        Q5_0     5.3 GB     ~11.2 GB
        Q5_K_M   5.4 GB     ~11.4 GB
        Q6_K     6.2 GB     ~12.2 GB
        Q8_0     8.1 GB     ~14 GB                near-lossless
        f16     15.1 GB     ~21 GB                full precision

      GPU memory includes ~6 GB of KV cache and compute buffers for Qwen's
      8192-token batch; Gemma adds ~0.5 GB. Under load it grows by ~1.8 GB
      more (measured: both models 11.4 GB after start, 13.1 GB after use).
      Vectors from different builds are close but not identical: don't mix
      them in one index.

  --backend NAME                     auto             auto | rocm-wsl | rocm |
                                                      cuda | vulkan | cpu
      llama.cpp image the models run in (auto on this machine: ${detected}).
      auto tries these in order and uses the first whose image llama.cpp can
      actually run on a GPU with (it asks llama.cpp --list-devices), so an
      image that can't see the GPU falls through to the next one:
        rocm-wsl  WSL2 with /dev/dxg + /opt/rocm/lib/librocdxg.so  (server-rocm)
        rocm      native Linux with /dev/kfd                         (server-rocm)
        cuda      nvidia-smi works + Docker's nvidia runtime          (server-cuda)
        vulkan    /dev/dri/renderD* on native Linux                  (server-vulkan)
        cpu       anything else                                      (server)

  --host HOST                        0.0.0.0          any local address
      Address the gateway is published on (0.0.0.0 = reachable from the LAN).

  --port PORT                        8001             1-65535

  --cache-dir DIR                    ./model_cache    any directory
      Where model files are downloaded; mounted into the containers.

  --pull                             off              (flag)
      Pull the latest llama.cpp image before starting (missing images are
      always pulled).

  -f, --foreground                   off              (flag)
      Stay attached to the container logs; Ctrl-C stops the service.

ENVIRONMENT (used when the matching option is not given)
  EMBEDDING_MODELS      --models          EMBEDDING_BACKEND  --backend
  EMBEDDING_QWEN_QUANT  --qwen-quant      HOST / PORT        --host / --port
  EMBEDDING_CACHE_PATH  --cache-dir
  EMBEDDING_QWEN_GGUF   exact Qwen GGUF file name (overrides --qwen-quant's default)
  EMBEDDING_GEMMA_GGUF  exact Gemma GGUF file name (only Q8_0 is published)
  HF_TOKEN              Hugging Face token for downloads (optional)
  STARTUP_TIMEOUT       seconds to wait for the models to load (default: 1800)

  Throughput tuning (kept by restart):
  EMBEDDING_MAX_BATCH_TEXTS  max texts the gateway sends to a model in one call,
                             batching concurrent requests (default: 256)
  EMBEDDING_BATCH_WAIT_MS    how long a busy gateway waits to fill a batch
                             (default: 2; an idle one never waits)
  LLAMA_PARALLEL             sequences each model server (gemma, qwen) runs at
                             once (default: 4)

FILES
  embedding_service.env   settings of the last start, read by stop / status /
                          logs / restart (gitignored)

EXAMPLES
  ./embedding_service.sh start                          # gemma + qwen Q4_K_M, auto GPU
  ./embedding_service.sh start --qwen-quant Q8_0        # higher-precision Qwen
  ./embedding_service.sh start --qwen-quant f16         # full-precision Qwen
  ./embedding_service.sh start --models gemma --backend cpu
  ./embedding_service.sh restart --qwen-quant Q4_K_M    # back to the default
  ./embedding_service.sh logs llama-qwen
EOF
}

# Qwen GGUF builds published in Qwen/Qwen3-Embedding-8B-GGUF.
QWEN_QUANTS=(Q4_K_M Q5_0 Q5_K_M Q6_K Q8_0 f16)

# Canonical spelling of a Qwen quant name (case-insensitive), or fail.
qwen_quant() {
  local q
  for q in "${QWEN_QUANTS[@]}"; do
    if [[ "${1,,}" == "${q,,}" ]]; then
      echo "${q}"
      return 0
    fi
  done
  echo "ERROR: unknown --qwen-quant '$1' (expected ${QWEN_QUANTS[*]})." >&2
  return 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Settings of the last start (compose files, profiles, ports, models).
STATE_FILE="${ROOT_DIR}/embedding_service.env"

compose() {
  docker compose --project-directory "${ROOT_DIR}" "$@"
}

# Export the saved settings so compose acts on the started containers. With
# none saved, enable every model profile so stop still finds everything.
load_state() {
  if [[ -f "${STATE_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${STATE_FILE}"
    set +a
  else
    export COMPOSE_PROFILES="gemma,qwen"
  fi
}

gateway_running() {
  [[ -n "$(compose ps --status running -q gateway 2>/dev/null)" ]]
}

# Print each loaded model's file and device from the healthcheck JSON on stdin.
print_models() {
  python3 -c "
import json, sys
for key, m in json.load(sys.stdin)['models'].items():
    print(f'  {key:6} : {m[\"file\"]} on {m[\"device\"]} ({m[\"backend\"]}), '
          f'native {m[\"dimension\"]} dims, default {m[\"default_dimension\"]}, {m[\"status\"]}')
"
}

# GPU backends this host looks able to run, most specific first, then cpu.
# Each is verified with probe_gpu before use.
backend_candidates() {
  local wsl=0 c=()
  grep -qi microsoft /proc/version 2>/dev/null && wsl=1
  (( wsl )) && [[ -e /dev/dxg && -e /opt/rocm/lib/librocdxg.so ]] && c+=(rocm-wsl)
  [[ -e /dev/kfd ]] && c+=(rocm)
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
    && docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia' && c+=(cuda)
  (( ! wsl )) && compgen -G '/dev/dri/renderD*' >/dev/null && c+=(vulkan)
  echo "${c[*]} cpu"
}

# Compose -f arguments for a backend: the base file plus its GPU overlay.
compose_files() {
  echo "-f ${ROOT_DIR}/docker-compose.yml"
  [[ "$1" == cpu ]] || echo "-f ${ROOT_DIR}/docker/compose.$1.yml"
}

# Ask llama.cpp, inside the backend's image with its devices, which GPU it
# can use. Prints "<name>|<total MiB>|<free MiB>" for the first device, or
# nothing when llama.cpp finds none (it would silently run on the CPU).
probe_gpu() {
  local backend="$1" pull="$2" image
  # shellcheck disable=SC2046
  image="$(COMPOSE_PROFILES=gemma docker compose --project-directory "${ROOT_DIR}" \
    $(compose_files "${backend}") config --images 2>/dev/null | grep llama.cpp | head -n 1)"
  [[ -n "${image}" ]] || return 0
  if (( pull )) || ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "  Pulling ${image} ..." >&2
    docker pull -q "${image}" >/dev/null || return 0
  fi
  # shellcheck disable=SC2046
  COMPOSE_PROFILES=gemma docker compose --project-directory "${ROOT_DIR}" \
    $(compose_files "${backend}") run --rm --no-deps -T llama-gemma --list-devices 2>/dev/null \
    | sed -nE 's/^ *[A-Za-z]+[0-9]+: (.*) \(([0-9]+) MiB, ([0-9]+) MiB free\)$/\1|\2|\3/p' \
    | head -n 1 || true
}

# GPU memory (MiB) Qwen needs per build, with gemma loaded too: weights plus
# ~6 GB KV cache and compute buffers for its 8192-token batch. Q4_K_M was
# measured on a Radeon 890M; the rest scale by file size.
qwen_need_mib() {
  case "$1" in
    Q4_K_M) echo 11000 ;; Q5_0) echo 11600 ;; Q5_K_M) echo 11700 ;;
    Q6_K) echo 12500 ;; Q8_0) echo 14300 ;; f16) echo 21000 ;;
  esac
}

# Fail early with a hint when the host lacks what a backend's overlay mounts.
check_backend() {
  local missing=()
  case "$1" in
    rocm-wsl)
      local f
      for f in /dev/dxg /usr/lib/wsl/lib/libdxcore.so /opt/rocm/lib/librocdxg.so \
          /opt/rocm/share/rocdxg/dids.conf; do
        [[ -e "${f}" ]] || missing+=("${f}")
      done
      (( ${#missing[@]} )) && echo "Install librocdxg (https://github.com/ROCm/librocdxg) and a recent AMD Windows driver; see docs/rocm_wsl_spike.md." >&2
      ;;
    rocm)
      [[ -e /dev/kfd ]] || missing+=(/dev/kfd)
      [[ -e /dev/dri ]] || missing+=(/dev/dri)
      ;;
    vulkan)
      [[ -e /dev/dri ]] || missing+=(/dev/dri)
      ;;
  esac
  if (( ${#missing[@]} )); then
    echo "ERROR: backend '$1' needs: ${missing[*]}" >&2
    exit 1
  fi
}

# ── start ─────────────────────────────────────────────────────────────────────
cmd_start() {
  # Defaults (env-overridable)
  local HOST="${HOST:-0.0.0.0}"
  local PORT="${PORT:-8001}"
  local MODELS="${EMBEDDING_MODELS:-gemma,qwen}"
  local GEMMA_FILE="${EMBEDDING_GEMMA_GGUF:-}"
  local QWEN_FILE="${EMBEDDING_QWEN_GGUF:-}"
  local QWEN_QUANT="${EMBEDDING_QWEN_QUANT:-}"
  local CACHE_DIR="${EMBEDDING_CACHE_PATH:-${ROOT_DIR}/model_cache}"
  local BACKEND="${EMBEDDING_BACKEND:-auto}"
  local STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
  local PULL=0
  local FOREGROUND=0

  while (( "$#" )); do
    case "$1" in
      --host)        HOST="$2"; shift 2 ;;
      --port)        PORT="$2"; shift 2 ;;
      --models)      MODELS="$2"; shift 2 ;;
      --qwen-quant)  QWEN_QUANT="$2"; shift 2 ;;
      --cache-dir)   CACHE_DIR="$2"; shift 2 ;;
      --backend)     BACKEND="$2"; shift 2 ;;
      --pull)        PULL=1; shift ;;
      -f|--foreground) FOREGROUND=1; shift ;;
      -h|--help)     usage; exit 0 ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  # Validate the model list up front rather than failing inside a container.
  local m
  for m in ${MODELS//,/ }; do
    case "${m}" in
      gemma|qwen) ;;
      *) echo "ERROR: unknown model '${m}' in --models (expected gemma, qwen)." >&2; exit 1 ;;
    esac
  done

  # Qwen build: --qwen-quant / EMBEDDING_QWEN_QUANT, else an exact
  # EMBEDDING_QWEN_GGUF (restart keeps the last one this way), else Q4_K_M.
  if [[ -n "${QWEN_QUANT}" ]]; then
    QWEN_QUANT="$(qwen_quant "${QWEN_QUANT}")" || exit 1
    QWEN_FILE="Qwen3-Embedding-8B-${QWEN_QUANT}.gguf"
  elif [[ -z "${QWEN_FILE}" ]]; then
    QWEN_QUANT=Q4_K_M
    QWEN_FILE="Qwen3-Embedding-8B-${QWEN_QUANT}.gguf"
  fi

  case "${BACKEND}" in
    auto|rocm-wsl|rocm|cuda|vulkan|cpu) ;;
    *) echo "ERROR: unknown backend '${BACKEND}' (expected auto, rocm-wsl, rocm, cuda, vulkan or cpu)." >&2; exit 1 ;;
  esac

  # Preflight: Docker + Compose v2
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "ERROR: Docker on macOS has no GPU access; the Docker setup is for Linux/WSL2." >&2
    exit 1
  fi
  if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: Docker with the Compose plugin ('docker compose') is required." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Can't reach the Docker daemon (is it running? are you in the docker group?)." >&2
    exit 1
  fi

  # restart: stop only now that the options are known to be valid.
  if (( ${RESTART:-0} )); then
    cmd_stop
    echo ""
  fi

  # Already running? (checked first, before any slow setup)
  load_state
  if gateway_running; then
    echo "ERROR: The service is already running." >&2
    echo "Use:  ./embedding_service.sh restart" >&2
    exit 1
  fi

  # Backend: try each candidate until llama.cpp in its image sees a GPU.
  local candidates gpu="" b
  case "${BACKEND}" in
    auto) candidates="$(backend_candidates)" ;;
    *) check_backend "${BACKEND}"; candidates="${BACKEND}" ;;
  esac
  echo "Detecting GPU (candidates: ${candidates}) ..."
  for b in ${candidates}; do
    BACKEND="${b}"
    [[ "${b}" == cpu ]] && break
    gpu="$(probe_gpu "${b}" "${PULL}")"
    [[ -n "${gpu}" ]] && break
    echo "  ⚠️  ${b}: llama.cpp found no usable GPU in its image."
    if [[ "${candidates}" == "${b}" ]]; then
      echo "ERROR: --backend ${b} can't use a GPU here; the models would run on the CPU." >&2
      echo "Check the GPU driver, or start with --backend auto / cpu." >&2
      exit 1
    fi
  done
  local GPU_NAME="" GPU_TOTAL=0
  if [[ -n "${gpu}" ]]; then
    IFS='|' read -r GPU_NAME GPU_TOTAL _ <<<"${gpu}"
    echo "  ✅ ${BACKEND}: ${GPU_NAME} (${GPU_TOTAL} MiB)"
  else
    echo "  Using the CPU."
  fi

  # Heads-up only: the chosen build looks bigger than the whole GPU.
  if [[ -n "${QWEN_QUANT}" && ",${MODELS}," == *,qwen,* ]] && (( GPU_TOTAL > 0 )) \
      && (( $(qwen_need_mib "${QWEN_QUANT}") > GPU_TOTAL )); then
    echo "  ⚠️  Qwen ${QWEN_QUANT} needs ~$(qwen_need_mib "${QWEN_QUANT}") MiB of GPU memory; this GPU has ${GPU_TOTAL} MiB."
  fi

  local COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
  [[ "${BACKEND}" != "cpu" ]] && COMPOSE_FILE+=":${ROOT_DIR}/docker/compose.${BACKEND}.yml"
  # rocm-wsl runs the same ROCm image; report it as rocm.
  local LLAMA_BACKEND="${BACKEND%-wsl}"

  mkdir -p "${CACHE_DIR}"
  CACHE_DIR="$(cd "${CACHE_DIR}" && pwd)"

  cat >"${STATE_FILE}" <<EOF
# Written by embedding_service.sh start; read by stop/status/logs.
COMPOSE_FILE=${COMPOSE_FILE}
COMPOSE_PROFILES=${MODELS}
HOST=${HOST}
PORT=${PORT}
EMBEDDING_MODELS=${MODELS}
EMBEDDING_GEMMA_GGUF=${GEMMA_FILE}
EMBEDDING_QWEN_GGUF=${QWEN_FILE}
EMBEDDING_CACHE_PATH=${CACHE_DIR}
LLAMA_BACKEND=${LLAMA_BACKEND}
LOCAL_UID=$(id -u)
LOCAL_GID=$(id -g)
EOF
  # Tuning knobs, kept only when set so the compose defaults apply otherwise.
  local var
  for var in EMBEDDING_MAX_BATCH_TEXTS EMBEDDING_BATCH_WAIT_MS LLAMA_PARALLEL; do
    if [[ -n "${!var:-}" ]]; then
      echo "${var}=${!var}" >>"${STATE_FILE}"
    fi
  done
  load_state

  local SERVICE_URL="http://${HOST}:${PORT}"
  # Health checks go to loopback when bound to all interfaces.
  local LOCAL_URL="${SERVICE_URL}"
  [[ "${HOST}" == "0.0.0.0" ]] && LOCAL_URL="http://127.0.0.1:${PORT}"

  echo "Starting embedding service ..."
  echo "  Models     : ${MODELS}"
  [[ ",${MODELS}," == *,qwen,* ]] \
    && echo "  Qwen build : ${QWEN_FILE}"
  echo "  Cache dir  : ${CACHE_DIR}"
  echo "  Backend    : ${BACKEND} ($(compose config --images 2>/dev/null | grep llama.cpp | sort -u | tr '\n' ' '))"
  [[ -n "${GPU_NAME}" ]] && echo "  GPU        : ${GPU_NAME} (${GPU_TOTAL} MiB)"
  echo "  URL        : ${SERVICE_URL}"
  echo ""

  if (( PULL )) && [[ "${BACKEND}" == cpu ]]; then
    echo "Pulling llama.cpp image ..."
    compose pull --ignore-buildable
    echo ""
  fi

  echo "Building gateway image ..."
  compose build --quiet
  echo "  ✅ Gateway image is up to date."

  # Run the download attached so first-run progress (~5–8 GB for Qwen) shows.
  echo ""
  echo "Fetching models (downloads on first run only) ..."
  compose run --rm --no-deps model-fetch

  echo ""
  if (( FOREGROUND )); then
    echo "Starting containers in the foreground (Ctrl-C to stop) ..."
    echo ""
    exec docker compose --project-directory "${ROOT_DIR}" up --abort-on-container-failure
  fi
  echo "Starting containers ..."
  compose --progress quiet up -d --remove-orphans

  # Loading Qwen takes ~30s on a GPU, longer on the CPU.
  echo ""
  echo "Waiting up to ${STARTUP_TIMEOUT}s for ${LOCAL_URL}/healthcheck ..."
  local start=${SECONDS} attempt=0 svc
  until curl -fsS "${LOCAL_URL}/healthcheck" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    for svc in gateway ${MODELS//,/ }; do
      [[ "${svc}" == gateway ]] || svc="llama-${svc}"
      if [[ -z "$(compose ps --status running -q "${svc}" 2>/dev/null)" ]] \
          || [[ "$(compose ps --format '{{.Status}}' "${svc}" 2>/dev/null)" == *Restarting* ]]; then
        echo "ERROR: ${svc} is not running. Last log lines:" >&2
        compose logs --tail 50 "${svc}" >&2
        echo "Stop the rest with: ./embedding_service.sh stop" >&2
        exit 1
      fi
    done
    if (( SECONDS - start >= STARTUP_TIMEOUT )); then
      echo "ERROR: Health check failed after ${STARTUP_TIMEOUT}s." >&2
      curl -sS "${LOCAL_URL}/healthcheck" >&2 || true
      echo "See: ./embedding_service.sh logs" >&2
      exit 1
    fi
    # Report progress every ~30s rather than every poll.
    if (( attempt % 15 == 0 )); then
      echo "  Not ready yet ($((SECONDS - start))s elapsed)..."
    fi
    sleep 2
  done
  echo "  ✅ Service is ready ($((SECONDS - start))s)."

  # llama.cpp falls back to the CPU silently when it can't use the GPU.
  if [[ "${BACKEND}" != "cpu" ]]; then
    for m in ${MODELS//,/ }; do
      if compose logs "llama-${m}" 2>&1 | grep -qi 'no usable GPU'; then
        echo "  ⚠️  llama-${m} found no usable GPU and is running on the CPU."
      fi
    done
  fi

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  Embedding service is up!"
  for m in ${MODELS//,/ }; do
    echo "  ${m} query    : ${SERVICE_URL}/embedding/${m}/query"
    echo "  ${m} document : ${SERVICE_URL}/embedding/${m}/document"
  done
  echo "  Health endpoint : ${SERVICE_URL}/healthcheck"
  echo "  Logs            : ./embedding_service.sh logs [service]"
  echo "  Stop            : ./embedding_service.sh stop"
  echo "════════════════════════════════════════════════════════════════"
  echo ""

  # Quick smoke test
  echo "Running quick smoke test..."
  curl -sS "${LOCAL_URL}/healthcheck" | print_models
  local resp
  for m in ${MODELS//,/ }; do
    resp="$(curl -sS --connect-timeout 5 --max-time 120 \
      -X POST "${LOCAL_URL}/embedding/${m}/query" \
      -H "Content-Type: application/json" \
      -d '{"payloads": ["smoke test"]}' 2>/dev/null || true)"
    if echo "${resp}" | grep -q '"embeddings"'; then
      echo "${resp}" | python3 -c "
import json, sys
vecs = json.load(sys.stdin)['embeddings']
print(f'  ${m}: {len(vecs)} embedding(s), {len(vecs[0]) if vecs else 0} dims ✅')
"
    else
      echo "  ⚠️  ${m}: smoke test did not return embeddings — see ./embedding_service.sh logs."
      echo "  Response: ${resp}"
    fi
  done
  echo ""
}

# ── stop ──────────────────────────────────────────────────────────────────────
cmd_stop() {
  load_state
  if [[ -z "$(compose ps -a -q 2>/dev/null)" ]]; then
    echo "Embedding service is not running."
    return 0
  fi
  echo "Stopping embedding service ..."
  compose --progress quiet down --remove-orphans
  echo "  ✅ Embedding service stopped."
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
  load_state
  if ! gateway_running; then
    echo "Embedding service: stopped"
    return 3
  fi
  local url="http://127.0.0.1:${PORT:-8001}" health
  [[ "${HOST:-0.0.0.0}" != "0.0.0.0" ]] && url="http://${HOST}:${PORT:-8001}"
  echo "Embedding service: running at ${url}"
  compose ps --format 'table {{.Service}}\t{{.Image}}\t{{.Status}}'
  echo ""
  # 503 while a model is still loading; print its JSON either way.
  health="$(curl -sS --max-time 10 "${url}/healthcheck" 2>/dev/null || true)"
  if echo "${health}" | grep -q '"models"'; then
    echo "${health}" | print_models
  else
    echo "  ⚠️  The gateway is not answering health checks — see ./embedding_service.sh logs."
  fi
}

# ── logs ──────────────────────────────────────────────────────────────────────
cmd_logs() {
  load_state
  compose logs -f --tail 100 "$@"
}

# ── dispatch ──────────────────────────────────────────────────────────────────
COMMAND="${1:-}"
[[ $# -gt 0 ]] && shift
case "${COMMAND}" in
  start)    cmd_start "$@" ;;
  stop)     cmd_stop ;;
  # Saved settings become the defaults; options given here override them.
  restart)  load_state; RESTART=1 cmd_start "$@" ;;
  status)   cmd_status ;;
  logs)     cmd_logs "$@" ;;
  -h|--help) usage ;;
  "")
    usage >&2
    exit 1
    ;;
  *)
    echo "Unknown command: ${COMMAND}" >&2
    usage >&2
    exit 1
    ;;
esac
