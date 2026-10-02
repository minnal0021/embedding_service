#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./install_backend.sh [options]

Detects this machine's OS, CPU and GPU, then builds llama-cpp-python against
the best available llama.cpp backend so in-process embedding (EmbeddingService
and the offline pipeline) can offload models to the GPU. llama.cpp picks its
backend at compile time, so this is a build step. The HTTP service doesn't use
this: it runs prebuilt llama.cpp Docker images (see embedding_service.sh).

generate_sample_embeddings.sh runs this automatically (with --if-needed), so
you rarely need to run it by hand.

Options:
  --backend NAME   auto | metal | cuda | hip | vulkan | cpu   (default: auto)
                   auto picks, in order:
                     macOS on Apple Silicon             → metal
                     NVIDIA GPU + CUDA toolkit (nvcc)   → cuda
                     AMD GPU in rocminfo + ROCm clang   → hip
                     Vulkan GPU + glslc + Vulkan headers → vulkan
                     otherwise                          → cpu
                   If an auto-detected GPU build fails, it falls back to cpu.
  --if-needed      Do nothing if setup already completed for the installed
                   llama-cpp-python version (see .backend_setup_complete)
  --no-sync        Don't run 'uv sync' first (the caller already did)
  -h, --help       Show this help message

On success it writes .backend_setup_complete (gitignored) recording the
backend, the llama-cpp-python version it was built for, and the detected
hardware. Setup reruns when that file is missing, when the installed
llama-cpp-python version changes (e.g. a lock upgrade replaced the build), or
when run without --if-needed.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER="${ROOT_DIR}/.backend_setup_complete"
BACKEND="auto"
IF_NEEDED=0
SYNC=1

while (( "$#" )); do
  case "$1" in
    --backend)    BACKEND="$2"; shift 2 ;;
    --if-needed)  IF_NEEDED=1; shift ;;
    --no-sync)    SYNC=0; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${BACKEND}" in
  auto|metal|cuda|hip|vulkan|cpu) ;;
  *)
    echo "ERROR: unknown backend '${BACKEND}' (expected auto, metal, cuda, hip, vulkan or cpu)." >&2
    exit 1
    ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed or not on PATH." >&2
  echo "Install it from https://docs.astral.sh/uv/ and re-run." >&2
  exit 1
fi

cd "${ROOT_DIR}"
# Make sure the environment exists (and is current) before checking/building.
if (( SYNC )); then
  uv sync --quiet
fi

# Read one key from the marker file (empty if absent).
marker_get() {
  [[ -f "${MARKER}" ]] || return 0
  sed -n "s/^$1=//p" "${MARKER}" | head -n 1
}

installed_version() {
  uv run --no-sync python -c \
    'import importlib.metadata as m; print(m.version("llama-cpp-python"))' 2>/dev/null || true
}

# ── Already set up? ───────────────────────────────────────────────────────────
VERSION="$(installed_version)"
if (( IF_NEEDED )) && [[ -f "${MARKER}" ]]; then
  if [[ -n "${VERSION}" && "$(marker_get llama_cpp_python)" == "${VERSION}" ]]; then
    echo "  ✅ llama.cpp backend: $(marker_get backend) (set up $(marker_get completed_at); skipping)."
    exit 0
  fi
  echo "llama-cpp-python changed ($(marker_get llama_cpp_python) → ${VERSION:-missing}); re-running backend setup."
fi

# ── Detect platform & hardware ───────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
ENV_KIND="native"
if [[ "${OS}" == "Linux" ]] && grep -qi microsoft /proc/version 2>/dev/null; then
  ENV_KIND="wsl2"
fi

if [[ "${OS}" == "Darwin" ]]; then
  CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
else
  CPU_MODEL="$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -n 1)"
  CPU_MODEL="${CPU_MODEL:-unknown}"
  CPU_CORES="$(nproc 2>/dev/null || echo '?')"
fi

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
NVIDIA_GPU=""
AMD_GFX=""
VULKAN_GPU=""
APPLE_GPU=""

if [[ "${OS}" == "Darwin" ]]; then
  APPLE_GPU="$(system_profiler SPDisplaysDataType 2>/dev/null \
    | sed -n 's/^[[:space:]]*Chipset Model: //p' | head -n 1)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_GPU="$(nvidia-smi -L 2>/dev/null | head -n 1 | sed 's/ (UUID.*//')"
fi
if command -v rocminfo >/dev/null 2>&1; then
  # Unique gfx targets of real GPU agents (gfx000 is the CPU agent).
  AMD_GFX="$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | grep -v '^gfx000$' \
    | sort -u | tr '\n' ';' | sed 's/;$//')"
fi
if command -v vulkaninfo >/dev/null 2>&1; then
  # Hardware devices only — skip software rasterisers such as llvmpipe.
  VULKAN_GPU="$(vulkaninfo --summary 2>/dev/null \
    | grep -B 6 -E 'PHYSICAL_DEVICE_TYPE_(INTEGRATED|DISCRETE)_GPU' \
    | sed -n 's/^[[:space:]]*deviceName[[:space:]]*= //p' | head -n 1)"
  # vulkaninfo prints deviceName after deviceType on some versions.
  if [[ -z "${VULKAN_GPU}" ]]; then
    VULKAN_GPU="$(vulkaninfo --summary 2>/dev/null \
      | grep -A 6 -E 'PHYSICAL_DEVICE_TYPE_(INTEGRATED|DISCRETE)_GPU' \
      | sed -n 's/^[[:space:]]*deviceName[[:space:]]*= //p' | head -n 1)"
  fi
fi

# Toolchains each GPU backend needs to compile.
has_cuda_toolchain()   { command -v nvcc >/dev/null 2>&1; }
has_hip_toolchain()    { [[ -x "${ROCM_PATH}/llvm/bin/clang" ]]; }
has_vulkan_toolchain() {
  command -v glslc >/dev/null 2>&1 || return 1
  [[ -f /usr/include/vulkan/vulkan.h || -f /usr/local/include/vulkan/vulkan.h ]] \
    || pkg-config --exists vulkan 2>/dev/null
}

echo "════════════════════════════════════════════════════════════════"
echo "  llama.cpp backend setup"
echo "════════════════════════════════════════════════════════════════"
echo "  OS / arch   : ${OS} ${ARCH} (${ENV_KIND})"
echo "  CPU         : ${CPU_MODEL} (${CPU_CORES} threads)"
[[ -n "${APPLE_GPU}" ]]  && echo "  Apple GPU   : ${APPLE_GPU}"
[[ -n "${NVIDIA_GPU}" ]] && echo "  NVIDIA GPU  : ${NVIDIA_GPU}"
[[ -n "${AMD_GFX}" ]]    && echo "  AMD (ROCm)  : ${AMD_GFX}"
[[ -n "${VULKAN_GPU}" ]] && echo "  Vulkan GPU  : ${VULKAN_GPU}"
if [[ -z "${APPLE_GPU}${NVIDIA_GPU}${AMD_GFX}${VULKAN_GPU}" ]]; then
  # No GPU runtime answered; name any hardware we can still see, as a hint.
  PCI_GPU=""
  if command -v lspci >/dev/null 2>&1; then
    PCI_GPU="$(lspci 2>/dev/null | grep -iE 'vga|3d controller|display' \
      | head -n 1 | sed 's/^[^:]*: //')"
  fi
  if [[ -n "${PCI_GPU}" ]]; then
    echo "  GPU         : ${PCI_GPU} (no ROCm/CUDA/Vulkan runtime found)"
  elif [[ "${ENV_KIND}" == "wsl2" && -e /dev/dxg ]]; then
    echo "  GPU         : WSL2 GPU passthrough (/dev/dxg) present, but no ROCm/CUDA/Vulkan runtime found"
  else
    echo "  GPU         : none detected"
  fi
fi

# ── Choose backend ────────────────────────────────────────────────────────────
AUTO=0
NOTES=()
if [[ "${BACKEND}" == "auto" ]]; then
  AUTO=1
  if [[ "${OS}" == "Darwin" && "${ARCH}" == "arm64" ]]; then
    BACKEND="metal"
  elif [[ -n "${NVIDIA_GPU}" ]] && has_cuda_toolchain; then
    BACKEND="cuda"
  elif [[ -n "${AMD_GFX}" ]] && has_hip_toolchain; then
    BACKEND="hip"
  elif [[ -n "${VULKAN_GPU}" ]] && has_vulkan_toolchain; then
    BACKEND="vulkan"
  else
    BACKEND="cpu"
  fi
  # Explain GPUs we saw but couldn't build for.
  if [[ "${BACKEND}" == "cpu" ]]; then
    [[ -n "${NVIDIA_GPU}" ]] && ! has_cuda_toolchain \
      && NOTES+=("NVIDIA GPU found but no CUDA toolkit (nvcc) — install it for a cuda build.")
    [[ -n "${AMD_GFX}" ]] && ! has_hip_toolchain \
      && NOTES+=("AMD GPU found but no ROCm clang at ${ROCM_PATH}/llvm/bin — install ROCm (or set ROCM_PATH).")
    [[ -n "${VULKAN_GPU}" ]] && ! has_vulkan_toolchain \
      && NOTES+=("Vulkan GPU found but glslc / Vulkan headers are missing (e.g. apt install libvulkan-dev glslc).")
    if [[ -z "${NVIDIA_GPU}${AMD_GFX}${VULKAN_GPU}" && "${OS}" == "Linux" ]] \
        && [[ -n "${PCI_GPU:-}" || ( "${ENV_KIND}" == "wsl2" && -e /dev/dxg ) ]]; then
      NOTES+=("For GPU use, install ROCm (AMD), the CUDA toolkit (NVIDIA) or Vulkan (libvulkan-dev, glslc, GPU Vulkan driver), then start with --install-backend.")
    fi
    if [[ "${OS}" == "Darwin" ]]; then
      NOTES+=("Intel Mac: Metal isn't used by llama.cpp here; building for CPU.")
    fi
  fi
fi

cmake_args_for() {
  case "$1" in
    metal)  echo "-DGGML_METAL=on" ;;
    cuda)   echo "-DGGML_CUDA=on" ;;
    hip)    echo "-DGGML_HIP=on${AMD_GFX:+ -DAMDGPU_TARGETS=${AMD_GFX}}" ;;
    vulkan) echo "-DGGML_VULKAN=on" ;;
    cpu)    echo "-DGGML_NATIVE=on" ;;
  esac
}

build() {
  local backend="$1" args
  args="$(cmake_args_for "${backend}")"
  echo ""
  echo "Building llama-cpp-python for '${backend}' (CMAKE_ARGS: ${args})"
  echo "  This compiles llama.cpp and can take several minutes ..."
  if [[ "${backend}" == "hip" ]]; then
    # HIP builds need ROCm's clang as the compiler.
    CC="${ROCM_PATH}/llvm/bin/clang" CXX="${ROCM_PATH}/llvm/bin/clang++" \
      CMAKE_ARGS="${args}" uv pip install --reinstall --no-cache llama-cpp-python
  else
    CMAKE_ARGS="${args}" uv pip install --reinstall --no-cache llama-cpp-python
  fi
}

echo "  Backend     : ${BACKEND}$( (( AUTO )) && echo ' (auto-detected)' )"
for n in "${NOTES[@]+"${NOTES[@]}"}"; do echo "  Note        : ${n}"; done
echo "════════════════════════════════════════════════════════════════"

if ! build "${BACKEND}"; then
  if (( AUTO )) && [[ "${BACKEND}" != "cpu" ]]; then
    echo "" >&2
    echo "WARNING: the ${BACKEND} build failed; falling back to a CPU build." >&2
    BACKEND="cpu"
    build cpu
  else
    echo "" >&2
    echo "ERROR: building llama-cpp-python for '${BACKEND}' failed." >&2
    exit 1
  fi
fi

# ── Verify & record ───────────────────────────────────────────────────────────
GPU_OK="$(uv run --no-sync python -c 'import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())')"
VERSION="$(installed_version)"
cat >"${MARKER}" <<EOF
backend=${BACKEND}
llama_cpp_python=${VERSION}
gpu_offload=${GPU_OK}
os=${OS} ${ARCH} (${ENV_KIND})
cpu=${CPU_MODEL}
gpu=${APPLE_GPU:-${NVIDIA_GPU:-${AMD_GFX:-${VULKAN_GPU:-none}}}}
completed_at=$(date '+%Y-%m-%d %H:%M:%S')
EOF

echo ""
echo "  GPU offload available: ${GPU_OK}"
if [[ "${BACKEND}" != "cpu" && "${GPU_OK}" != "True" ]]; then
  echo "  WARNING: built for ${BACKEND}, but llama.cpp reports no GPU offload;" >&2
  echo "  in-process embedding will use the CPU (EMBEDDING_DEVICE=auto)." >&2
fi
echo "  ✅ Backend setup complete (${BACKEND}); recorded in ${MARKER##*/}."
