# Embedding Service

A text embedding service that serves two models through
[llama.cpp](https://github.com/ggml-org/llama.cpp), run with Docker Compose:

| Model key | Model | Native dim |
| --- | --- | --- |
| `gemma` | Google [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) | 768 |
| `qwen` | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 4096 |

It also includes an offline data-preparation pipeline that generates sample
embeddings and K-means cluster centroids per model.

The HTTP API matches the contract expected by the Rust client: a batch interface
with separate document and query endpoints, returning one L2-normalised vector
per payload. **The caller names the model in the path**, so which model produced
a vector is never implicit.

---

## Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [HTTP API](#http-api)
- [Migrating from the single-model API](#migrating-from-the-single-model-api)
- [Models](#models)
- [The `EmbeddingService` class](#the-embeddingservice-class)
- [Offline pipeline: sample embeddings → centroids](#offline-pipeline-sample-embeddings--centroids)
- [Scripts](#scripts)
- [Configuration](#configuration)
- [GPU backends (Docker)](#gpu-backends-docker)
- [In-process GPU builds (offline pipeline)](#in-process-gpu-builds-offline-pipeline)
- [Sample data & Git LFS](#sample-data--git-lfs)
- [Project layout](#project-layout)

---

## Architecture

```
                ┌──────────────────────────────────┐
  HTTP clients  │  gateway  (FastAPI, server.py)   │   port 8001, the only
  (Rust, curl)  │  /embedding/{model}/document     │   port published
  ───────────►  │  /embedding/{model}/query        │   {model} = gemma | qwen
                │  /healthcheck                    │   anything else → 404
                └───────┬─────────────────┬────────┘
       prompt prefix,   │                 │   internal Docker network
       truncate + L2    │                 │
            ┌───────────▼──────┐   ┌──────▼───────────┐
            │  llama-gemma     │   │  llama-qwen      │   ghcr.io/ggml-org/llama.cpp
            │  llama-server    │   │  llama-server    │   :server-rocm | -vulkan |
            └───────────┬──────┘   └──────┬───────────┘   -cuda | :server (CPU)
                        └───────┬─────────┘
                    ./model_cache  (GGUFs, fetched by the model-fetch container)

  Offline pipeline (no server required, in-process llama-cpp-python), per model:

  sample_data/*.jsonl ─► SampleEmbeddingGenerator ─► embedding/<model>/sample_data_embedding.parquet
                                                          │
                                                          ▼
                              ClusterCentroidGenerator ─► embedding/<model>/cluster_centroids.jsonl
```

The gateway adds each model's prompt prefix, forwards the texts to that
model's `llama-server`, then truncates and L2-normalises the vectors. It
batches concurrent requests into shared calls (see
[Cross-request batching](#cross-request-batching)).

Each model embeds documents and queries **asymmetrically** via task-specific
prompts. Both support **Matryoshka Representation Learning (MRL)**, so callers
can request a smaller `dimensions` value; vectors are truncated to that width
and re-normalised. **Every model defaults to 768 dimensions** (Qwen's native
4096 is truncated to 768 unless more is requested), so vectors from either
model fit the same 768-dim indexes. All returned vectors are **L2-normalised**.

---

## Requirements

For the service:

- Linux or WSL2 with **Docker Engine** and the **Compose plugin**
  (`docker compose`). Docker on macOS has no GPU access, so the service isn't
  set up for macOS.
- For a GPU, see [GPU backends](#gpu-backends-docker). Without one, it runs on
  the CPU.
- Network access on first run (models are downloaded from Hugging Face:
  ~330 MB for Gemma, ~5 GB for Qwen at Q4_K_M)

For the offline pipeline (and `EmbeddingService` in-process):

- Python **≥ 3.14** and [`uv`](https://docs.astral.sh/uv/)
- A C/C++ compiler (`llama-cpp-python` is built from source on install;
  on macOS install the Xcode command-line tools: `xcode-select --install`)
- [Git LFS](https://git-lfs.com/) to check out the sample data

Memory: Qwen at the default Q4_K_M uses about 10.6 GB of GPU memory in total
(4.5 GB weights, plus KV cache and compute buffers for its 8192-token batch).
`--qwen-quant Q8_0` (near-lossless) needs ~14 GB and `--qwen-quant f16` (full
precision) ~21 GB. On small machines load only Gemma (`--models gemma`).

---

## Quick start

Start the service. It finds the llama.cpp image that can use your GPU, builds
the gateway image, downloads the models (first run only), starts the
containers, waits for health and runs a smoke test against each model:

```bash
./embedding_service.sh start                  # loads gemma + qwen
./embedding_service.sh start --models gemma   # gemma only
./embedding_service.sh start --qwen-quant Q8_0   # higher-precision Qwen
./embedding_service.sh --help                     # every option, default and allowed value
./embedding_service.sh status                 # containers, models, device
./embedding_service.sh logs llama-qwen        # follow one container's logs
```

In another terminal, run the test suite:

```bash
./test_embeddings.sh          # every loaded model
./test_embeddings.sh qwen     # one model
```

Stop the service:

```bash
./embedding_service.sh stop
```

The containers restart automatically (e.g. after a reboot or a crash) until
stopped. `restart` reuses the last start's settings.

---

## HTTP API

Base URL defaults to `http://localhost:8001`.

### `POST /embedding/{model}/document`
### `POST /embedding/{model}/query`

`{model}` is `gemma` or `qwen`. Any other value — or a model the server was not
started with (see `EMBEDDING_MODELS`) — returns **404**:

```json
{"detail": "Unknown model 'foo'; available: gemma, qwen"}
```

Both endpoints share the same request/response shape. Use `document` for content
being indexed and `query` for search queries — the model embeds them differently.

**Request**

```json
{
  "payloads": ["first text", "second text"],
  "dimensions": 256
}
```

- `payloads` — list of strings; one embedding is returned per payload, in order.
- `dimensions` — *optional*, defaults to **768** for every model. Values below
  the native width (768 for `gemma`, 4096 for `qwen`) trigger Matryoshka
  truncation; ask for up to 4096 explicitly to get more from `qwen`. Must be ≥ 1 (422 otherwise).

**Response**

```json
{
  "embeddings": [[0.012, -0.034, ...], [0.044, 0.001, ...]]
}
```

Each vector is exactly `min(dimensions, native)` long and L2-normalised. An
empty `payloads` list returns `{"embeddings": []}` without invoking the model.

### `GET /healthcheck`

Lists the loaded models. Returns **200** with `"status": "healthy"` once every
model's server is ready, and **503** with `"status": "starting"` while any is
still loading (or down). Each model's `status` is `ok`, `loading` or
`unreachable`:

```json
{
  "status": "healthy",
  "models": {
    "gemma": {"name": "google/embeddinggemma-300m", "file": "embeddinggemma-300M-Q8_0.gguf",
              "dimension": 768, "default_dimension": 768, "device": "gpu", "backend": "rocm",
              "status": "ok"},
    "qwen":  {"name": "Qwen/Qwen3-Embedding-8B", "file": "Qwen3-Embedding-8B-Q4_K_M.gguf",
              "dimension": 4096, "default_dimension": 768, "device": "gpu", "backend": "rocm",
              "status": "ok"}
  }
}
```

An input longer than the model's context returns **500** with llama.cpp's
message, and a model server that is down or still loading returns **503**.

**Example**

```bash
curl -X POST http://localhost:8001/embedding/qwen/query \
  -H 'Content-Type: application/json' \
  -d '{"payloads": ["what is machine learning?"], "dimensions": 1024}'
```

---

## Migrating from the single-model API

- **Paths changed.** `/embedding/document` and `/embedding/query` are gone (404).
  Use `/embedding/gemma/document` and `/embedding/gemma/query` for the previous
  model.
- **Re-embed existing Gemma data.** The previous FastEmbed/ONNX pipeline
  mean-pooled the raw hidden states and skipped EmbeddingGemma's dense projection
  layers, so its vectors were not the model's real embeddings and live in a
  different vector space (cosine ≈ 0 against the reference). The llama.cpp
  pipeline matches the model's reference output (cosine 0.9997). Vectors, indexes
  and centroids produced by the old service must be regenerated; do not mix them
  with new ones.
- **Healthcheck shape changed.** `model`/`providers` were replaced by a `models` map.
  It now answers 503 until every model is loaded.
- **Docker.** Moving from the in-process server to the llama.cpp containers
  doesn't change the vector space. Gemma matches the in-process output at
  cosine ≈ 0.9997 and Qwen Q4_K_M at ≈ 0.997 on the GPU, which is float noise
  across llama.cpp builds and devices. Existing vectors don't need
  re-embedding.

---

## Models

Defined in `MODELS` in [`src/models.py`](src/models.py) (the context sizes
and pooling are repeated in [`docker-compose.yml`](docker-compose.yml)). Both
are loaded from official GGUF builds and downloaded into the model cache on
first use.

| | `gemma` | `qwen` |
| --- | --- | --- |
| GGUF repo | [`ggml-org/embeddinggemma-300M-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | [`Qwen/Qwen3-Embedding-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF) |
| Default file | `embeddinggemma-300M-Q8_0.gguf` | `Qwen3-Embedding-8B-Q4_K_M.gguf` |
| Other builds | (Q8_0 only in this repo) | `Q5_0`, `Q5_K_M`, `Q6_K`, `Q8_0`, `f16` |
| Pooling | mean | last token (EOS) |
| Native dim | 768 | 4096 |
| Context | 2048 tokens | 8192 tokens (model supports 32k; capped for memory) |
| Query prompt | `task: search result \| query: <payload>` | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: <payload>` |
| Document prompt | `title: none \| text: <payload>` | `<payload>` (no prefix) |

Pick another Qwen build with `--qwen-quant` (`Q4_K_M` default, `Q5_0`, `Q5_K_M`,
`Q6_K`, `Q8_0`, or `f16` for full precision; `./embedding_service.sh --help`
lists sizes and GPU memory), or name any file with `EMBEDDING_QWEN_GGUF` /
`EMBEDDING_GEMMA_GGUF`.

---

## The `EmbeddingService` class

[`src/embedding_service.py`](src/embedding_service.py) wraps one llama.cpp model
in-process through `llama-cpp-python`. The offline pipeline uses it, and you can
use it directly without the service (see
[In-process GPU builds](#in-process-gpu-builds-offline-pipeline) for GPU use):

```python
from embedding_service import EmbeddingService

service = EmbeddingService("qwen")               # or "gemma"
docs = service.embed_documents(["the cat sat on the mat"], dimensions=256)
qry  = service.embed_query(["where did the cat sit?"], dimensions=256)
```

- `embed_documents(payloads, dimensions=768)` — applies the document prompt.
- `embed_query(payloads, dimensions=768)` — applies the query prompt.
- Both return `list[list[float]]`, truncated to `dimensions` (pass `None` for
  the native width) and L2-normalised.
- `load_models(["gemma", "qwen"])` returns a `{key: EmbeddingService}` dict.

Run its `main()` demo:

```bash
uv run python src/embedding_service.py gemma
```

---

## Offline pipeline: sample embeddings → centroids

A two-stage offline pipeline turns the raw QA dataset into cluster centroids,
**per model** — centroids from one model's vector space are meaningless for the
other. Both stages run in-process (no server needed).

```bash
./generate_sample_embeddings.sh qwen     # → embedding/qwen/sample_data_embedding.parquet
./generate_cluster_centroids.sh qwen     # → embedding/qwen/cluster_centroids.jsonl
```

Both scripts require the model name (`gemma` or `qwen`) and reject anything else.

### 1. Generate sample embeddings

[`src/sample_embedding_generator.py`](src/sample_embedding_generator.py) reads
ELI5-style `["question", "answer"]` JSONL records, embeds the combined
`question + "\n" + answer` text as a document, and writes a Parquet file.

```bash
uv run python src/sample_embedding_generator.py \
  --model qwen \
  --input sample_data/eli5_question_answer.jsonl \
  --dimensions 768 \
  --batch-size 32 \
  --max-records 0          # 0 = all records
# --output defaults to embedding/<model>/sample_data_embedding.parquet
```

`--dimensions` defaults to 768; `0` keeps the model's native width. The shell script
reads `DIMENSIONS`, `BATCH_SIZE` and `MAX_RECORDS` (default 25000) from the
environment. Qwen-8B is much slower than Gemma, especially on CPU (~1 s per QA
pair on a 22-core x86 CPU vs ~12 pairs/s for Gemma), so use a GPU or a smaller
`MAX_RECORDS` for it.

Output Parquet schema: `question : string`, `answer : string`,
`qa_embedding : list[float]`.

### 2. Generate cluster centroids

[`src/cluster_centroid_generator.py`](src/cluster_centroid_generator.py) loads the
model's Parquet, runs K-means, and writes centroids to JSONL.

```bash
uv run python src/cluster_centroid_generator.py \
  --model qwen \
  --dimensions 768 \
  --n-clusters 256 \
  --seed 42
# --input/--output default to embedding/<model>/...; pass them to override
```

Output JSONL (one object per line):

```json
{"cluster_id": 1, "centroid": [0.01, -0.02, ...]}
{"cluster_id": 2, "centroid": [0.03,  0.01, ...]}
```

`--dimensions` Matryoshka-truncates + renormalises the stored vectors before
clustering (default 768; `0` = use the stored width as-is). `--n-clusters` defaults
to 256 and is capped to the number of available embeddings. The shell script
reads `DIMENSIONS`, `N_CLUSTERS` and `SEED` from the environment.

---

## Scripts

| Script | Purpose |
| --- | --- |
| [`embedding_service.sh start\|stop\|restart\|status\|logs`](embedding_service.sh) | `start`: build the gateway image, fetch the models, pick the GPU backend, start the containers, wait for health, smoke test each model (`-f` stays attached). `stop`: remove the containers. `restart`: both, with the last settings. `status`: containers and loaded models. `logs [service]`: follow the logs. |
| [`test_embeddings.sh [MODEL]`](test_embeddings.sh) | Test suite per model (query/document embedding, default and truncated dimensions) plus 404 checks. No `MODEL` = every loaded model. |
| [`install_backend.sh`](install_backend.sh) | Detect OS/CPU/GPU and build `llama-cpp-python` for the best backend (Metal / CUDA / HIP / Vulkan / CPU), for the in-process offline pipeline. Run automatically by the sample-embedding script. |
| [`generate_sample_embeddings.sh MODEL`](generate_sample_embeddings.sh) | Embed the sample QA data with `MODEL`. |
| [`generate_cluster_centroids.sh MODEL`](generate_cluster_centroids.sh) | Cluster `MODEL`'s sample embeddings into centroids. |

`embedding_service.sh start` options: `--host`, `--port`, `--models`,
`--qwen-quant`, `--cache-dir`, `--backend`, `--pull`,
`-f/--foreground`. Run any script with `-h` for details.

---

## Configuration

Environment variables (read by the start script, the containers and the
offline pipeline):

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODELS` | `gemma,qwen` | Models the service loads. Others return 404. |
| `EMBEDDING_GEMMA_GGUF` | `embeddinggemma-300M-Q8_0.gguf` | Gemma GGUF file in its repo. |
| `EMBEDDING_QWEN_QUANT` | `Q4_K_M` | Qwen build for the start script, as `--qwen-quant`. |
| `EMBEDDING_QWEN_GGUF` | `Qwen3-Embedding-8B-Q4_K_M.gguf` | Qwen GGUF file in its repo (e.g. `Qwen3-Embedding-8B-Q8_0.gguf`). |
| `EMBEDDING_CACHE_PATH` | `./model_cache` | Where GGUF files are downloaded (mounted into the containers). |
| `EMBEDDING_BACKEND` | `auto` | llama.cpp image for the service: `auto`, `rocm-wsl`, `rocm`, `cuda`, `vulkan` or `cpu`. |
| `EMBEDDING_DEVICE` | `auto` | In-process only: `auto` (GPU if available, CPU otherwise or if the GPU load fails), `gpu` (fail without one) or `cpu`. |
| `HF_TOKEN` | — | Optional Hugging Face token for downloads. |
| `HOST` / `PORT` | `0.0.0.0` / `8001` | Address the gateway is published on. |
| `STARTUP_TIMEOUT` | `1800` | Seconds the start script waits for health. |
| `EMBEDDING_REQUEST_TIMEOUT` | `600` | Gateway: seconds to wait for one call to a model server. |
| `EMBEDDING_MAX_BATCH_TEXTS` | `256` | Gateway: max texts per model call when batching concurrent requests. |
| `EMBEDDING_BATCH_WAIT_MS` | `2` | Gateway: how long a busy worker waits for more requests to batch (an idle one never waits). |
| `LLAMA_GEMMA_PARALLEL` | `4` | Sequences gemma's `llama-server` runs at once. |
| `SERVICE_URL` | `http://localhost:8001` | Service the test script targets. |

`start` saves its settings in `embedding_service.env` (gitignored), which
`stop`, `status`, `logs` and `restart` read (the three tuning variables only
when set).

---

## GPU backends (Docker)

`start` picks a llama.cpp image and adds the matching overlay from
[`docker/`](docker/) to [`docker-compose.yml`](docker-compose.yml) (CPU on its
own). For each candidate below, in order, it runs `llama-server --list-devices`
in that image with its devices and uses the first that reports a GPU, along
with the GPU's name and memory. So a
backend that can't use the GPU falls through to the next instead of running
silently on the CPU. `--backend NAME` (or `EMBEDDING_BACKEND`) forces one and
fails if it finds no GPU. Candidates:

| Backend | Detected by | Image | Host needs |
| --- | --- | --- | --- |
| `rocm-wsl` | WSL2 + `/dev/dxg` + `/opt/rocm/lib/librocdxg.so` | `server-rocm` | Recent AMD Adrenalin driver on Windows, [librocdxg](https://github.com/ROCm/librocdxg) in WSL |
| `rocm` | `/dev/kfd` | `server-rocm` | amdgpu kernel driver (RDNA4 cards such as the R9700 / gfx1201 need a recent kernel; see AMD's ROCm Linux support matrix) |
| `cuda` | `nvidia-smi` + Docker's `nvidia` runtime | `server-cuda` | NVIDIA Container Toolkit |
| `vulkan` | `/dev/dri/renderD*` (not on WSL) | `server-vulkan` | GPU Vulkan driver |
| `cpu` | anything else | `server` | — |

The `server-rocm` image targets gfx908, gfx90a, gfx942, gfx1030, gfx1100–1102,
gfx1150, gfx1151, gfx1200 and gfx1201. Only `rocm-wsl` and `cpu` have been
tested so far (Radeon 890M / gfx1150 under
WSL2: both models fully offloaded, Gemma ≈ 2.3× and Qwen ≈ 2.4× the CPU
throughput; see [`docs/rocm_wsl_spike.md`](docs/rocm_wsl_spike.md)).

### Cross-request batching

Clients such as minnal send one small request per document (the whole text
plus ~4 chunks, about 5 texts). Each model call has a fixed cost, so the
gateway batches **across** requests, per model: every request goes onto that
model's queue, and a single worker sends one `llama-server` call for
everything queued (document and query texts together, each already carrying
its task prompt). It then splits the vectors back and applies each request's
own `dimensions`. If a shared call fails, each request is retried alone, so
one bad request cannot fail its neighbours. An idle worker sends a lone
request immediately; a busy one waits up to `EMBEDDING_BATCH_WAIT_MS` (2 ms)
for more, up to `EMBEDDING_MAX_BATCH_TEXTS` (256) texts per call.

`llama-server` then runs the texts of a call `LLAMA_GEMMA_PARALLEL` (4) at a
time. Gemma's attention is bidirectional, so it has no KV cache and more slots
cost no memory; with a fast GPU, raising this lets each GPU pass hold more of
a batch. Measure before changing it, with many requests in flight.

### Throughput

Measured with the pattern above (ELI5 documents, 5 texts per request), Gemma
on a Radeon 890M iGPU under WSL2: about 20–30 docs/s whatever the batching or
slot settings, because one request already keeps the iGPU busy.

Baseline for the Radeon AI PRO R9700, from the previous PyTorch ROCm runtime
(bfloat16, same batching design), with SciFact abstracts:

| Requests in flight | docs/s |
| ---: | ---: |
| 1 | ~30 |
| 8 | ~36 |
| 16 | ~43 |
| 64 | **~58** |

The llama.cpp setup hasn't been measured on the R9700 yet; compare against
these numbers with many requests in flight, and tune `LLAMA_GEMMA_PARALLEL`
there.

llama.cpp falls back to the CPU silently if it finds no usable GPU. `start`
checks the model servers' logs for this and warns. `--pull` updates the
llama.cpp image to the latest build.

LAN access under WSL2 needs mirrored networking and inbound firewall rules for
the port, in both Windows Defender and the Hyper-V firewall (see Step 1 of the
spike doc).

---

## In-process GPU builds (offline pipeline)

The offline pipeline (and `EmbeddingService`) runs llama.cpp in-process via
`llama-cpp-python`, which picks its GPU backend **when it is compiled**, so GPU
support is a build step. `./generate_sample_embeddings.sh` handles it:

1. **First run**: there is no `.backend_setup_complete` file, so it runs
   [`install_backend.sh`](install_backend.sh), which detects the OS
   (macOS/Linux, WSL2), CPU and GPU, builds `llama-cpp-python` for the best
   available backend, and records the result in `.backend_setup_complete`
   (gitignored).
2. **Later runs**: setup is skipped (one line in the output says which
   backend is in use).
3. **Rebuild** when you add a GPU toolchain or want to re-detect:

   ```bash
   ./install_backend.sh                  # re-detect + rebuild
   ./install_backend.sh --backend vulkan # force one backend
   ```

Setup also reruns automatically if the installed `llama-cpp-python` version no
longer matches the one recorded, e.g. after a lock upgrade reinstalls the
default (CPU) build. `uv sync` itself is safe: uv keeps an installed build whose
version matches the lock. Run `install_backend.sh -h` for options.

Auto-detection order:

| Platform | Backend | Needs (to build) |
| --- | --- | --- |
| macOS, Apple Silicon | `metal` | Xcode command-line tools |
| NVIDIA GPU (`nvidia-smi`) | `cuda` | CUDA toolkit (`nvcc`) |
| AMD GPU agent in `rocminfo` | `hip` | ROCm 6.x/7.x (`/opt/rocm/llvm/bin/clang`); gfx targets read from `rocminfo` |
| Hardware Vulkan device in `vulkaninfo` (e.g. WSL2, iGPUs ROCm doesn't support) | `vulkan` | Vulkan driver + `libvulkan-dev` + `glslc` |
| anything else (incl. Intel Macs) | `cpu` | — |

A GPU whose toolchain is missing is reported with a hint about what to install,
and the CPU backend is used. If an auto-detected GPU build fails to compile,
setup falls back to a CPU build. A failed forced build (`--backend`) stops with
an error and leaves the previous build in place.

`.backend_setup_complete` looks like:

```
backend=hip
llama_cpp_python=0.3.36
gpu_offload=True
os=Linux x86_64 (native)
cpu=AMD Ryzen AI 9 HX 375 w/ Radeon 890M
gpu=gfx1150
completed_at=2026-10-02 10:56:34
```

Device selection when a model loads (`EMBEDDING_DEVICE`):

- `auto` (default) — offload all layers to the GPU when the build supports it.
  If there's no GPU build, or loading a model on the GPU fails (e.g. out of
  memory), that model **falls back to CPU** with a warning in the log.
- `gpu` — the same, but either case is a startup error.
- `cpu` — never use the GPU.

Notes:

- AMD integrated GPUs such as the Radeon 890M (gfx1150) and 8060S (gfx1151)
  work with HIP on native Linux with recent ROCm. Under WSL2 the GPU must
  appear in `rocminfo`; if it doesn't, Vulkan is usually the easier route.
- On Apple Silicon, GPU and CPU share memory: Qwen at Q4_K_M uses ~6 GB, and
  Q8_0 (~9–10 GB) wants a 16 GB+ Mac with little else running.

---

## Sample data & Git LFS

`sample_data/eli5_question_answer.jsonl` (~166 MB) is tracked via Git LFS (see
[`.gitattributes`](.gitattributes)). After cloning:

```bash
git lfs install
git lfs pull
```

The model cache (`model_cache/`), generated embeddings (`embedding/`), the
backend setup marker (`.backend_setup_complete`), and the start script's saved
settings (`embedding_service.env`) are gitignored. The old
`fastembed_cache/` directory is no longer used and can be deleted.

---

## Project layout

```
.
├── src/
│   ├── models.py                     # MODELS registry (prompts, pooling, GGUF builds)
│   ├── server.py                     # FastAPI gateway → llama-server per model
│   ├── fetch_models.py               # GGUF download for the containers (model-fetch)
│   ├── embedding_service.py          # EmbeddingService (in-process llama.cpp)
│   ├── sample_embedding_generator.py # ELI5 QA → Parquet embeddings (per model)
│   └── cluster_centroid_generator.py # Parquet → K-means centroids (JSONL)
├── docker/
│   ├── gateway.Dockerfile            # gateway + model-fetch image
│   └── compose.<backend>.yml         # GPU overlays: rocm-wsl, rocm, cuda, vulkan
├── docs/
│   └── rocm_wsl_spike.md             # ROCm-on-WSL2 Docker validation
├── sample_data/
│   └── eli5_question_answer.jsonl    # sample QA pairs (Git LFS)
├── docker-compose.yml
├── embedding_service.sh              # start | stop | restart | status | logs
├── test_embeddings.sh
├── install_backend.sh
├── generate_sample_embeddings.sh
├── generate_cluster_centroids.sh
├── pyproject.toml
└── README.md
```
