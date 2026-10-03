# Embedding Service

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

The embedding service turns text into vector embeddings over HTTP. It serves two
models through [llama.cpp](https://github.com/ggml-org/llama.cpp), run with
Docker Compose, on an AMD, NVIDIA or Vulkan GPU, or on the CPU.

It has **two parts**:

- **The service.** A small HTTP gateway in front of one llama.cpp model server
  per model. Callers send a batch of texts and get back one L2-normalised vector
  per text. See **[Quick start](#quick-start)** and **[HTTP API](#http-api)**.
- **The offline pipeline.** Two scripts embed a sample QA dataset through the
  running service and cluster the results into K-means centroids, one set per
  model. See **[Offline pipeline](#offline-pipeline-sample-embeddings--centroids)**.

> **Platform support:** Linux and WSL2 with Docker. Docker on macOS has no GPU
> access, so the service isn't set up for macOS.

> **Companion database:** [minnal0021/minnal](https://github.com/minnal0021/minnal)
> uses this service for semantic search. Point its
> `semantic_search.embedding_service_url` at the service (default
> `http://localhost:8001`), set `semantic_search.model` to `gemma` or `qwen`, and
> use the cluster centroids this pipeline generates for that model.

---

## Contents

- [Overview](#overview)
- [Concepts Primer](#concepts-primer)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [HTTP API](#http-api)
- [Models](#models)
- [Offline pipeline: sample embeddings → centroids](#offline-pipeline-sample-embeddings--centroids)
- [Scripts](#scripts)
- [Configuration](#configuration)
- [GPU backends](#gpu-backends)
- [Sample data & Git LFS](#sample-data--git-lfs)
- [Project layout](#project-layout)
- [License](#license)

---

## Overview

| Model key | Model | Native dim | Size of the default build |
| --- | --- | --- | --- |
| `gemma` | Google [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) | 768 | ~330 MB |
| `qwen` | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 4096 | ~4.7 GB |

**The caller names the model in the URL path** (`/embedding/gemma/...` or
`/embedding/qwen/...`), so which model produced a vector is never implicit.

What every response guarantees:

- One vector per input text, in the same order.
- **768 dimensions by default for both models**, so vectors from either model
  fit the same 768-dim index. Ask for fewer, or for up to 4096 from `qwen`.
- Every vector is **L2-normalised** (length 1), so a dot product is the cosine
  similarity.
- Separate **document** and **query** endpoints, because each model embeds the
  two differently.

---

## Concepts Primer

If you already know these terms, skip ahead to [Architecture](#architecture).

| Term | In one sentence |
|---|---|
| **Embedding** | A list of numbers (a vector) that represents a text's meaning, so that texts with similar meaning have vectors that point in similar directions. |
| **Vector space** | The set of vectors one model produces. Vectors are only comparable with others from the same model (and the same build of it): never mix models in one index. |
| **Document vs query embedding** | Both models are trained to embed the text being searched (*documents*) and the search text (*queries*) slightly differently, by prepending a different instruction ("prompt") to each. |
| **Matryoshka truncation** (MRL) | The models are trained so that the first *N* numbers of a vector are a usable smaller embedding on their own. Truncating to *N* and re-normalising trades a little accuracy for a smaller vector. |
| **L2 normalisation** | Scaling a vector to length 1, so the dot product of two vectors equals their cosine similarity. |
| **GGUF / quantisation** | GGUF is llama.cpp's model file format. A *quantised* build stores the weights with fewer bits (e.g. `Q4_K_M` ≈ 4.5 bits, `Q8_0` = 8 bits, `f16` = 16 bits), trading a little accuracy for a smaller download and less GPU memory. |
| **Cluster centroids** | The centre points of groups of similar vectors, found with K-means. A vector database such as minnal uses them to search only the few groups nearest to a query instead of every vector. |

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

  Offline pipeline, per model (embeds through the running service):

  sample_data/*.jsonl ─► SampleEmbeddingGenerator ─► embedding/<model>/sample_data_embedding.parquet
                                                          │
                                                          ▼
                              ClusterCentroidGenerator ─► embedding/<model>/cluster_centroids.jsonl
```

The service runs as four containers:

| Container | Role |
|---|---|
| `gateway` | The HTTP API. Adds each model's document or query prompt, forwards the texts to that model's server, then truncates and L2-normalises the vectors. |
| `llama-gemma`, `llama-qwen` | One llama.cpp `llama-server` per model, reachable only from the gateway. Only the models you start are run. |
| `model-fetch` | Runs once at start: downloads the GGUF files into `./model_cache` (first run only) and exits. |

### Cross-request batching

Clients such as minnal send one small request per document (the whole text
plus ~4 chunks, about 5 texts). Each model call has a fixed cost, so the
gateway batches **across** requests, per model:

1. Every request goes onto its model's queue.
2. A single worker per model sends one `llama-server` call for everything
   queued, document and query texts together (each already carries its prompt).
3. It splits the vectors back per request and applies each request's own
   `dimensions`.

If a shared call fails, each request in it is retried alone, so one bad request
cannot fail its neighbours. An idle worker sends a lone request immediately; a
busy one waits up to `EMBEDDING_BATCH_WAIT_MS` (2 ms) for more, up to
`EMBEDDING_MAX_BATCH_TEXTS` (256) texts per call.

Each `llama-server` then runs the texts of a call `LLAMA_PARALLEL` (4) at a
time. The parallel slots share the model's context (`-kvu`), so each text can
still use all of it, and extra slots add little memory: Gemma's attention is
bidirectional, so it has no KV cache, and Qwen's 8192-token KV cache is shared
between the slots rather than multiplied.

---

## Requirements

For the service:

- Linux or WSL2 with **Docker Engine** and the **Compose plugin**
  (`docker compose`).
- Optionally a GPU; see [GPU backends](#gpu-backends). Without one, the models
  run on the CPU.
- Network access on first run, to download the models from Hugging Face.
- GPU (or system) memory: about **10.6 GB** with both models at the default
  builds, growing to about 13 GB under load. Gemma alone needs well under 1 GB;
  see [GPU memory](#gpu-memory).

For the offline pipeline:

- Python **≥ 3.14** and [`uv`](https://docs.astral.sh/uv/)
- The running service, here or on another machine (`SERVICE_URL`)
- [Git LFS](https://git-lfs.com/) to check out the sample data

---

## Quick start

Start the service. The start script finds the llama.cpp image that can use your
GPU, builds the gateway image, downloads the models (first run only), starts the
containers, waits until every model is loaded and runs a smoke test against
each:

```bash
./embedding_service.sh start                     # loads gemma + qwen
./embedding_service.sh start --models gemma      # gemma only (small machines)
./embedding_service.sh start --qwen-quant Q8_0   # higher-precision Qwen
./embedding_service.sh --help                    # every option, default and allowed value
./embedding_service.sh status                    # containers, models, device
./embedding_service.sh logs llama-qwen           # follow one container's logs
```

Embed something:

```bash
curl -X POST http://localhost:8001/embedding/gemma/document \
  -H 'Content-Type: application/json' \
  -d '{"payloads": ["The quick brown fox jumps over the lazy dog."]}'
```

Run the test suite in another terminal:

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

`{model}` is `gemma` or `qwen`. Any other value, or a model the service was not
started with (see `--models`), returns **404**:

```json
{"detail": "Unknown model 'foo'; available: gemma, qwen"}
```

Both endpoints share the same request and response shape. Use `document` for
content being indexed and `query` for search text; the model embeds them
differently.

**Request**

```json
{
  "payloads": ["first text", "second text"],
  "dimensions": 256
}
```

- `payloads` — list of strings; one embedding is returned per payload, in order.
- `dimensions` — *optional*, defaults to **768** for both models. Values below
  the native width (768 for `gemma`, 4096 for `qwen`) are Matryoshka-truncated;
  ask for up to 4096 explicitly to get more from `qwen`. Must be ≥ 1 (422
  otherwise).

**Response**

```json
{
  "embeddings": [[0.012, -0.034, ...], [0.044, 0.001, ...]]
}
```

Each vector is exactly `min(dimensions, native)` long and L2-normalised. An
empty `payloads` list returns `{"embeddings": []}` without calling the model.

**Long inputs are truncated.** A payload longer than the model's context (2048
tokens for `gemma`, 8192 for `qwen`, counting the task prompt) is embedded from
its first tokens, up to one token under the context, keeping the model's own
start and end tokens. Payloads that fit are embedded exactly as before, and
each truncation is logged as a warning. sentence-transformers truncates the
same way.

**Errors**

| Status | When |
|---|---|
| 404 | Unknown model, or a model the service was not started with |
| 422 | Malformed request (e.g. `dimensions` < 1) |
| 500 | The model server rejected the input; the message is llama.cpp's. Inputs longer than the context are truncated, not rejected |
| 503 | The model server is down or still loading |

**Example**

```bash
curl -X POST http://localhost:8001/embedding/qwen/query \
  -H 'Content-Type: application/json' \
  -d '{"payloads": ["what is machine learning?"], "dimensions": 1024}'
```

### `GET /healthcheck`

Lists the loaded models. Returns **200** with `"status": "healthy"` once every
model is ready, and **503** with `"status": "starting"` while any is still
loading or down. Each model's `status` is `ok`, `loading` or `unreachable`:

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

---

## Models

Defined in `MODELS` in [`src/models.py`](src/models.py); pooling and context
size are set on each model's `llama-server` in
[`docker-compose.yml`](docker-compose.yml). Both are loaded from official GGUF
builds, downloaded into the model cache on first use.

| | `gemma` | `qwen` |
| --- | --- | --- |
| GGUF repo | [`ggml-org/embeddinggemma-300M-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | [`Qwen/Qwen3-Embedding-8B-GGUF`](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF) |
| Default file | `embeddinggemma-300M-Q8_0.gguf` | `Qwen3-Embedding-8B-Q4_K_M.gguf` |
| Other builds | (only Q8_0 is published) | `Q5_0`, `Q5_K_M`, `Q6_K`, `Q8_0`, `f16` |
| Pooling | mean | last token (EOS) |
| Native dim | 768 | 4096 |
| Context | 2048 tokens | 8192 tokens (the model supports 32k; capped for memory) |
| Query prompt | `task: search result \| query: <payload>` | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: <payload>` |
| Document prompt | `title: none \| text: <payload>` | `<payload>` (no prefix) |

Pick another Qwen build with `--qwen-quant` (`./embedding_service.sh --help`
lists each build's download size and GPU memory), or name any GGUF file with
`EMBEDDING_QWEN_GGUF` / `EMBEDDING_GEMMA_GGUF`.

**Keep one index to one model and one build.** Vectors from different models
live in unrelated vector spaces. Different builds of the same model (e.g. Qwen
`Q4_K_M` and `Q8_0`) give close but not identical vectors. When you change
either, re-embed the index and regenerate its centroids.

---

## Offline pipeline: sample embeddings → centroids

A two-stage pipeline turns the sample QA dataset into cluster centroids,
**per model**, since centroids from one model's vector space are meaningless
for the other. Stage 1 embeds through the running service, so start it first
with the model loaded; stage 2 only reads the saved embeddings.

```bash
./embedding_service.sh start             # gemma + qwen (or --models qwen)
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

- `--dimensions` defaults to 768; `0` keeps the model's native width.
- `--service-url` (or `SERVICE_URL`, default `http://localhost:8001`) points it
  at the service, which may run on another machine. It checks that the model is
  loaded and ready before starting.
- A batch the service rejects is retried one record at a time, and only the
  failing records are skipped. (Records longer than the model's context are
  truncated by the service, not rejected.)
- The shell script reads `DIMENSIONS`, `BATCH_SIZE`, `MAX_RECORDS` (default
  25000) and `SERVICE_URL` from the environment.

Qwen-8B is much slower than Gemma (~2 QA pairs/s on a Radeon 890M, ~1 s per
pair on a 22-core CPU), so use a GPU-backed service or a smaller `MAX_RECORDS`
for it.

Output Parquet schema: `question : string`, `answer : string`,
`qa_embedding : list[float]`.

### 2. Generate cluster centroids

[`src/cluster_centroid_generator.py`](src/cluster_centroid_generator.py) loads the
model's Parquet file, runs K-means, and writes the centroids to JSONL.

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

- `--dimensions` Matryoshka-truncates and re-normalises the stored vectors
  before clustering (default 768; `0` = use the stored width as-is).
- `--n-clusters` defaults to 256 and is capped to the number of embeddings.
- The shell script reads `DIMENSIONS`, `N_CLUSTERS` and `SEED` from the
  environment.

---

## Scripts

| Script | Purpose |
| --- | --- |
| [`embedding_service.sh start\|stop\|restart\|status\|logs`](embedding_service.sh) | `start`: pick the GPU backend, build the gateway image, fetch the models, start the containers, wait for health, smoke test each model (`-f` stays attached). `stop`: remove the containers. `restart`: both, with the last settings. `status`: containers and loaded models. `logs [service]`: follow the logs. |
| [`test_embeddings.sh [MODEL]`](test_embeddings.sh) | Test suite per model (query/document embedding, default and truncated dimensions) plus 404 checks. No `MODEL` = every loaded model. |
| [`generate_sample_embeddings.sh MODEL`](generate_sample_embeddings.sh) | Embed the sample QA data with `MODEL`, through the running service. |
| [`generate_cluster_centroids.sh MODEL`](generate_cluster_centroids.sh) | Cluster `MODEL`'s sample embeddings into centroids. |

`embedding_service.sh start` options: `--models`, `--qwen-quant`, `--backend`,
`--host`, `--port`, `--cache-dir`, `--pull`, `-f/--foreground`. Run any script
with `-h` for details.

---

## Configuration

Environment variables, read by the start script, the containers and the
offline pipeline. Each start option has a matching variable, used when the
option is not given.

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODELS` | `gemma,qwen` | Models the service loads (`--models`). Others return 404. |
| `EMBEDDING_QWEN_QUANT` | `Q4_K_M` | Qwen build (`--qwen-quant`). |
| `EMBEDDING_QWEN_GGUF` | `Qwen3-Embedding-8B-Q4_K_M.gguf` | Exact Qwen GGUF file in its repo; overrides `--qwen-quant`. |
| `EMBEDDING_GEMMA_GGUF` | `embeddinggemma-300M-Q8_0.gguf` | Exact Gemma GGUF file in its repo. |
| `EMBEDDING_BACKEND` | `auto` | llama.cpp image (`--backend`): `auto`, `rocm-wsl`, `rocm`, `cuda`, `vulkan` or `cpu`. |
| `EMBEDDING_CACHE_PATH` | `./model_cache` | Where GGUF files are downloaded (`--cache-dir`); mounted into the containers. |
| `HOST` / `PORT` | `0.0.0.0` / `8001` | Address the gateway is published on (`--host` / `--port`). |
| `HF_TOKEN` | — | Optional Hugging Face token for downloads. |
| `STARTUP_TIMEOUT` | `1800` | Seconds the start script waits for the models to load. |
| `EMBEDDING_MAX_BATCH_TEXTS` | `256` | Max texts per model call when the gateway batches concurrent requests. |
| `EMBEDDING_BATCH_WAIT_MS` | `2` | How long a busy gateway worker waits for more requests to batch (an idle one never waits). |
| `LLAMA_PARALLEL` | `4` | Texts each model server (gemma and qwen) runs at once. |
| `SERVICE_URL` | `http://localhost:8001` | Service the test and sample-embedding scripts use. |

`start` saves its settings in `embedding_service.env` (gitignored), which
`stop`, `status`, `logs` and `restart` read. The three tuning variables
(`EMBEDDING_MAX_BATCH_TEXTS`, `EMBEDDING_BATCH_WAIT_MS`, `LLAMA_PARALLEL`) are
saved only when set.

---

## GPU backends

`start` picks a llama.cpp image and adds the matching overlay from
[`docker/`](docker/) to [`docker-compose.yml`](docker-compose.yml) (the CPU
image needs no overlay). It tries each candidate below in order: it runs
`llama-server --list-devices` in that image and uses the first one that reports
a GPU. So a backend that can't use the GPU falls through to the next instead of
running silently on the CPU. `--backend NAME` (or `EMBEDDING_BACKEND`) forces
one and fails if it finds no GPU.

| Backend | Detected by | Image | Host needs |
| --- | --- | --- | --- |
| `rocm-wsl` | WSL2 + `/dev/dxg` + `/opt/rocm/lib/librocdxg.so` | `server-rocm` | Recent AMD Adrenalin driver on Windows, [librocdxg](https://github.com/ROCm/librocdxg) in WSL ([setup](#rocm-on-wsl2-setup)) |
| `rocm` | `/dev/kfd` | `server-rocm` | amdgpu kernel driver (RDNA4 cards such as the R9700 / gfx1201 need a recent kernel; see AMD's ROCm Linux support matrix) |
| `cuda` | `nvidia-smi` + Docker's `nvidia` runtime | `server-cuda` | NVIDIA Container Toolkit |
| `vulkan` | `/dev/dri/renderD*` (not on WSL) | `server-vulkan` | GPU Vulkan driver |
| `cpu` | anything else | `server` | — |

The `server-rocm` image targets gfx908, gfx90a, gfx942, gfx1030, gfx1100–1102,
gfx1150, gfx1151, gfx1200 and gfx1201. The `rocm-wsl` and `cpu` backends have
been tested, on a Radeon 890M (gfx1150) under WSL2: both models fully on the
GPU, Gemma ≈ 2.3× and Qwen ≈ 2.4× the CPU throughput, output matching the CPU,
stable under a 5-minute load test. The other backends have not been tested yet.

llama.cpp falls back to the CPU silently if it finds no usable GPU; `start`
checks the model servers' logs for this and warns. `--pull` updates the
llama.cpp image to the latest build.

### GPU memory

Qwen at the default `Q4_K_M` uses about 10.6 GB of GPU memory: 4.5 GB of
weights plus the KV cache and compute buffers for its 8192-token batch.
`--qwen-quant Q8_0` (near-lossless) needs ~14 GB and `f16` (full precision)
~21 GB. Gemma adds about 0.5 GB. On small machines load only Gemma
(`--models gemma`).

Usage grows once the models have served traffic, because the ROCm runtime keeps
the working buffers it allocates. With both models on a Radeon 890M: ~11.4 GB
right after start, ~13.1 GB after use. Leave room for that when other programs
share the GPU. `LLAMA_PARALLEL` doesn't change it.

### Throughput

Measured on a Radeon 890M iGPU under WSL2, with ELI5 documents at 5 texts per
request: Gemma serves about 20–30 docs/s whatever the batching or slot
settings, because a single request already keeps the iGPU busy.

On a faster GPU, more requests in flight and a higher `LLAMA_PARALLEL` let each
GPU pass hold more of a batch. Measure with many concurrent requests before
changing it.

### ROCm on WSL2 setup

One-time host setup for the `rocm-wsl` backend. The image brings ROCm itself;
WSL only needs AMD's bridge library:

1. **Windows:** install the latest AMD Software: Adrenalin Edition driver for
   your GPU, then in an admin PowerShell run `wsl --update` and `wsl --shutdown`.
2. **WSL:** install [librocdxg](https://github.com/ROCm/librocdxg):

   ```bash
   cd /tmp
   curl -LO https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
   sudo dpkg -i rocdxg-roct_1.2.2_amd64.deb
   ls /opt/rocm/lib/librocdxg.so /opt/rocm/share/rocdxg/dids.conf /usr/lib/wsl/lib/libdxcore.so
   ```

If `start` reports no usable GPU, tell librocdxg the device ID (find it in
Windows Device Manager → the GPU → Details → Hardware Ids; `0x150E` is the
Radeon 890M, gfx1150):
`echo '0x150E,11,5,0' | sudo tee -a /opt/rocm/share/rocdxg/dids.conf`.

### LAN access under WSL2

Reaching the service from other machines needs WSL's mirrored networking and
inbound firewall rules for the port, in both Windows Defender and the Hyper-V
firewall. In an admin PowerShell (`{40E0AC32-…}` is WSL's VM creator id), then
`wsl --shutdown`:

```powershell
New-NetFirewallRule -DisplayName "Embedding service 8001" -Direction Inbound `
  -Protocol TCP -LocalPort 8001 -Action Allow
New-NetFirewallHyperVRule -Name "EmbeddingService8001" `
  -DisplayName "Embedding service 8001 (WSL)" -Direction Inbound `
  -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -Protocol TCP -LocalPorts 8001 -Action Allow
```

---

## Sample data & Git LFS

`sample_data/eli5_question_answer.jsonl` (~166 MB) is tracked with Git LFS (see
[`.gitattributes`](.gitattributes)). After cloning:

```bash
git lfs install
git lfs pull
```

The model cache (`model_cache/`), generated embeddings (`embedding/`), local
notes (`docs/`) and the start script's saved settings (`embedding_service.env`)
are gitignored.

---

## Project layout

```
.
├── src/
│   ├── models.py                     # MODELS registry (prompts, dims, GGUF builds)
│   ├── server.py                     # FastAPI gateway → llama-server per model
│   ├── fetch_models.py               # GGUF download for the containers (model-fetch)
│   ├── sample_embedding_generator.py # ELI5 QA → Parquet embeddings (per model)
│   └── cluster_centroid_generator.py # Parquet → K-means centroids (JSONL)
├── docker/
│   ├── gateway.Dockerfile            # gateway + model-fetch image
│   └── compose.<backend>.yml         # GPU overlays: rocm-wsl, rocm, cuda, vulkan
├── sample_data/
│   └── eli5_question_answer.jsonl    # sample QA pairs (Git LFS)
├── docker-compose.yml
├── embedding_service.sh              # start | stop | restart | status | logs
├── tests/
│   └── test_truncate.py              # unit tests: python -m unittest discover tests
├── test_embeddings.sh
├── generate_sample_embeddings.sh
├── generate_cluster_centroids.sh
├── pyproject.toml
└── README.md
```

---

## License

Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). See [LICENSE](LICENSE) for the terms.
