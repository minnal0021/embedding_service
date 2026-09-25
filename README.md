# Embedding Service

A text embedding service built on [PyTorch](https://pytorch.org/) (ROCm build) with
[sentence-transformers](https://www.sbert.net/) and Google's
[EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m),
plus an offline data-preparation pipeline that generates sample embeddings and
K-means cluster centroids.

The HTTP API matches the contract expected by the Rust client: a batch interface
with separate document and query endpoints, returning one L2-normalised vector
per payload.

---

## Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [HTTP API](#http-api)
- [The `EmbeddingService` class](#the-embeddingservice-class)
- [Offline pipeline: sample embeddings → centroids](#offline-pipeline-sample-embeddings--centroids)
- [Scripts](#scripts)
- [Configuration](#configuration)
- [GPU acceleration (AMD)](#gpu-acceleration-amd)
- [How EmbeddingGemma is loaded](#how-embeddinggemma-is-loaded)
- [Migrating from the ONNX runtime](#migrating-from-the-onnx-runtime)
- [Sample data & Git LFS](#sample-data--git-lfs)
- [Project layout](#project-layout)

---

## Architecture

```
                ┌─────────────────────────────┐
  HTTP clients  │  FastAPI server (server.py) │
  (Rust, curl)  │  /embedding/document        │
  ───────────►  │  /embedding/query           │
                │  /healthcheck               │
                └──────────────┬──────────────┘
                               │
                     ┌─────────▼──────────┐
                     │  EmbeddingService  │  sentence-transformers + EmbeddingGemma-300M
                     │  (embedding_service.py)
                     └─────────┬──────────┘
                               │
              PyTorch (ROCm/HIP: AMD GPU in bfloat16, else CPU in float32)

  Offline pipeline (no server required):

  sample_data/*.jsonl ─► SampleEmbeddingGenerator ─► embedding/*.parquet
                                                          │
                                                          ▼
                              ClusterCentroidGenerator ─► embedding/*.jsonl (centroids)
```

The model embeds documents and queries **asymmetrically** via task-specific
prompts. EmbeddingGemma supports **Matryoshka Representation Learning (MRL)**, so
callers can request a smaller `dimensions` value; vectors are truncated to that
width and re-normalised. All returned vectors are **L2-normalised**.

---

## Requirements

- Python **≥ 3.14**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Network access on first run (the ~1.2 GB model is downloaded from Hugging Face;
  the ROCm PyTorch wheel makes the `uv sync` download several GB)
- Linux x86_64 with an AMD GPU for GPU inference (see
  [GPU acceleration](#gpu-acceleration-amd)); anything else runs on the CPU
- [Git LFS](https://git-lfs.com/) to check out the sample data

Install dependencies:

```bash
uv sync
```

---

## Quick start

Start the service (syncs deps, downloads the model on first run, waits for
health, runs a smoke test):

```bash
./start_embedding_service.sh
```

In another terminal, run the test suite:

```bash
./test_embeddings.sh
```

Stop the service:

```bash
./stop_embedding_service.sh
```

---

## HTTP API

Base URL defaults to `http://localhost:8000`.

### `POST /embedding/document`
### `POST /embedding/query`

Both endpoints share the same request/response shape. Use `document` for content
being indexed and `query` for search queries — the model embeds them differently.

**Request**

```json
{
  "payloads": ["first text", "second text"],
  "dimensions": 768
}
```

- `payloads` — list of strings; one embedding is returned per payload, in order.
- `dimensions` — *optional*, defaults to **768** (the model's native width).
  Smaller values (e.g. 512, 256, 128) trigger Matryoshka truncation.

**Response**

```json
{
  "embeddings": [[0.012, -0.034, ...], [0.044, 0.001, ...]]
}
```

Each vector is exactly `dimensions` long and L2-normalised. An empty `payloads`
list returns `{"embeddings": []}` without invoking the model.

### `GET /healthcheck`

```json
{ "status": "healthy", "model": "google/embeddinggemma-300m", "device": "gpu" }
```

**Example**

```bash
curl -X POST http://localhost:8000/embedding/query \
  -H 'Content-Type: application/json' \
  -d '{"payloads": ["what is machine learning?"], "dimensions": 256}'
```

---

## The `EmbeddingService` class

[`src/embedding_service.py`](src/embedding_service.py) wraps sentence-transformers
and can be used directly (in-process), without the HTTP server:

```python
from embedding_service import EmbeddingService

service = EmbeddingService()                      # loads EmbeddingGemma-300M
docs = service.embed_documents(["the cat sat on the mat"], dimensions=256)
qry  = service.embed_query(["where did the cat sit?"], dimensions=256)
```

- `embed_documents(payloads, dimensions=None)` — applies the document prompt.
- `embed_query(payloads, dimensions=None)` — applies the query prompt.
- Both return `list[list[float]]`, truncated to `dimensions` (or native 768 if
  `None`) and L2-normalised.

Run its `main()` demo:

```bash
uv run python src/embedding_service.py
```

---

## Offline pipeline: sample embeddings → centroids

A two-stage offline pipeline turns the raw QA dataset into cluster centroids.
Both stages run in-process (no server needed).

### 1. Generate sample embeddings

[`src/sample_embedding_generator.py`](src/sample_embedding_generator.py) reads
ELI5-style `["question", "answer"]` JSONL records, embeds the combined
`question + "\n" + answer` text as a document, and writes a Parquet file.

```bash
uv run python src/sample_embedding_generator.py \
  --input sample_data/eli5_question_answer.jsonl \
  --output embedding/sample_data_embedding.parquet \
  --dimensions 768 \
  --batch-size 32 \
  --max-records 0          # 0 = all records
```

Output Parquet schema: `question : string`, `answer : string`,
`qa_embedding : list[float]`.

### 2. Generate cluster centroids

[`src/cluster_centroid_generator.py`](src/cluster_centroid_generator.py) loads the
Parquet, runs K-means, and writes centroids to JSONL.

```bash
./generate_cluster_centroids.sh
# or:
uv run python src/cluster_centroid_generator.py \
  --input embedding/sample_data_embedding.parquet \
  --output embedding/cluster_centroids.jsonl \
  --embedding-col qa_embedding \
  --dimensions 768 \
  --n-clusters 256 \
  --seed 42
```

Output JSONL (one object per line):

```json
{"cluster_id": 1, "centroid": [0.01, -0.02, ...]}
{"cluster_id": 2, "centroid": [0.03,  0.01, ...]}
```

`--dimensions` Matryoshka-truncates + renormalises the stored vectors before
clustering (default 768; `0` = use the stored width as-is). `--n-clusters`
defaults to 256 and is capped to the number of available embeddings.

---

## Scripts

| Script | Purpose |
| --- | --- |
| [`start_embedding_service.sh`](start_embedding_service.sh) | Sync deps, start the service in the background, wait for health, smoke test. `-f` runs foreground. |
| [`stop_embedding_service.sh`](stop_embedding_service.sh) | Stop the background service via its PID file. |
| [`test_embeddings.sh`](test_embeddings.sh) | Test suite: health, batch query/document embedding, dimension truncation. |
| [`generate_cluster_centroids.sh`](generate_cluster_centroids.sh) | Run the cluster centroid generator with project defaults. |

`start_embedding_service.sh` options: `--host`, `--port`, `--model`,
`--cache-dir`, `--device`, `--dtype`, `-f/--foreground`. Run any script with `-h`
for details.

---

## Configuration

Environment variables (read by the service and start script):

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `unsloth/embeddinggemma-300m` | Hugging Face repo to load. The default is an ungated mirror of `google/embeddinggemma-300m`. The official repo is gated: accept the Gemma licence on huggingface.co, set `HF_TOKEN` (or run `hf auth login`), then set this to `google/embeddinggemma-300m`. |
| `EMBEDDING_CACHE_PATH` | `/app/model_cache` (`./model_cache` via the start script) | Where the model is cached. |
| `EMBEDDING_DEVICE` | `auto` | `auto` uses a supported AMD GPU when PyTorch finds one, else the CPU. `gpu` requires the GPU (startup fails without it); `cpu` forces the CPU. |
| `EMBEDDING_DTYPE` | `bfloat16` on GPU, `float32` on CPU | Inference precision. `float16` is not offered: EmbeddingGemma's activations overflow in fp16. |
| `EMBEDDING_GPU_INDEX` | *(auto)* | Pin a GPU by PyTorch index instead of auto-selecting. |
| `EMBEDDING_BATCH_SIZE` | `32` on GPU, `16` on CPU | Texts per forward pass. |
| `EMBEDDING_MAX_SEQ_LENGTH` | `2048` (model native) | Truncate inputs to this many tokens. |
| `EMBEDDING_MAX_BATCH_TEXTS` | `256` | Server: cap on texts gathered from concurrent requests into one model call. |
| `EMBEDDING_BATCH_WAIT_MS` | `2` | Server: how long a busy batch worker waits for more requests after taking the backlog. An idle worker skips the wait and runs the first request immediately. |
| `HOST` / `PORT` | `0.0.0.0` / `8001` | Bind address for the start script. |

---

## GPU acceleration (AMD)

The service runs on PyTorch's **ROCm** build (`torch==…+rocm7.1`, pulled from
PyTorch's ROCm wheel index; see `[tool.uv.sources]` in `pyproject.toml`). Inference
runs natively on the GPU through HIP, using the matrix cores in **bfloat16**.
The wheel bundles its own ROCm libraries. The host needs only:

- the in-kernel `amdgpu` driver (stock on recent Ubuntu kernels);
- read/write access to `/dev/kfd` and `/dev/dri/renderD*` (the `render` group,
  or the ACL a logged-in desktop session grants);
- a GPU architecture the wheel ships kernels for. The ROCm 7.1 wheel covers
  gfx900–gfx950, including RDNA3 (gfx110x) and RDNA4 (gfx1200/gfx1201, e.g.
  Radeon AI PRO R9700).

With `EMBEDDING_DEVICE=auto` (the default), the service picks among the GPUs
PyTorch can see. It skips any whose architecture the wheel has no kernels for
(e.g. an integrated Radeon next to a discrete card) and takes the one with the
most compute units. If there is none, it uses the CPU. The chosen device,
dtype, and batch size are logged at startup, and `/healthcheck` returns the
device. The ROCm index only has Linux x86_64 wheels; other platforms install
the regular PyPI `torch` and run on the CPU (or pin a GPU build yourself).

### Cross-request batching

Clients such as minnal send one small request per document (the whole text plus
~4 chunks, about 5 texts). Each model call has a fixed cost of about 15 ms, and
larger calls pad less because texts are length-sorted within a call. So the
server batches **across** requests: every request goes onto a queue, and a
single worker runs one model call for everything queued (document and query
texts together, each already carrying its task prompt). It then splits the
vectors back and applies each request's own `dimensions`. If a shared call
fails, each request is retried alone, so one bad request cannot fail its
neighbours. The model call runs in a worker thread, so `/healthcheck` stays
responsive under load (p50 < 1 ms).

### Throughput

Measured on a Radeon AI PRO R9700 with SciFact abstracts, one document per
request (whole text + 4-sentence chunks):

| Runtime | Requests in flight | docs/s |
| --- | ---: | ---: |
| Previous: FastEmbed + onnxruntime-webgpu (WebGPU over Vulkan/RADV), fp32 | 8 | ~4 |
| PyTorch ROCm, bfloat16 | 1 | ~30 |
| PyTorch ROCm, bfloat16 | 8 | ~36 |
| PyTorch ROCm, bfloat16 | 16 | ~43 |
| PyTorch ROCm, bfloat16 | 64 | **~58** |
| *Model only, in-process, 64 docs per call (ceiling)* | — | *~67* |

Throughput tracks the batch size the server can form, so **keep many requests
in flight** (32–64) to get the most from the GPU. bfloat16 and float32
embeddings agree to a cosine similarity of ≥ 0.9999, and a request batched with
others gets the same vectors as when sent alone (cosine ≥ 0.9998, bfloat16
rounding).

---

## How EmbeddingGemma is loaded

`EmbeddingService` loads the model with sentence-transformers, which applies
EmbeddingGemma's full published pipeline:

```
transformer → mean pooling → Dense 768→3072 → Dense 3072→768 → L2 normalise
```

EmbeddingGemma embeds queries and documents asymmetrically. The service applies
the task prompts itself:

- Documents: `title: none | text: <payload>`
- Queries: `task: search result | query: <payload>`

---

## Migrating from the ONNX runtime

The previous runtime registered `onnx-community/embeddinggemma-300m-ONNX` with
FastEmbed as a custom model with MEAN pooling. That mean-pooled the transformer's
raw hidden states and **skipped both Dense projections**, so its vectors were not
EmbeddingGemma's trained embeddings. They match the new service's
transformer + pooling stage (cosine ≈ 0.9996) but are unrelated to its final
output (cosine ≈ 0.0).

**Existing vectors and centroids are incompatible with this service.** Anything
built on the old embeddings must be regenerated with the new service:

1. Sample embeddings: `./generate_sample_embeddings.sh`.
2. Cluster centroids: `./generate_cluster_centroids.sh`, then install the new
   `clusters.json` wherever the consumer (e.g. minnal) loads its centroids.
3. Re-index every stored vector in the consumer, and clear any query-embedding
   caches.

The old `fastembed_cache/` directory is no longer used and can be deleted.

---

## Sample data & Git LFS

`sample_data/eli5_question_answer.jsonl` (~166 MB) is tracked via Git LFS (see
[`.gitattributes`](.gitattributes)). After cloning:

```bash
git lfs install
git lfs pull
```

The model cache (`model_cache/`), generated embeddings (`embedding/`), and
service runtime files (`embedding_service.pid`, `embedding_service.log`) are
gitignored.

---

## Project layout

```
.
├── src/
│   ├── embedding_service.py          # EmbeddingService (PyTorch / sentence-transformers)
│   ├── server.py                     # FastAPI HTTP server
│   ├── sample_embedding_generator.py # ELI5 QA → Parquet embeddings
│   └── cluster_centroid_generator.py # Parquet → K-means centroids (JSONL)
├── sample_data/
│   └── eli5_question_answer.jsonl    # sample QA pairs (Git LFS)
├── start_embedding_service.sh
├── stop_embedding_service.sh
├── test_embeddings.sh
├── generate_cluster_centroids.sh
├── pyproject.toml
└── README.md
```
