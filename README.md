# Embedding Service

A text embedding service built on [FastEmbed](https://github.com/qdrant/fastembed)
and Google's [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m),
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
- [How EmbeddingGemma is loaded](#how-embeddinggemma-is-loaded)
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
                     │  EmbeddingService  │  FastEmbed + EmbeddingGemma-300M
                     │  (embedding_service.py)
                     └─────────┬──────────┘
                               │
                        ONNX (onnxruntime)

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
- Network access on first run (the model is downloaded from Hugging Face)
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
{ "status": "healthy", "model": "google/embeddinggemma-300m" }
```

**Example**

```bash
curl -X POST http://localhost:8000/embedding/query \
  -H 'Content-Type: application/json' \
  -d '{"payloads": ["what is machine learning?"], "dimensions": 256}'
```

---

## The `EmbeddingService` class

[`src/embedding_service.py`](src/embedding_service.py) wraps FastEmbed and can be
used directly (in-process), without the HTTP server:

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

`start_embedding_service.sh` options: `--host`, `--port`, `--onnx-file`,
`--cache-dir`, `-f/--foreground`. Run any script with `-h` for details.

---

## Configuration

Environment variables (read by the service and start script):

| Variable | Default | Description |
| --- | --- | --- |
| `FASTEMBED_CACHE_PATH` | `/app/fastembed_cache` (`./fastembed_cache` via the start script) | Where the model is cached. |
| `FASTEMBED_ONNX_FILE` | `onnx/model.onnx` | ONNX build to load. Use `onnx/model_quantized.onnx` for a smaller/faster download. |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address for the start script. |

Available ONNX builds in the upstream repo include `onnx/model.onnx`
(full precision), `onnx/model_fp16.onnx`, and `onnx/model_quantized.onnx`.

---

## How EmbeddingGemma is loaded

EmbeddingGemma is **not** in FastEmbed's built-in registry, so `EmbeddingService`
registers it as a custom ONNX model on first use, pulling
[`onnx-community/embeddinggemma-300m-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)
from Hugging Face (MEAN pooling, normalisation enabled, native dim 768).

Because it is a custom model, FastEmbed does not auto-apply EmbeddingGemma's task
prompts — the service applies them itself:

- Documents: `title: none | text: <payload>`
- Queries: `task: search result | query: <payload>`

---

## Sample data & Git LFS

`sample_data/eli5_question_answer.jsonl` (~166 MB) is tracked via Git LFS (see
[`.gitattributes`](.gitattributes)). After cloning:

```bash
git lfs install
git lfs pull
```

The model cache (`fastembed_cache/`), generated embeddings (`embedding/`), and
service runtime files (`embedding_service.pid`, `embedding_service.log`) are
gitignored.

---

## Project layout

```
.
├── src/
│   ├── embedding_service.py          # EmbeddingService (FastEmbed wrapper)
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
