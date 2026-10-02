"""HTTP interface for the embedding service (the gateway).

The caller picks the model in the path; `{model}` is one of the loaded models
(`gemma`, `qwen`). Any other value, including a known model that this server
was not started with (see EMBEDDING_MODELS), returns 404.

    POST {base}/embedding/{model}/document   {"payloads":[...],"dimensions":D}  ->  {"embeddings":[[f32], ...]}
    POST {base}/embedding/{model}/query      (same request/response shape)
    GET  {base}/healthcheck                  -> 200 once every model is ready
                                                (503 while loading), lists them

One embedding is returned per payload. `dimensions` defaults to 768 for every
model (Matryoshka-truncated from qwen's native 4096); ask for up to the native
width (gemma 768, qwen 4096) explicitly.

The models themselves run in llama.cpp's `llama-server`, one per model (see
docker-compose.yml). This gateway adds each model's prompt prefix, forwards the
texts to that model's server, then truncates and L2-normalises the vectors.

Requests are batched **across** clients, per model: every request's payloads
go onto that model's queue, and a single worker sends one llama-server call
for everything queued (document and query texts alike, since each already
carries its task prompt), then splits the vectors back per request. Clients
typically send one small request per document (~5 texts), so this is what
fills the GPU; llama-server then runs the texts of a call in parallel across
its slots (`-np` in docker-compose.yml).

Environment:
    EMBEDDING_MODELS          models to serve (default: gemma,qwen)
    LLAMA_<MODEL>_URL         llama-server base URL per model
                              (default: http://llama-<model>:8080)
    LLAMA_BACKEND             llama.cpp image backend, reported by /healthcheck
                              (rocm, vulkan, cuda or cpu; default: cpu)
    EMBEDDING_REQUEST_TIMEOUT seconds to wait for one llama-server call
                              (default: 600)
    EMBEDDING_MAX_BATCH_TEXTS caps the texts per llama-server call (default 256)
    EMBEDDING_BATCH_WAIT_MS   how long a busy worker waits for more requests
                              after taking the backlog (default 2). An idle
                              worker sends the first request immediately, so a
                              lone request never pays the wait.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from models import DEFAULT_DIMENSIONS, MODELS, ModelSpec, gguf_file, parse_model_keys

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class LlamaServer:
    """One model's llama-server, as seen from the gateway."""

    key: str
    spec: ModelSpec
    url: str
    gguf_file: str


SERVERS = {
    key: LlamaServer(
        key=key,
        spec=MODELS[key],
        url=os.getenv(f"LLAMA_{key.upper()}_URL", f"http://llama-{key}:8080").rstrip("/"),
        gguf_file=gguf_file(MODELS[key]),
    )
    for key in parse_model_keys(os.getenv("EMBEDDING_MODELS", "gemma,qwen"))
}
BACKEND = os.getenv("LLAMA_BACKEND", "cpu")
DEVICE = "cpu" if BACKEND == "cpu" else "gpu"
REQUEST_TIMEOUT = float(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "600"))


class ModelServerError(Exception):
    """A llama-server call failed; `status_code` is what the client gets."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class _Job:
    """One request's prompted texts, awaiting its slice of a batch."""

    texts: list[str]
    dimensions: int
    future: asyncio.Future


class DynamicBatcher:
    """Coalesce concurrent requests for one model into shared llama-server calls."""

    def __init__(self, server: LlamaServer, max_texts: int, max_wait_s: float):
        self.server = server
        self.max_texts = max_texts
        self.max_wait_s = max_wait_s
        self.queue: asyncio.Queue[_Job] = asyncio.Queue()

    async def submit(self, texts: list[str], dimensions: int) -> list[list[float]]:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(_Job(texts, dimensions, future))
        return await future

    async def run(self, client: httpx.AsyncClient) -> None:
        """Worker loop: take the first job, gather more, send one call."""
        loop = asyncio.get_running_loop()
        while True:
            # An empty queue here means the worker is idle: run the next
            # request immediately rather than adding the wait to its latency.
            idle = self.queue.empty()
            jobs = [await self.queue.get()]
            count = len(jobs[0].texts)
            # Under load, take the backlog that queued while the previous batch
            # ran, then wait briefly so more near-simultaneous requests share.
            deadline = loop.time() + (0 if idle else self.max_wait_s)
            while count < self.max_texts:
                try:
                    job = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        job = await asyncio.wait_for(self.queue.get(), remaining)
                    except TimeoutError:
                        break
                jobs.append(job)
                count += len(job.texts)
            results = await self._process(client, jobs)
            for job, result in zip(jobs, results):
                if job.future.done():  # client disconnected / cancelled
                    continue
                if isinstance(result, Exception):
                    job.future.set_exception(result)
                else:
                    job.future.set_result(result)

    async def _process(
        self, client: httpx.AsyncClient, jobs: list[_Job]
    ) -> list[list[list[float]] | Exception]:
        """Embed all jobs' texts in one call; on failure, retry each job alone
        so one bad request cannot fail the others sharing its batch."""
        try:
            vectors = await self._embed(client, [t for job in jobs for t in job.texts])
        except Exception as e:  # noqa: BLE001 — isolate the failing request below
            if len(jobs) == 1:
                logger.warning("%s: embedding %d texts failed: %s",
                               self.server.key, len(jobs[0].texts), e)
                return [e]
            return [r for job in jobs for r in await self._process(client, [job])]
        results, offset = [], 0
        for job in jobs:
            part = vectors[offset : offset + len(job.texts)]
            offset += len(job.texts)
            results.append(_finalize(part, job.dimensions))
        return results

    async def _embed(self, client: httpx.AsyncClient, texts: list[str]) -> np.ndarray:
        """Embed already-prompted `texts` at full width, in input order."""
        body = {"input": texts, "encoding_format": "float"}
        try:
            r = await client.post(f"{self.server.url}/v1/embeddings", json=body)
        except httpx.HTTPError as e:
            raise ModelServerError(
                503, f"{self.server.key} model server unavailable: {e!r}"
            )
        if r.status_code != 200:
            # 503 = still loading; anything else (e.g. an input longer than the
            # context) is a model error, surfaced as a 500.
            raise ModelServerError(
                503 if r.status_code == 503 else 500,
                f"{self.server.key}: {_error_message(r)}",
            )
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return np.asarray([d["embedding"] for d in data], dtype=np.float32)


BATCHERS = {
    key: DynamicBatcher(
        server,
        max_texts=int(os.getenv("EMBEDDING_MAX_BATCH_TEXTS", "256")),
        max_wait_s=float(os.getenv("EMBEDDING_BATCH_WAIT_MS", "2")) / 1000,
    )
    for key, server in SERVERS.items()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for all llama-servers, plus one batch worker per model.
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=5)
    ) as client:
        app.state.client = client
        workers = [asyncio.create_task(b.run(client)) for b in BATCHERS.values()]
        try:
            yield
        finally:
            for worker in workers:
                worker.cancel()


app = FastAPI(title="Embedding API", lifespan=lifespan)


class BatchEmbedRequest(BaseModel):
    payloads: list[str]
    # Same default for every model; values above the native width return it.
    dimensions: int = Field(default=DEFAULT_DIMENSIONS, ge=1)


class BatchEmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.post("/embedding/{model}/document", response_model=BatchEmbedResponse)
async def embed_document(model: str, request: BatchEmbedRequest) -> BatchEmbedResponse:
    server = _server(model)
    return await _embed(server, server.spec.document_prompt, request)


@app.post("/embedding/{model}/query", response_model=BatchEmbedResponse)
async def embed_query(model: str, request: BatchEmbedRequest) -> BatchEmbedResponse:
    server = _server(model)
    return await _embed(server, server.spec.query_prompt, request)


@app.get("/healthcheck")
async def healthcheck() -> JSONResponse:
    statuses = await asyncio.gather(*(_llama_status(s) for s in SERVERS.values()))
    ready = all(status == "ok" for status in statuses)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "healthy" if ready else "starting",
            "models": {
                s.key: {
                    "name": s.spec.name,
                    "file": s.gguf_file,
                    "dimension": s.spec.dimension,
                    "default_dimension": DEFAULT_DIMENSIONS,
                    "device": DEVICE,
                    "backend": BACKEND,
                    "status": status,
                }
                for s, status in zip(SERVERS.values(), statuses)
            },
        },
    )


def _server(model: str) -> LlamaServer:
    # A plain str path param (not an Enum) so unknown names are 404, not 422.
    if model not in SERVERS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model {model!r}; available: {', '.join(SERVERS)}",
        )
    return SERVERS[model]


async def _llama_status(server: LlamaServer) -> str:
    """"ok", "loading" (llama-server answers 503) or "unreachable"."""
    try:
        r = await app.state.client.get(f"{server.url}/health", timeout=5)
    except httpx.HTTPError:
        return "unreachable"
    return "ok" if r.status_code == 200 else "loading"


async def _embed(
    server: LlamaServer, prompt: str, request: BatchEmbedRequest
) -> BatchEmbedResponse:
    # Empty payloads short-circuit to an empty result (the Rust client never
    # posts these, but mirror its no-op semantics rather than erroring).
    if not request.payloads:
        return BatchEmbedResponse(embeddings=[])
    texts = [prompt + p for p in request.payloads]
    try:
        embeddings = await BATCHERS[server.key].submit(texts, request.dimensions)
    except ModelServerError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:  # noqa: BLE001 — surface model errors as 500s
        raise HTTPException(status_code=500, detail=str(e))
    return BatchEmbedResponse(embeddings=embeddings)


def _finalize(vectors: np.ndarray, dimensions: int) -> list[list[float]]:
    """Matryoshka-truncate `vectors` to `dimensions` and L2-normalise."""
    vectors = vectors[:, :dimensions]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    return vectors.tolist()


def _error_message(r: httpx.Response) -> str:
    try:
        return r.json()["error"]["message"]
    except Exception:  # noqa: BLE001 — not llama-server's JSON error shape
        return f"HTTP {r.status_code}: {r.text[:500]}"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
