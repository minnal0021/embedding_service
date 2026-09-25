"""HTTP interface for the embedding service.

Routes match the Rust client contract exactly:

    POST {base}/embedding/document   {"payloads":[...],"dimensions":D}  ->  {"embeddings":[[f32], ...]}
    POST {base}/embedding/query      (same request/response shape)
    GET  {base}/healthcheck          -> 200

One embedding is returned per payload; the model is fixed server-side.

Requests are batched **across** clients: every request's payloads go onto a
queue, and a single worker runs one model call for everything queued (document
and query texts alike, since each already carries its task prompt), then splits
the vectors back per request. Clients typically send one small request per
document (~5 texts), so this is what fills the GPU. The model call runs in a
worker thread, keeping the event loop — and `/healthcheck` — responsive.

Knobs (env): EMBEDDING_MAX_BATCH_TEXTS (default 256) caps the texts per model
call; EMBEDDING_BATCH_WAIT_MS (default 2) is how long a busy worker waits for
more requests after taking the backlog. An idle worker runs the first request
immediately, so a lone request never pays the wait.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embedding_service import EmbeddingService

logger = logging.getLogger("uvicorn.error")

# Loaded once at import; runs on the GPU when one is found (see EmbeddingService).
service = EmbeddingService()


@dataclass
class _Job:
    """One request's prompted texts, awaiting its slice of a batch."""

    texts: list[str]
    dimensions: int
    future: asyncio.Future


class DynamicBatcher:
    """Coalesce concurrent requests into shared model calls."""

    def __init__(self, embedder: EmbeddingService, max_texts: int, max_wait_s: float):
        self.embedder = embedder
        self.max_texts = max_texts
        self.max_wait_s = max_wait_s
        self.queue: asyncio.Queue[_Job] = asyncio.Queue()

    async def submit(self, texts: list[str], dimensions: int) -> list[list[float]]:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(_Job(texts, dimensions, future))
        return await future

    async def run(self) -> None:
        """Worker loop: take the first job, gather more, run one model call."""
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
            results = await asyncio.to_thread(self._process, jobs)
            for job, result in zip(jobs, results):
                if job.future.done():  # client disconnected / cancelled
                    continue
                if isinstance(result, Exception):
                    job.future.set_exception(result)
                else:
                    job.future.set_result(result)

    def _process(self, jobs: list[_Job]) -> list[list[list[float]] | Exception]:
        """Embed all jobs' texts in one call; on failure, retry each job alone
        so one bad request cannot fail the others sharing its batch."""
        try:
            vectors = self.embedder.encode_texts([t for job in jobs for t in job.texts])
        except Exception as e:  # noqa: BLE001 — isolate the failing request below
            if len(jobs) == 1:
                logger.exception("Embedding %d texts failed", len(jobs[0].texts))
                return [e]
            return [r for job in jobs for r in self._process([job])]
        results, offset = [], 0
        for job in jobs:
            part = vectors[offset : offset + len(job.texts)]
            offset += len(job.texts)
            results.append(self.embedder.finalize(part, job.dimensions))
        return results


batcher = DynamicBatcher(
    service,
    max_texts=int(os.getenv("EMBEDDING_MAX_BATCH_TEXTS", "256")),
    max_wait_s=float(os.getenv("EMBEDDING_BATCH_WAIT_MS", "2")) / 1000,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    worker = asyncio.create_task(batcher.run())
    try:
        yield
    finally:
        worker.cancel()


app = FastAPI(title="EmbeddingGemma-300M API", lifespan=lifespan)


class BatchEmbedRequest(BaseModel):
    payloads: list[str]
    # Defaults to the model's full width when the client omits it.
    dimensions: int = EmbeddingService.FULL_DIMENSION


class BatchEmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.post("/embedding/document", response_model=BatchEmbedResponse)
async def embed_document(request: BatchEmbedRequest) -> BatchEmbedResponse:
    return await _embed(service.prompt_documents, request)


@app.post("/embedding/query", response_model=BatchEmbedResponse)
async def embed_query(request: BatchEmbedRequest) -> BatchEmbedResponse:
    return await _embed(service.prompt_queries, request)


@app.get("/healthcheck")
async def healthcheck() -> dict[str, str]:
    return {"status": "healthy", "model": service.model_name, "device": service.device}


async def _embed(prompt_fn, request: BatchEmbedRequest) -> BatchEmbedResponse:
    # Empty payloads short-circuit to an empty result (the Rust client never
    # posts these, but mirror its no-op semantics rather than erroring).
    if not request.payloads:
        return BatchEmbedResponse(embeddings=[])
    try:
        embeddings = await batcher.submit(prompt_fn(request.payloads), request.dimensions)
    except Exception as e:  # noqa: BLE001 — surface model errors as 500s
        raise HTTPException(status_code=500, detail=str(e))
    return BatchEmbedResponse(embeddings=embeddings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
