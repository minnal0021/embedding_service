"""HTTP interface for the embedding service.

Routes match the Rust client contract exactly:

    POST {base}/embedding/document   {"payloads":[...],"dimensions":D}  ->  {"embeddings":[[f32], ...]}
    POST {base}/embedding/query      (same request/response shape)
    GET  {base}/healthcheck          -> 200

One embedding is returned per payload; the model is fixed server-side.
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embedding_service import EmbeddingService

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="EmbeddingGemma-300M API")

# Loaded once at import; runs on the GPU when one is found (see EmbeddingService).
service = EmbeddingService()


class BatchEmbedRequest(BaseModel):
    payloads: list[str]
    # Defaults to the model's full width when the client omits it.
    dimensions: int = EmbeddingService.FULL_DIMENSION


class BatchEmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.post("/embedding/document", response_model=BatchEmbedResponse)
async def embed_document(request: BatchEmbedRequest) -> BatchEmbedResponse:
    return _embed(service.embed_documents, request)


@app.post("/embedding/query", response_model=BatchEmbedResponse)
async def embed_query(request: BatchEmbedRequest) -> BatchEmbedResponse:
    return _embed(service.embed_query, request)


@app.get("/healthcheck")
async def healthcheck() -> dict[str, str]:
    return {"status": "healthy", "model": service.model_name, "device": service.device}


def _embed(embed_fn, request: BatchEmbedRequest) -> BatchEmbedResponse:
    # Empty payloads short-circuit to an empty result (the Rust client never
    # posts these, but mirror its no-op semantics rather than erroring).
    if not request.payloads:
        return BatchEmbedResponse(embeddings=[])
    try:
        embeddings = embed_fn(request.payloads, request.dimensions)
    except Exception as e:  # noqa: BLE001 — surface model errors as 500s
        logger.exception("Embedding %d payloads failed", len(request.payloads))
        raise HTTPException(status_code=500, detail=str(e))
    return BatchEmbedResponse(embeddings=embeddings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
