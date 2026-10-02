"""Static descriptions of the servable embedding models.

Shared by the gateway (src/server.py), the model download (src/fetch_models.py)
and the offline pipeline. How each model runs (pooling, context size, slots)
is set on its llama-server in docker-compose.yml.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one servable embedding model."""

    # Display name of the upstream model.
    name: str
    # Hugging Face repo + default file of the GGUF build.
    repo: str
    gguf_file: str
    # Env var that overrides `gguf_file` (e.g. to pick another quantization).
    gguf_env: str
    # Native (untruncated) embedding width.
    dimension: int
    # Prompt prefixes for asymmetric query/document embedding (model card).
    query_prompt: str
    document_prompt: str


# Width returned when the caller doesn't ask for one, for every model (gemma's
# native width; qwen's 4096 is Matryoshka-truncated to it). Keeps vectors from
# either model drop-in compatible with 768-dim indexes.
DEFAULT_DIMENSIONS = 768

# Models selectable via the `/embedding/{model}/...` path. Both are served by
# llama.cpp from official GGUF builds and support Matryoshka truncation.
MODELS: dict[str, ModelSpec] = {
    "gemma": ModelSpec(
        name="google/embeddinggemma-300m",
        repo="ggml-org/embeddinggemma-300M-GGUF",
        gguf_file="embeddinggemma-300M-Q8_0.gguf",
        gguf_env="EMBEDDING_GEMMA_GGUF",
        dimension=768,
        query_prompt="task: search result | query: ",
        document_prompt="title: none | text: ",
    ),
    "qwen": ModelSpec(
        name="Qwen/Qwen3-Embedding-8B",
        repo="Qwen/Qwen3-Embedding-8B-GGUF",
        # Q4_K_M: ~5 GB, fits next to gemma on a 16 GB iGPU; Q8_0 is closer
        # to full precision but needs ~4 GB more.
        gguf_file="Qwen3-Embedding-8B-Q4_K_M.gguf",
        gguf_env="EMBEDDING_QWEN_GGUF",
        dimension=4096,
        query_prompt=(
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: "
        ),
        # Documents are embedded without an instruction.
        document_prompt="",
    ),
}


def gguf_file(spec: ModelSpec) -> str:
    """The GGUF build to use: the model's EMBEDDING_<MODEL>_GGUF or its default."""
    return os.getenv(spec.gguf_env) or spec.gguf_file


def parse_model_keys(value: str | list[str]) -> list[str]:
    """Parse a model list ("gemma,qwen" or a list), failing fast on a typo."""
    if isinstance(value, str):
        value = value.split(",")
    keys = list(dict.fromkeys(k.strip() for k in value if k.strip()))
    unknown = [k for k in keys if k not in MODELS]
    if unknown or not keys:
        raise ValueError(
            f"Invalid model list {keys!r}; choose from: {', '.join(MODELS)}"
        )
    return keys
