import os

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType


class EmbeddingService:
    """Generate text embeddings with the EmbeddingGemma-300M model via FastEmbed.

    EmbeddingGemma is not in FastEmbed's built-in registry, so it is registered
    as a custom ONNX model from `onnx-community/embeddinggemma-300m-ONNX` on the
    first instantiation. The model embeds documents and queries asymmetrically by
    way of task-specific prompt prefixes (it is *not* handled by FastEmbed's
    `query_embed`/`passage_embed` for a custom model), so we apply those prompts
    ourselves. EmbeddingGemma supports Matryoshka representation, so callers may
    request a truncated `dimensions`; vectors are sliced and re-normalised.
    """

    DEFAULT_MODEL_NAME = "google/embeddinggemma-300m"
    DEFAULT_CACHE_DIR = "/app/fastembed_cache"
    ONNX_SOURCE = "onnx-community/embeddinggemma-300m-ONNX"
    # Native (untruncated) embedding width of the model.
    FULL_DIMENSION = 768

    # EmbeddingGemma's documented task prompts (from the model card).
    QUERY_PROMPT = "task: search result | query: "
    DOCUMENT_PROMPT = "title: none | text: "

    _registered = False

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        onnx_file: str | None = None,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        # Pull custom cache path or fall back to the local cached folder.
        self.cache_dir = cache_dir or os.getenv(
            "FASTEMBED_CACHE_PATH", self.DEFAULT_CACHE_DIR
        )
        # Defaults to full-precision ONNX; override (e.g. model_quantized.onnx)
        # via arg or FASTEMBED_ONNX_FILE for a smaller/faster build.
        onnx_file = onnx_file or os.getenv("FASTEMBED_ONNX_FILE", "onnx/model.onnx")

        self._register(self.model_name, onnx_file)

        print(f"Loading {self.model_name} from {self.cache_dir}...")
        self.encoder = TextEmbedding(
            model_name=self.model_name, cache_dir=self.cache_dir
        )

    @classmethod
    def _register(cls, model_name: str, onnx_file: str) -> None:
        """Register EmbeddingGemma's ONNX build with FastEmbed (idempotent)."""
        if cls._registered:
            return
        # External-data file sits next to the .onnx graph; ship it too.
        TextEmbedding.add_custom_model(
            model=model_name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=cls.ONNX_SOURCE),
            dim=cls.FULL_DIMENSION,
            model_file=onnx_file,
            additional_files=[onnx_file + "_data"],
        )
        cls._registered = True

    def embed_documents(
        self, payloads: list[str], dimensions: int | None = None
    ) -> list[list[float]]:
        """Embed payloads as documents (for indexing)."""
        prompted = [self.DOCUMENT_PROMPT + p for p in payloads]
        return self._embed(self.encoder.embed(prompted), dimensions)

    def embed_query(
        self, payloads: list[str], dimensions: int | None = None
    ) -> list[list[float]]:
        """Embed payloads as search queries."""
        prompted = [self.QUERY_PROMPT + p for p in payloads]
        return self._embed(self.encoder.embed(prompted), dimensions)

    def _embed(self, vectors, dimensions: int | None) -> list[list[float]]:
        """Truncate (Matryoshka) + L2-normalise FastEmbed's vectors to lists.

        Order is preserved: vectors are emitted in the same order the model
        yields them, which matches the input payload order.
        """
        out: list[list[float]] = []
        for vector in vectors:
            vec = np.asarray(vector, dtype=np.float32)
            if dimensions is not None and dimensions < vec.shape[0]:
                # Matryoshka truncation; renormalise the shortened vector below.
                vec = vec[:dimensions]
            # Always return L2-normalised vectors (a no-op if already unit-norm).
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out.append(vec.tolist())
        return out


def main() -> None:
    service = EmbeddingService()
    sample_texts = [
        "FastEmbed makes embedding generation fast and lightweight.",
        "EmbeddingGemma is a compact text embedding model.",
    ]
    embeddings = service.embed_documents(sample_texts, dimensions=256)
    for text, vector in zip(sample_texts, embeddings):
        print(f"[{len(vector)} dims] {text!r} -> {vector[:5]}...")


if __name__ == "__main__":
    main()
