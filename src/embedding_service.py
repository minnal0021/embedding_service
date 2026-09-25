import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate text embeddings with EmbeddingGemma-300M on PyTorch.

    The model runs through sentence-transformers, which applies EmbeddingGemma's
    full pipeline: transformer → mean pooling → two Dense projections
    (768 → 3072 → 768) → L2 normalisation. (The previous FastEmbed/ONNX runtime
    mean-pooled the raw hidden states and skipped both projections, so its
    vectors were not EmbeddingGemma's trained embeddings.)

    The model embeds documents and queries asymmetrically by way of
    task-specific prompt prefixes, which we apply ourselves. EmbeddingGemma
    supports Matryoshka representation, so callers may request a truncated
    `dimensions`; vectors are sliced and re-normalised.

    Inference runs on an AMD GPU when PyTorch's ROCm build finds a supported one
    (natively through HIP, in bfloat16), otherwise on the CPU (float32).
    `device` (or the EMBEDDING_DEVICE env var) overrides this: "auto" (default),
    "gpu", "cpu".
    """

    # Ungated mirror of google/embeddinggemma-300m. The official repo is gated
    # (Gemma licence + Hugging Face token); point EMBEDDING_MODEL at it once a
    # token is configured.
    DEFAULT_MODEL_NAME = "unsloth/embeddinggemma-300m"
    DEFAULT_CACHE_DIR = "/app/model_cache"
    # Native (untruncated) embedding width of the model.
    FULL_DIMENSION = 768

    # EmbeddingGemma's documented task prompts (from the model card).
    QUERY_PROMPT = "task: search result | query: "
    DOCUMENT_PROMPT = "title: none | text: "

    DEVICES = ("auto", "gpu", "cpu")
    # EmbeddingGemma does not support float16 (activations overflow); the model
    # card recommends bfloat16 or float32.
    DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}

    # Texts per forward pass. 32 measured fastest on a Radeon AI PRO R9700:
    # larger batches mix whole documents with short chunks and waste compute on
    # padding. Override with EMBEDDING_BATCH_SIZE.
    GPU_BATCH_SIZE = 32
    CPU_BATCH_SIZE = 16

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", self.DEFAULT_MODEL_NAME
        )
        self.cache_dir = cache_dir or os.getenv(
            "EMBEDDING_CACHE_PATH", self.DEFAULT_CACHE_DIR
        )

        requested = (device or os.getenv("EMBEDDING_DEVICE", "auto")).lower()
        if requested not in self.DEVICES:
            raise ValueError(f"device must be one of {self.DEVICES}, got {requested!r}")
        torch_device = self._select_device(requested)
        self.device = "cpu" if torch_device == "cpu" else "gpu"

        dtype_name = (
            dtype
            or os.getenv("EMBEDDING_DTYPE")
            or ("bfloat16" if self.device == "gpu" else "float32")
        ).lower()
        if dtype_name not in self.DTYPES:
            raise ValueError(f"dtype must be one of {tuple(self.DTYPES)}, got {dtype_name!r}")
        self.dtype = dtype_name

        batch_override = os.getenv("EMBEDDING_BATCH_SIZE")
        if batch_override:
            self.batch_size = int(batch_override)
        else:
            self.batch_size = self.GPU_BATCH_SIZE if self.device == "gpu" else self.CPU_BATCH_SIZE

        print(f"Loading {self.model_name} from {self.cache_dir}...")
        self.model = SentenceTransformer(
            self.model_name,
            device=torch_device,
            cache_folder=self.cache_dir,
            model_kwargs={"torch_dtype": self.DTYPES[dtype_name]},
        )
        max_len = os.getenv("EMBEDDING_MAX_SEQ_LENGTH")
        if max_len:
            self.model.max_seq_length = int(max_len)
        # Warm up so the first request doesn't pay kernel selection/compilation.
        self.model.encode(["warmup"], batch_size=1)

        gpu = f" [{torch.cuda.get_device_name(torch_device)}]" if self.device == "gpu" else ""
        print(
            f"Embedding device: {self.device} ({torch_device}{gpu}), dtype {self.dtype}, "
            f"batch size {self.batch_size}, max_seq_length {self.model.max_seq_length}"
        )

    @staticmethod
    def _select_device(requested: str) -> str:
        """Pick the torch device for `requested` ("auto" | "gpu" | "cpu").

        Among the GPUs PyTorch can see, only those whose architecture this torch
        build ships kernels for are usable (e.g. an integrated Radeon next to a
        discrete card may not be); of those, the one with the most compute units
        wins. EMBEDDING_GPU_INDEX pins a specific GPU instead.
        """
        if requested == "cpu":
            return "cpu"
        candidates: list[tuple[int, int]] = []
        if torch.cuda.is_available():
            supported = set(torch.cuda.get_arch_list())
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                arch = getattr(props, "gcnArchName", "").split(":")[0]
                if not supported or arch in supported:
                    candidates.append((props.multi_processor_count, i))
        pinned = os.getenv("EMBEDDING_GPU_INDEX")
        if pinned is not None:
            index = int(pinned)
            if index not in [i for _, i in candidates]:
                raise RuntimeError(f"EMBEDDING_GPU_INDEX={index} is not a usable GPU")
            return f"cuda:{index}"
        if candidates:
            return f"cuda:{max(candidates)[1]}"
        if requested == "gpu":
            raise RuntimeError("GPU requested but PyTorch found no supported GPU")
        return "cpu"

    def embed_documents(
        self, payloads: list[str], dimensions: int | None = None
    ) -> list[list[float]]:
        """Embed payloads as documents (for indexing)."""
        return self.finalize(self.encode_texts(self.prompt_documents(payloads)), dimensions)

    def embed_query(
        self, payloads: list[str], dimensions: int | None = None
    ) -> list[list[float]]:
        """Embed payloads as search queries."""
        return self.finalize(self.encode_texts(self.prompt_queries(payloads)), dimensions)

    # The three steps below are public so the HTTP server can batch across
    # requests: prompt each request's payloads, encode every request's texts in
    # one call, then finalize each request's slice at its own `dimensions`.

    def prompt_documents(self, payloads: list[str]) -> list[str]:
        """Apply EmbeddingGemma's document task prompt."""
        return [self.DOCUMENT_PROMPT + p for p in payloads]

    def prompt_queries(self, payloads: list[str]) -> list[str]:
        """Apply EmbeddingGemma's query task prompt."""
        return [self.QUERY_PROMPT + p for p in payloads]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Embed already-prompted `texts` at full width, as a float32 array.

        Order is preserved: `encode` returns vectors in input order (it sorts
        by length internally to minimise padding, then restores the order).
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    @staticmethod
    def finalize(vectors: np.ndarray, dimensions: int | None) -> list[list[float]]:
        """Matryoshka-truncate `vectors` to `dimensions` and L2-normalise."""
        if dimensions is not None and dimensions < vectors.shape[1]:
            vectors = vectors[:, :dimensions]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        return vectors.tolist()


def main() -> None:
    service = EmbeddingService()
    sample_texts = [
        "Sentence-transformers runs EmbeddingGemma's full pipeline.",
        "EmbeddingGemma is a compact text embedding model.",
    ]
    embeddings = service.embed_documents(sample_texts, dimensions=256)
    for text, vector in zip(sample_texts, embeddings):
        print(f"[{len(vector)} dims] {text!r} -> {vector[:5]}...")


if __name__ == "__main__":
    main()
