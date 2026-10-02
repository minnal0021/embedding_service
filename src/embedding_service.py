import os
import sys

import llama_cpp
import numpy as np
from huggingface_hub import hf_hub_download

from models import DEFAULT_DIMENSIONS, MODELS, parse_model_keys
from models import gguf_file as default_gguf_file


# llama-cpp-python pooling constants for ModelSpec.pooling.
POOLING = {
    "mean": llama_cpp.LLAMA_POOLING_TYPE_MEAN,
    "last": llama_cpp.LLAMA_POOLING_TYPE_LAST,
}


class EmbeddingService:
    """Generate text embeddings for one model in `MODELS` via llama.cpp.

    The GGUF build is fetched from Hugging Face into the cache dir on first
    use. Documents and queries are embedded asymmetrically by way of each
    model's task-specific prompt prefixes. Both models support Matryoshka
    representation, so vectors are sliced to `dimensions` (DEFAULT_DIMENSIONS
    unless given; None = the model's native width) and re-normalised.

    The GPU backend (Metal on macOS, HIP/ROCm or Vulkan on AMD) is fixed when
    llama-cpp-python is built (see install_backend.sh), so device selection
    only decides whether to offload layers to whatever backend is compiled in.
    """

    DEFAULT_CACHE_DIR = "./model_cache"

    def __init__(
        self,
        model_key: str,
        cache_dir: str | None = None,
        gguf_file: str | None = None,
        device: str | None = None,
    ):
        if model_key not in MODELS:
            raise ValueError(
                f"Unknown model {model_key!r}; available: {', '.join(MODELS)}"
            )
        self.key = model_key
        self.spec = MODELS[model_key]
        self.model_name = self.spec.name
        self.dimension = self.spec.dimension
        self.cache_dir = cache_dir or os.getenv(
            "EMBEDDING_CACHE_PATH", self.DEFAULT_CACHE_DIR
        )
        # Defaults to the model's build in MODELS; override per model (e.g. a
        # different quantization) via arg or the model's EMBEDDING_<MODEL>_GGUF.
        self.gguf_file = gguf_file or default_gguf_file(self.spec)

        # "auto" uses the GPU when llama.cpp was built with one and falls back
        # to the CPU otherwise (or if loading on it fails); "gpu" requires it;
        # "cpu" never uses it. Override via EMBEDDING_DEVICE.
        device = (device or os.getenv("EMBEDDING_DEVICE", "auto")).lower()
        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(
                f"EMBEDDING_DEVICE must be auto, gpu or cpu (got {device!r})"
            )

        model_path = hf_hub_download(
            repo_id=self.spec.repo, filename=self.gguf_file, cache_dir=self.cache_dir
        )
        self.llm, self.device = self._load(model_path, device)
        self.backend = gpu_backend() if self.device == "gpu" else "cpu"
        print(f"Loaded {self.key} ({self.gguf_file}) on {self.device} ({self.backend})")

    def _load(self, model_path: str, device: str) -> tuple[llama_cpp.Llama, str]:
        """Load the GGUF, offloading to the GPU when requested and available."""
        gpu_ok = llama_cpp.llama_supports_gpu_offload()
        if device == "gpu" and not gpu_ok:
            raise RuntimeError(
                "EMBEDDING_DEVICE=gpu but llama-cpp-python was built without "
                "GPU support. Rebuild it with ./install_backend.sh; see README."
            )
        if device == "cpu" or not gpu_ok:
            return self._llama(model_path, n_gpu_layers=0), "cpu"

        print(f"Loading {self.key} from {model_path} on gpu ({gpu_backend()})...")
        try:
            return self._llama(model_path, n_gpu_layers=-1), "gpu"
        except Exception as e:  # noqa: BLE001 — e.g. out of GPU memory
            if device == "gpu":
                raise
            print(
                f"WARNING: loading {self.key} on the GPU failed ({e}); "
                "falling back to CPU.",
                file=sys.stderr,
            )
        return self._llama(model_path, n_gpu_layers=0), "cpu"

    def _llama(self, model_path: str, n_gpu_layers: int) -> llama_cpp.Llama:
        n_ctx = self.spec.n_ctx
        return llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=POOLING[self.spec.pooling],
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_ubatch=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def embed_documents(
        self, payloads: list[str], dimensions: int | None = DEFAULT_DIMENSIONS
    ) -> list[list[float]]:
        """Embed payloads as documents (for indexing)."""
        prompted = [self.spec.document_prompt + p for p in payloads]
        return self._embed(self.llm.embed(prompted, normalize=False), dimensions)

    def embed_query(
        self, payloads: list[str], dimensions: int | None = DEFAULT_DIMENSIONS
    ) -> list[list[float]]:
        """Embed payloads as search queries."""
        prompted = [self.spec.query_prompt + p for p in payloads]
        return self._embed(self.llm.embed(prompted, normalize=False), dimensions)

    def _embed(self, vectors, dimensions: int | None) -> list[list[float]]:
        """Truncate (Matryoshka) + L2-normalise pooled vectors to lists.

        Order is preserved: vectors are emitted in the same order the model
        yields them, which matches the input payload order.
        """
        out: list[list[float]] = []
        for vector in vectors:
            vec = np.asarray(vector, dtype=np.float32)
            if dimensions is not None and dimensions < vec.shape[0]:
                # Matryoshka truncation; renormalise the shortened vector below.
                vec = vec[:dimensions]
            # Always return L2-normalised vectors.
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out.append(vec.tolist())
        return out


def gpu_backend() -> str:
    """Name the GPU backend llama-cpp-python was compiled with, if any."""
    if not llama_cpp.llama_supports_gpu_offload():
        return "cpu"
    info = llama_cpp.llama_print_system_info().decode()
    for marker, name in (
        ("Metal", "metal"),
        ("ROCm", "hip"),
        ("HIP", "hip"),
        ("Vulkan", "vulkan"),
        ("CUDA", "cuda"),
    ):
        if marker in info:
            return name
    return "gpu"


def load_models(keys: list[str], **kwargs) -> dict[str, EmbeddingService]:
    """Load each named model (validated up front so a typo fails fast)."""
    return {k: EmbeddingService(k, **kwargs) for k in parse_model_keys(keys)}


def main() -> None:
    service = EmbeddingService(sys.argv[1] if len(sys.argv) > 1 else "gemma")
    sample_texts = [
        "llama.cpp makes embedding generation fast and portable.",
        "EmbeddingGemma is a compact text embedding model.",
    ]
    embeddings = service.embed_documents(sample_texts, dimensions=256)
    for text, vector in zip(sample_texts, embeddings):
        print(f"[{len(vector)} dims] {text!r} -> {vector[:5]}...")


if __name__ == "__main__":
    main()
