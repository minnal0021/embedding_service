"""Download the selected models' GGUF builds for the llama-server containers.

Run by the `model-fetch` service in docker-compose.yml before the model
servers start. Each GGUF goes into the Hugging Face cache layout under the
cache dir (shared with the in-process EmbeddingService), and
<cache>/active/<model>.gguf is pointed at it with a relative symlink, so the
containers can load a fixed path without knowing the snapshot hash.

Environment: EMBEDDING_MODELS, EMBEDDING_CACHE_PATH, EMBEDDING_<MODEL>_GGUF.
"""

import os

from huggingface_hub import hf_hub_download

from models import MODELS, gguf_file, parse_model_keys


def main() -> None:
    cache_dir = os.getenv("EMBEDDING_CACHE_PATH", "./model_cache")
    active_dir = os.path.join(cache_dir, "active")
    os.makedirs(active_dir, exist_ok=True)

    for key in parse_model_keys(os.getenv("EMBEDDING_MODELS", "gemma,qwen")):
        spec, filename = MODELS[key], gguf_file(MODELS[key])
        print(f"Fetching {key}: {spec.repo}/{filename} ...", flush=True)
        path = hf_hub_download(repo_id=spec.repo, filename=filename, cache_dir=cache_dir)

        # Swap the link atomically so a running server never sees it missing.
        link = os.path.join(active_dir, f"{key}.gguf")
        tmp = f"{link}.tmp"
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(os.path.relpath(path, active_dir), tmp)
        os.replace(tmp, link)
        print(f"  {key} -> {os.path.relpath(path, cache_dir)}", flush=True)


if __name__ == "__main__":
    main()
