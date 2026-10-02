# Embedding gateway (src/server.py) and model downloader (src/fetch_models.py).
# Installs only the project's main dependencies from uv.lock (not the offline
# pipeline's "tools" group); the models run in the llama.cpp server images.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src/ ./src/

USER 1000:1000
EXPOSE 8001
CMD ["uvicorn", "server:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8001"]
