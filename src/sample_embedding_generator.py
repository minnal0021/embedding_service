#!/usr/bin/env python3
"""
SampleEmbeddingGenerator

Reads ELI5-style question-answer pairs from a JSONL file where each line is a
JSON array of the form:

    ["<question text>", "<answer text>"]

For each record it generates one document embedding from the combined
``question + "\\n" + answer`` text and writes it, alongside the original
question and answer, to a Parquet file. The resulting vectors are intended as
sample data for downstream centroid creation.

Embeddings come from the running embedding service
(``./embedding_service.sh start``) for the model named by ``--model``, through
its ``/embedding/<model>/document`` endpoint. The service applies the model's
document prompt, Matryoshka truncation to the requested ``--dimensions``, and
L2-normalisation.

Output Parquet schema
---------------------
  question     : string
  answer       : string
  qa_embedding : list[float]   – embedding of question + "\\n" + answer

Usage
-----
    python src/sample_embedding_generator.py [OPTIONS]

Options
-------
    --model         Model to embed with: gemma | qwen  (required)
    --input         Path to the JSONL file
                    (default: sample_data/eli5_question_answer.jsonl)
    --output        Path to the output Parquet file
                    (default: embedding/<model>/sample_data_embedding.parquet)
    --dimensions    Embedding width; Matryoshka-truncated from the model's
                    native width (default: 768; 0 = native, gemma 768 / qwen 4096)
    --batch-size    Number of QA pairs per embedding call
                    (default: 32)
    --max-records   Maximum number of records to process  (0 = all)
                    (default: 0)
    --service-url   Embedding service base URL
                    (default: $SERVICE_URL or http://localhost:8001)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx
import pandas as pd
from tqdm import tqdm

from models import DEFAULT_DIMENSIONS as SERVICE_DEFAULT_DIMENSIONS
from models import MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
# httpx logs every request at INFO; keep the progress bar readable.
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "sample_data/eli5_question_answer.jsonl"
DEFAULT_OUTPUT = "embedding/{model}/sample_data_embedding.parquet"
DEFAULT_DIMENSIONS = SERVICE_DEFAULT_DIMENSIONS  # 768; 0 = model native width
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_RECORDS = 0  # 0 = process all records
DEFAULT_SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8001")
# Seconds to wait for one batch; Qwen on a CPU takes ~1 s per QA pair.
REQUEST_TIMEOUT = 1800


class ServiceUnavailable(RuntimeError):
    """The embedding service can't be reached or isn't ready."""


# ── Data class ────────────────────────────────────────────────────────────────


@dataclass
class QAPair:
    question: str
    answer: str


# ── JSONL reader ──────────────────────────────────────────────────────────────


class ELI5Reader:
    """
    Reads a JSONL file where every line is a two-element JSON array:
        ["<question>", "<answer>"]

    Lines that are blank, malformed, or not a two-element array are skipped
    with a warning.
    """

    @staticmethod
    def parse_file(filepath: str | Path) -> Iterator[QAPair]:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")

        skipped = 0
        with open(filepath, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning("Line %d: JSON decode error — %s", lineno, exc)
                    skipped += 1
                    continue

                if not isinstance(record, list) or len(record) != 2:
                    logger.warning(
                        "Line %d: expected a 2-element array, got %s — skipping",
                        lineno,
                        type(record).__name__,
                    )
                    skipped += 1
                    continue

                question, answer = record
                if not isinstance(question, str) or not isinstance(answer, str):
                    logger.warning(
                        "Line %d: question/answer are not strings — skipping", lineno
                    )
                    skipped += 1
                    continue

                question = question.strip()
                answer = answer.strip()
                if not question or not answer:
                    skipped += 1
                    continue

                yield QAPair(question=question, answer=answer)

        if skipped:
            logger.info("Skipped %d malformed / empty lines", skipped)


# ── Main Generator ────────────────────────────────────────────────────────────


class SampleEmbeddingGenerator:
    """
    End-to-end pipeline:
        parse JSONL → batch embed QA pairs → write Parquet.

    For each QA pair one document embedding is generated from
    ``question + "\\n" + answer``. The vectors are intended as sample data for
    downstream centroid creation.
    """

    def __init__(
        self,
        model: str,
        input_path: str = DEFAULT_INPUT,
        output_path: str | None = None,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_records: int = DEFAULT_MAX_RECORDS,
        service_url: str = DEFAULT_SERVICE_URL,
    ):
        self.model = model
        self.input_path = Path(input_path)
        self.output_path = Path(output_path or DEFAULT_OUTPUT.format(model=model))
        self.batch_size = batch_size
        self.max_records = max_records
        self.service_url = service_url.rstrip("/")
        self.client = httpx.Client(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=5))
        self.service_model = self._check_service()
        # 0 means "model native width" (no truncation).
        self.dimensions = dimensions or self.service_model["dimension"]

    def _check_service(self) -> dict:
        """Return the service's healthcheck entry for `model`, or fail clearly."""
        hint = "start it with ./embedding_service.sh start"
        try:
            r = self.client.get(f"{self.service_url}/healthcheck")
        except httpx.HTTPError as exc:
            raise ServiceUnavailable(
                f"Can't reach the embedding service at {self.service_url} ({exc}); {hint}"
            ) from exc
        models = r.json().get("models", {})
        if self.model not in models:
            raise ServiceUnavailable(
                f"The service at {self.service_url} doesn't serve {self.model!r} "
                f"(loaded: {', '.join(models) or 'none'}); {hint} --models ..."
            )
        if models[self.model].get("status") != "ok":
            raise ServiceUnavailable(
                f"{self.model} is not ready yet ({models[self.model].get('status')}); "
                "wait for ./embedding_service.sh status to show it ok"
            )
        return models[self.model]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """POST texts to the service's document endpoint.

        Raises ServiceUnavailable when the service is down or still loading
        (retrying per item wouldn't help), httpx.HTTPStatusError otherwise.
        """
        try:
            r = self.client.post(
                f"{self.service_url}/embedding/{self.model}/document",
                json={"payloads": texts, "dimensions": self.dimensions},
            )
        except httpx.HTTPError as exc:
            raise ServiceUnavailable(f"Embedding service request failed: {exc}") from exc
        if r.status_code == 503:
            raise ServiceUnavailable(f"Embedding service unavailable: {r.text[:300]}")
        r.raise_for_status()
        return r.json()["embeddings"]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of combined QA texts as documents.

        Falls back to one-at-a-time on batch failure so a single problematic
        record (e.g. longer than the model's context) doesn't drop the whole
        batch.
        """
        try:
            return self._embed_documents(texts)
        except httpx.HTTPStatusError as exc:
            logger.warning("Batch failed (%s) — retrying individually...", _detail(exc))
            embeddings: list[list[float]] = []
            for text in texts:
                try:
                    embeddings.append(self._embed_documents([text])[0])
                except httpx.HTTPStatusError as single_exc:
                    logger.warning(
                        "Skipping text (len %d chars): %s", len(text), _detail(single_exc)
                    )
                    embeddings.append([])  # placeholder; filtered out below
            return embeddings

    # ── main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Execute the full pipeline and return the output Parquet path."""
        logger.info(
            "Model        : %s (%s on %s, via %s)",
            self.model,
            self.service_model["file"],
            self.service_model["device"],
            self.service_url,
        )
        logger.info("Input file   : %s", self.input_path)
        logger.info("Output file  : %s", self.output_path)
        logger.info(
            "Dimensions   : %d (native %d)",
            self.dimensions,
            self.service_model["dimension"],
        )
        logger.info("Batch size   : %d", self.batch_size)
        logger.info(
            "Max records  : %s", self.max_records if self.max_records else "all"
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []
        batch: list[QAPair] = []
        processed = 0
        skipped_pairs = 0

        reader = ELI5Reader.parse_file(self.input_path)
        progress = tqdm(reader, desc="Processing QA pairs", unit="pair")

        for pair in progress:
            # Count the pending batch too, so the limit is exact.
            if 0 < self.max_records <= processed + skipped_pairs + len(batch):
                break

            batch.append(pair)

            if len(batch) >= self.batch_size:
                n_added, n_skipped = self._flush_batch(batch, records)
                processed += n_added
                skipped_pairs += n_skipped
                batch = []
                progress.set_postfix(embedded=processed, skipped=skipped_pairs)

        # Flush remainder
        if batch:
            n_added, n_skipped = self._flush_batch(batch, records)
            processed += n_added
            skipped_pairs += n_skipped

        logger.info("Total pairs embedded : %d", processed)
        logger.info("Total pairs skipped  : %d", skipped_pairs)

        # ── Write Parquet ─────────────────────────────────────────────────
        df = pd.DataFrame(records)
        df.to_parquet(self.output_path, index=False, engine="pyarrow")
        size_mb = self.output_path.stat().st_size / (1024 * 1024)
        logger.info("Parquet written : %s (%.1f MB)", self.output_path, size_mb)
        logger.info("Schema          : %s", list(df.columns))
        return self.output_path

    def _flush_batch(
        self,
        batch: list[QAPair],
        records: list[dict],
    ) -> tuple[int, int]:
        """
        Embed the combined question + answer text as a document for each pair
        in one batched call, then append to records.

        Returns (n_added, n_skipped).
        """
        combined = [f"{p.question}\n{p.answer}" for p in batch]

        qa_embeddings = self._embed_batch(combined)

        n_added = n_skipped = 0
        for pair, qa_emb in zip(batch, qa_embeddings):
            if not qa_emb:
                logger.warning(
                    "Skipping pair — missing embedding for: %s", pair.question[:60]
                )
                n_skipped += 1
                continue
            records.append(
                {
                    "question": pair.question,
                    "answer": pair.answer,
                    "qa_embedding": qa_emb,
                }
            )
            n_added += 1

        return n_added, n_skipped


def _detail(exc: httpx.HTTPStatusError) -> str:
    """The service's error detail, falling back to the status line."""
    try:
        return exc.response.json()["detail"]
    except Exception:  # noqa: BLE001 — not the service's JSON error shape
        return str(exc)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate sample QA embeddings and write to Parquet (for centroid creation).",
    )
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODELS),
        help="Model to embed with",
    )
    p.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"JSONL input file (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output",
        default=None,
        help=f"Output Parquet path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"Embedding dimensions, 0 = model native (default: {DEFAULT_DIMENSIONS})",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"QA pairs per embedding call (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help="Max records to process, 0 = all (default: 0)",
    )
    p.add_argument(
        "--service-url",
        default=DEFAULT_SERVICE_URL,
        help=f"Embedding service base URL (default: {DEFAULT_SERVICE_URL})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        generator = SampleEmbeddingGenerator(
            model=args.model,
            input_path=args.input,
            output_path=args.output,
            dimensions=args.dimensions,
            batch_size=args.batch_size,
            max_records=args.max_records,
            service_url=args.service_url,
        )
        output = generator.run()
        logger.info("Done. Output → %s", output)
    except (FileNotFoundError, ServiceUnavailable) as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
