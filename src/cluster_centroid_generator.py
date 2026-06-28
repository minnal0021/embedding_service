#!/usr/bin/env python3
"""
ClusterCentroidGenerator

Reads the sample QA embeddings produced by ``sample_embedding_generator.py``
from a Parquet file, runs K-means clustering, and writes the cluster centroids
to a JSONL file.

Usage:
    python src/cluster_centroid_generator.py [OPTIONS]

Options:
    --input         Path to the embedding parquet file
                    (default: embedding/sample_data_embedding.parquet)
    --output        Path to the output JSONL file
                    (default: embedding/cluster_centroids.jsonl)
    --embedding-col Name of the embedding column in the parquet file
                    (default: qa_embedding)
    --dimensions    Embedding width to cluster on; Matryoshka-truncated +
                    L2-renormalised from the stored width if smaller
                    (default: 768; 0 = use the stored width as-is)
    --n-clusters    Number of K-means clusters
                    (default: 256)
    --seed          Random seed for reproducibility
                    (default: 42)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "embedding/sample_data_embedding.parquet"
DEFAULT_OUTPUT = "embedding/cluster_centroids.jsonl"
DEFAULT_EMBEDDING_COL = "qa_embedding"
DEFAULT_DIMENSIONS = 768
DEFAULT_N_CLUSTERS = 256
DEFAULT_SEED = 42


# ── Cluster Generator ────────────────────────────────────────────────────────


class ClusterCentroidGenerator:
    """
    Reads embedding vectors from a Parquet file, clusters them with K-means,
    and writes the cluster centroids to a JSONL file.

    Output JSONL structure (one JSON object per line):
        {"cluster_id": 1, "centroid": [0.01, -0.02, ...]}
        {"cluster_id": 2, "centroid": [0.03,  0.01, ...]}
        ...

    In Python:
        with open(path) as f:
            clusters = [json.loads(line) for line in f]
    """

    def __init__(
        self,
        input_path: str = DEFAULT_INPUT,
        output_path: str = DEFAULT_OUTPUT,
        embedding_col: str = DEFAULT_EMBEDDING_COL,
        dimensions: int = DEFAULT_DIMENSIONS,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        seed: int = DEFAULT_SEED,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.embedding_col = embedding_col
        # 0 → None means "use the stored width as-is" (no truncation).
        self.dimensions = dimensions or None
        self.n_clusters = n_clusters
        self.seed = seed

    def _load_embeddings(self) -> np.ndarray:
        """Load the embedding column as a 2-D array, optionally MRL-truncated."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        logger.info("Loading embeddings from %s", self.input_path)
        df = pd.read_parquet(self.input_path)

        if self.embedding_col not in df.columns:
            raise ValueError(
                f"Parquet file must contain a '{self.embedding_col}' column "
                f"(found: {list(df.columns)})"
            )

        embeddings = np.array(df[self.embedding_col].tolist(), dtype=np.float32)
        n_embeddings, stored_dim = embeddings.shape
        logger.info("Loaded %d embeddings of dimension %d", n_embeddings, stored_dim)

        if self.dimensions is not None and self.dimensions < stored_dim:
            # Matryoshka truncation: keep the leading sub-vector and renormalise.
            embeddings = embeddings[:, : self.dimensions]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            logger.info("Truncated embeddings to dimension %d", self.dimensions)
        elif self.dimensions is not None and self.dimensions > stored_dim:
            logger.warning(
                "Requested dimension %d exceeds stored dimension %d — using %d",
                self.dimensions,
                stored_dim,
                stored_dim,
            )

        return embeddings

    def run(self) -> Path:
        """Execute the full pipeline and return the output JSONL path."""

        # ── 1. Load embeddings ───────────────────────────────────────────
        embeddings = self._load_embeddings()
        n_embeddings, dimensions = embeddings.shape

        # Cap clusters to number of samples
        actual_clusters = min(self.n_clusters, n_embeddings)
        if actual_clusters < self.n_clusters:
            logger.warning(
                "Requested %d clusters but only %d embeddings — using %d clusters",
                self.n_clusters,
                n_embeddings,
                actual_clusters,
            )

        # ── 2. Run K-means ───────────────────────────────────────────────
        logger.info(
            "Running K-means with %d clusters (seed=%d)...",
            actual_clusters,
            self.seed,
        )
        kmeans = KMeans(
            n_clusters=actual_clusters,
            random_state=self.seed,
            n_init=10,
            max_iter=300,
        )
        kmeans.fit(embeddings)
        logger.info("K-means converged — inertia: %.4f", kmeans.inertia_)

        # ── 3. Write JSONL ─────────────────────────────────────────────
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            for idx, centroid in enumerate(kmeans.cluster_centers_):
                record = {
                    "cluster_id": idx + 1,
                    "centroid": centroid.tolist(),
                }
                fh.write(json.dumps(record) + "\n")

        file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
        logger.info("JSONL written: %s (%.1f MB)", self.output_path, file_size_mb)
        logger.info("Clusters: %d | Dimensions: %d", actual_clusters, dimensions)

        return self.output_path


# ── CLI Entry Point ──────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster sample embeddings with K-means and write centroids to JSONL.",
    )
    p.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to embedding parquet (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--embedding-col",
        default=DEFAULT_EMBEDDING_COL,
        help=f"Embedding column name in parquet (default: {DEFAULT_EMBEDDING_COL})",
    )
    p.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"Dimensions to cluster on, 0 = stored width (default: {DEFAULT_DIMENSIONS})",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help=f"Number of K-means clusters (default: {DEFAULT_N_CLUSTERS})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    generator = ClusterCentroidGenerator(
        input_path=args.input,
        output_path=args.output,
        embedding_col=args.embedding_col,
        dimensions=args.dimensions,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )

    try:
        output = generator.run()
        logger.info("Done. Output → %s", output)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
