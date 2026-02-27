"""
Steps 5–6 — Topic modelling & cluster labelling.

Method:
    TF-IDF (unigrams + bigrams, 60k features)
    → Truncated SVD (200 dims)
    → KMeans (K=60 fine clusters)
    → Top TF-IDF terms per cluster → human-readable label

Outputs:
    monthly_topic_entropy_tfidf.csv   — Shannon entropy per month
    cluster_summary_tfidf.csv         — cluster sizes, labels, top terms
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# Default hyperparameters (overridable via config dict)
DEFAULT_CONFIG = {
    "n_features": 60_000,
    "svd_components": 200,
    "n_clusters": 60,
    "min_df": 5,
    "max_df": 0.6,
    "random_state": 42,
}


def run(
    db_path: str | Path,
    out_dir: str | Path,
    config: dict | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Fit topic model and write entropy + cluster summary CSVs."""
    raise NotImplementedError("topics.run() will be implemented in Week 2.")
