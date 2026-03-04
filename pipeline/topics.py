"""
Steps 5–6 — Topic modelling & cluster labelling.

Fits a single topic model on ALL messages (user + assistant combined)
so both roles share the same topic space — required for dyadic alignment
in Step 7.

Method:
    TF-IDF  (unigrams + bigrams, up to 60k features)
    → TruncatedSVD  (200 dims)
    → L2 normalise
    → KMeans  (K=60 fine clusters)
    → top-15 TF-IDF terms per cluster → auto-label (first 5 terms)

Writes to SQLite:
    node_to_fine_cluster  (node_id TEXT, cluster_id INT)
    cluster_summary       (cluster_id, size, auto_label, top_terms)

Writes to out_dir:
    monthly_topic_entropy_tfidf.csv   — per-month Shannon entropy (user msgs)
    cluster_summary_tfidf.csv         — cluster sizes, labels, top terms
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "max_features":   60_000,
    "svd_components": 200,
    "n_clusters":     60,
    "min_df":         3,       # lowered from 5 — handles smaller corpora
    "max_df":         0.6,
    "top_terms":      15,
    "random_state":   42,
    "n_init":         10,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_config(user_config: Optional[dict]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if user_config:
        cfg.update(user_config)
    return cfg


def _safe_entropy(counts: pd.Series) -> float:
    probs = counts / counts.sum()
    return float(scipy_entropy(probs))   # natural log (nats)


def _write_cluster_tables(
    db_path: Path,
    node_ids: list[str],
    labels: np.ndarray,
    cluster_summary: pd.DataFrame,
) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS node_to_fine_cluster;
        DROP TABLE IF EXISTS cluster_summary;

        CREATE TABLE node_to_fine_cluster (
            node_id    TEXT PRIMARY KEY,
            cluster_id INTEGER
        );

        CREATE TABLE cluster_summary (
            cluster_id   INTEGER PRIMARY KEY,
            size         INTEGER,
            auto_label   TEXT,
            top_terms    TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_ntfc_cluster
            ON node_to_fine_cluster(cluster_id);
    """)

    cur.executemany(
        "INSERT INTO node_to_fine_cluster (node_id, cluster_id) VALUES (?, ?)",
        zip(node_ids, labels.tolist()),
    )

    cur.executemany(
        """INSERT INTO cluster_summary (cluster_id, size, auto_label, top_terms)
           VALUES (:cluster_id, :size, :auto_label, :top_terms)""",
        cluster_summary.to_dict("records"),
    )

    con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    db_path: str | Path,
    out_dir: str | Path,
    config: Optional[dict] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Fit topic model on all messages and write cluster assignments to SQLite.

    Args:
        db_path:     SQLite database written by pipeline.parse.run()
        out_dir:     Directory for output CSVs
        config:      Optional overrides for DEFAULT_CONFIG
        progress_cb: Optional callable(fraction 0–1, status_string)

    Returns:
        Summary dict: n_messages, n_clusters, vocab_size, entropy stats
    """
    cfg = _merge_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path)

    def _cb(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    # ── 1. Load all messages ──────────────────────────────────────────────────
    _cb(0.02, "Loading messages…")
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT node_id, role, year_month, text
           FROM messages
           WHERE text IS NOT NULL AND LENGTH(text) > 0""",
        con,
    )
    con.close()

    if df.empty:
        raise ValueError("No messages found in database.")

    n_msgs = len(df)

    # Clamp K so we never have fewer than 5 messages per cluster on average
    k = min(cfg["n_clusters"], max(2, n_msgs // 5))
    cfg["n_clusters"] = k

    texts    = df["text"].astype(str).tolist()
    node_ids = df["node_id"].tolist()

    # ── 2. TF-IDF ─────────────────────────────────────────────────────────────
    _cb(0.08, f"Vectorising {n_msgs:,} messages (TF-IDF)…")
    vec = TfidfVectorizer(
        max_features=cfg["max_features"],
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())

    # ── 3. SVD ────────────────────────────────────────────────────────────────
    n_components = min(cfg["svd_components"], X.shape[1] - 1)
    _cb(0.25, f"Reducing dimensions (SVD → {n_components})…")
    svd = TruncatedSVD(n_components=n_components, random_state=cfg["random_state"])
    X_red = svd.fit_transform(X)
    X_red = normalize(X_red)

    # ── 4. KMeans ─────────────────────────────────────────────────────────────
    _cb(0.45, f"Clustering into {cfg['n_clusters']} topics (KMeans)…")
    km = KMeans(
        n_clusters=cfg["n_clusters"],
        random_state=cfg["random_state"],
        n_init=cfg["n_init"],
    )
    labels = km.fit_predict(X_red)
    df["cluster_id"] = labels

    # ── 5. Cluster labels ─────────────────────────────────────────────────────
    _cb(0.70, "Extracting cluster labels…")
    top_n = cfg["top_terms"]
    summary_rows = []

    for i in range(cfg["n_clusters"]):
        center       = km.cluster_centers_[i]
        top_idx      = center.argsort()[-top_n:][::-1]
        top_words    = terms[top_idx].tolist()
        cluster_size = int((df["cluster_id"] == i).sum())

        summary_rows.append({
            "cluster_id": i,
            "size":       cluster_size,
            "auto_label": ", ".join(top_words[:5]),
            "top_terms":  ", ".join(top_words),
        })

    cluster_summary = pd.DataFrame(summary_rows).sort_values(
        "size", ascending=False
    )

    # ── 6. Write SQLite tables ────────────────────────────────────────────────
    _cb(0.78, "Writing cluster assignments to database…")
    _write_cluster_tables(db_path, node_ids, labels, cluster_summary)

    # ── 7. Monthly entropy (user messages only) ───────────────────────────────
    _cb(0.85, "Computing monthly topic entropy…")
    user_df = df[df["role"] == "user"].copy()

    entropy_rows = []
    for month, grp in user_df.groupby("year_month"):
        counts = grp["cluster_id"].value_counts()
        entropy_rows.append({
            "year_month":         month,
            "user_messages":      len(grp),
            "clusters_present":   int(counts.shape[0]),
            "topic_entropy_nats": _safe_entropy(counts),
        })

    entropy_df = pd.DataFrame(entropy_rows).sort_values("year_month")
    entropy_path = out_dir / "monthly_topic_entropy_tfidf.csv"
    entropy_df.to_csv(entropy_path, index=False)

    # ── 8. Cluster summary CSV ────────────────────────────────────────────────
    _cb(0.93, "Writing cluster summary…")
    cluster_summary.to_csv(out_dir / "cluster_summary_tfidf.csv", index=False)

    _cb(1.0, "Topic modelling complete.")

    return {
        "n_messages":       n_msgs,
        "n_clusters":       cfg["n_clusters"],
        "vocab_size":       int(X.shape[1]),
        "months_with_data": len(entropy_df),
        "entropy_min":      round(float(entropy_df["topic_entropy_nats"].min()), 3),
        "entropy_max":      round(float(entropy_df["topic_entropy_nats"].max()), 3),
    }
