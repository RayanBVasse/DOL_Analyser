"""
Step 7 — Dyadic alignment.

Reads fine-cluster assignments already stored in SQLite by topics.py
(node_to_fine_cluster table) and computes the monthly Jensen-Shannon
divergence between user and assistant topic distributions.

Lower JS divergence = the two roles are covering more similar topics
that month (more "in sync"). Higher = more divergent agendas.

Outputs (out_dir):
    dyadic_alignment_monthly.csv  — month, user_msgs, asst_msgs, js_divergence
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

MIN_MSGS_PER_ROLE = 50   # skip months where either role has too few messages


def run(
    db_path: str | Path,
    out_dir: str | Path,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Compute monthly JS divergence between user and assistant topic distributions.

    Requires topics.run() to have been called first (node_to_fine_cluster table
    must exist in the database).

    Args:
        db_path:     SQLite database
        out_dir:     Directory for output CSVs
        progress_cb: Optional callable(fraction 0–1, status_string)

    Returns:
        Summary dict: months_computed, mean_js, min_js, max_js
    """
    def _cb(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path)

    # ── 1. Load messages joined with cluster assignments ──────────────────────
    _cb(0.05, "Loading cluster assignments…")
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT m.role, m.year_month, n.cluster_id
           FROM messages m
           JOIN node_to_fine_cluster n ON m.node_id = n.node_id
           WHERE m.role IN ('user', 'assistant')""",
        con,
    )
    con.close()

    if df.empty:
        raise ValueError(
            "No cluster assignments found. Run topics.run() before alignment.run()."
        )

    months = sorted(df["year_month"].dropna().unique())
    n_clusters = int(df["cluster_id"].max()) + 1

    # ── 2. Monthly JS divergence ──────────────────────────────────────────────
    _cb(0.20, "Computing monthly JS divergence…")
    rows = []
    n = len(months)

    for i, month in enumerate(months):
        _cb(0.20 + 0.70 * (i / max(n, 1)), f"Aligning month {month}…")
        g = df[df["year_month"] == month]
        user_g = g[g["role"] == "user"]
        asst_g = g[g["role"] == "assistant"]

        if len(user_g) < MIN_MSGS_PER_ROLE or len(asst_g) < MIN_MSGS_PER_ROLE:
            continue

        # Build full-length probability vectors (one entry per cluster)
        user_counts = user_g["cluster_id"].value_counts()
        asst_counts = asst_g["cluster_id"].value_counts()

        user_dist = np.array(
            [user_counts.get(c, 0) for c in range(n_clusters)], dtype=float
        )
        asst_dist = np.array(
            [asst_counts.get(c, 0) for c in range(n_clusters)], dtype=float
        )

        # Normalise to probability distributions
        if user_dist.sum() > 0:
            user_dist /= user_dist.sum()
        if asst_dist.sum() > 0:
            asst_dist /= asst_dist.sum()

        js = float(jensenshannon(user_dist, asst_dist))

        rows.append({
            "year_month":    month,
            "user_msgs":     len(user_g),
            "asst_msgs":     len(asst_g),
            "js_divergence": round(js, 6),
        })

    result_df = pd.DataFrame(rows).sort_values("year_month")
    out_path = out_dir / "dyadic_alignment_monthly.csv"
    result_df.to_csv(out_path, index=False)

    _cb(1.0, "Alignment complete.")

    if result_df.empty:
        return {"months_computed": 0}

    return {
        "months_computed": len(result_df),
        "mean_js":  round(float(result_df["js_divergence"].mean()), 4),
        "min_js":   round(float(result_df["js_divergence"].min()), 4),
        "max_js":   round(float(result_df["js_divergence"].max()), 4),
    }
