"""
Steps 10a/b — Behavioural dynamics.

Sub-steps (all run from a single run() call):

  10a2  state_segmentation   — assign each month to Exploration /
                               Consolidation / Transitional based on
                               quantile thresholds on macro entropy + JS.
                               Also a rate-based variant (sign of Δ).

  10a3  rolling_entropy      — sliding-window Shannon entropy over all
                               user messages ordered by timestamp.
                               window = stride = 250 messages.

  10b   shift_initiation     — detect domain shifts within threads and
                               test whether user or assistant initiates
                               more than 50 % (permutation test, N=5,000).

Note: scale_separation (10a1) and episode_initiation (10b2) are omitted
      here — they require ≥6 qualifying months and ≥20-message threads
      respectively, which may not hold for all user corpora.  They can be
      added as optional steps once the base pipeline is validated.

Outputs (all written to out_dir):
    monthly_states.csv
    monthly_states_refined.csv
    state_transition_summary.csv
    state_transition_summary_refined.csv
    rolling_entropy_250.csv
    shift_initiation_summary.csv
    shift_initiation_detail.csv
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

DEFAULT_CONFIG = {
    "rolling_window":    250,
    "rolling_stride":    250,
    "shift_permutations": 5_000,
    "random_seed":       42,
}


# ─────────────────────────────────────────────────────────────────────────────
# 10a2 — State segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _state_segmentation(metrics_path: Path, out_dir: Path) -> pd.DataFrame:
    """
    Assign Exploration / Consolidation / Transitional states.

    Static variant:  based on quantile levels of entropy + JS divergence.
    Rate-based variant: based on sign of month-to-month deltas.
    """
    df = pd.read_csv(metrics_path).sort_values("year_month").reset_index(drop=True)

    # Drop rows missing either signal
    df = df.dropna(subset=["macro_entropy_user", "macro_js_divergence"])

    if df.empty:
        return df

    # ── Static (quantile thresholds) ──────────────────────────────────────────
    e_hi = df["macro_entropy_user"].quantile(0.66)
    e_lo = df["macro_entropy_user"].quantile(0.33)
    j_hi = df["macro_js_divergence"].quantile(0.66)
    j_lo = df["macro_js_divergence"].quantile(0.33)

    def _static_state(row):
        if row["macro_entropy_user"] >= e_hi and row["macro_js_divergence"] >= j_hi:
            return "Exploration"
        if row["macro_entropy_user"] <= e_lo and row["macro_js_divergence"] <= j_lo:
            return "Consolidation"
        return "Transitional"

    df["state"] = df.apply(_static_state, axis=1)

    states_out = df[["year_month", "macro_entropy_user", "macro_js_divergence", "state"]]
    states_out.to_csv(out_dir / "monthly_states.csv", index=False)

    # Transition summary
    pairs = [(df.loc[i - 1, "state"], df.loc[i, "state"]) for i in range(1, len(df))]
    trans = (
        pd.DataFrame(pairs, columns=["from_state", "to_state"])
        .groupby(["from_state", "to_state"]).size()
        .reset_index(name="count")
    )
    trans.to_csv(out_dir / "state_transition_summary.csv", index=False)

    # ── Rate-based ────────────────────────────────────────────────────────────
    df["delta_entropy"] = df["macro_entropy_user"].diff()
    df["delta_js"]      = df["macro_js_divergence"].diff()

    def _rate_state(row):
        if pd.isna(row["delta_entropy"]):
            return "Initial"
        if row["delta_entropy"] > 0 and row["delta_js"] > 0:
            return "Exploration"
        if row["delta_entropy"] < 0 and row["delta_js"] < 0:
            return "Consolidation"
        return "Transitional"

    df["state_rate"] = df.apply(_rate_state, axis=1)

    df[["year_month", "macro_entropy_user", "macro_js_divergence",
        "delta_entropy", "delta_js", "state_rate"]].to_csv(
        out_dir / "monthly_states_refined.csv", index=False
    )

    pairs_r = [
        (df.loc[i - 1, "state_rate"], df.loc[i, "state_rate"])
        for i in range(1, len(df))
        if df.loc[i - 1, "state_rate"] != "Initial"
    ]
    trans_r = (
        pd.DataFrame(pairs_r, columns=["from_state", "to_state"])
        .groupby(["from_state", "to_state"]).size()
        .reset_index(name="count")
    )
    trans_r.to_csv(out_dir / "state_transition_summary_refined.csv", index=False)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10a3 — Rolling entropy
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_entropy(
    db_path: Path,
    out_dir: Path,
    window: int,
    stride: int,
) -> pd.DataFrame:
    """
    Sliding-window Shannon entropy over user messages ordered by timestamp.
    Each window covers `window` messages; windows advance by `stride`.
    """
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT m.node_id, m.timestamp, n.cluster_id
           FROM messages m
           JOIN node_to_fine_cluster n ON m.node_id = n.node_id
           WHERE m.role = 'user'
           ORDER BY m.timestamp ASC""",
        con,
    )
    con.close()

    if len(df) < window:
        return pd.DataFrame()

    rows = []
    for start in range(0, len(df) - window + 1, stride):
        chunk = df.iloc[start: start + window]
        counts = chunk["cluster_id"].value_counts()
        probs  = counts / counts.sum()
        H      = float(shannon_entropy(probs, base=2))  # bits
        mid_ts = chunk["timestamp"].median()
        rows.append({
            "window_start":       start,
            "window_end":         start + window - 1,
            "mid_timestamp":      mid_ts,
            "entropy_bits":       round(H, 4),
        })

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "rolling_entropy_250.csv", index=False)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 10b — Shift initiation
# ─────────────────────────────────────────────────────────────────────────────

def _shift_initiation(
    db_path: Path,
    out_dir: Path,
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    """
    Detect macro-domain shifts within threads; test who initiates them.
    """
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT m.node_id, m.thread_id, m.role, m.timestamp, n.macro_domain
           FROM messages m
           JOIN node_to_macro_domain n ON m.node_id = n.node_id
           WHERE m.role IN ('user', 'assistant')
           ORDER BY m.thread_id, m.timestamp ASC""",
        con,
    )
    con.close()

    if df.empty:
        return {}

    shifts = []
    for _, grp in df.groupby("thread_id"):
        grp = grp.reset_index(drop=True)
        for i in range(1, len(grp)):
            if grp.loc[i, "macro_domain"] != grp.loc[i - 1, "macro_domain"]:
                shifts.append({
                    "thread_id":      grp.loc[i, "thread_id"],
                    "initiator_role": grp.loc[i, "role"],
                    "from_domain":    grp.loc[i - 1, "macro_domain"],
                    "to_domain":      grp.loc[i, "macro_domain"],
                })

    shift_df = pd.DataFrame(shifts)
    if shift_df.empty:
        return {"total_shifts": 0}

    shift_df.to_csv(out_dir / "shift_initiation_detail.csv", index=False)

    total  = len(shift_df)
    roles  = shift_df["initiator_role"].values
    u_prop = float((roles == "user").mean())
    a_prop = 1.0 - u_prop

    # Two-tailed permutation test vs 50/50
    observed_abs = abs(u_prop - 0.5)
    null_abs = np.array([
        abs((rng.permutation(roles) == "user").mean() - 0.5)
        for _ in range(n_perm)
    ])
    p_perm = float((null_abs >= observed_abs).mean())

    summary = pd.DataFrame([{
        "total_shifts":           total,
        "user_initiated_prop":    round(u_prop, 4),
        "asst_initiated_prop":    round(a_prop, 4),
        "perm_p_vs_random":       round(p_perm, 4),
        "n_permutations":         n_perm,
    }])
    summary.to_csv(out_dir / "shift_initiation_summary.csv", index=False)

    return {
        "total_shifts":        total,
        "user_initiated_prop": round(u_prop, 4),
        "p_perm":              round(p_perm, 4),
    }


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
    Run all Step 10 sub-analyses.

    Requires:
        - topics.run()  (node_to_fine_cluster in DB)
        - domains.run() (node_to_macro_domain in DB + macro_monthly_metrics.csv)

    Args:
        db_path:     SQLite database
        out_dir:     Directory containing macro_monthly_metrics.csv (from domains.run())
                     and where output CSVs will be written
        config:      Optional overrides for DEFAULT_CONFIG
        progress_cb: Optional callable(fraction 0–1, status_string)

    Returns:
        Summary dict with results from each sub-step.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    def _cb(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    out_dir  = Path(out_dir)
    db_path  = Path(db_path)
    rng      = np.random.default_rng(cfg["random_seed"])
    results  = {}

    # ── 10a2 State segmentation ───────────────────────────────────────────────
    _cb(0.05, "10a2 — Assigning behavioural states…")
    metrics_path = out_dir / "macro_monthly_metrics.csv"
    if metrics_path.exists():
        states_df = _state_segmentation(metrics_path, out_dir)
        results["state_months"] = len(states_df)
        if "state" in states_df.columns:
            results["state_counts"] = states_df["state"].value_counts().to_dict()
    else:
        results["state_segmentation"] = "skipped — macro_monthly_metrics.csv not found"

    # ── 10a3 Rolling entropy ──────────────────────────────────────────────────
    _cb(0.30, "10a3 — Computing rolling topic entropy…")
    rolling_df = _rolling_entropy(
        db_path, out_dir,
        window=cfg["rolling_window"],
        stride=cfg["rolling_stride"],
    )
    results["rolling_windows"] = len(rolling_df)

    # ── 10b Shift initiation ──────────────────────────────────────────────────
    _cb(0.60, "10b — Detecting domain shifts…")
    shift_results = _shift_initiation(
        db_path, out_dir,
        n_perm=cfg["shift_permutations"],
        rng=rng,
    )
    results.update(shift_results)

    _cb(1.0, "Dynamics analysis complete.")
    return results
