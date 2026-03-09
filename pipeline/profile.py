"""
Steps 3–4 — Cognitive style profiling & longitudinal trajectory.

Two markers only (relational framing dropped — dominated by structural
AI-address pronoun 'you', not a cognitive signal):

    - structural_thinking  : systems / process / framework language (30 terms)
    - epistemic_uncertainty: hedging / expressed uncertainty (12 terms)

Word lists sourced from DOL/step3_expanded_lexicon.py (validated).

Outputs:
    trajectory_monthly.csv  — monthly counts + per-1,000 user-message rates
    spearman_results.csv    — Spearman ρ + permutation p-values per marker
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ─────────────────────────────────────────────────────────────────────────────
# Word lists  (from step3_expanded_lexicon.py)
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURAL_THINKING: list[str] = [
    # Original 10
    "model", "framework", "structure", "system", "pipeline",
    "architecture", "taxonomy", "theory", "design", "mechanism",
    # Expanded +20
    "pattern", "layer", "component", "schema", "hierarchy",
    "module", "abstraction", "paradigm", "protocol", "workflow",
    "ontology", "construct", "blueprint", "scaffold", "template",
    "topology", "mapping", "representation", "heuristic", "iteration",
    "decomposition", "integration", "inference", "criterion", "variable",
    "dimension", "constraint", "parameter", "formalisation", "operationalisation",
]

EPISTEMIC_UNCERTAINTY: list[str] = [
    # Original 5
    "maybe", "perhaps", "not sure", "i guess", "possibly",
    # Expanded +7  (could/might deliberately excluded)
    "uncertain", "unclear", "unsure", "ambiguous",
    "i wonder", "i suspect", "not certain",
]

# Map marker name → word list
MARKERS: dict[str, list[str]] = {
    "structural_thinking":   STRUCTURAL_THINKING,
    "epistemic_uncertainty": EPISTEMIC_UNCERTAINTY,
}

# Months with fewer than this many user messages are excluded
MIN_MESSAGES = 100

# Permutation test settings
N_PERMUTATIONS = 10_000
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_pattern(word_list: list[str]) -> re.Pattern:
    """Compile a single word-boundary regex from a word/phrase list."""
    escaped = sorted(map(re.escape, word_list), key=len, reverse=True)
    return re.compile(
        r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)",
        re.IGNORECASE,
    )


def _count_matches(texts: list[str], pattern: re.Pattern) -> int:
    return sum(len(pattern.findall(t)) for t in texts if isinstance(t, str))


def _permutation_p(
    observed_rho: float,
    x: np.ndarray,
    y: np.ndarray,
    n_perms: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
    two_tailed: bool = False,
) -> float:
    """Permutation p-value for Spearman ρ."""
    rng = np.random.default_rng(seed)
    null = np.array([
        spearmanr(x, rng.permutation(y)).statistic
        for _ in range(n_perms)
    ])
    if two_tailed:
        return float(np.mean(np.abs(null) >= abs(observed_rho)))
    return float(np.mean(null >= observed_rho))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    db_path: str | Path,
    out_dir: str | Path,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Compute cognitive style marker trajectories from the parsed SQLite DB.

    Args:
        db_path:     SQLite database written by pipeline.parse.run()
        out_dir:     Directory to write output CSVs
        progress_cb: Optional callable(fraction 0–1, status_string)

    Returns:
        Summary dict with monthly row count and Spearman results per marker.
    """
    def _cb(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load user messages from DB ────────────────────────────────────────
    _cb(0.05, "Loading messages from database…")
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT year_month, text FROM messages WHERE role = 'user'",
        con,
    )
    con.close()

    if df.empty:
        raise ValueError("No user messages found in database.")

    months = sorted(df["year_month"].dropna().unique())

    # ── 2. Compile regex patterns ─────────────────────────────────────────────
    _cb(0.10, "Compiling lexicon patterns…")
    patterns = {name: _build_pattern(wlist) for name, wlist in MARKERS.items()}

    # ── 3. Monthly counts ─────────────────────────────────────────────────────
    _cb(0.15, "Counting marker occurrences by month…")
    traj_rows = []
    n_months = len(months)

    for i, month in enumerate(months):
        _cb(0.15 + 0.55 * (i / max(n_months, 1)), f"Scoring month {month}…")
        texts = df[df["year_month"] == month]["text"].tolist()
        n = len(texts)

        if n < MIN_MESSAGES:
            continue

        row: dict = {"year_month": month, "user_messages": n}
        for name, pat in patterns.items():
            count = _count_matches(texts, pat)
            row[f"{name}_count"] = count
            row[f"{name}_per1k"] = count / n * 1000

        traj_rows.append(row)

    traj = pd.DataFrame(traj_rows)
    traj_path = out_dir / "trajectory_monthly.csv"
    traj.to_csv(traj_path, index=False)

    # ── 4. Spearman + permutation p ──────────────────────────────────────────
    _cb(0.75, "Computing Spearman correlations…")
    time_idx = np.arange(1, len(traj) + 1, dtype=float)
    stat_rows = []

    for name in MARKERS:
        col = f"{name}_per1k"
        if col not in traj.columns or len(traj) < 3:
            continue

        vals = traj[col].values
        rho, _ = spearmanr(time_idx, vals)

        # structural: one-tailed (expected positive trend)
        # epistemic:  two-tailed (non-monotonic predicted)
        two_tailed = name == "epistemic_uncertainty"
        p = _permutation_p(rho, time_idx, vals, two_tailed=two_tailed)

        stat_rows.append({
            "marker":          name,
            "spearman_rho":    round(rho, 3),
            "p_value":         round(p, 4),
            "two_tailed":      two_tailed,
            "significant":     p < 0.05,
            "min_per1k":       round(float(vals.min()), 1),
            "max_per1k":       round(float(vals.max()), 1),
            "fold_change":     round(float(vals.max() / vals.min()), 2) if vals.min() > 0 else None,
            "n_terms":         len(MARKERS[name]),
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_path = out_dir / "spearman_results.csv"
    stat_df.to_csv(stat_path, index=False)

    _cb(1.0, "Profiling complete.")

    return {
        "months_scored": len(traj),
        "markers": list(MARKERS.keys()),
        "spearman": stat_rows,
    }
