"""
Steps 9–9.1 — Directional coupling (monthly lead–lag).

Tests whether user topic changes predict assistant changes one month
later (user leads) or vice versa (assistant leads), using lag-1 cosine
similarity on macro-domain share vectors.

Requires domains.run() to have been called first (reads the two CSVs
it writes: macro_monthly_domain_shares.csv, macro_monthly_metrics.csv).

Method:
    ΔU(t) = U(t) - U(t-1)   change in user's domain share vector
    ΔA(t) = A(t) - A(t-1)   change in assistant's domain share vector

    forward  = mean cos(ΔU(t), ΔA(t+1))  — user leads assistant
    reverse  = mean cos(ΔA(t), ΔU(t+1))  — assistant leads user
    lead_diff = forward - reverse

    Permutation p-values: shuffle ΔA row order, recompute N=2,000 times.
    Per-domain Pearson r with permutation p-values.

Outputs (out_dir):
    coupling_summary.csv    — system-level forward/reverse/diff + p-values
    coupling_by_domain.csv  — per-domain r (user→asst, asst→user) + p-values
    coupling_pairs.csv      — raw cosine similarities per month-pair
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

DEFAULT_N_PERMUTATIONS    = 2_000
MIN_MSGS_PER_ROLE_PER_MONTH = 50   # months below this are excluded
RANDOM_SEED               = 42


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _perm_p_greater(null: list[float], observed: float) -> float:
    arr = np.array([v for v in null if np.isfinite(v)], dtype=float)
    if not np.isfinite(observed) or len(arr) == 0:
        return np.nan
    return float((arr >= observed).mean())


def _perm_p_abs(null: list[float], observed: float) -> float:
    arr = np.array([v for v in null if np.isfinite(v)], dtype=float)
    if not np.isfinite(observed) or len(arr) == 0:
        return np.nan
    return float((np.abs(arr) >= abs(observed)).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    out_dir: str | Path,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Compute monthly lead-lag directional coupling statistics.

    Reads:
        out_dir/macro_monthly_domain_shares.csv
        out_dir/macro_monthly_metrics.csv

    Writes:
        out_dir/coupling_summary.csv
        out_dir/coupling_by_domain.csv
        out_dir/coupling_pairs.csv

    Args:
        out_dir:         Directory containing domain CSVs (from domains.run())
        n_permutations:  Number of permutations for p-values (default 2,000)
        progress_cb:     Optional callable(fraction 0–1, status_string)

    Returns:
        Summary dict with forward_mean, reverse_mean, lead_diff, p-values
    """
    def _cb(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    out_dir = Path(out_dir)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Load CSVs ──────────────────────────────────────────────────────────
    _cb(0.05, "Loading domain share data…")
    shares_path  = out_dir / "macro_monthly_domain_shares.csv"
    metrics_path = out_dir / "macro_monthly_metrics.csv"

    if not shares_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            "macro_monthly_domain_shares.csv / macro_monthly_metrics.csv not found. "
            "Run domains.run() first."
        )

    shares  = pd.read_csv(shares_path)
    metrics = pd.read_csv(metrics_path)

    # ── 2. Filter months by message threshold ─────────────────────────────────
    keep = metrics[
        (metrics["user_msgs"]  >= MIN_MSGS_PER_ROLE_PER_MONTH) &
        (metrics["asst_msgs"]  >= MIN_MSGS_PER_ROLE_PER_MONTH)
    ]["year_month"].tolist()

    shares = shares[shares["year_month"].isin(keep)].copy()
    months  = sorted(shares["year_month"].unique())
    domains = sorted(shares["macro_domain"].unique())
    M = len(domains)
    T = len(months)

    if T < 4:
        raise ValueError(
            f"Need at least 4 qualifying months for coupling analysis; got {T}. "
            "Try lowering MIN_MSGS_PER_ROLE_PER_MONTH."
        )

    # ── 3. Build U(t), A(t) matrices ─────────────────────────────────────────
    _cb(0.15, "Building share matrices…")
    U = np.zeros((T, M), dtype=float)
    A = np.zeros((T, M), dtype=float)

    for ti, month in enumerate(months):
        g = shares[shares["year_month"] == month]
        for ji, d in enumerate(domains):
            row = g[g["macro_domain"] == d]
            U[ti, ji] = row["user_share"].values[0]  if len(row) else 0.0
            A[ti, ji] = row["asst_share"].values[0]  if len(row) else 0.0

    dU = U[1:] - U[:-1]   # (T-1) × M
    dA = A[1:] - A[:-1]

    # ── 4. Pairwise cosine similarities ───────────────────────────────────────
    _cb(0.25, "Computing lag-1 cosine similarities…")
    pair_rows     = []
    forward_sims  = []
    reverse_sims  = []

    for t in range(T - 2):
        sim_f = _cosine(dU[t], dA[t + 1])
        sim_r = _cosine(dA[t], dU[t + 1])
        forward_sims.append(sim_f)
        reverse_sims.append(sim_r)
        pair_rows.append({
            "month_t":        months[t + 1],
            "month_tplus1":   months[t + 2],
            "cos_user_leads": sim_f,
            "cos_asst_leads": sim_r,
        })

    forward_mean = float(np.nanmean(forward_sims))
    reverse_mean = float(np.nanmean(reverse_sims))
    lead_diff    = forward_mean - reverse_mean

    # ── 5. Permutation test (system-level) ────────────────────────────────────
    _cb(0.40, f"Permutation test ({n_permutations:,} iterations)…")
    null_f, null_r, null_d = [], [], []

    for _ in range(n_permutations):
        perm = rng.permutation(dA.shape[0])
        dA_p = dA[perm]
        sf, sr = [], []
        for t in range(T - 2):
            sf.append(_cosine(dU[t], dA_p[t + 1]))
            sr.append(_cosine(dA_p[t], dU[t + 1]))
        mf = float(np.nanmean(sf))
        mr = float(np.nanmean(sr))
        null_f.append(mf)
        null_r.append(mr)
        null_d.append(mf - mr)

    p_forward = _perm_p_greater(null_f, forward_mean)
    p_reverse = _perm_p_greater(null_r, reverse_mean)
    p_diff    = _perm_p_abs(null_d, lead_diff)

    # ── 6. Per-domain coupling ─────────────────────────────────────────────────
    _cb(0.70, "Computing per-domain coupling…")
    Xu  = dU[0:T - 2, :]
    Ya  = dA[1:T - 1, :]
    Xa  = dA[0:T - 2, :]
    Yu  = dU[1:T - 1, :]

    domain_rows = []
    for j, d in enumerate(domains):
        r_u2a = _pearson(Xu[:, j], Ya[:, j])
        r_a2u = _pearson(Xa[:, j], Yu[:, j])

        null_u2a, null_a2u = [], []
        for _ in range(n_permutations):
            p1 = rng.permutation(len(Ya))
            null_u2a.append(_pearson(Xu[:, j], Ya[p1, j]))
            p2 = rng.permutation(len(Yu))
            null_a2u.append(_pearson(Xa[:, j], Yu[p2, j]))

        domain_rows.append({
            "macro_domain":              d,
            "r_user_leads_asst":         r_u2a,
            "p_user_leads_asst":         _perm_p_abs(null_u2a, r_u2a),
            "r_asst_leads_user":         r_a2u,
            "p_asst_leads_user":         _perm_p_abs(null_a2u, r_a2u),
        })

    # ── 7. Write outputs ───────────────────────────────────────────────────────
    _cb(0.92, "Writing coupling results…")

    summary = pd.DataFrame([{
        "months_used":        ", ".join(months),
        "n_months":           T,
        "n_pairs":            T - 2,
        "forward_mean":       round(forward_mean, 4),
        "reverse_mean":       round(reverse_mean, 4),
        "lead_diff":          round(lead_diff, 4),
        "p_forward":          round(p_forward, 4),
        "p_reverse":          round(p_reverse, 4),
        "p_diff_abs":         round(p_diff, 4),
        "n_permutations":     n_permutations,
        "min_msgs_threshold": MIN_MSGS_PER_ROLE_PER_MONTH,
    }])

    domain_df = (
        pd.DataFrame(domain_rows)
        .sort_values("p_user_leads_asst")
    )
    pairs_df = pd.DataFrame(pair_rows)

    summary.to_csv(   out_dir / "coupling_summary.csv",    index=False)
    domain_df.to_csv( out_dir / "coupling_by_domain.csv",  index=False)
    pairs_df.to_csv(  out_dir / "coupling_pairs.csv",      index=False)

    _cb(1.0, "Coupling analysis complete.")

    return {
        "n_months":     T,
        "forward_mean": round(forward_mean, 4),
        "reverse_mean": round(reverse_mean, 4),
        "lead_diff":    round(lead_diff, 4),
        "p_diff":       round(p_diff, 4),
        "user_leads":   lead_diff > 0,
    }
