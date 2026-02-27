"""
Steps 9–9.1 — Directional coupling (weekly & monthly).

Tests whether user topic changes predict assistant changes at lag-1
(or vice versa) at both weekly and monthly timescales.

Method: lag-1 cosine similarity on macro-domain share vectors,
        permutation p-values (N=2,000).

Outputs:
    weekly_directional_summary.csv
    weekly_directional_by_domain.csv
    step9_directional_coupling_summary.csv
    step9_directional_coupling_by_domain.csv
    step9_directional_coupling_pairs.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

DEFAULT_N_PERMUTATIONS = 2_000


def run(
    out_dir: str | Path,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Compute weekly and monthly lead-lag coupling statistics."""
    raise NotImplementedError("coupling.run() will be implemented in Week 2.")
