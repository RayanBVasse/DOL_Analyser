"""
Steps 8.5–8.6 — Macro-domain mapping.

Meta-clusters the 60 fine topic clusters into M=8 macro-domains.
Assigns every message node a macro-domain label.

Outputs:
    macro_cluster_map.csv
    macro_domain_summary.csv
    macro_monthly_metrics.csv
    macro_monthly_domain_shares.csv
    node_to_fine_cluster.csv
    node_to_macro_domain.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

DEFAULT_N_MACRO = 8


def run(
    db_path: str | Path,
    out_dir: str | Path,
    n_macro: int = DEFAULT_N_MACRO,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Build macro-domain hierarchy and assign every message a domain label."""
    raise NotImplementedError("domains.run() will be implemented in Week 2.")
