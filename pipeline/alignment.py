"""
Step 7 — Dyadic alignment.

Measures monthly Jensen-Shannon divergence between user and assistant
topic distributions. Lower JS divergence = more topically aligned.

Output:
    dyadic_alignment_monthly.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def run(
    db_path: str | Path,
    out_dir: str | Path,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Compute monthly JS divergence between user and assistant topic distributions."""
    raise NotImplementedError("alignment.run() will be implemented in Week 2.")
