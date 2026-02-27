"""
Step 8r — Robustness & null model tests.

Grid-searches over K ∈ {40, 60, 80} × SVD dims ∈ {100, 200}.
Runs three null models (month shuffle, role shuffle, volume downsample)
with N=200 permutations each.

Output:
    robustness_null_tests.csv
    robustness_curves_k{K}_svd{D}.csv  (one per hyperparameter combo)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

DEFAULT_CONFIG = {
    "k_values": [40, 60, 80],
    "svd_values": [100, 200],
    "n_permutations": 200,
    "random_state": 42,
}


def run(
    db_path: str | Path,
    out_dir: str | Path,
    config: dict | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Run robustness grid search and permutation null tests."""
    raise NotImplementedError("robustness.run() will be implemented in Week 2.")
