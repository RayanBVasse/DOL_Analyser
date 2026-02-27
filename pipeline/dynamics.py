"""
Steps 10a/b — Behavioural dynamics.

10a1  scale_separation     — weekly vs monthly structure index
10a2  state_segmentation   — Exploration / Consolidation / Transitional states
                             (static-level and rate-based variants)
10a3  rolling_entropy      — sliding-window Shannon entropy (window=250 msgs)
10b   shift_initiation     — who initiates domain shifts (permutation test)
10b2  episode_initiation   — who initiates sustained topic episodes
                             (binomial test, min_episode_len=20)

Outputs (one CSV per sub-step):
    scale_separation_report.csv
    monthly_states.csv
    monthly_states_refined.csv
    state_transition_summary.csv
    state_transition_summary_refined.csv
    rolling_entropy_250.csv
    step10b_shift_initiation_summary.csv
    step10b_shift_initiation_detail.csv
    step10b2_episode_summary.csv
    step10b2_episode_detail.csv
    step10b2_thread_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

DEFAULT_CONFIG = {
    "rolling_window": 250,
    "rolling_stride": 250,
    "min_thread_msgs": 20,
    "min_episode_len": 20,
    "shift_permutations": 5_000,
}


def run(
    db_path: str | Path,
    out_dir: str | Path,
    config: dict | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Run all Step 10 sub-analyses."""
    raise NotImplementedError("dynamics.run() will be implemented in Week 2.")
