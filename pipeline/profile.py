"""
Steps 3–4 — Cognitive style profiling & longitudinal trajectory.

Scores each message for lexical markers:
    - system_thinking   : systems / process / framework language
    - future_orient     : forward-looking language
    - meta_cognitive    : reflective / self-aware language
    - relational        : interpersonal / collaborative language
    - uncertainty       : hedging / expressed uncertainty

Outputs:
    ai_view_profile_v01.csv   — per-message marker scores
    trajectory_monthly.csv    — monthly aggregates (per 1,000 user messages)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def run(
    db_path: str | Path,
    out_dir: str | Path,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Compute cognitive style markers and monthly trajectory."""
    raise NotImplementedError("profile.run() will be implemented in Week 2.")
