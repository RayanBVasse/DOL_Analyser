"""
HTML report assembler.

Combines all Plotly charts and auto-generated narrative text into a single
self-contained .html file that the user can save and open in any browser
without an internet connection.

To be implemented in Week 7.
"""

from __future__ import annotations

from pathlib import Path


def build(
    figures: list,
    narratives: list[str],
    meta: dict,
    out_path: str | Path,
) -> Path:
    """
    Write a self-contained HTML report.

    Args:
        figures:    List of plotly Figure objects (one per analysis section).
        narratives: List of plain-English interpretation strings (same order as figures).
        meta:       Report metadata — format, date_range, message_count, etc.
        out_path:   Destination path for the .html file.

    Returns:
        Path to the written file.
    """
    raise NotImplementedError("html_export.build() will be implemented in Week 7.")
