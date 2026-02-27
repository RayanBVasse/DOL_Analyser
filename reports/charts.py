"""
Plotly chart builders.

Each function accepts a pandas DataFrame (output of the corresponding
pipeline step) and returns a plotly.graph_objects.Figure.

Charts to be implemented in Week 6:
    cognitive_style_timeline    — multi-line monthly markers
    topic_entropy_timeline      — monthly + rolling window entropy
    macro_domain_heatmap        — user vs assistant share per month
    dyadic_alignment_timeline   — monthly JS divergence
    coupling_bar                — lead-lag coupling by macro-domain
    state_timeline              — Exploration / Consolidation / Transitional
    shift_initiation_donut      — user vs assistant % of domain shifts
"""

from __future__ import annotations

# Implementations will be added in Week 6.
# All functions will follow the signature:
#     def <chart_name>(df: pd.DataFrame, **kwargs) -> go.Figure
