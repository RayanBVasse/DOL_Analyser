"""
reports/charts.py — Plotly figure builders for all DOL Analyser pipeline outputs.

Each function accepts DataFrames produced by the pipeline steps and returns a
``plotly.graph_objects.Figure`` suitable for both:
  - Streamlit display        →  st.plotly_chart(fig, use_container_width=True)
  - Self-contained HTML      →  fig.to_html(full_html=False, include_plotlyjs=False)

Colour conventions (consistent across all charts):
  User          #4C78A8   steel-blue
  Assistant     #F58518   amber-orange
  Exploration   #54A24B   green
  Consolidation #4C78A8   steel-blue
  Transitional  #E45756   coral-red
  Neutral       #BAB0AC   warm-grey
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Shared style constants
# ─────────────────────────────────────────────────────────────────────────────

C_USER    = "#4C78A8"
C_ASST    = "#F58518"
C_EXPLORE = "#54A24B"
C_CONSOL  = "#4C78A8"
C_TRANS   = "#E45756"
C_GREY    = "#BAB0AC"
C_BG      = "#FFFFFF"
C_GRID    = "#EBEBEB"

STATE_COLOURS: dict[str, str] = {
    "Exploration":   C_EXPLORE,
    "Consolidation": C_CONSOL,
    "Transitional":  C_TRANS,
    "Initial":       C_GREY,
}

# Eight-colour qualitative palette for domain charts
DOMAIN_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#72B7B2", "#B279A2", "#FF9DA7", "#9D755D",
]

_BASE = dict(
    plot_bgcolor  = C_BG,
    paper_bgcolor = C_BG,
    font          = dict(family="Inter, Arial, sans-serif", size=13, color="#333333"),
    margin        = dict(t=64, b=52, l=64, r=32),
    hovermode     = "x unified",
    xaxis         = dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
    yaxis         = dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
)


def _base(fig: go.Figure, title: str, **extra) -> go.Figure:
    """Apply shared layout and title to a figure."""
    layout = dict(_BASE)
    layout.update(extra)
    layout["title"] = dict(text=title, x=0.05, font_size=16)
    fig.update_layout(**layout)
    return fig


def _short_label(row: pd.Series, max_terms: int = 3) -> str:
    """Build 'D{id}: term1, term2, term3' from a domain summary row."""
    terms = ", ".join(str(row["auto_label"]).split(", ")[:max_terms])
    return f"D{row['macro_domain']}: {terms}"


# ─────────────────────────────────────────────────────────────────────────────
# 1 — Cognitive style trajectory
# ─────────────────────────────────────────────────────────────────────────────

def cognitive_style_timeline(traj_df: pd.DataFrame) -> go.Figure:
    """
    Dual-line monthly rates (per 1,000 user messages):
      • Structural thinking   — solid blue
      • Epistemic uncertainty — dashed orange
    """
    df = traj_df.copy().sort_values("year_month")
    fig = go.Figure()

    if "structural_thinking_per1k" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["year_month"],
            y=df["structural_thinking_per1k"],
            name="Structural thinking",
            mode="lines+markers",
            line=dict(color=C_USER, width=2.5),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="%{y:.1f} /1k<extra>Structural thinking</extra>",
        ))

    if "epistemic_uncertainty_per1k" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["year_month"],
            y=df["epistemic_uncertainty_per1k"],
            name="Epistemic uncertainty",
            mode="lines+markers",
            line=dict(color=C_ASST, width=2.5, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="%{y:.1f} /1k<extra>Epistemic uncertainty</extra>",
        ))

    _base(
        fig, "Cognitive Style Over Time",
        yaxis=dict(title="Rate per 1,000 messages", showgrid=True, gridcolor=C_GRID),
        legend=dict(orientation="h", y=-0.20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2 — Topic entropy timeline (monthly bars + optional rolling line)
# ─────────────────────────────────────────────────────────────────────────────

def topic_entropy_timeline(
    entropy_df: pd.DataFrame,
    rolling_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Monthly topic entropy (nats) as bars.
    If rolling_df is provided, overlays rolling window entropy (bits) on a
    secondary Y-axis.
    """
    df = entropy_df.copy().sort_values("year_month")
    has_rolling = rolling_df is not None and not rolling_df.empty

    fig = make_subplots(specs=[[{"secondary_y": has_rolling}]])

    fig.add_trace(go.Bar(
        x=df["year_month"],
        y=df["topic_entropy_nats"],
        name="Monthly entropy (nats)",
        marker_color=C_USER,
        opacity=0.72,
        hovertemplate="%{y:.3f} nats<extra>Monthly</extra>",
    ), secondary_y=False)

    if has_rolling:
        r = rolling_df.copy()
        # Prefer numeric window midpoint as x-axis label
        x_vals = r["window_start"].astype(str)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=r["entropy_bits"],
            name="Rolling entropy (bits, w=250)",
            mode="lines",
            line=dict(color=C_ASST, width=2.2),
            hovertemplate="%{y:.2f} bits<extra>Rolling (w=250)</extra>",
        ), secondary_y=True)
        fig.update_yaxes(
            title_text="Rolling entropy (bits)",
            secondary_y=True,
            showgrid=False,
            zeroline=False,
        )

    fig.update_layout(
        plot_bgcolor=C_BG, paper_bgcolor=C_BG,
        font=dict(family="Inter, Arial, sans-serif", size=13, color="#333333"),
        margin=dict(t=64, b=52, l=64, r=32),
        hovermode="x unified",
        title=dict(text="Topic Entropy Over Time", x=0.05, font_size=16),
        barmode="overlay",
        yaxis=dict(
            title="Monthly topic entropy (nats)",
            showgrid=True, gridcolor=C_GRID, zeroline=False,
        ),
        legend=dict(orientation="h", y=-0.20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Dyadic alignment timeline
# ─────────────────────────────────────────────────────────────────────────────

def dyadic_alignment_timeline(align_df: pd.DataFrame) -> go.Figure:
    """
    Monthly Jensen-Shannon divergence between user and assistant topic
    distributions.  Lower = more in sync.
    """
    df = align_df.copy().sort_values("year_month").dropna(subset=["js_divergence"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year_month"],
        y=df["js_divergence"],
        name="JS divergence",
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(76,120,168,0.10)",
        line=dict(color=C_USER, width=2.5),
        marker=dict(size=7),
        hovertemplate="%{y:.4f}<extra>JS divergence</extra>",
    ))

    fig.add_annotation(
        text="Lower = topics more in sync",
        x=0.01, y=0.96, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=11, color=C_GREY),
        xanchor="left",
    )

    _base(
        fig, "Dyadic Alignment (Topic Similarity)",
        yaxis=dict(
            title="Jensen-Shannon divergence",
            range=[0, None],
            showgrid=True, gridcolor=C_GRID, zeroline=False,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Macro-domain size bar
# ─────────────────────────────────────────────────────────────────────────────

def macro_domain_bar(
    domain_summary_df: pd.DataFrame,
    top_n: int = 8,
) -> go.Figure:
    """
    Horizontal bar chart of macro-domain message counts, sorted largest→top.
    """
    df = (
        domain_summary_df
        .copy()
        .sort_values("size", ascending=True)
        .tail(top_n)
    )

    n = len(df)
    labels  = [_short_label(row, 4) for _, row in df.iterrows()]
    colours = [
        f"rgba(76,120,168,{0.38 + 0.62 * i / max(n - 1, 1):.2f})"
        for i in range(n)
    ]

    fig = go.Figure(go.Bar(
        x=df["size"],
        y=labels,
        orientation="h",
        marker_color=colours,
        hovertemplate="%{x:,} messages<br>%{y}<extra></extra>",
    ))

    _base(
        fig, "Macro-Domain Distribution",
        xaxis=dict(title="Messages", showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(automargin=True, showgrid=False),
        margin=dict(t=64, b=52, l=230, r=32),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5 — Domain shares stacked area (user)
# ─────────────────────────────────────────────────────────────────────────────

def domain_shares_area(
    shares_df: pd.DataFrame,
    domain_summary_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Stacked area chart of user macro-domain shares over time.
    Each band is one macro-domain, coloured consistently with the palette.
    """
    df      = shares_df.copy()
    months  = sorted(df["year_month"].unique())
    domains = sorted(df["macro_domain"].unique())

    # Label map
    label_map: dict = {}
    if domain_summary_df is not None:
        for _, row in domain_summary_df.iterrows():
            label_map[row["macro_domain"]] = _short_label(row, 3)
    else:
        label_map = {d: f"Domain {d}" for d in domains}

    fig = go.Figure()

    for i, d in enumerate(domains):
        sub = df[df["macro_domain"] == d].set_index("year_month")
        y   = [
            float(sub.loc[m, "user_share"]) if m in sub.index else 0.0
            for m in months
        ]
        colour = DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)]
        fig.add_trace(go.Scatter(
            x=months,
            y=y,
            name=label_map.get(d, f"D{d}"),
            stackgroup="user",
            mode="lines",
            line=dict(width=0.6, color=colour),
            fillcolor=colour,
            hovertemplate="%{y:.1%}<extra>" + label_map.get(d, f"D{d}") + "</extra>",
        ))

    _base(
        fig, "Your Topic Focus Over Time",
        yaxis=dict(
            title="Share of your messages",
            tickformat=".0%",
            range=[0, 1],
            showgrid=True, gridcolor=C_GRID, zeroline=False,
        ),
        legend=dict(orientation="h", y=-0.24, traceorder="normal"),
        margin=dict(t=64, b=110, l=64, r=32),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6 — Behavioural state timeline
# ─────────────────────────────────────────────────────────────────────────────

def state_timeline(states_df: pd.DataFrame) -> go.Figure:
    """
    Two-panel figure:
      Top (35 %):  colour-coded state strip — one bar per month
      Bottom (65 %): entropy + JS divergence as lines
    """
    df = states_df.copy().sort_values("year_month").reset_index(drop=True)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.70],
        vertical_spacing=0.06,
    )

    # ── Top: state colour strip ───────────────────────────────────────────────
    seen_states: set = set()
    for _, row in df.iterrows():
        state  = str(row.get("state", "Transitional"))
        colour = STATE_COLOURS.get(state, C_GREY)
        show   = state not in seen_states
        seen_states.add(state)
        fig.add_trace(go.Bar(
            x=[row["year_month"]],
            y=[1],
            name=state,
            legendgroup=state,
            showlegend=show,
            marker_color=colour,
            hovertemplate=f"{row['year_month']}: <b>{state}</b><extra></extra>",
        ), row=1, col=1)

    # ── Bottom: signals ───────────────────────────────────────────────────────
    if "macro_entropy_user" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["year_month"],
            y=df["macro_entropy_user"],
            name="Macro entropy (user)",
            legendgroup="Macro entropy (user)",
            mode="lines+markers",
            line=dict(color=C_USER, width=2.2),
            marker=dict(size=6),
            hovertemplate="%{y:.3f}<extra>Entropy</extra>",
        ), row=2, col=1)

    if "macro_js_divergence" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["year_month"],
            y=df["macro_js_divergence"],
            name="JS divergence",
            legendgroup="JS divergence",
            mode="lines+markers",
            line=dict(color=C_ASST, width=2.2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="%{y:.4f}<extra>JS divergence</extra>",
        ), row=2, col=1)

    fig.update_yaxes(
        showticklabels=False, showgrid=False, row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Value", showgrid=True, gridcolor=C_GRID,
        zeroline=False, row=2, col=1,
    )
    fig.update_layout(
        plot_bgcolor=C_BG, paper_bgcolor=C_BG,
        font=dict(family="Inter, Arial, sans-serif", size=13, color="#333333"),
        title=dict(text="Behavioural States Over Time", x=0.05, font_size=16),
        margin=dict(t=64, b=52, l=64, r=32),
        hovermode="x unified",
        barmode="stack",
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7 — Lead–lag coupling bar (per domain)
# ─────────────────────────────────────────────────────────────────────────────

def coupling_bar(
    domain_df: pd.DataFrame,
    domain_summary_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Grouped bar chart of Pearson r for user→AI and AI→user per macro-domain.
    Bars with p < 0.05 shown at full opacity; non-significant at 35%.
    """
    df = domain_df.copy().sort_values("macro_domain")

    label_map: dict = {}
    if domain_summary_df is not None:
        for _, row in domain_summary_df.iterrows():
            label_map[row["macro_domain"]] = _short_label(row, 2)
    else:
        label_map = {int(d): f"D{d}" for d in df["macro_domain"]}

    x_labels = [label_map.get(int(d), f"D{d}") for d in df["macro_domain"]]

    def _opacities(p_col: str) -> list[float]:
        return [1.0 if p <= 0.05 else 0.32 for p in df[p_col]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="You → AI",
        x=x_labels,
        y=df["r_user_leads_asst"],
        marker_color=[
            f"rgba(76,120,168,{op:.2f})" for op in _opacities("p_user_leads_asst")
        ],
        hovertemplate="<b>%{x}</b><br>r = %{y:.3f}<extra>You → AI</extra>",
    ))

    fig.add_trace(go.Bar(
        name="AI → You",
        x=x_labels,
        y=df["r_asst_leads_user"],
        marker_color=[
            f"rgba(245,133,24,{op:.2f})" for op in _opacities("p_asst_leads_user")
        ],
        hovertemplate="<b>%{x}</b><br>r = %{y:.3f}<extra>AI → You</extra>",
    ))

    fig.add_hline(y=0, line_width=1, line_color=C_GREY)

    fig.add_annotation(
        text="Full opacity = p < 0.05",
        x=1.0, y=1.06, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=10, color=C_GREY),
        xanchor="right",
    )

    _base(
        fig, "Lead–Lag Coupling by Domain",
        barmode="group",
        yaxis=dict(
            title="Pearson r (lag-1)",
            showgrid=True, gridcolor=C_GRID, zeroline=False,
        ),
        xaxis=dict(showgrid=False, zeroline=False, automargin=True),
        legend=dict(orientation="h", y=-0.20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8 — Shift initiation donut
# ─────────────────────────────────────────────────────────────────────────────

def shift_initiation_donut(summary_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart: % of domain shifts initiated by user vs assistant,
    annotated with total count and permutation p-value.
    """
    fig = go.Figure()

    if summary_df.empty:
        fig.add_annotation(
            text="No shift data available",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color=C_GREY),
        )
        return fig

    row   = summary_df.iloc[0]
    u_pct = float(row["user_initiated_prop"])
    a_pct = float(row["asst_initiated_prop"])
    total = int(row["total_shifts"])
    p_val = float(row["perm_p_vs_random"])

    fig.add_trace(go.Pie(
        labels=["You initiated", "AI initiated"],
        values=[u_pct, a_pct],
        hole=0.58,
        marker=dict(
            colors=[C_USER, C_ASST],
            line=dict(color=C_BG, width=2),
        ),
        textinfo="label+percent",
        textfont_size=13,
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))

    # Centre annotation
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:12px;color:{C_GREY}'>shifts</span>",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=18, color="#333333"),
    )

    # p-value below chart
    sig = "✱ significant" if p_val < 0.05 else "not significant"
    fig.add_annotation(
        text=f"Permutation p = {p_val:.3f}  ({sig} vs 50/50)",
        x=0.5, y=-0.10,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=11, color=C_GREY),
    )

    fig.update_layout(
        plot_bgcolor=C_BG,
        paper_bgcolor=C_BG,
        font=dict(family="Inter, Arial, sans-serif", size=13, color="#333333"),
        title=dict(text="Domain Shift Initiation", x=0.5, font_size=16),
        margin=dict(t=64, b=80, l=40, r=40),
        showlegend=False,
    )
    return fig
