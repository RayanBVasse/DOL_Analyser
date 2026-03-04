"""
reports/html_export.py — Self-contained HTML report assembler.

Combines all Plotly charts, auto-generated narrative text, and summary
statistics into a single .html file that opens in any browser with no
internet connection required (Plotly JS is inlined).

Public API:
    build(work_dir, session_meta, out_path) -> Path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from reports import charts


# ─────────────────────────────────────────────────────────────────────────────
# Narrative templates
# Each function receives the relevant DataFrame(s) and returns an HTML string.
# ─────────────────────────────────────────────────────────────────────────────

def _narrative_cognitive(traj_df: pd.DataFrame) -> str:
    if traj_df.empty:
        return "<p>No cognitive style data available.</p>"

    st_col = "structural_thinking_per1k"
    eu_col = "epistemic_uncertainty_per1k"
    has_st = st_col in traj_df.columns
    has_eu = eu_col in traj_df.columns

    lines = [
        "<p><strong>What this shows:</strong> How often you used language associated "
        "with two cognitive styles across each month of your AI conversations.</p>",
        "<ul>"
        "<li><strong>Structural thinking</strong> — words like "
        "<em>model, framework, architecture, pipeline, taxonomy</em>. "
        "A rising rate suggests you are increasingly building mental models and "
        "organising knowledge into explicit structures.</li>"
        "<li><strong>Epistemic uncertainty</strong> — words like "
        "<em>maybe, perhaps, not sure, possibly, I guess</em>. "
        "A high rate reflects open-minded, exploratory thinking; a falling rate "
        "may signal growing confidence or consolidation.</li>"
        "</ul>",
    ]

    if has_st and has_eu:
        first = traj_df.iloc[0]
        last  = traj_df.iloc[-1]
        st_dir = "increased" if last[st_col] > first[st_col] else "decreased"
        eu_dir = "increased" if last[eu_col] > first[eu_col] else "decreased"
        lines.append(
            f"<p><strong>In your data:</strong> Structural thinking <em>{st_dir}</em> "
            f"from {first[st_col]:.1f} to {last[st_col]:.1f} /1k. "
            f"Epistemic uncertainty <em>{eu_dir}</em> "
            f"from {first[eu_col]:.1f} to {last[eu_col]:.1f} /1k.</p>"
        )

    return "\n".join(lines)


def _narrative_entropy(entropy_df: pd.DataFrame) -> str:
    if entropy_df.empty:
        return "<p>No topic entropy data available.</p>"

    hi = entropy_df.loc[entropy_df["topic_entropy_nats"].idxmax(), "year_month"]
    lo = entropy_df.loc[entropy_df["topic_entropy_nats"].idxmin(), "year_month"]

    return (
        "<p><strong>What this shows:</strong> How spread across topics your messages "
        "were each month. High entropy = you ranged widely across many topics; "
        "low entropy = your conversations were more focused on a narrow set.</p>"
        "<p>The bars show monthly entropy in <em>nats</em> (natural-log units). "
        "The overlaid line shows the same signal computed in a rolling 250-message "
        "window — useful for seeing short-term shifts within a month.</p>"
        f"<p><strong>In your data:</strong> Broadest month was <strong>{hi}</strong>; "
        f"most focused month was <strong>{lo}</strong>.</p>"
    )


def _narrative_alignment(align_df: pd.DataFrame) -> str:
    if align_df.empty:
        return "<p>No alignment data available.</p>"

    mean_js = align_df["js_divergence"].mean()
    trend   = align_df["js_divergence"].iloc[-1] - align_df["js_divergence"].iloc[0]
    trend_s = "converging (divergence falling)" if trend < 0 else "diverging (divergence rising)"

    return (
        "<p><strong>What this shows:</strong> Each month, how different were the topics "
        "<em>you</em> discussed from the topics <em>the AI</em> discussed? "
        "This is measured with Jensen-Shannon (JS) divergence — a value of 0 means "
        "identical topic distributions; 1 means completely different.</p>"
        "<p>High alignment (low JS) means you and the AI were covering the same ground. "
        "Low alignment (high JS) suggests the AI was steering conversations into "
        "territory you hadn't initiated, or vice versa.</p>"
        f"<p><strong>In your data:</strong> Mean JS divergence was <strong>{mean_js:.3f}</strong>. "
        f"Over time the alignment is <em>{trend_s}</em>.</p>"
    )


def _narrative_domains(domain_df: pd.DataFrame) -> str:
    if domain_df.empty:
        return "<p>No domain data available.</p>"

    top = domain_df.sort_values("size", ascending=False).iloc[0]
    total = domain_df["size"].sum()
    top_pct = 100 * top["size"] / total if total > 0 else 0

    return (
        "<p><strong>What this shows:</strong> All your conversations were grouped "
        "into up to 8 broad <em>macro-domains</em> using hierarchical topic modelling "
        "(TF-IDF → SVD → KMeans → meta-clustering). Each domain is labelled with "
        "its most characteristic terms.</p>"
        "<p>The bar lengths show how many messages fell into each domain across your "
        "entire conversation history.</p>"
        f"<p><strong>In your data:</strong> Your dominant domain was "
        f"<strong>\"{top['auto_label'][:60]}\"</strong>, "
        f"accounting for {top_pct:.0f}% of all messages.</p>"
    )


def _narrative_domain_focus(shares_df: pd.DataFrame) -> str:
    if shares_df.empty:
        return "<p>No domain share data available.</p>"

    return (
        "<p><strong>What this shows:</strong> How your conversational focus shifted "
        "across the 8 macro-domains month by month. Each coloured band represents "
        "one domain; the band's thickness shows what proportion of your messages "
        "that month fell into that domain.</p>"
        "<p>A stable pattern suggests you stuck to the same topics over time. "
        "Large shifts — bands suddenly growing or shrinking — indicate you pivoted "
        "your intellectual focus. Compare this with the Behavioural States chart "
        "to see whether these shifts align with Exploration or Consolidation phases.</p>"
    )


def _narrative_states(states_df: pd.DataFrame) -> str:
    if states_df.empty:
        return "<p>No state data available.</p>"

    counts = states_df["state"].value_counts().to_dict() if "state" in states_df.columns else {}
    explore = counts.get("Exploration", 0)
    consol  = counts.get("Consolidation", 0)
    trans   = counts.get("Transitional", 0)

    return (
        "<p><strong>What this shows:</strong> Each month is classified into one of "
        "three behavioural states based on topic entropy and JS divergence:</p>"
        "<ul>"
        "<li><strong style='color:#54A24B'>Exploration</strong> — high entropy "
        "<em>and</em> high divergence. You were ranging broadly across many topics, "
        "and you and the AI were covering different ground.</li>"
        "<li><strong style='color:#4C78A8'>Consolidation</strong> — low entropy "
        "<em>and</em> low divergence. You were focused and in sync with the AI — "
        "a period of deepening rather than broadening.</li>"
        "<li><strong style='color:#E45756'>Transitional</strong> — mixed signals. "
        "Neither clearly exploratory nor consolidating.</li>"
        "</ul>"
        "<p>The lower panel shows the raw signals: macro-domain entropy (how spread "
        "your topics were) and JS divergence (how different your topics were from "
        "the AI's).</p>"
        f"<p><strong>In your data:</strong> {explore} Exploration month(s), "
        f"{consol} Consolidation month(s), {trans} Transitional month(s).</p>"
    )


def _narrative_coupling(coup_df: pd.DataFrame, summary_path: Optional[Path] = None) -> str:
    forward = reverse = lead_diff = p_diff = None
    if summary_path and summary_path.exists():
        s = pd.read_csv(summary_path).iloc[0]
        forward   = float(s["forward_mean"])
        reverse   = float(s["reverse_mean"])
        lead_diff = float(s["lead_diff"])
        p_diff    = float(s["p_diff_abs"])

    direction = ""
    if lead_diff is not None:
        if lead_diff > 0:
            direction = (
                f"<p><strong>In your data:</strong> You tend to <em>lead</em> the AI "
                f"(forward r = {forward:.3f}, reverse r = {reverse:.3f}, "
                f"diff = {lead_diff:+.3f}, p = {p_diff:.3f}). "
                "Your topic changes this month predict the AI's next-month topics — "
                "you are steering the conversation agenda.</p>"
            )
        else:
            direction = (
                f"<p><strong>In your data:</strong> The AI tends to <em>lead</em> you "
                f"(forward r = {forward:.3f}, reverse r = {reverse:.3f}, "
                f"diff = {lead_diff:+.3f}, p = {p_diff:.3f}). "
                "The AI's topic shifts this month predict where you go next month.</p>"
            )

    return (
        "<p><strong>What this shows:</strong> Does your conversational agenda "
        "<em>lead</em> the AI's, or does the AI lead yours? This is tested with "
        "lag-1 cosine similarity on monthly topic-share change vectors:</p>"
        "<ul>"
        "<li><strong>Forward coupling</strong> — your topic changes this month predict "
        "the AI's topic changes next month (you lead).</li>"
        "<li><strong>Reverse coupling</strong> — the AI's topic changes this month "
        "predict yours next month (AI leads).</li>"
        "</ul>"
        "<p>Bars at full opacity have a permutation p-value below 0.05.</p>"
        + direction
    )


def _narrative_shifts(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "<p>No shift initiation data available.</p>"

    row   = summary_df.iloc[0]
    u_pct = float(row["user_initiated_prop"]) * 100
    a_pct = float(row["asst_initiated_prop"]) * 100
    total = int(row["total_shifts"])
    p     = float(row["perm_p_vs_random"])
    sig   = "statistically significant" if p < 0.05 else "not statistically significant"

    initiator = "You" if u_pct >= 50 else "The AI"
    other     = "the AI" if u_pct >= 50 else "you"

    return (
        "<p><strong>What this shows:</strong> Within each conversation thread, "
        "whenever the topic changed to a different macro-domain, who made that change — "
        "you or the AI? This reveals whether you are driving the intellectual agenda "
        "within conversations, or following the AI's lead.</p>"
        "<p>The percentages are tested against a 50/50 random baseline using a "
        "permutation test (N = 5,000 shuffles).</p>"
        f"<p><strong>In your data:</strong> Across {total:,} domain shifts, "
        f"<strong>{initiator}</strong> initiated {max(u_pct, a_pct):.1f}% and "
        f"{other} initiated {min(u_pct, a_pct):.1f}%. "
        f"This difference is <em>{sig}</em> (p = {p:.3f}).</p>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML shell
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: Inter, -apple-system, Arial, sans-serif;
    font-size: 15px;
    line-height: 1.65;
    color: #222;
    background: #f7f7f7;
}
.page { max-width: 960px; margin: 0 auto; padding: 40px 24px 80px; }
header {
    border-bottom: 3px solid #4C78A8;
    padding-bottom: 20px;
    margin-bottom: 36px;
}
header h1 { font-size: 2rem; font-weight: 700; color: #4C78A8; }
header p  { color: #666; margin-top: 6px; }
.meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 40px;
}
.meta-card {
    background: #fff;
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
}
.meta-card .label { font-size: 11px; text-transform: uppercase;
                    letter-spacing: .07em; color: #888; margin-bottom: 4px; }
.meta-card .value { font-size: 1.5rem; font-weight: 600; color: #222; }
.section {
    background: #fff;
    border-radius: 10px;
    padding: 28px 28px 20px;
    margin-bottom: 28px;
    box-shadow: 0 1px 5px rgba(0,0,0,.06);
}
.section h2 {
    font-size: 1.15rem;
    font-weight: 600;
    color: #4C78A8;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ebebeb;
}
.narrative {
    margin-bottom: 18px;
    color: #444;
}
.narrative p  { margin-bottom: 10px; }
.narrative ul { padding-left: 20px; margin-bottom: 10px; }
.narrative li { margin-bottom: 4px; }
.chart-wrap   { width: 100%; }
footer {
    text-align: center;
    color: #aaa;
    font-size: 12px;
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid #e0e0e0;
}
"""


def _section(number: int, title: str, narrative_html: str, chart_html: str) -> str:
    return f"""
<div class="section">
  <h2>{number}. {title}</h2>
  <div class="narrative">{narrative_html}</div>
  <div class="chart-wrap">{chart_html}</div>
</div>
"""


def _plotly_script_tag() -> str:
    """
    Return a <script> tag that makes Plotly available on the page.

    Strategy (in order):
      1. Inline the JS bundle shipped with the locally-installed plotly package
         → fully self-contained, works offline, always version-matched.
      2. CDN fallback if the local bundle is missing (requires internet).
    """
    try:
        import plotly as _plotly
        bundle = Path(_plotly.__file__).parent / "package_data" / "plotly.min.js"
        if bundle.exists():
            js = bundle.read_text(encoding="utf-8")
            return f"<script>{js}</script>"
    except Exception:
        pass
    # Fallback: CDN (requires internet connection)
    return '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'


def _fig_html(fig: go.Figure) -> str:
    """Serialise a figure to an HTML fragment (Plotly JS loaded separately)."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build(
    work_dir: str | Path,
    meta: dict,
    out_path: str | Path,
) -> Path:
    """
    Build and write a self-contained HTML report for one analysis session.

    Args:
        work_dir:  Session directory containing all pipeline CSV outputs.
        meta:      Dict with keys: format, user_messages, threads, date_range,
                   months_scored, session_name.
        out_path:  Destination .html file path.

    Returns:
        Resolved Path to the written file.
    """
    work_dir = Path(work_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    def _csv(name: str) -> pd.DataFrame:
        p = work_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def _csv_sub(sub: str, name: str) -> pd.DataFrame:
        p = work_dir / sub / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    traj_df    = _csv_sub("profile", "trajectory_monthly.csv")
    entropy_df = _csv("monthly_topic_entropy_tfidf.csv")
    rolling_df = _csv("rolling_entropy_250.csv")
    align_df   = _csv("dyadic_alignment_monthly.csv")
    domain_df  = _csv("macro_domain_summary.csv")
    shares_df  = _csv("macro_monthly_domain_shares.csv")
    states_df  = _csv("monthly_states.csv")
    coup_df    = _csv("coupling_by_domain.csv")
    shift_df   = _csv("shift_initiation_summary.csv")

    # ── Build chart HTML ──────────────────────────────────────────────────────
    sections_html = ""

    if not traj_df.empty:
        sections_html += _section(
            1, "Cognitive Style Over Time",
            _narrative_cognitive(traj_df),
            _fig_html(charts.cognitive_style_timeline(traj_df)),
        )

    if not entropy_df.empty:
        sections_html += _section(
            2, "Topic Entropy Over Time",
            _narrative_entropy(entropy_df),
            _fig_html(charts.topic_entropy_timeline(entropy_df, rolling_df)),
        )

    if not align_df.empty:
        sections_html += _section(
            3, "Dyadic Alignment",
            _narrative_alignment(align_df),
            _fig_html(charts.dyadic_alignment_timeline(align_df)),
        )

    if not domain_df.empty:
        sections_html += _section(
            4, "Macro-Domain Distribution",
            _narrative_domains(domain_df),
            _fig_html(charts.macro_domain_bar(domain_df)),
        )

    if not shares_df.empty:
        sections_html += _section(
            5, "Your Topic Focus Over Time",
            _narrative_domain_focus(shares_df),
            _fig_html(charts.domain_shares_area(shares_df, domain_df)),
        )

    if not states_df.empty:
        sections_html += _section(
            6, "Behavioural States",
            _narrative_states(states_df),
            _fig_html(charts.state_timeline(states_df)),
        )

    if not coup_df.empty:
        sections_html += _section(
            7, "Lead–Lag Coupling",
            _narrative_coupling(coup_df, work_dir / "coupling_summary.csv"),
            _fig_html(charts.coupling_bar(coup_df, domain_df)),
        )

    if not shift_df.empty:
        sections_html += _section(
            8, "Domain Shift Initiation",
            _narrative_shifts(shift_df),
            _fig_html(charts.shift_initiation_donut(shift_df)),
        )

    # ── Meta cards ────────────────────────────────────────────────────────────
    def _card(label: str, value: str) -> str:
        return (
            f'<div class="meta-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f"</div>"
        )

    meta_html = (
        _card("Format",        str(meta.get("format", "—")).upper())
        + _card("Your messages", f"{meta.get('user_messages', 0):,}")
        + _card("Threads",       f"{meta.get('threads', 0):,}")
        + _card("Date range",    str(meta.get("date_range", "—")))
        + _card("Months scored", str(meta.get("months_scored", "—")))
    )

    session_name = meta.get("session_name", "")

    # ── Assemble full HTML ────────────────────────────────────────────────────
    plotly_tag = _plotly_script_tag()   # inline bundle or CDN fallback

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DOL Analyser Report — {session_name}</title>
  {plotly_tag}
  <style>{_CSS}</style>
</head>
<body>
<div class="page">

  <header>
    <h1>💬 DOL Analyser Report</h1>
    <p>Distribution of Cognitive Load &mdash; {session_name}</p>
  </header>

  <div class="meta-grid">
    {meta_html}
  </div>

  {sections_html}

  <footer>
    Generated by DOL Analyser &middot; All analysis ran locally on your device &middot;
    No data was uploaded to any server.
  </footer>

</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    return out_path.resolve()
