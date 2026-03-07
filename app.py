"""
DOL Analyser — Streamlit entry point.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import streamlit as st

from pipeline import (
    precheck,
    parse,
    profile,
    topics,
    alignment,
    domains,
    coupling,
    dynamics,
)

# Persistent session output root (committed as empty dir; contents git-ignored)
TEMP_OUT = Path(__file__).parent / "temp_out"
TEMP_OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DOL Analyser",
    page_icon="💬",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

defaults = {
    "json_tmp_path":    None,
    "work_dir":         None,   # temp dir holding the SQLite db + CSVs
    "precheck_result":  None,
    "parse_result":     None,
    "profile_result":   None,
    "topics_result":    None,
    "alignment_result": None,
    "domains_result":   None,
    "coupling_result":  None,
    "dynamics_result":  None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def _reset_downstream(from_step: str):
    """Clear all state at and after a given step so reruns start fresh."""
    order = [
        "precheck_result",
        "parse_result",
        "profile_result",
        "topics_result",
        "alignment_result",
        "domains_result",
        "coupling_result",
        "dynamics_result",
    ]
    clear = False
    for key in order:
        if key == from_step:
            clear = True
        if clear:
            st.session_state[key] = None


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("💬 DOL Analyser")
st.caption(
    "Distribution of Cognitive Load — "
    "understand how your conversations with AI have evolved over time."
)
st.info(
    "**Your data never leaves your device.** "
    "The analysis runs entirely on your computer. "
    "Nothing is uploaded to any server.",
    icon="🔒",
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("1. Upload your conversations export")

with st.expander("How to export your conversations", expanded=False):
    st.markdown("""
**ChatGPT**
1. Go to [chatgpt.com](https://chatgpt.com) → Settings → Data controls
2. Click **Export data** → confirm via email
3. Download the ZIP and extract `conversations.json`

**Claude (Anthropic)**
1. Go to [claude.ai](https://claude.ai) → Settings → Privacy
2. Click **Export data** → confirm via email
3. Download the ZIP and extract `conversations.json`

**Tip — short export windows (30 / 90 days)**
If your platform limits exports to a recent time window, request multiple
exports covering different periods and upload all the `conversations.json`
files at once. The app merges them automatically and removes duplicates.
    """)

uploaded_files = st.file_uploader(
    "Drop your conversations.json file(s) here",
    type=["json"],
    accept_multiple_files=True,
    help=(
        "Upload one or more conversations.json files from ChatGPT or Claude. "
        "Multiple files from the same platform are merged automatically — "
        "useful when your export only covers 30 or 90 days."
    ),
)

# Process uploads when files arrive and no session exists yet
if uploaded_files and st.session_state.json_tmp_path is None:
    try:
        raw_list = [f.read() for f in uploaded_files]
        merged, _fmt, n_dupes = parse.merge_exports(raw_list)

        # Session name: single filename or "merged_N_files"
        if len(uploaded_files) == 1:
            stem = Path(uploaded_files[0].name).stem[:30]
        else:
            stem = f"merged_{len(uploaded_files)}_files"

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = TEMP_OUT / f"{stem}_{stamp}"
        work_dir.mkdir(parents=True, exist_ok=True)

        json_path = work_dir / "upload.json"
        json_path.write_text(json.dumps(merged), encoding="utf-8")

        st.session_state.json_tmp_path = str(json_path)
        st.session_state.work_dir      = str(work_dir)

        # Show merge summary if multiple files were combined
        if len(uploaded_files) > 1:
            st.success(
                f"Merged **{len(uploaded_files)} files** → "
                f"**{len(merged):,} conversations** "
                + (f"({n_dupes:,} duplicates removed)" if n_dupes else ""),
                icon="🔀",
            )

        _reset_downstream("precheck_result")

    except ValueError as exc:
        st.error(str(exc), icon="❌")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Pre-check
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.json_tmp_path:
    st.divider()
    st.subheader("2. Pre-check")

    if st.session_state.precheck_result is None:
        if st.button("Run pre-check", type="primary", key="btn_precheck"):
            bar = st.progress(0.0, text="Starting pre-check…")
            result = precheck.run(
                st.session_state.json_tmp_path,
                progress_cb=lambda f, m: bar.progress(f, text=m),
            )
            bar.empty()
            st.session_state.precheck_result = result
            st.rerun()

    if st.session_state.precheck_result:
        r = st.session_state.precheck_result

        col1, col2, col3 = st.columns(3)
        col1.metric("Format", r.format.upper() if r.format != "unknown" else "Unknown")
        col2.metric("User messages", f"{r.user_messages:,}")
        col3.metric("Months covered", r.months_covered)

        if r.earliest_date and r.latest_date:
            st.caption(
                f"Date range: **{r.earliest_date:%B %Y}** → **{r.latest_date:%B %Y}**"
            )

        for w in r.warnings:
            st.warning(w, icon="⚠️")

        if r.ready:
            st.success("File looks good — ready to parse.", icon="✅")
        else:
            st.error(
                "File does not meet minimum requirements. See warnings above.",
                icon="❌",
            )

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Parse
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.precheck_result and st.session_state.precheck_result.ready:
    st.divider()
    st.subheader("3. Parse conversations")

    if st.session_state.parse_result is None:
        if st.button("Parse", type="primary", key="btn_parse"):
            db_path = Path(st.session_state.work_dir) / "conversations.db"
            bar = st.progress(0.0, text="Starting parse…")
            try:
                result = parse.run(
                    json_path=st.session_state.json_tmp_path,
                    db_path=db_path,
                    fmt=st.session_state.precheck_result.format,
                    progress_cb=lambda f, m: bar.progress(f, text=m),
                )
                bar.empty()
                st.session_state.parse_result = result
                st.rerun()
            except Exception as exc:
                bar.empty()
                st.error(f"Parse failed: {exc}", icon="❌")

    if st.session_state.parse_result:
        p = st.session_state.parse_result
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Threads",        f"{p['threads']:,}")
        col2.metric("Total messages", f"{p['messages']:,}")
        col3.metric("Your messages",  f"{p['user_messages']:,}")
        col4.metric("AI messages",    f"{p['asst_messages']:,}")
        st.success("Conversations parsed and stored.", icon="✅")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Cognitive style profile
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.parse_result:
    st.divider()
    st.subheader("4. Cognitive style profile")

    if st.session_state.profile_result is None:
        if st.button("Run profile", type="primary", key="btn_profile"):
            db_path = Path(st.session_state.work_dir) / "conversations.db"
            out_dir = Path(st.session_state.work_dir) / "profile"
            bar = st.progress(0.0, text="Starting profile…")
            try:
                result = profile.run(
                    db_path=db_path,
                    out_dir=out_dir,
                    progress_cb=lambda f, m: bar.progress(f, text=m),
                )
                bar.empty()
                st.session_state.profile_result = result
                st.rerun()
            except Exception as exc:
                bar.empty()
                st.error(f"Profile failed: {exc}", icon="❌")

    if st.session_state.profile_result:
        import pandas as pd

        pr = st.session_state.profile_result
        st.success(
            f"Scored {pr['months_scored']} month(s) across "
            f"{len(pr['markers'])} markers.",
            icon="✅",
        )

        traj_path = Path(st.session_state.work_dir) / "profile" / "trajectory_monthly.csv"
        if traj_path.exists():
            traj = pd.read_csv(traj_path)
            st.markdown("**Monthly marker rates (per 1,000 user messages)**")
            display_cols = [
                "year_month", "user_messages",
                "structural_thinking_per1k",
                "epistemic_uncertainty_per1k",
            ]
            display_cols = [c for c in display_cols if c in traj.columns]
            st.dataframe(
                traj[display_cols].rename(columns={
                    "year_month":                  "Month",
                    "user_messages":               "Messages",
                    "structural_thinking_per1k":   "Structural thinking /1k",
                    "epistemic_uncertainty_per1k": "Epistemic uncertainty /1k",
                }),
                width="stretch",
                hide_index=True,
            )

        spearman_path = (
            Path(st.session_state.work_dir) / "profile" / "spearman_results.csv"
        )
        if spearman_path.exists():
            stat = pd.read_csv(spearman_path)
            st.markdown("**Spearman correlations with time**")
            st.dataframe(
                stat[["marker", "spearman_rho", "p_value", "significant"]].rename(
                    columns={
                        "marker":       "Marker",
                        "spearman_rho": "ρ",
                        "p_value":      "p",
                        "significant":  "p < 0.05",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Topic modelling  (DOL Steps 5–6: TF-IDF + SVD + KMeans)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.profile_result:
    st.divider()
    st.subheader("5. Topic modelling")

    if st.session_state.topics_result is None:
        st.info(
            "Fits TF-IDF + SVD + KMeans on all messages to discover up to 60 topics. "
            "This step may take 1–3 minutes on large datasets.",
            icon="ℹ️",
        )
        if st.button("Run topic modelling", type="primary", key="btn_topics"):
            db_path = Path(st.session_state.work_dir) / "conversations.db"
            out_dir = Path(st.session_state.work_dir)
            bar = st.progress(0.0, text="Starting topic modelling…")
            try:
                result = topics.run(
                    db_path=db_path,
                    out_dir=out_dir,
                    progress_cb=lambda f, m: bar.progress(f, text=m),
                )
                bar.empty()
                st.session_state.topics_result = result
                st.rerun()
            except Exception as exc:
                bar.empty()
                st.error(f"Topic modelling failed: {exc}", icon="❌")

    if st.session_state.topics_result:
        import pandas as pd

        tr = st.session_state.topics_result
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Topics found",   tr.get("n_clusters", "—"))
        col2.metric("Vocabulary",     f"{tr.get('vocab_size', 0):,}")
        col3.metric("Months covered", tr.get("months_with_data", "—"))
        col4.metric(
            "Entropy range",
            f"{tr.get('entropy_min', 0):.2f}–{tr.get('entropy_max', 0):.2f}",
        )

        cluster_path = Path(st.session_state.work_dir) / "cluster_summary_tfidf.csv"
        if cluster_path.exists():
            cls_df = pd.read_csv(cluster_path).head(10)
            st.markdown("**Top 10 topics by message count**")
            st.dataframe(
                cls_df[["cluster_id", "size", "auto_label"]].rename(columns={
                    "cluster_id": "Topic #",
                    "size":       "Messages",
                    "auto_label": "Key terms",
                }),
                width="stretch",
                hide_index=True,
            )

        st.success("Topic model fitted — cluster assignments saved to database.", icon="✅")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Dyadic alignment  (DOL Step 7: monthly JS divergence)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.topics_result:
    st.divider()
    st.subheader("6. Dyadic alignment")

    if st.session_state.alignment_result is None:
        st.info(
            "Measures how similar your topics are to the AI's topics each month "
            "(Jensen-Shannon divergence — lower = more in sync).",
            icon="ℹ️",
        )
        if st.button("Run alignment", type="primary", key="btn_alignment"):
            db_path = Path(st.session_state.work_dir) / "conversations.db"
            out_dir = Path(st.session_state.work_dir)
            bar = st.progress(0.0, text="Computing dyadic alignment…")
            try:
                result = alignment.run(
                    db_path=db_path,
                    out_dir=out_dir,
                    progress_cb=lambda f, m: bar.progress(f, text=m),
                )
                bar.empty()
                st.session_state.alignment_result = result
                st.rerun()
            except Exception as exc:
                bar.empty()
                st.error(f"Alignment failed: {exc}", icon="❌")

    if st.session_state.alignment_result:
        import pandas as pd

        ar = st.session_state.alignment_result
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Months computed", ar.get("months_computed", "—"))
        col2.metric("Mean JS",         f"{ar.get('mean_js', 0):.4f}")
        col3.metric("Min JS",          f"{ar.get('min_js', 0):.4f}")
        col4.metric("Max JS",          f"{ar.get('max_js', 0):.4f}")

        align_path = Path(st.session_state.work_dir) / "dyadic_alignment_monthly.csv"
        if align_path.exists():
            align_df = pd.read_csv(align_path)
            st.markdown("**Monthly JS divergence (you vs AI topic distributions)**")
            st.dataframe(
                align_df.rename(columns={
                    "year_month":    "Month",
                    "user_msgs":     "Your msgs",
                    "asst_msgs":     "AI msgs",
                    "js_divergence": "JS divergence",
                }),
                width="stretch",
                hide_index=True,
            )

        st.success("Dyadic alignment computed.", icon="✅")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Macro-domain mapping  (DOL Steps 8.5–8.6)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.alignment_result:
    st.divider()
    st.subheader("7. Macro-domain mapping")

    if st.session_state.domains_result is None:
        st.info(
            "Groups the fine topics into 8 broad macro-domains "
            "(e.g. code, science, writing, creative…). "
            "This step re-runs TF-IDF for labelling and may take 1–3 minutes.",
            icon="ℹ️",
        )
        if st.button("Run domain mapping", type="primary", key="btn_domains"):
            db_path = Path(st.session_state.work_dir) / "conversations.db"
            out_dir = Path(st.session_state.work_dir)
            bar = st.progress(0.0, text="Building macro-domain map…")
            try:
                result = domains.run(
                    db_path=db_path,
                    out_dir=out_dir,
                    progress_cb=lambda f, m: bar.progress(f, text=m),
                )
                bar.empty()
                st.session_state.domains_result = result
                st.rerun()
            except Exception as exc:
                bar.empty()
                st.error(f"Domain mapping failed: {exc}", icon="❌")

    if st.session_state.domains_result:
        import pandas as pd

        dr = st.session_state.domains_result
        st.success(
            f"Identified {dr.get('n_macro', 8)} macro-domains "
            f"across {dr.get('months_computed', '—')} months.",
            icon="✅",
        )

        domain_path = Path(st.session_state.work_dir) / "macro_domain_summary.csv"
        if domain_path.exists():
            dom_df = pd.read_csv(domain_path)
            st.markdown("**Macro-domains (ranked by message count)**")
            st.dataframe(
                dom_df[["macro_domain", "size", "auto_label"]].rename(columns={
                    "macro_domain": "Domain #",
                    "size":         "Messages",
                    "auto_label":   "Top terms",
                }),
                width="stretch",
                hide_index=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Lead–lag coupling  (DOL Steps 9–9.1)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.domains_result:
    st.divider()
    st.subheader("8. Lead–lag coupling")

    if st.session_state.coupling_result is None:
        st.info(
            "Tests whether your topic changes predict the AI's next-month topics "
            "(user leads) or vice versa — lag-1 cosine similarity with "
            "permutation p-values (N=2,000).",
            icon="ℹ️",
        )
        if st.button("Run coupling analysis", type="primary", key="btn_coupling"):
            out_dir = Path(st.session_state.work_dir)
            bar = st.progress(0.0, text="Computing lead-lag coupling…")
            try:
                result = coupling.run(
                    out_dir=out_dir,
                    progress_cb=lambda f, m: bar.progress(f, text=m),
                )
                bar.empty()
                st.session_state.coupling_result = result
                st.rerun()
            except Exception as exc:
                bar.empty()
                st.error(f"Coupling analysis failed: {exc}", icon="❌")

    if st.session_state.coupling_result:
        import pandas as pd

        cr = st.session_state.coupling_result
        direction = "You lead the AI 🡒" if cr.get("user_leads") else "AI leads you 🡐"
        lead_diff = cr.get("lead_diff", 0)
        p_diff    = cr.get("p_diff", 1)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Direction",      direction)
        col2.metric("Forward cosine", f"{cr.get('forward_mean', 0):.4f}")
        col3.metric("Reverse cosine", f"{cr.get('reverse_mean', 0):.4f}")
        col4.metric("Lead diff",      f"{lead_diff:.4f}  (p={p_diff:.3f})")

        st.success(
            f"Lead–lag coupling computed over {cr.get('n_months', '—')} months.",
            icon="✅",
        )

        domain_coup_path = Path(st.session_state.work_dir) / "coupling_by_domain.csv"
        if domain_coup_path.exists():
            dc_df = pd.read_csv(domain_coup_path)
            st.markdown("**Per-domain coupling (Pearson r, lag-1)**")
            st.dataframe(
                dc_df.round(4).rename(columns={
                    "macro_domain":      "Domain #",
                    "r_user_leads_asst": "r (you→AI)",
                    "p_user_leads_asst": "p (you→AI)",
                    "r_asst_leads_user": "r (AI→you)",
                    "p_asst_leads_user": "p (AI→you)",
                }),
                width="stretch",
                hide_index=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Behavioural dynamics  (DOL Step 10a2, 10a3, 10b)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.domains_result:
    st.divider()
    st.subheader("9. Behavioural dynamics")

    if st.session_state.dynamics_result is None:
        st.info(
            "Assigns each month to an Exploration / Consolidation / Transitional "
            "state, computes rolling topic entropy (window = 250 messages), and "
            "tests who initiates domain shifts within conversations.",
            icon="ℹ️",
        )
        if st.button("Run dynamics analysis", type="primary", key="btn_dynamics"):
            db_path = Path(st.session_state.work_dir) / "conversations.db"
            out_dir = Path(st.session_state.work_dir)
            bar = st.progress(0.0, text="Running dynamics analysis…")
            try:
                result = dynamics.run(
                    db_path=db_path,
                    out_dir=out_dir,
                    progress_cb=lambda f, m: bar.progress(f, text=m),
                )
                bar.empty()
                st.session_state.dynamics_result = result
                st.rerun()
            except Exception as exc:
                bar.empty()
                st.error(f"Dynamics analysis failed: {exc}", icon="❌")

    if st.session_state.dynamics_result:
        import pandas as pd

        dyn = st.session_state.dynamics_result

        # State counts
        state_counts = dyn.get("state_counts", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Exploration months",   state_counts.get("Exploration",    "—"))
        col2.metric("Consolidation months", state_counts.get("Consolidation",  "—"))
        col3.metric("Transitional months",  state_counts.get("Transitional",   "—"))
        col4.metric("Rolling windows",      dyn.get("rolling_windows",         "—"))

        # Shift initiation
        if "user_initiated_prop" in dyn:
            u_pct  = round(dyn["user_initiated_prop"] * 100, 1)
            a_pct  = round((1.0 - dyn["user_initiated_prop"]) * 100, 1)
            p_perm = dyn.get("p_perm", "—")
            total  = dyn.get("total_shifts", "—")
            st.markdown(
                f"**Domain shift initiation** ({total} total shifts) — "
                f"You: **{u_pct}%** | AI: **{a_pct}%** "
                f"(permutation p vs 50/50 = {p_perm})"
            )

        # Monthly states table
        states_path = Path(st.session_state.work_dir) / "monthly_states.csv"
        if states_path.exists():
            states_df = pd.read_csv(states_path)
            st.markdown("**Monthly behavioural states (quantile thresholds)**")
            st.dataframe(
                states_df.rename(columns={
                    "year_month":          "Month",
                    "macro_entropy_user":  "Entropy (user)",
                    "macro_js_divergence": "JS divergence",
                    "state":               "State",
                }),
                width="stretch",
                hide_index=True,
            )

        st.success("Behavioural dynamics analysis complete.", icon="✅")

# ─────────────────────────────────────────────────────────────────────────────
# Report preview — charts (shown once dynamics is complete)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.dynamics_result:
    st.divider()
    st.subheader("📊 Report Preview")
    st.caption(
        f"Session: `{Path(st.session_state.work_dir).name}`  ·  "
        f"All intermediate files saved to `temp_out/{Path(st.session_state.work_dir).name}/`"
    )

    import pandas as pd
    from reports import charts

    work = Path(st.session_state.work_dir)

    # Helper: safe CSV load
    def _csv(name: str) -> pd.DataFrame:
        p = work / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def _csv_sub(sub: str, name: str) -> pd.DataFrame:
        p = work / sub / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    tabs = st.tabs([
        "Cognitive style",
        "Topic entropy",
        "Alignment",
        "Domains",
        "Domain focus",
        "States",
        "Coupling",
        "Shifts",
    ])

    with tabs[0]:
        traj = _csv_sub("profile", "trajectory_monthly.csv")
        if not traj.empty:
            st.plotly_chart(charts.cognitive_style_timeline(traj),
                            width="stretch")

    with tabs[1]:
        entropy_df  = _csv("monthly_topic_entropy_tfidf.csv")
        rolling_df  = _csv("rolling_entropy_250.csv")
        if not entropy_df.empty:
            st.plotly_chart(
                charts.topic_entropy_timeline(entropy_df, rolling_df),
                width="stretch",
            )

    with tabs[2]:
        align_df = _csv("dyadic_alignment_monthly.csv")
        if not align_df.empty:
            st.plotly_chart(charts.dyadic_alignment_timeline(align_df),
                            width="stretch")

    with tabs[3]:
        domain_summary = _csv("macro_domain_summary.csv")
        if not domain_summary.empty:
            st.plotly_chart(charts.macro_domain_bar(domain_summary),
                            width="stretch")

    with tabs[4]:
        shares_df = _csv("macro_monthly_domain_shares.csv")
        if not shares_df.empty:
            st.plotly_chart(
                charts.domain_shares_area(shares_df, domain_summary),
                width="stretch",
            )

    with tabs[5]:
        states_df = _csv("monthly_states.csv")
        if not states_df.empty:
            st.plotly_chart(charts.state_timeline(states_df),
                            width="stretch")

    with tabs[6]:
        coup_df = _csv("coupling_by_domain.csv")
        if not coup_df.empty:
            st.plotly_chart(
                charts.coupling_bar(coup_df, domain_summary),
                width="stretch",
            )

    with tabs[7]:
        shift_df = _csv("shift_initiation_summary.csv")
        if not shift_df.empty:
            st.plotly_chart(charts.shift_initiation_donut(shift_df),
                            width="stretch")

    # ── Generate HTML report ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📄 Export Report")

    default_report = work / f"report_{Path(st.session_state.work_dir).name}.html"
    report_path    = st.text_input(
        "Save report to",
        value=str(default_report),
        help="Full path for the HTML report file.",
    )

    if st.button("Generate HTML report", type="primary", key="btn_report"):
        from reports import html_export

        pr = st.session_state.profile_result or {}
        parse_r = st.session_state.parse_result or {}

        meta = {
            "format":        st.session_state.precheck_result.format
                             if st.session_state.precheck_result else "—",
            "user_messages": parse_r.get("user_messages", 0),
            "threads":       parse_r.get("threads", 0),
            "date_range":    (
                f"{st.session_state.precheck_result.earliest_date:%b %Y}"
                f" – "
                f"{st.session_state.precheck_result.latest_date:%b %Y}"
                if st.session_state.precheck_result
                   and st.session_state.precheck_result.earliest_date
                else "—"
            ),
            "months_scored": pr.get("months_scored", "—"),
            "session_name":  Path(st.session_state.work_dir).name,
        }

        try:
            written = html_export.build(
                work_dir=st.session_state.work_dir,
                meta=meta,
                out_path=report_path,
            )
            st.success(f"Report saved to **{written}**", icon="✅")
        except Exception as exc:
            st.error(f"Report generation failed: {exc}", icon="❌")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "DOL Analyser · open source · MIT licence · "
    "[github.com/RayanBVasse/DOL_Analyser](https://github.com/RayanBVasse/DOL_Analyser)"
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: the ``dol-analyser`` command launches the Streamlit UI."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__],
        check=True,
    )


if __name__ == "__main__":
    # Streamlit executes this file with __name__ == "__main__" via exec(),
    # so we must check whether we're already inside a running Streamlit session
    # before launching a second server.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
        _inside_streamlit = _get_ctx() is not None
    except Exception:
        _inside_streamlit = False

    if not _inside_streamlit:
        main()
