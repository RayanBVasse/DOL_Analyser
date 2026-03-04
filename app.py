"""
DOL Analyser — Streamlit entry point.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from pipeline import precheck, parse, profile

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
    "json_tmp_path": None,
    "work_dir":      None,   # temp dir holding the SQLite db + CSVs
    "precheck_result": None,
    "parse_result":    None,
    "profile_result":  None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def _reset_downstream(from_step: str):
    """Clear all state at and after a given step so reruns start fresh."""
    order = ["precheck_result", "parse_result", "profile_result"]
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
    """)

uploaded = st.file_uploader(
    "Drop your conversations.json here",
    type=["json"],
    help="Select the conversations.json file from your ChatGPT or Claude export.",
)

# If user uploads a new file, wipe all downstream results
if uploaded is not None and st.session_state.json_tmp_path is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="wb") as tmp:
        tmp.write(uploaded.read())
        st.session_state.json_tmp_path = tmp.name
    st.session_state.work_dir = tempfile.mkdtemp(prefix="dol_")
    _reset_downstream("precheck_result")

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
        col1.metric("Threads",           f"{p['threads']:,}")
        col2.metric("Total messages",    f"{p['messages']:,}")
        col3.metric("Your messages",     f"{p['user_messages']:,}")
        col4.metric("AI messages",       f"{p['asst_messages']:,}")
        st.success("Conversations parsed and stored.", icon="✅")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Cognitive style profile
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.parse_result:
    st.divider()
    st.subheader("4. Cognitive style profile")

    if st.session_state.profile_result is None:
        if st.button("Run profile", type="primary", key="btn_profile"):
            db_path  = Path(st.session_state.work_dir) / "conversations.db"
            out_dir  = Path(st.session_state.work_dir) / "profile"
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

        # Trajectory table
        traj_path = Path(st.session_state.work_dir) / "profile" / "trajectory_monthly.csv"
        if traj_path.exists():
            traj = pd.read_csv(traj_path)
            st.markdown("**Monthly marker rates (per 1,000 user messages)**")
            display_cols = ["year_month", "user_messages",
                            "structural_thinking_per1k",
                            "epistemic_uncertainty_per1k"]
            display_cols = [c for c in display_cols if c in traj.columns]
            st.dataframe(
                traj[display_cols].rename(columns={
                    "year_month":                    "Month",
                    "user_messages":                 "Messages",
                    "structural_thinking_per1k":     "Structural thinking /1k",
                    "epistemic_uncertainty_per1k":   "Epistemic uncertainty /1k",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # Spearman results
        spearman_path = Path(st.session_state.work_dir) / "profile" / "spearman_results.csv"
        if spearman_path.exists():
            stat = pd.read_csv(spearman_path)
            st.markdown("**Spearman correlations with time**")
            st.dataframe(
                stat[["marker", "spearman_rho", "p_value", "significant"]].rename(columns={
                    "marker":       "Marker",
                    "spearman_rho": "ρ",
                    "p_value":      "p",
                    "significant":  "p < 0.05",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.info(
            "Visualisation charts coming in Week 6. "
            "HTML report export coming in Week 7.",
            icon="🚧",
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "DOL Analyser · open source · MIT licence · "
    "[github.com/RayanBVasse/DOL_Analyser](https://github.com/RayanBVasse/DOL_Analyser)"
)
