"""
Chat Analyser — Streamlit entry point.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from pipeline import precheck

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Chat Analyser",
    page_icon="💬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "precheck_result" not in st.session_state:
    st.session_state.precheck_result = None
if "json_tmp_path" not in st.session_state:
    st.session_state.json_tmp_path = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("💬 Chat Analyser")
st.caption("Understand how your conversations with AI have evolved over time.")

st.info(
    "**Your data never leaves your device.** "
    "The analysis runs entirely on your computer. "
    "Nothing is uploaded to any server.",
    icon="🔒",
)

st.divider()

# ---------------------------------------------------------------------------
# Step 1 — File upload
# ---------------------------------------------------------------------------

st.subheader("1. Upload your conversations export")

with st.expander("How to export your conversations", expanded=False):
    st.markdown(
        """
**ChatGPT**
1. Go to [chatgpt.com](https://chatgpt.com) → Settings → Data controls
2. Click **Export data** → confirm via email
3. Download the ZIP and extract `conversations.json`

**Claude (Anthropic)**
1. Go to [claude.ai](https://claude.ai) → Settings → Privacy
2. Click **Export data** → confirm via email
3. Download the ZIP and extract `conversations.json`
        """
    )

uploaded = st.file_uploader(
    "Drop your conversations.json here",
    type=["json"],
    help="Select the conversations.json file from your ChatGPT or Claude export.",
)

# ---------------------------------------------------------------------------
# Step 2 — Pre-check
# ---------------------------------------------------------------------------

if uploaded is not None:
    st.subheader("2. Pre-check")

    run_precheck = st.button("Run pre-check", type="primary")

    if run_precheck or st.session_state.precheck_result is not None:

        if run_precheck:
            # Save upload to a temp file so pipeline can read it by path
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".json", mode="wb"
            ) as tmp:
                tmp.write(uploaded.read())
                st.session_state.json_tmp_path = tmp.name

            progress_bar = st.progress(0, text="Starting pre-check…")

            def _cb(frac: float, msg: str):
                progress_bar.progress(frac, text=msg)

            with st.spinner("Inspecting your file…"):
                result = precheck.run(st.session_state.json_tmp_path, progress_cb=_cb)

            st.session_state.precheck_result = result
            progress_bar.empty()

        result = st.session_state.precheck_result

        # Display result
        col1, col2, col3 = st.columns(3)
        col1.metric("Format", result.format.upper() if result.format != "unknown" else "Unknown")
        col2.metric("User messages", f"{result.user_messages:,}")
        col3.metric("Months covered", result.months_covered)

        if result.earliest_date and result.latest_date:
            st.caption(
                f"Date range: **{result.earliest_date:%B %Y}** → **{result.latest_date:%B %Y}**"
            )

        if result.warnings:
            for w in result.warnings:
                st.warning(w, icon="⚠️")

        if result.ready:
            st.success(
                "File looks good! Ready to run the full analysis.",
                icon="✅",
            )
        else:
            st.error(
                "This file does not meet the minimum requirements. "
                "See warnings above.",
                icon="❌",
            )

# ---------------------------------------------------------------------------
# Step 3 — Run analysis (placeholder, Weeks 5–6)
# ---------------------------------------------------------------------------

if st.session_state.precheck_result and st.session_state.precheck_result.ready:
    st.divider()
    st.subheader("3. Run analysis")
    st.info(
        "Full pipeline coming in Week 5. "
        "For now, the pre-check above confirms your file is compatible.",
        icon="🚧",
    )

    st.button("Run full analysis", disabled=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Chat Analyser · open source · MIT licence · "
    "[github.com/RayanBVasse/Chat_analyser](https://github.com/RayanBVasse/Chat_analyser)"
)
