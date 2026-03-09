# DOL Analyser

**Distribution of Cognitive Load** — understand how your conversations with AI have evolved over time.

Upload your `conversations.json` export from ChatGPT or Claude and get a
self-contained HTML report covering topic evolution, cognitive style,
dyadic alignment, and who drives the conversation.

**Your data never leaves your device.** Everything runs locally.

---

## For readers of the paper

**Requirement:** Python 3.10 or later — download from [python.org](https://python.org)
(tick *"Add Python to PATH"* on Windows during installation).

1. **Download** this repository
   - Click the green **Code** button → **Download ZIP**, then extract it — or run
     `git clone https://github.com/RayanBVasse/DOL_Analyser.git`
2. **Install** (one-time, takes ~1 minute):
   - **Windows** — double-click `install.bat`
   - **Mac / Linux** — open a Terminal in the folder and run `./install.sh`
3. **Launch**:
   - **Windows** — double-click `run.bat`
   - **Mac / Linux** — run `./run.sh`

The app opens automatically in your browser at `http://localhost:8501`.

---

## Quickstart (Python — developers)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the browser tab that appears and upload your `conversations.json`.

## Minimum requirements

| Requirement | Value |
|---|---|
| User messages | ≥ 2,000 |
| Months of data | ≥ 3 |
| Supported formats | ChatGPT export, Claude export |

## How to export your conversations

**ChatGPT** — Settings → Data controls → Export data → extract `conversations.json`

**Claude** — Settings → Privacy → Export data → extract `conversations.json`

---

## Development status

| Week | Milestone | Status |
|---|---|---|
| 1 | Project scaffold, pre-check, Streamlit skeleton | ✅ Done |
| 2 | Parser (ChatGPT + Claude), 33 unit tests | ✅ Done |
| 3–4 | Full pipeline integration (topics → dynamics) | ✅ Done |
| 5–6 | Plotly visualisations (8 chart builders) | ✅ Done |
| 7 | Self-contained HTML report with narratives | ✅ Done |
| 8 | Reader distribution scripts | ✅ Done |

---

## Licence

MIT © 2026 Rayan B Vasse

---

> **DOL** = Distribution of (Cognitive) Load — a framework for analysing
> how intellectual engagement is distributed across a human–AI dialogue over time.
