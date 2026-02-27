# Chat Analyser

Understand how your conversations with AI have evolved over time.

Upload your `conversations.json` export from ChatGPT or Claude and get a
self-contained HTML report covering topic evolution, cognitive style,
dyadic alignment, and who drives the conversation.

**Your data never leaves your device.** Everything runs locally.

---

## Quickstart (Python)

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
| 2 | Parser (ChatGPT + Claude), pipeline modules | 🔜 Next |
| 3–4 | Full pipeline integration | Planned |
| 5–6 | Plotly visualisations | Planned |
| 7 | HTML report generator | Planned |
| 8 | Packaging & GitHub Releases | Planned |

---

## Licence

MIT © 2026 Rayan B Vasse
