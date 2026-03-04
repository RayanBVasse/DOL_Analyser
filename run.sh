#!/usr/bin/env bash
set -euo pipefail
if [ ! -d ".venv" ]; then
    echo " Virtual environment not found. Please run ./install.sh first."
    exit 1
fi
source .venv/bin/activate
streamlit run app.py
