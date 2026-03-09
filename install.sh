#!/usr/bin/env bash
set -euo pipefail

echo ""
echo " DOL Analyser — one-time setup"
echo " ================================"
echo ""

# Require Python 3.10+
if ! command -v python3 &>/dev/null; then
    echo " ERROR: python3 not found."
    echo " Please install Python 3.10 or later from https://python.org"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(sys.version_info[:2] >= (3,10))")
if [ "$PY_VER" != "True" ]; then
    echo " ERROR: Python 3.10 or later is required."
    echo " Detected: $(python3 --version)"
    exit 1
fi

# Create virtual environment (only on first run)
if [ ! -d ".venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install
echo " Installing dependencies (this may take a minute)..."
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo " Setup complete!"
echo " Run  ./run.sh  to launch the app."
echo ""
