"""
PyInstaller entry point for the DOL Analyser standalone app.

When frozen by PyInstaller this module:
  1. Resolves all resource paths relative to sys._MEIPASS (the unpacked bundle)
  2. Opens the default browser after a short delay
  3. Starts the Streamlit server on localhost:8501

When run from source (python launcher.py) it behaves identically to run.sh/run.bat.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _resource(rel: str) -> str:
    """Return absolute path whether running frozen or from source."""
    if hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / rel)
    return str(Path(__file__).parent / rel)


def _open_browser(port: int = 8501, delay: float = 4.0) -> None:
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")


def main() -> None:
    port = 8501

    # Streamlit environment flags
    os.environ.setdefault("STREAMLIT_SERVER_PORT", str(port))
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")  # no watchdog in frozen

    # Open browser in background after Streamlit is ready
    threading.Thread(target=_open_browser, kwargs={"port": port}, daemon=True).start()

    # Launch Streamlit
    from streamlit.web import cli as stcli  # noqa: PLC0415

    sys.argv = [
        "streamlit",
        "run",
        _resource("app.py"),
        f"--server.port={port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
    ]
    stcli.main()


if __name__ == "__main__":
    main()
