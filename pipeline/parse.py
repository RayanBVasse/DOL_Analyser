"""
Step 1 — Parse conversations.json → canonical internal schema.

Output schema (written to SQLite via Step 2):

messages table
--------------
node_id       TEXT PRIMARY KEY
thread_id     TEXT
role          TEXT   ('user' | 'assistant')
timestamp     REAL   (Unix seconds)
year_month    TEXT   (e.g. '2024-03')
char_count    INT
text          TEXT

threads table
-------------
thread_id     TEXT PRIMARY KEY
title         TEXT
created_ts    REAL
updated_ts    REAL
msg_count     INT
user_chars    INT
asst_chars    INT
"""

from __future__ import annotations

# Full implementation will be completed in Week 2 when sample JSON files
# are available for both ChatGPT and Claude formats.
#
# Placeholders below define the public API used by app.py.

from pathlib import Path
from typing import Callable


def run(
    json_path: str | Path,
    db_path: str | Path,
    fmt: str,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """
    Parse *json_path* (format *fmt*: 'chatgpt' | 'claude') and write the
    canonical schema to the SQLite database at *db_path*.

    Returns a summary dict with row counts.
    """
    raise NotImplementedError("parse.run() will be implemented in Week 2.")
