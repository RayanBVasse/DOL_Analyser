"""
Step 1 — Parse conversations.json → canonical internal schema (SQLite).

Supports:
    ChatGPT  — branching tree via 'mapping'; main path reconstructed with
                cycle-safe iterative postorder traversal
    Claude   — linear 'chat_messages' list; no tree traversal required

Output schema
─────────────
messages table
    node_id       TEXT  PRIMARY KEY   (UUID or synthetic key)
    thread_id     TEXT
    thread_title  TEXT
    role          TEXT  ('user' | 'assistant')
    timestamp     REAL  (Unix seconds, UTC)
    year_month    TEXT  ('YYYY-MM')
    char_count    INT
    text          TEXT

threads table
    thread_id     TEXT  PRIMARY KEY
    title         TEXT
    source        TEXT  ('chatgpt' | 'claude')
    created_ts    REAL  (Unix seconds)
    updated_ts    REAL  (Unix seconds)
    msg_count     INT
    user_msgs     INT
    asst_msgs     INT
    user_chars    INT
    asst_chars    INT
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    """Load JSON from *path*, trying UTF-8 then UTF-16 (handles Windows BOM)."""
    for enc in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            with path.open(encoding=enc) as fh:
                return json.load(fh)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} as UTF-8 or UTF-16 JSON.")


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_unix(value: Any) -> Optional[float]:
    """Convert a Unix float or ISO-8601 string to a Unix timestamp."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return None
    return None


def _year_month(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m")
    except (OSError, OverflowError):
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# ChatGPT — content extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_chatgpt_text(content: Any) -> str:
    """
    Extract plain text from a ChatGPT message content block.
    Handles: parts-as-strings, parts-as-dicts, direct string content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            chunks: List[str] = []
            for p in parts:
                if isinstance(p, str):
                    chunks.append(p)
                elif isinstance(p, dict):
                    chunks.append(p.get("text") or json.dumps(p, ensure_ascii=False))
                else:
                    chunks.append(str(p))
            return "\n".join(chunks).strip()
        for key in ("text", "value"):
            if isinstance(content.get(key), str):
                return content[key].strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# ChatGPT — main-path reconstruction (cycle-safe iterative postorder)
# Ported from DOL/scripts/step1_parse_conversations.py
# ─────────────────────────────────────────────────────────────────────────────

def _pick_main_path(message_nodes: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return the longest (deepest) path through the conversation tree.
    Ties are broken by earliest create_time. Cycle-safe.
    """
    children_map: Dict[str, List[str]] = {}
    roots: List[str] = []

    def _node_time(nid: str) -> float:
        msg = (message_nodes.get(nid, {}).get("message") or {})
        t = msg.get("create_time")
        try:
            return float(t) if t is not None else float("inf")
        except Exception:
            return float("inf")

    for node_id, node in message_nodes.items():
        parent = node.get("parent")
        if not parent or parent not in message_nodes:
            roots.append(node_id)
        else:
            children_map.setdefault(parent, []).append(node_id)

    if not roots:
        roots = list(message_nodes.keys())[:1]

    for p in children_map:
        children_map[p] = sorted(children_map[p], key=_node_time)

    best_len: Dict[str, int] = {}
    best_next: Dict[str, Optional[str]] = {}

    for root in roots:
        stack: List[Tuple[str, int]] = [(root, 0)]
        visiting: set = set()

        while stack:
            nid, stage = stack.pop()
            if stage == 0:
                if nid in visiting:
                    continue
                visiting.add(nid)
                stack.append((nid, 1))
                for kid in children_map.get(nid, []):
                    if kid not in best_len:
                        stack.append((kid, 0))
            else:
                visiting.discard(nid)
                kids = children_map.get(nid, [])
                if not kids:
                    best_len[nid] = 1
                    best_next[nid] = None
                else:
                    chosen = None
                    chosen_len = 0
                    for kid in kids:
                        klen = best_len.get(kid, 1)
                        if klen > chosen_len:
                            chosen_len = klen
                            chosen = kid
                        elif klen == chosen_len and chosen is not None:
                            if _node_time(kid) < _node_time(chosen):
                                chosen = kid
                    best_len[nid] = 1 + (chosen_len if chosen is not None else 0)
                    best_next[nid] = chosen

    roots_sorted = sorted(roots, key=lambda r: (-best_len.get(r, 1), _node_time(r)))
    best_root = roots_sorted[0]

    path: List[str] = []
    seen: set = set()
    cur: Optional[str] = best_root
    while cur and cur not in seen:
        seen.add(cur)
        path.append(cur)
        cur = best_next.get(cur)

    return path


# ─────────────────────────────────────────────────────────────────────────────
# ChatGPT parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_chatgpt(
    data: List[dict],
    progress_cb: Optional[Callable],
) -> Tuple[List[dict], List[dict]]:
    """Parse ChatGPT export. Returns (messages_rows, threads_rows)."""
    messages: List[dict] = []
    threads: List[dict] = []
    n = len(data)

    for i, conv in enumerate(data):
        if progress_cb:
            progress_cb(i / n, f"Parsing ChatGPT conversation {i + 1:,}/{n:,}…")

        thread_id = str(conv.get("id") or conv.get("conversation_id") or f"gpt_{i}")
        title = conv.get("title") or "Untitled"
        created_ts = _to_unix(conv.get("create_time"))
        updated_ts = _to_unix(conv.get("update_time"))

        mapping = conv.get("mapping") or {}
        if not isinstance(mapping, dict) or not mapping:
            continue

        message_nodes: Dict[str, Dict[str, Any]] = {
            str(nid): {
                "parent": node.get("parent"),
                "children": node.get("children") or [],
                "message": node.get("message"),
            }
            for nid, node in mapping.items()
            if isinstance(node, dict)
        }

        main_path = _pick_main_path(message_nodes)

        user_msgs = asst_msgs = user_chars = asst_chars = 0

        for node_id in main_path:
            node = message_nodes.get(node_id) or {}
            msg = node.get("message") or {}
            author = msg.get("author") or {}
            role = author.get("role") or ""

            if role not in ("user", "assistant"):
                continue

            text = _extract_chatgpt_text(msg.get("content"))
            if not text:
                continue

            ts = _to_unix(msg.get("create_time"))

            messages.append({
                "node_id": node_id,
                "thread_id": thread_id,
                "thread_title": title,
                "role": role,
                "timestamp": ts,
                "year_month": _year_month(ts),
                "char_count": len(text),
                "text": text,
            })

            if role == "user":
                user_msgs += 1
                user_chars += len(text)
            else:
                asst_msgs += 1
                asst_chars += len(text)

        threads.append({
            "thread_id": thread_id,
            "title": title,
            "source": "chatgpt",
            "created_ts": created_ts,
            "updated_ts": updated_ts,
            "msg_count": user_msgs + asst_msgs,
            "user_msgs": user_msgs,
            "asst_msgs": asst_msgs,
            "user_chars": user_chars,
            "asst_chars": asst_chars,
        })

    return messages, threads


# ─────────────────────────────────────────────────────────────────────────────
# Claude parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_claude(
    data: List[dict],
    progress_cb: Optional[Callable],
) -> Tuple[List[dict], List[dict]]:
    """Parse Claude export. Returns (messages_rows, threads_rows)."""
    messages: List[dict] = []
    threads: List[dict] = []
    n = len(data)

    for i, conv in enumerate(data):
        if progress_cb:
            progress_cb(i / n, f"Parsing Claude conversation {i + 1:,}/{n:,}…")

        thread_id = str(conv.get("uuid") or f"claude_{i}")
        title = conv.get("name") or "Untitled"
        created_ts = _to_unix(conv.get("created_at"))
        updated_ts = _to_unix(conv.get("updated_at"))

        chat_messages = conv.get("chat_messages") or conv.get("messages") or []
        if not isinstance(chat_messages, list):
            continue

        user_msgs = asst_msgs = user_chars = asst_chars = 0

        for msg in chat_messages:
            if not isinstance(msg, dict):
                continue

            sender = msg.get("sender") or msg.get("role") or ""
            if sender in ("human", "user"):
                role = "user"
            elif sender == "assistant":
                role = "assistant"
            else:
                continue

            # 'text' is the plain-text field; fall back to content blocks
            text = msg.get("text") or ""
            if not text:
                content_blocks = msg.get("content") or []
                chunks: List[str] = []
                for block in content_blocks:
                    if isinstance(block, dict):
                        chunks.append(block.get("text") or "")
                    elif isinstance(block, str):
                        chunks.append(block)
                text = "\n".join(c for c in chunks if c).strip()

            if not text:
                continue

            ts = _to_unix(msg.get("created_at") or msg.get("timestamp"))
            node_id = str(msg.get("uuid") or msg.get("id") or f"{thread_id}_{len(messages)}")

            messages.append({
                "node_id": node_id,
                "thread_id": thread_id,
                "thread_title": title,
                "role": role,
                "timestamp": ts,
                "year_month": _year_month(ts),
                "char_count": len(text),
                "text": text,
            })

            if role == "user":
                user_msgs += 1
                user_chars += len(text)
            else:
                asst_msgs += 1
                asst_chars += len(text)

        threads.append({
            "thread_id": thread_id,
            "title": title,
            "source": "claude",
            "created_ts": created_ts,
            "updated_ts": updated_ts,
            "msg_count": user_msgs + asst_msgs,
            "user_msgs": user_msgs,
            "asst_msgs": asst_msgs,
            "user_chars": user_chars,
            "asst_chars": asst_chars,
        })

    return messages, threads


# ─────────────────────────────────────────────────────────────────────────────
# SQLite writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_db(db_path: Path, messages: List[dict], threads: List[dict]) -> None:
    """Create (or overwrite) the SQLite database and write both tables."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS messages;
        DROP TABLE IF EXISTS threads;

        CREATE TABLE messages (
            node_id      TEXT PRIMARY KEY,
            thread_id    TEXT,
            thread_title TEXT,
            role         TEXT,
            timestamp    REAL,
            year_month   TEXT,
            char_count   INTEGER,
            text         TEXT
        );

        CREATE TABLE threads (
            thread_id   TEXT PRIMARY KEY,
            title       TEXT,
            source      TEXT,
            created_ts  REAL,
            updated_ts  REAL,
            msg_count   INTEGER,
            user_msgs   INTEGER,
            asst_msgs   INTEGER,
            user_chars  INTEGER,
            asst_chars  INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
        CREATE INDEX IF NOT EXISTS idx_messages_ym     ON messages(year_month);
        CREATE INDEX IF NOT EXISTS idx_messages_role   ON messages(role);
    """)

    cur.executemany(
        """INSERT OR REPLACE INTO messages
           (node_id, thread_id, thread_title, role, timestamp, year_month, char_count, text)
           VALUES (:node_id, :thread_id, :thread_title, :role, :timestamp,
                   :year_month, :char_count, :text)""",
        messages,
    )

    cur.executemany(
        """INSERT OR REPLACE INTO threads
           (thread_id, title, source, created_ts, updated_ts,
            msg_count, user_msgs, asst_msgs, user_chars, asst_chars)
           VALUES (:thread_id, :title, :source, :created_ts, :updated_ts,
                   :msg_count, :user_msgs, :asst_msgs, :user_chars, :asst_chars)""",
        threads,
    )

    con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def merge_exports(raw_files: list[bytes]) -> tuple[list[dict], str, int]:
    """Merge multiple conversations.json byte-strings into one deduplicated list.

    Parameters
    ----------
    raw_files:
        List of raw bytes from each uploaded conversations.json file.

    Returns
    -------
    (conversations, format, n_duplicates_removed)
        conversations          – merged, deduplicated list ready to be written
                                 as upload.json and passed to run()
        format                 – 'chatgpt' | 'claude'
        n_duplicates_removed   – conversations present in more than one file

    Raises
    ------
    ValueError  if any file is not a JSON array, formats are mixed across
                files, or no recognised format is found.
    """
    all_convs: list[dict] = []
    formats_seen: set[str] = set()

    for raw in raw_files:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(
                "Each conversations.json must be a JSON array. "
                "Make sure you are uploading the correct file from your export ZIP."
            )

        # Detect format from first few conversations
        fmt = "unknown"
        for conv in data[:5]:
            if isinstance(conv, dict):
                if "mapping" in conv and isinstance(conv.get("mapping"), dict):
                    fmt = "chatgpt"
                    break
                if "chat_messages" in conv or (
                    "uuid" in conv and ("name" in conv or "messages" in conv)
                ):
                    fmt = "claude"
                    break

        if fmt == "unknown":
            raise ValueError(
                "Could not detect the format (ChatGPT or Claude) in one of the "
                "uploaded files. Please check you are uploading conversations.json "
                "from the correct export ZIP."
            )

        formats_seen.add(fmt)
        all_convs.extend(data)

    if len(formats_seen) > 1:
        raise ValueError(
            f"Mixed export formats detected ({', '.join(sorted(formats_seen))}). "
            "All uploaded files must be exports from the same platform."
        )

    detected_fmt = formats_seen.pop()

    # Deduplicate by conversation ID (first occurrence = oldest file wins)
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for conv in all_convs:
        if detected_fmt == "chatgpt":
            cid = str(conv.get("id") or conv.get("conversation_id") or id(conv))
        else:  # claude
            cid = str(conv.get("uuid") or conv.get("id") or id(conv))

        if cid not in seen_ids:
            seen_ids.add(cid)
            unique.append(conv)

    n_dupes = len(all_convs) - len(unique)
    return unique, detected_fmt, n_dupes


def run(
    json_path: str | Path,
    db_path: str | Path,
    fmt: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Parse *json_path* and write the canonical schema to *db_path* (SQLite).

    Args:
        json_path:   Path to conversations.json
        db_path:     Destination SQLite file (created or overwritten)
        fmt:         'chatgpt' or 'claude'
        progress_cb: Optional callable(fraction 0–1, status_string)

    Returns:
        Summary dict: threads, messages, user_messages, asst_messages
    """
    def _cb(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac * 0.85, msg)  # reserve last 15% for DB write

    path = Path(json_path)

    if progress_cb:
        progress_cb(0.0, "Loading file…")

    data = _load_json(path)

    if not isinstance(data, list):
        raise ValueError("conversations.json must be a JSON array at the top level.")

    fmt = fmt.lower().strip()

    if fmt == "chatgpt":
        messages, threads = _parse_chatgpt(data, _cb)
    elif fmt == "claude":
        messages, threads = _parse_claude(data, _cb)
    else:
        raise ValueError(f"Unknown format '{fmt}'. Expected 'chatgpt' or 'claude'.")

    if progress_cb:
        progress_cb(0.87, "Writing database…")

    _write_db(Path(db_path), messages, threads)

    if progress_cb:
        progress_cb(1.0, "Parse complete.")

    return {
        "threads":       len(threads),
        "messages":      len(messages),
        "user_messages": sum(1 for m in messages if m["role"] == "user"),
        "asst_messages": sum(1 for m in messages if m["role"] == "assistant"),
        "user_chars":    sum(m.get("char_count", 0) for m in messages if m["role"] == "user"),
        "asst_chars":    sum(m.get("char_count", 0) for m in messages if m["role"] == "assistant"),
    }
