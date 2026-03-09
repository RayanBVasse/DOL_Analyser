"""
Tests for pipeline/parse.py

Run with:  pytest tests/

Covers:
    - ChatGPT parser   (tree structure, main-path traversal, content extraction)
    - Claude parser    (linear chat_messages, sender normalisation)
    - SQLite output    (schema, row counts, column values)
    - Encoding         (UTF-8, UTF-8-BOM)
    - Edge cases       (empty conversations, missing fields, branching trees)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from pipeline.parse import run


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────────

BASE_TS = 1_700_000_000  # ~Nov 2023  (Unix seconds)
ONE_MONTH = 30 * 24 * 3600


def _chatgpt_conv(
    conv_id: str = "conv_1",
    title: str = "Test conversation",
    n_pairs: int = 3,
    month_offset: int = 0,
    branching: bool = False,
) -> dict:
    """
    Build one ChatGPT conversation dict.

    If branching=True, adds an alternative child node to the first user
    message so the tree traversal must pick the longer (main) path.
    """
    ts = BASE_TS + month_offset * ONE_MONTH
    mapping = {}

    # Root system node
    mapping["root"] = {
        "id": "root",
        "parent": None,
        "children": ["u0"],
        "message": {
            "author": {"role": "system"},
            "create_time": ts,
            "content": {"content_type": "text", "parts": [""]},
        },
    }

    prev = "root"
    for i in range(n_pairs):
        uid = f"u{i}"
        aid = f"a{i}"

        mapping[uid] = {
            "id": uid,
            "parent": prev,
            "children": [aid],
            "message": {
                "author": {"role": "user"},
                "create_time": ts + i * 120,
                "content": {"content_type": "text", "parts": [f"User message {i}"]},
            },
        }
        mapping[aid] = {
            "id": aid,
            "parent": uid,
            "children": [f"u{i+1}"] if i < n_pairs - 1 else [],
            "message": {
                "author": {"role": "assistant"},
                "create_time": ts + i * 120 + 60,
                "content": {"content_type": "text", "parts": [f"Assistant reply {i}"]},
            },
        }
        prev = aid

    if branching:
        # Dead-end branch off u0 — should be ignored by main-path picker
        mapping["u0"]["children"].append("branch_dead")
        mapping["branch_dead"] = {
            "id": "branch_dead",
            "parent": "u0",
            "children": [],
            "message": {
                "author": {"role": "assistant"},
                "create_time": ts + 30,
                "content": {"content_type": "text", "parts": ["Branch reply"]},
            },
        }

    return {
        "id": conv_id,
        "title": title,
        "create_time": float(ts),
        "update_time": float(ts + 3600),
        "mapping": mapping,
    }


def _claude_conv(
    uuid: str = "claude_conv_1",
    name: str = "Test conversation",
    n_pairs: int = 3,
    month_offset: int = 0,
) -> dict:
    """Build one Claude conversation dict matching the real export format."""
    from datetime import datetime, timezone, timedelta

    base_dt = datetime(2023, 11, 1, tzinfo=timezone.utc) + timedelta(
        days=month_offset * 30
    )

    messages = []
    for i in range(n_pairs):
        user_dt = base_dt + timedelta(minutes=i * 2)
        asst_dt = base_dt + timedelta(minutes=i * 2 + 1)

        messages.append({
            "uuid": f"msg_u{i}",
            "sender": "human",
            "text": f"User message {i}",
            "content": [{"type": "text", "text": f"User message {i}"}],
            "created_at": user_dt.isoformat().replace("+00:00", "Z"),
            "updated_at": user_dt.isoformat().replace("+00:00", "Z"),
            "attachments": [],
            "files": [],
        })
        messages.append({
            "uuid": f"msg_a{i}",
            "sender": "assistant",
            "text": f"Assistant reply {i}",
            "content": [{"type": "text", "text": f"Assistant reply {i}"}],
            "created_at": asst_dt.isoformat().replace("+00:00", "Z"),
            "updated_at": asst_dt.isoformat().replace("+00:00", "Z"),
            "attachments": [],
            "files": [],
        })

    created_at = base_dt.isoformat().replace("+00:00", "Z")
    return {
        "uuid": uuid,
        "name": name,
        "account": {"uuid": "acct-123"},
        "chat_messages": messages,
        "created_at": created_at,
        "updated_at": created_at,
        "summary": "Test summary",
    }


def _write_json(data: object, encoding: str = "utf-8") -> Path:
    suffix = ".json"
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, mode="w", encoding=encoding
    ) as fh:
        json.dump(data, fh)
        return Path(fh.name)


def _tmp_db() -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as fh:
        return Path(fh.name)


def _read_table(db_path: Path, table: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(f"SELECT * FROM {table}").fetchall()]
    con.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ChatGPT parser tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChatGPTParser:

    def test_basic_message_count(self, tmp_path):
        data = [_chatgpt_conv(n_pairs=4)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"

        result = run(json_path, db_path, fmt="chatgpt")

        assert result["user_messages"] == 4
        assert result["asst_messages"] == 4
        assert result["threads"] == 1

    def test_messages_written_to_db(self, tmp_path):
        data = [_chatgpt_conv(n_pairs=3)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        msgs = _read_table(db_path, "messages")
        assert len(msgs) == 6  # 3 user + 3 assistant

    def test_roles_correct(self, tmp_path):
        data = [_chatgpt_conv(n_pairs=2)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        msgs = _read_table(db_path, "messages")
        roles = {m["role"] for m in msgs}
        assert roles == {"user", "assistant"}

    def test_text_content_extracted(self, tmp_path):
        data = [_chatgpt_conv(n_pairs=1)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        msgs = _read_table(db_path, "messages")
        texts = {m["text"] for m in msgs}
        assert "User message 0" in texts
        assert "Assistant reply 0" in texts

    def test_year_month_derived(self, tmp_path):
        data = [_chatgpt_conv(month_offset=0)]  # ~Nov 2023
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        msgs = _read_table(db_path, "messages")
        assert all(m["year_month"] == "2023-11" for m in msgs)

    def test_thread_row_written(self, tmp_path):
        data = [_chatgpt_conv(conv_id="abc", title="My thread", n_pairs=2)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        threads = _read_table(db_path, "threads")
        assert len(threads) == 1
        assert threads[0]["thread_id"] == "abc"
        assert threads[0]["title"] == "My thread"
        assert threads[0]["source"] == "chatgpt"
        assert threads[0]["user_msgs"] == 2
        assert threads[0]["asst_msgs"] == 2

    def test_branching_tree_main_path_only(self, tmp_path):
        """Branch node should NOT appear in messages — only main path."""
        data = [_chatgpt_conv(n_pairs=3, branching=True)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        msgs = _read_table(db_path, "messages")
        texts = [m["text"] for m in msgs]
        assert "Branch reply" not in texts

    def test_multiple_conversations(self, tmp_path):
        data = [
            _chatgpt_conv(conv_id="c1", n_pairs=2, month_offset=0),
            _chatgpt_conv(conv_id="c2", n_pairs=3, month_offset=1),
        ]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        result = run(json_path, db_path, fmt="chatgpt")

        assert result["threads"] == 2
        assert result["user_messages"] == 5
        assert result["asst_messages"] == 5

    def test_empty_mapping_skipped(self, tmp_path):
        data = [
            {"id": "empty", "title": "Empty", "create_time": BASE_TS,
             "update_time": BASE_TS, "mapping": {}},
            _chatgpt_conv(conv_id="real", n_pairs=1),
        ]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        result = run(json_path, db_path, fmt="chatgpt")

        assert result["user_messages"] == 1

    def test_parts_as_dict_extracted(self, tmp_path):
        """Content parts can be dicts with a 'text' key."""
        conv = _chatgpt_conv(n_pairs=0)
        conv["mapping"]["u0"] = {
            "id": "u0", "parent": "root", "children": [],
            "message": {
                "author": {"role": "user"},
                "create_time": float(BASE_TS + 10),
                "content": {
                    "content_type": "text",
                    "parts": [{"text": "Dict part content"}],
                },
            },
        }
        conv["mapping"]["root"]["children"] = ["u0"]
        json_path = _write_json([conv])
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        msgs = _read_table(db_path, "messages")
        assert any("Dict part content" in m["text"] for m in msgs)


# ─────────────────────────────────────────────────────────────────────────────
# Claude parser tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClaudeParser:

    def test_basic_message_count(self, tmp_path):
        data = [_claude_conv(n_pairs=4)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        result = run(json_path, db_path, fmt="claude")

        assert result["user_messages"] == 4
        assert result["asst_messages"] == 4
        assert result["threads"] == 1

    def test_human_sender_mapped_to_user(self, tmp_path):
        data = [_claude_conv(n_pairs=2)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="claude")

        msgs = _read_table(db_path, "messages")
        roles = {m["role"] for m in msgs}
        assert "user" in roles
        assert "human" not in roles  # must be normalised

    def test_text_field_used(self, tmp_path):
        data = [_claude_conv(n_pairs=1)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="claude")

        msgs = _read_table(db_path, "messages")
        texts = {m["text"] for m in msgs}
        assert "User message 0" in texts
        assert "Assistant reply 0" in texts

    def test_iso_timestamp_parsed(self, tmp_path):
        data = [_claude_conv(month_offset=0)]  # Nov 2023
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="claude")

        msgs = _read_table(db_path, "messages")
        assert all(m["year_month"] == "2023-11" for m in msgs)

    def test_thread_row_written(self, tmp_path):
        data = [_claude_conv(uuid="uuid-abc", name="Claude thread", n_pairs=2)]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="claude")

        threads = _read_table(db_path, "threads")
        assert len(threads) == 1
        assert threads[0]["thread_id"] == "uuid-abc"
        assert threads[0]["title"] == "Claude thread"
        assert threads[0]["source"] == "claude"

    def test_content_block_fallback(self, tmp_path):
        """If 'text' field is absent, reconstruct from content blocks."""
        conv = _claude_conv(n_pairs=0)
        conv["chat_messages"] = [{
            "uuid": "msg_fallback",
            "sender": "human",
            "text": "",  # empty text field
            "content": [{"type": "text", "text": "Fallback content"}],
            "created_at": "2023-11-01T10:00:00Z",
            "updated_at": "2023-11-01T10:00:00Z",
        }]
        json_path = _write_json([conv])
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="claude")

        msgs = _read_table(db_path, "messages")
        assert any("Fallback content" in m["text"] for m in msgs)

    def test_multiple_conversations(self, tmp_path):
        data = [
            _claude_conv(uuid="c1", n_pairs=2, month_offset=0),
            _claude_conv(uuid="c2", n_pairs=3, month_offset=1),
        ]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        result = run(json_path, db_path, fmt="claude")

        assert result["threads"] == 2
        assert result["user_messages"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# SQLite schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSQLiteSchema:

    def test_both_tables_exist(self, tmp_path):
        json_path = _write_json([_chatgpt_conv()])
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        con = sqlite3.connect(db_path)
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        con.close()
        assert "messages" in tables
        assert "threads" in tables

    def test_messages_columns(self, tmp_path):
        json_path = _write_json([_chatgpt_conv()])
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        con = sqlite3.connect(db_path)
        cols = {r[1] for r in con.execute("PRAGMA table_info(messages)").fetchall()}
        con.close()
        assert {"node_id", "thread_id", "role", "timestamp",
                "year_month", "char_count", "text"}.issubset(cols)

    def test_char_count_matches_text_length(self, tmp_path):
        json_path = _write_json([_chatgpt_conv(n_pairs=2)])
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        msgs = _read_table(db_path, "messages")
        for m in msgs:
            assert m["char_count"] == len(m["text"])

    def test_node_id_unique(self, tmp_path):
        data = [
            _chatgpt_conv(conv_id="c1", n_pairs=3),
            _chatgpt_conv(conv_id="c2", n_pairs=3),
        ]
        json_path = _write_json(data)
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt")

        msgs = _read_table(db_path, "messages")
        node_ids = [m["node_id"] for m in msgs]
        assert len(node_ids) == len(set(node_ids))


# ─────────────────────────────────────────────────────────────────────────────
# Encoding tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEncoding:

    def test_utf8_bom_loads(self, tmp_path):
        data = [_chatgpt_conv(n_pairs=1)]
        json_path = _write_json(data, encoding="utf-8-sig")
        db_path = tmp_path / "out.db"
        result = run(json_path, db_path, fmt="chatgpt")
        assert result["messages"] > 0

    def test_invalid_format_raises(self, tmp_path):
        json_path = _write_json([_chatgpt_conv()])
        db_path = tmp_path / "out.db"
        with pytest.raises(ValueError, match="Unknown format"):
            run(json_path, db_path, fmt="unknown_format")


# ─────────────────────────────────────────────────────────────────────────────
# Progress callback tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressCallback:

    def test_callback_called(self, tmp_path):
        calls = []
        json_path = _write_json([_chatgpt_conv()])
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt",
            progress_cb=lambda f, m: calls.append(f))
        assert len(calls) > 0

    def test_callback_ends_at_1(self, tmp_path):
        fracs = []
        json_path = _write_json([_chatgpt_conv()])
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt",
            progress_cb=lambda f, m: fracs.append(f))
        assert fracs[-1] == 1.0

    def test_fractions_between_0_and_1(self, tmp_path):
        fracs = []
        json_path = _write_json([_chatgpt_conv()])
        db_path = tmp_path / "out.db"
        run(json_path, db_path, fmt="chatgpt",
            progress_cb=lambda f, m: fracs.append(f))
        assert all(0.0 <= f <= 1.0 for f in fracs)
