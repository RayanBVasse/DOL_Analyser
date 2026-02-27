"""
Tests for pipeline/precheck.py

Run with:  pytest tests/
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pipeline.precheck import run, MIN_USER_MESSAGES, MIN_MONTHS


# ---------------------------------------------------------------------------
# Helpers to build minimal valid JSON fixtures
# ---------------------------------------------------------------------------

def _chatgpt_fixture(n_user: int = 2500, n_months: int = 6) -> list:
    """Build a minimal ChatGPT-format conversations list."""
    import time

    conversations = []
    msgs_per_month = n_user // n_months
    base_ts = 1_680_000_000  # ~April 2023

    for month_idx in range(n_months):
        month_ts_offset = month_idx * 30 * 24 * 3600
        mapping = {}
        for i in range(msgs_per_month * 2):  # user + assistant pairs
            node_id = f"m{month_idx}_{i}"
            role = "user" if i % 2 == 0 else "assistant"
            mapping[node_id] = {
                "parent": None,
                "children": [],
                "message": {
                    "id": node_id,
                    "author": {"role": role},
                    "create_time": base_ts + month_ts_offset + i * 60,
                    "content": {
                        "content_type": "text",
                        "parts": ["Hello world"],
                    },
                },
            }
        conversations.append({
            "id": f"conv_{month_idx}",
            "title": f"Conversation {month_idx}",
            "create_time": base_ts + month_ts_offset,
            "update_time": base_ts + month_ts_offset + 3600,
            "mapping": mapping,
        })

    return conversations


def _write_tmp(data: object) -> Path:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".json", mode="w", encoding="utf-8"
    ) as fh:
        json.dump(data, fh)
        return Path(fh.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFormatDetection:
    def test_chatgpt_detected(self):
        path = _write_tmp(_chatgpt_fixture())
        result = run(path)
        assert result.format == "chatgpt"

    def test_unknown_format(self):
        path = _write_tmp([{"foo": "bar"}])
        result = run(path)
        assert result.format == "unknown"
        assert not result.ready

    def test_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all {{{", encoding="utf-8")
        result = run(bad)
        assert result.format == "unknown"
        assert any("parse" in w.lower() or "JSON" in w for w in result.warnings)


class TestMinimumRequirements:
    def test_passes_with_sufficient_data(self):
        path = _write_tmp(_chatgpt_fixture(n_user=2500, n_months=6))
        result = run(path)
        assert result.passes_message_minimum
        assert result.passes_month_minimum
        assert result.ready

    def test_fails_too_few_messages(self):
        path = _write_tmp(_chatgpt_fixture(n_user=100, n_months=6))
        result = run(path)
        assert not result.passes_message_minimum
        assert not result.ready

    def test_fails_too_few_months(self):
        path = _write_tmp(_chatgpt_fixture(n_user=3000, n_months=1))
        result = run(path)
        assert not result.passes_month_minimum
        assert not result.ready


class TestProgressCallback:
    def test_callback_called(self):
        calls = []
        path = _write_tmp(_chatgpt_fixture())
        run(path, progress_cb=lambda f, m: calls.append(f))
        assert len(calls) > 0
        assert calls[-1] == 1.0
