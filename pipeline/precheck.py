"""
Step 0 — Pre-check

Runs before the full pipeline. Inspects the raw conversations.json and returns
a structured report so the UI can show the user what was found and whether the
file meets the minimum requirements for analysis.

Minimum requirements:
    - Recognised format (ChatGPT or Claude)
    - Total user messages >= MIN_USER_MESSAGES
    - Date range spans >= MIN_MONTHS distinct calendar months
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

MIN_USER_MESSAGES = 2000
MIN_MONTHS = 3


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PrecheckResult:
    # Core detection
    format: Literal["chatgpt", "claude", "unknown"] = "unknown"
    format_confidence: str = ""

    # Counts
    total_conversations: int = 0
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0

    # Temporal coverage
    earliest_date: datetime | None = None
    latest_date: datetime | None = None
    months_covered: int = 0

    # Requirement checks
    passes_message_minimum: bool = False
    passes_month_minimum: bool = False

    # Warnings raised during inspection
    warnings: list[str] = field(default_factory=list)

    # Overall pass/fail
    @property
    def ready(self) -> bool:
        return (
            self.format != "unknown"
            and self.passes_message_minimum
            and self.passes_month_minimum
        )

    def summary_lines(self) -> list[str]:
        lines = [
            f"Format detected : {self.format.upper()} ({self.format_confidence})",
            f"Conversations   : {self.total_conversations:,}",
            f"Total messages  : {self.total_messages:,}",
            f"  — user        : {self.user_messages:,}",
            f"  — assistant   : {self.assistant_messages:,}",
        ]
        if self.earliest_date and self.latest_date:
            lines += [
                f"Date range      : {self.earliest_date:%Y-%m-%d} → {self.latest_date:%Y-%m-%d}",
                f"Months covered  : {self.months_covered}",
            ]
        lines += [
            f"Meets msg min ({MIN_USER_MESSAGES:,}) : {'YES' if self.passes_message_minimum else 'NO'}",
            f"Meets month min ({MIN_MONTHS})  : {'YES' if self.passes_month_minimum else 'NO'}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return lines


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(data: object) -> tuple[Literal["chatgpt", "claude", "unknown"], str]:
    """Return (format_name, confidence_description)."""
    if not isinstance(data, list) or len(data) == 0:
        return "unknown", "top-level is not a non-empty list"

    first = data[0]
    if not isinstance(first, dict):
        return "unknown", "first element is not an object"

    # ChatGPT: conversations have a 'mapping' key containing node objects
    if "mapping" in first and isinstance(first.get("mapping"), dict):
        return "chatgpt", "found 'mapping' tree structure"

    # Claude: conversations have a 'chat_messages' or 'messages' list
    if "chat_messages" in first or (
        "messages" in first and isinstance(first.get("messages"), list)
    ):
        return "claude", "found linear 'messages' list"

    return "unknown", "unrecognised structure"


# ---------------------------------------------------------------------------
# Message extraction helpers
# ---------------------------------------------------------------------------

def _extract_chatgpt_messages(data: list[dict]) -> list[dict]:
    """Flatten ChatGPT mapping trees into a list of {role, timestamp} dicts."""
    out = []
    for conv in data:
        mapping = conv.get("mapping") or {}
        for node in mapping.values():
            msg = node.get("message")
            if not msg:
                continue
            role = (msg.get("author") or {}).get("role", "")
            ts = msg.get("create_time")
            if role in ("user", "assistant") and ts:
                out.append({"role": role, "timestamp": float(ts)})
    return out


def _extract_claude_messages(data: list[dict]) -> list[dict]:
    """Flatten Claude linear conversations into a list of {role, timestamp} dicts."""
    out = []
    for conv in data:
        messages = conv.get("chat_messages") or conv.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("sender") or msg.get("role") or ""
            # Claude exports may use ISO strings or unix timestamps
            ts_raw = msg.get("created_at") or msg.get("timestamp") or msg.get("create_time")
            if role in ("human", "user", "assistant") and ts_raw is not None:
                # Normalise role label
                role = "user" if role in ("human", "user") else "assistant"
                out.append({"role": role, "timestamp": _to_unix(ts_raw)})
    return out


def _to_unix(value: object) -> float | None:
    """Convert an ISO string or numeric value to a Unix timestamp."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(json_path: str | Path, progress_cb=None) -> PrecheckResult:
    """
    Inspect *json_path* and return a PrecheckResult.

    Args:
        json_path:   Path to the conversations.json file.
        progress_cb: Optional callable(float, str) for progress updates.
                     Called with a fraction 0–1 and a status string.
    """
    result = PrecheckResult()
    path = Path(json_path)

    def _progress(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    # ------------------------------------------------------------------
    # 1. Load JSON
    # ------------------------------------------------------------------
    _progress(0.0, "Loading file…")
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        result.warnings.append(f"Could not parse JSON: {exc}")
        return result
    except OSError as exc:
        result.warnings.append(f"Could not open file: {exc}")
        return result

    # ------------------------------------------------------------------
    # 2. Detect format
    # ------------------------------------------------------------------
    _progress(0.2, "Detecting format…")
    result.format, result.format_confidence = _detect_format(data)

    if result.format == "unknown":
        result.warnings.append(
            "Format not recognised. Expected a ChatGPT or Claude conversations.json export."
        )
        return result

    result.total_conversations = len(data)

    # ------------------------------------------------------------------
    # 3. Extract messages
    # ------------------------------------------------------------------
    _progress(0.4, "Counting messages…")
    if result.format == "chatgpt":
        messages = _extract_chatgpt_messages(data)
    else:
        messages = _extract_claude_messages(data)

    result.total_messages = len(messages)
    result.user_messages = sum(1 for m in messages if m["role"] == "user")
    result.assistant_messages = sum(1 for m in messages if m["role"] == "assistant")

    result.passes_message_minimum = result.user_messages >= MIN_USER_MESSAGES

    if result.user_messages < MIN_USER_MESSAGES:
        result.warnings.append(
            f"Only {result.user_messages:,} user messages found; "
            f"minimum required is {MIN_USER_MESSAGES:,}."
        )

    # ------------------------------------------------------------------
    # 4. Temporal coverage
    # ------------------------------------------------------------------
    _progress(0.7, "Checking date range…")
    timestamps = [m["timestamp"] for m in messages if m["timestamp"] is not None]

    if timestamps:
        result.earliest_date = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
        result.latest_date = datetime.fromtimestamp(max(timestamps), tz=timezone.utc)

        month_set = set()
        for ts in timestamps:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            month_set.add((dt.year, dt.month))

        result.months_covered = len(month_set)
        result.passes_month_minimum = result.months_covered >= MIN_MONTHS

        if not result.passes_month_minimum:
            result.warnings.append(
                f"Only {result.months_covered} distinct month(s) of data found; "
                f"minimum required is {MIN_MONTHS}."
            )
    else:
        result.warnings.append("No valid timestamps found in messages.")

    _progress(1.0, "Pre-check complete.")
    return result
