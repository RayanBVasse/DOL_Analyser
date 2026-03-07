"""
Step 0 — Pre-check

Runs before the full pipeline. Inspects the raw conversations.json and returns
a structured report so the UI can show the user what was found and whether the
file meets the minimum requirements for analysis.

Minimum requirements
────────────────────
- Recognised format (ChatGPT or Claude)
- Distinct conversations (threads) >= MIN_CONVERSATIONS   [topic diversity]
- Date range spans >= MIN_MONTHS distinct calendar months [temporal coverage]
- Average user messages per month >= MIN_AVG_USER_MSGS_PER_MONTH [signal density]
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

MIN_CONVERSATIONS           = 50
MIN_MONTHS                  = 3
MIN_AVG_USER_MSGS_PER_MONTH = 30


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

    # Character counts (content volume)
    user_chars: int = 0
    assistant_chars: int = 0

    # Temporal coverage
    earliest_date: datetime | None = None
    latest_date: datetime | None = None
    months_covered: int = 0
    avg_user_msgs_per_month: float = 0.0

    # Requirement checks (new criteria)
    passes_conversation_minimum: bool = False
    passes_month_minimum: bool = False
    passes_avg_msgs_minimum: bool = False

    # Warnings raised during inspection
    warnings: list[str] = field(default_factory=list)

    # ── Derived metrics ───────────────────────────────────────────────────

    @property
    def msg_ratio(self) -> float | None:
        """Assistant-to-user message ratio (asst / user). None if no user messages."""
        if self.user_messages == 0:
            return None
        return self.assistant_messages / self.user_messages

    @property
    def char_ratio(self) -> float | None:
        """Assistant-to-user character ratio (asst / user). None if no user chars."""
        if self.user_chars == 0:
            return None
        return self.assistant_chars / self.user_chars

    # ── Pass/fail ─────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return (
            self.format != "unknown"
            and self.passes_conversation_minimum
            and self.passes_month_minimum
            and self.passes_avg_msgs_minimum
        )

    def summary_lines(self) -> list[str]:
        lines = [
            f"Format detected      : {self.format.upper()} ({self.format_confidence})",
            f"Conversations        : {self.total_conversations:,}",
            f"Total messages       : {self.total_messages:,}",
            f"  — user             : {self.user_messages:,}",
            f"  — assistant        : {self.assistant_messages:,}",
        ]
        if self.user_chars or self.assistant_chars:
            lines += [
                f"User chars           : {self.user_chars:,}",
                f"Assistant chars      : {self.assistant_chars:,}",
            ]
            if self.char_ratio is not None:
                lines.append(f"Asst:user char ratio : {self.char_ratio:.1f}×")
        if self.earliest_date and self.latest_date:
            lines += [
                f"Date range           : {self.earliest_date:%Y-%m-%d} → "
                f"{self.latest_date:%Y-%m-%d}",
                f"Months covered       : {self.months_covered}",
                f"Avg user msgs/month  : {self.avg_user_msgs_per_month:.0f}",
            ]
        lines += [
            f"Meets conv min  (>={MIN_CONVERSATIONS:,}) : "
            f"{'YES' if self.passes_conversation_minimum else 'NO'}",
            f"Meets month min (>={MIN_MONTHS})  : "
            f"{'YES' if self.passes_month_minimum else 'NO'}",
            f"Meets avg/month (>={MIN_AVG_USER_MSGS_PER_MONTH}) : "
            f"{'YES' if self.passes_avg_msgs_minimum else 'NO'}",
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
    """Flatten ChatGPT mapping trees into a list of {role, timestamp, chars} dicts."""
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
                # Extract text for character count
                content = msg.get("content") or {}
                parts = content.get("parts") or []
                text = " ".join(str(p) for p in parts if p)
                out.append({
                    "role":      role,
                    "timestamp": float(ts),
                    "chars":     len(text),
                })
    return out


def _extract_claude_messages(data: list[dict]) -> list[dict]:
    """Flatten Claude conversations into a list of {role, timestamp, chars} dicts."""
    out = []
    for conv in data:
        messages = conv.get("chat_messages") or conv.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("sender") or msg.get("role") or ""
            ts_raw = (
                msg.get("created_at")
                or msg.get("timestamp")
                or msg.get("create_time")
            )
            if role in ("human", "user", "assistant") and ts_raw is not None:
                role = "user" if role in ("human", "user") else "assistant"
                # Text content
                text = ""
                raw_content = msg.get("content") or msg.get("text") or ""
                if isinstance(raw_content, str):
                    text = raw_content
                elif isinstance(raw_content, list):
                    text = " ".join(
                        c.get("text", "") for c in raw_content
                        if isinstance(c, dict)
                    )
                out.append({
                    "role":      role,
                    "timestamp": _to_unix(ts_raw),
                    "chars":     len(text),
                })
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
    # 3. Check conversation (thread) diversity
    # ------------------------------------------------------------------
    result.passes_conversation_minimum = (
        result.total_conversations >= MIN_CONVERSATIONS
    )
    if not result.passes_conversation_minimum:
        result.warnings.append(
            f"Only {result.total_conversations:,} conversations found; "
            f"minimum required is {MIN_CONVERSATIONS} (for topic diversity)."
        )

    # ------------------------------------------------------------------
    # 4. Extract messages (with character counts)
    # ------------------------------------------------------------------
    _progress(0.4, "Counting messages…")
    if result.format == "chatgpt":
        messages = _extract_chatgpt_messages(data)
    else:
        messages = _extract_claude_messages(data)

    result.total_messages     = len(messages)
    result.user_messages      = sum(1 for m in messages if m["role"] == "user")
    result.assistant_messages = sum(1 for m in messages if m["role"] == "assistant")
    result.user_chars         = sum(m["chars"] for m in messages if m["role"] == "user")
    result.assistant_chars    = sum(m["chars"] for m in messages if m["role"] == "assistant")

    # ------------------------------------------------------------------
    # 5. Temporal coverage + avg msgs/month
    # ------------------------------------------------------------------
    _progress(0.7, "Checking date range…")
    timestamps = [m["timestamp"] for m in messages if m["timestamp"] is not None]

    if timestamps:
        result.earliest_date = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
        result.latest_date   = datetime.fromtimestamp(max(timestamps), tz=timezone.utc)

        month_counts: Counter = Counter()
        for m in messages:
            if m["timestamp"] is not None and m["role"] == "user":
                dt = datetime.fromtimestamp(m["timestamp"], tz=timezone.utc)
                month_counts[(dt.year, dt.month)] += 1

        result.months_covered = len(month_counts)
        result.passes_month_minimum = result.months_covered >= MIN_MONTHS

        if not result.passes_month_minimum:
            result.warnings.append(
                f"Only {result.months_covered} distinct month(s) of data found; "
                f"minimum required is {MIN_MONTHS}."
            )

        if result.months_covered > 0:
            result.avg_user_msgs_per_month = (
                result.user_messages / result.months_covered
            )
            result.passes_avg_msgs_minimum = (
                result.avg_user_msgs_per_month >= MIN_AVG_USER_MSGS_PER_MONTH
            )
            if not result.passes_avg_msgs_minimum:
                result.warnings.append(
                    f"Average of {result.avg_user_msgs_per_month:.0f} user messages/month "
                    f"is below the minimum of {MIN_AVG_USER_MSGS_PER_MONTH}/month. "
                    "Monthly analyses may be unreliable."
                )
    else:
        result.warnings.append("No valid timestamps found in messages.")

    _progress(1.0, "Pre-check complete.")
    return result
