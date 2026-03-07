"""
DOL Analyser — command-line diagnostic tool.

Usage (from the repo root, with .venv active):

    python scripts/diagnose.py conversations.json
    python scripts/diagnose.py file1.json file2.json          # merges both
    python scripts/diagnose.py file1.json file2.json --verbose

Reports conversation counts, message counts, date range, and a
per-month breakdown so you can cross-check the Streamlit UI figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.parse import merge_exports  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _ts_to_year_month(ts: object) -> str | None:
    """Convert an ISO-8601 string or Unix timestamp to 'YYYY-MM'."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime, timezone
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m")
        s = str(ts)
        return s[:7]          # "2025-08-13T..." → "2025-08"
    except Exception:
        return None


def _analyse(conversations: list[dict]) -> dict:
    """Return a stats dict for a list of conversations."""
    total_convs   = len(conversations)
    total_msgs    = 0
    user_msgs     = 0
    asst_msgs     = 0
    months_user: Counter  = Counter()
    months_asst: Counter  = Counter()

    for conv in conversations:
        messages = (
            conv.get("chat_messages")          # Claude
            or conv.get("messages")            # Claude alt
            or list((conv.get("mapping") or {}).values())  # ChatGPT nodes
        )
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Role — Claude uses "sender" with values "human"/"assistant"
            #         ChatGPT uses "author.role" with values "user"/"assistant"
            role = (
                msg.get("sender")                       # Claude
                or msg.get("role")                      # Claude alt / ChatGPT flat
                or (msg.get("author") or {}).get("role")  # ChatGPT nested
                or ""
            )
            # Normalise "human" → "user"
            if role == "human":
                role = "user"
            if role not in ("user", "assistant"):
                continue

            total_msgs += 1
            if role == "user":
                user_msgs += 1
            else:
                asst_msgs += 1

            # Timestamp → year_month
            ts = (
                msg.get("created_at")
                or msg.get("timestamp")
                or msg.get("create_time")
            )
            ym = _ts_to_year_month(ts)
            if ym:
                if role == "user":
                    months_user[ym] += 1
                else:
                    months_asst[ym] += 1

    all_months = sorted(set(months_user) | set(months_asst))
    return {
        "total_convs":  total_convs,
        "total_msgs":   total_msgs,
        "user_msgs":    user_msgs,
        "asst_msgs":    asst_msgs,
        "months_user":  months_user,
        "months_asst":  months_asst,
        "all_months":   all_months,
        "date_range":   (all_months[0], all_months[-1]) if all_months else (None, None),
    }


def _bar(n: int, max_n: int, width: int = 30) -> str:
    filled = int(width * n / max_n) if max_n else 0
    return "█" * filled + "░" * (width - filled)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and verify DOL Analyser input files."
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE",
        help="One or more conversations.json files to inspect.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-month breakdown.",
    )
    args = parser.parse_args()

    # ── Load & merge ─────────────────────────────────────────────────────────
    print()
    raw_list: list[bytes] = []
    for p in args.files:
        path = Path(p)
        if not path.exists():
            print(f"  ✗  File not found: {p}")
            sys.exit(1)
        raw = path.read_bytes()
        raw_list.append(raw)
        size_mb = len(raw) / 1_048_576
        # Quick individual file stats before merge
        data = json.loads(raw)
        stats_i = _analyse(data)
        print(f"  📄  {path.name}  ({size_mb:.1f} MB)")
        print(f"      {len(data):,} conversations · "
              f"{stats_i['user_msgs']:,} user msgs · "
              f"{stats_i['asst_msgs']:,} asst msgs · "
              f"{stats_i['date_range'][0]} → {stats_i['date_range'][1]}")

    # ── Merge if needed ───────────────────────────────────────────────────────
    print()
    if len(raw_list) > 1:
        try:
            merged, fmt, n_dupes = merge_exports(raw_list)
        except ValueError as exc:
            print(f"  ✗  Merge failed: {exc}")
            sys.exit(1)
        print(f"  🔀  Merged {len(args.files)} files  →  {len(merged):,} conversations"
              f"  ({n_dupes:,} duplicates removed)  [{fmt.upper()}]")
    else:
        data_single = json.loads(raw_list[0])
        merged = data_single
        fmt = "unknown"
        n_dupes = 0

    # ── Analyse merged result ─────────────────────────────────────────────────
    s = _analyse(merged)
    early, late = s["date_range"]

    n_months     = len(s["all_months"])
    avg_per_month = s["user_msgs"] / n_months if n_months else 0
    pass_convs   = s["total_convs"] >= 50
    pass_months  = n_months >= 3
    pass_avg     = avg_per_month >= 30
    all_pass     = pass_convs and pass_months and pass_avg

    def _tick(ok): return "✓" if ok else "✗"

    print()
    print("  ┌─ MERGED TOTALS ──────────────────────────────────────────────────┐")
    print(f"  │  Conversations    : {s['total_convs']:>8,}  {_tick(pass_convs)} (min 50)              │")
    print(f"  │  Total messages   : {s['total_msgs']:>8,}                                  │")
    print(f"  │  User messages    : {s['user_msgs']:>8,}                                  │")
    print(f"  │  Asst messages    : {s['asst_msgs']:>8,}                                  │")
    if s["user_msgs"]:
        ratio = s["asst_msgs"] / s["user_msgs"]
        print(f"  │  Asst:user ratio  : {ratio:>7.1f}×                                  │")
    print(f"  │  Date range       : {str(early):>7} → {late}                   │")
    print(f"  │  Months covered   : {n_months:>8,}  {_tick(pass_months)} (min 3)               │")
    print(f"  │  Avg user/month   : {avg_per_month:>8.0f}  {_tick(pass_avg)} (min 30)              │")
    print(f"  │  Overall          : {'✓ PASS — ready to analyse' if all_pass else '✗ FAIL — see above':}           │")
    print("  └──────────────────────────────────────────────────────────────────┘")

    # ── Per-month breakdown ───────────────────────────────────────────────────
    if args.verbose and s["all_months"]:
        print()
        print("  Per-month breakdown (user messages | asst messages):")
        print()
        max_count = max(
            max(s["months_user"].values(), default=0),
            max(s["months_asst"].values(), default=0),
        )
        for ym in s["all_months"]:
            u = s["months_user"].get(ym, 0)
            a = s["months_asst"].get(ym, 0)
            bar_u = _bar(u, max_count, 20)
            print(f"  {ym}  user {u:>5,}  {bar_u}  asst {a:>5,}")

    print()


if __name__ == "__main__":
    main()
