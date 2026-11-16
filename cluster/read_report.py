"""
Print the same summary the Excel report would show, directly from a
checkpoint JSONL file -- no Excel/GUI needed, works over plain SSH.

Usage:
    python cluster/read_report.py                         # all 3 default paths
    python cluster/read_report.py reports/full_eval_baseline_checkpoint.jsonl
    python cluster/read_report.py path1.jsonl path2.jsonl  # any subset
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_PATHS = [
    "reports/full_eval_baseline_checkpoint.jsonl",
    "reports/full_eval_checkpoint.jsonl",
    "reports/full_eval_v2_checkpoint.jsonl",
]


def report_one(path: str) -> None:
    if not Path(path).exists():
        print(f"File: {path}  -- not found yet, skipping\n")
        return

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    verdict_counts = {"FULL": 0, "PARTIAL": 0, "REJECTED": 0, "ERROR": 0}
    judge_counts = {"yes": 0, "no": 0, "uncertain": 0, "": 0}

    for r in rows:
        v = r.get("verdict") or "ERROR"
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        j = (r.get("judge_match") or "").lower()
        judge_counts[j] = judge_counts.get(j, 0) + 1

    def pct(n: int) -> str:
        return f"{n/total*100:.1f}%" if total else "n/a"

    judged_total = total - judge_counts.get("", 0)

    def jpct(n: int) -> str:
        return f"{n/judged_total*100:.1f}%" if judged_total else "n/a"

    print(f"File: {path}")
    print(f"Total rows: {total}")
    print(
        f"FULL: {verdict_counts['FULL']} ({pct(verdict_counts['FULL'])})  |  "
        f"PARTIAL: {verdict_counts['PARTIAL']} ({pct(verdict_counts['PARTIAL'])})  |  "
        f"REJECTED: {verdict_counts['REJECTED']} ({pct(verdict_counts['REJECTED'])})  |  "
        f"ERROR: {verdict_counts['ERROR']} ({pct(verdict_counts['ERROR'])})"
    )
    print(
        f"Judge agreement (of {judged_total} judged): "
        f"YES: {judge_counts['yes']} ({jpct(judge_counts['yes'])})  |  "
        f"NO: {judge_counts['no']} ({jpct(judge_counts['no'])})  |  "
        f"UNCERTAIN: {judge_counts['uncertain']} ({jpct(judge_counts['uncertain'])})"
    )
    print()


def main() -> None:
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PATHS
    for path in paths:
        report_one(path)


if __name__ == "__main__":
    main()
