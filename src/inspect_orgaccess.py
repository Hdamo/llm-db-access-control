"""
Utility: scan the OrgAccess dataset and report all unique field values.

Fetches rows from respai-lab/orgaccess, walks every field in every row,
and collects:
  - All top-level column names and their value types
  - For string columns: every unique value and its frequency
  - For dict columns (e.g. the permissions object): every unique key, and
    for each key every unique value with its frequency
  - For list columns: every unique element with its frequency
  - For numeric columns: min / max / mean

This is a read-only analysis tool — it writes nothing to config/ or src/.
Output goes to stdout; use --save to also write JSON to data/.

Usage
-----
    # Quick look at 500 rows from the easy split:
    python -m src.inspect_orgaccess --splits easy --limit 500

    # Full scan of all splits (slow, ~75k rows):
    python -m src.inspect_orgaccess --splits easy medium hard

    # Save the structured report to data/orgaccess_schema.json:
    python -m src.inspect_orgaccess --splits easy medium hard --save

    # Only show fields with fewer than 50 unique values (skip free-text columns):
    python -m src.inspect_orgaccess --splits easy --max-unique 50
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("easy", "medium", "hard")

# -----------------------------------------------------------------------
# Scanning logic
# -----------------------------------------------------------------------

class FieldStats:
    """Accumulates statistics about one field across all rows."""

    def __init__(self, name: str):
        self.name = name
        self.total = 0
        self.null_count = 0
        self.type_counts: dict[str, int] = defaultdict(int)
        # for str / scalar values
        self.value_counts: dict[str, int] = defaultdict(int)
        # for dict values — key -> FieldStats
        self.sub_fields: dict[str, "FieldStats"] = {}
        # for numeric values
        self._nums: list[float] = []

    def observe(self, value: Any) -> None:
        import json as _json
        self.total += 1
        if value is None:
            self.null_count += 1
            self.type_counts["null"] += 1
            return

        # Auto-detect JSON-encoded strings and parse them so we can drill in
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0] in ('{', '['):
                try:
                    parsed = _json.loads(stripped)
                    value = parsed
                except _json.JSONDecodeError:
                    pass

        t = type(value).__name__
        self.type_counts[t] += 1

        if isinstance(value, dict):
            for k, v in value.items():
                if k not in self.sub_fields:
                    self.sub_fields[k] = FieldStats(k)
                self.sub_fields[k].observe(v)

        elif isinstance(value, list):
            for item in value:
                if item is not None:
                    self.value_counts[str(item)] += 1

        elif isinstance(value, (int, float)):
            self._nums.append(float(value))
            self.value_counts[str(value)] += 1

        else:
            self.value_counts[str(value)] += 1

    @property
    def dominant_type(self) -> str:
        if not self.type_counts:
            return "unknown"
        return max(self.type_counts, key=self.type_counts.__getitem__)

    @property
    def unique_count(self) -> int:
        return len(self.value_counts)

    def num_stats(self) -> Optional[dict]:
        if not self._nums:
            return None
        return {
            "min": min(self._nums),
            "max": max(self._nums),
            "mean": round(sum(self._nums) / len(self._nums), 4),
        }

    def to_dict(self) -> dict:
        d: dict = {
            "total_seen": self.total,
            "null_count": self.null_count,
            "types": dict(self.type_counts),
            "dominant_type": self.dominant_type,
            "unique_value_count": self.unique_count,
        }
        if self.value_counts:
            d["value_counts"] = dict(
                sorted(self.value_counts.items(), key=lambda x: -x[1])
            )
        ns = self.num_stats()
        if ns:
            d["numeric_stats"] = ns
        if self.sub_fields:
            d["sub_fields"] = {k: v.to_dict() for k, v in sorted(self.sub_fields.items())}
        return d


def scan_rows(rows: list[dict]) -> dict[str, FieldStats]:
    """Walk all rows and accumulate FieldStats per top-level column."""
    stats: dict[str, FieldStats] = {}
    for row in rows:
        for key, value in row.items():
            if key not in stats:
                stats[key] = FieldStats(key)
            stats[key].observe(value)
    return stats


# -----------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------

def _bar(count: int, total: int, width: int = 20) -> str:
    if total == 0:
        return " " * width
    filled = round(count / total * width)
    return "#" * filled + "." * (width - filled)


def print_report(
    stats: dict[str, FieldStats],
    total_rows: int,
    splits_scanned: list[str],
    max_unique: Optional[int] = None,
) -> None:
    sep = "=" * 72
    thin = "-" * 72

    print(f"\n{sep}")
    print(f"  OrgAccess Dataset Schema Inspector")
    print(f"  Splits scanned : {', '.join(splits_scanned)}")
    print(f"  Total rows     : {total_rows:,}")
    print(f"  Top-level cols : {len(stats)}")
    print(sep)

    for field_name, fs in sorted(stats.items()):
        non_null = fs.total - fs.null_count
        null_pct = f"{fs.null_count / fs.total * 100:.1f}%" if fs.total else "n/a"
        print(f"\n  FIELD: {field_name!r}")
        print(f"    type      : {fs.dominant_type}  {dict(fs.type_counts)}")
        print(f"    non-null  : {non_null:,} / {fs.total:,}  (nulls: {null_pct})")

        ns = fs.num_stats()
        if ns:
            print(f"    numeric   : min={ns['min']}  max={ns['max']}  mean={ns['mean']}")

        # sub-fields (dict-valued columns like permissions)
        if fs.sub_fields:
            print(f"    sub-keys  : {len(fs.sub_fields)} unique keys")
            for sub_name, sub_fs in sorted(fs.sub_fields.items()):
                print(f"      [{sub_name}]  unique values: {sub_fs.unique_count}")
                if max_unique is None or sub_fs.unique_count <= max_unique:
                    for val, cnt in sorted(
                        sub_fs.value_counts.items(), key=lambda x: -x[1]
                    )[:30]:
                        pct = f"{cnt / non_null * 100:.1f}%"
                        bar = _bar(cnt, non_null)
                        print(f"         {bar} {pct:>6}  {cnt:>6}  {val!r}")
                else:
                    top5 = sorted(
                        sub_fs.value_counts.items(), key=lambda x: -x[1]
                    )[:5]
                    print(f"        (skipping — {sub_fs.unique_count} unique values, "
                          f"max-unique={max_unique})")
                    print(f"        top 5: "
                          + ", ".join(f"{v!r}({c})" for v, c in top5))

        # scalar / list-element values
        elif fs.value_counts:
            if max_unique is not None and fs.unique_count > max_unique:
                top5 = sorted(fs.value_counts.items(), key=lambda x: -x[1])[:5]
                print(f"    values    : {fs.unique_count} unique  "
                      f"(skipping full list — max-unique={max_unique})")
                print(f"    top 5     : "
                      + ", ".join(f"{v!r}({c})" for v, c in top5))
            else:
                print(f"    values    : {fs.unique_count} unique")
                for val, cnt in sorted(
                    fs.value_counts.items(), key=lambda x: -x[1]
                )[:50]:
                    pct = f"{cnt / non_null * 100:.1f}%"
                    bar = _bar(cnt, non_null)
                    print(f"      {bar} {pct:>6}  {cnt:>6}  {val!r}")
                if fs.unique_count > 50:
                    print(f"      ... and {fs.unique_count - 50} more unique values")

        print(thin)

    print()


def save_report(
    stats: dict[str, FieldStats],
    total_rows: int,
    splits_scanned: list[str],
    out_path: Path,
) -> None:
    payload = {
        "meta": {
            "total_rows": total_rows,
            "splits_scanned": splits_scanned,
        },
        "fields": {k: v.to_dict() for k, v in sorted(stats.items())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Schema report saved to: {out_path.relative_to(ROOT)}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect unique field values in the OrgAccess dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.inspect_orgaccess --splits easy --limit 500
  python -m src.inspect_orgaccess --splits easy medium hard --save
  python -m src.inspect_orgaccess --splits easy --max-unique 30
        """,
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=["easy"],
        metavar="SPLIT",
        help="which splits to scan: easy medium hard  (default: easy)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap rows per split (default: no cap — scans the full split)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="write the full structured report to data/orgaccess_schema.json",
    )
    parser.add_argument(
        "--max-unique", type=int, default=None, dest="max_unique",
        help="hide value lists for fields with more than N unique values "
             "(useful for suppressing free-text / query columns)",
    )
    args = parser.parse_args()

    # Import here so the module can be imported without datasets installed
    try:
        from src.load_orgaccess import fetch_split, DatasetFetchError
    except ImportError as e:
        print(f"[error] Could not import load_orgaccess: {e}", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict] = []
    for split in args.splits:
        print(f"Fetching '{split}' split (limit={args.limit}) ...")
        try:
            rows = fetch_split(split, limit=args.limit)
        except DatasetFetchError as e:
            print(f"  [error] {e}", file=sys.stderr)
            continue
        print(f"  {len(rows):,} rows loaded")
        all_rows.extend(rows)

    if not all_rows:
        print("No rows fetched — nothing to inspect.", file=sys.stderr)
        sys.exit(1)

    print(f"\nScanning {len(all_rows):,} rows ...")
    stats = scan_rows(all_rows)

    print_report(
        stats,
        total_rows=len(all_rows),
        splits_scanned=args.splits,
        max_unique=args.max_unique,
    )

    if args.save:
        out_path = ROOT / "data" / "orgaccess_schema.json"
        save_report(stats, total_rows=len(all_rows), splits_scanned=args.splits, out_path=out_path)


if __name__ == "__main__":
    main()
