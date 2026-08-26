"""
Excel report generator for OrgAccess pipeline runs.

Produces a single-sheet .xlsx file that is suitable for large runs (up to the
full ~75k OrgAccess dataset) by using openpyxl's write-only streaming mode —
rows are written directly to disk without building an in-memory DOM.

Sheet layout
------------
Row 1   Title
Row 2   Run metadata (date, model, splits, seed)
Row 3   Verdict summary (Total / FULL / PARTIAL / REJECTED with percentages)
Row 4   Blank separator
Row 5   Column headers  (bold)
Row 6+  One data row per QueryResult

Columns
-------
A  #                  1-based index
B  Split              easy | medium | hard | unknown
C  User               user id
D  Role               resolved role
E  Query              first 120 chars of the original query
F  Verdict            FULL | PARTIAL | REJECTED
G  Sub-queries        total count of sub-queries
H  Allowed            count where decision == allow
I  Denied             count where decision == deny
J  Denied Resources   up to 3 denied resource ids, comma-joined; "…" appended if more
K  OrgAccess Expected expected_response / answer / label from orgaccess_meta (60 chars)
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

from src.access_check import Decision, QueryResult, Verdict

_HEADERS = [
    "#",
    "Split",
    "User",
    "Role",
    "Query",
    "Verdict",
    "Sub-queries",
    "Allowed",
    "Denied",
    "Denied Resources",
    "OrgAccess Expected",
]

# Pre-set column widths (characters) — write-only mode doesn't support
# auto-fit, so we use sensible fixed widths to keep the file readable.
_COL_WIDTHS = [6, 9, 26, 18, 42, 11, 13, 10, 10, 36, 32]

_MAX_QUERY_LEN = 120
_MAX_EXPECTED_LEN = 60
_MAX_DENIED_SHOWN = 3


def _verdict_label(v: Verdict) -> str:
    return v.value.upper()


def _extract_expected(meta: dict) -> str:
    for key in ("expected_response", "answer", "label", "response"):
        val = meta.get(key)
        if val and isinstance(val, str):
            return val[:_MAX_EXPECTED_LEN]
    return ""


def _denied_resources(result: QueryResult) -> str:
    denied = [
        sr.sub_query.resource
        for sr in result.sub_results
        if sr.decision == Decision.DENY
    ]
    if not denied:
        return ""
    shown = denied[:_MAX_DENIED_SHOWN]
    suffix = " \u2026" if len(denied) > _MAX_DENIED_SHOWN else ""
    return ", ".join(shown) + suffix


def generate_report(
    results: list[QueryResult],
    output_path: str | Path,
    *,
    splits_per_result: list[str],
    splits_used: list[str],
    model_used: str,
    random_seed: Optional[int] = None,
) -> Path:
    """Write an Excel report for a completed OrgAccess pipeline run.

    Args:
        results:           Ordered list of QueryResult objects from run_orgaccess().
        output_path:       Destination .xlsx file path (parent dirs created if needed).
        splits_per_result: Parallel list of split names, one per entry in results.
        splits_used:       Which splits were requested (for the metadata row).
        model_used:        Ollama model name (for the metadata row).
        random_seed:       Seed used for shuffling, or None (for the metadata row).

    Returns:
        The resolved Path of the written file.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.styles import Font
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required to generate reports. "
            "Run: pip install 'openpyxl>=3.1'"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- pre-compute stats (needed for summary rows written first) ---
    total = len(results)
    counts = {Verdict.FULL: 0, Verdict.PARTIAL: 0, Verdict.REJECTED: 0}
    for r in results:
        counts[r.verdict] += 1

    def _pct(n: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "n/a"

    summary_line = (
        f"Total: {total}  |  "
        f"FULL: {counts[Verdict.FULL]} ({_pct(counts[Verdict.FULL])})  |  "
        f"PARTIAL: {counts[Verdict.PARTIAL]} ({_pct(counts[Verdict.PARTIAL])})  |  "
        f"REJECTED: {counts[Verdict.REJECTED]} ({_pct(counts[Verdict.REJECTED])})"
    )

    seed_str = str(random_seed) if random_seed is not None else "none (non-deterministic)"
    meta_line = (
        f"Generated: {datetime.date.today()}  |  "
        f"Model: {model_used}  |  "
        f"Splits: {', '.join(splits_used)}  |  "
        f"Seed: {seed_str}"
    )

    # --- build workbook in write-only (streaming) mode ---
    wb = Workbook(write_only=True)
    ws = wb.create_sheet(title="Results")

    bold = Font(bold=True)
    bold_large = Font(bold=True, size=13)

    def _cell(value, font=None, number_format=None):
        c = WriteOnlyCell(ws, value=value)
        if font:
            c.font = font
        if number_format:
            c.number_format = number_format
        return c

    # Row 1: title
    ws.append([_cell("LLM-DB Phase 1 \u2014 OrgAccess Evaluation Report", font=bold_large)])

    # Row 2: metadata
    ws.append([_cell(meta_line)])

    # Row 3: verdict summary
    ws.append([_cell(summary_line, font=bold)])

    # Row 4: blank
    ws.append([])

    # Row 5: column headers
    ws.append([_cell(h, font=bold) for h in _HEADERS])

    # Row 6+: data rows (plain cells — no formatting for speed/size)
    for idx, (result, split) in enumerate(zip(results, splits_per_result), start=1):
        allowed_count = sum(
            1 for sr in result.sub_results if sr.decision == Decision.ALLOW
        )
        denied_count = len(result.sub_results) - allowed_count
        meta = {}
        # orgaccess_meta is not on QueryResult — we only have what run_orgaccess stored.
        # pipeline.py passes orgaccess_meta separately via splits_per_result parallel list;
        # here we only have the split name. Expected response must be passed if needed.
        # (see generate_report_with_meta for the richer variant)
        ws.append([
            idx,
            split,
            result.user,
            result.role,
            result.original_query[:_MAX_QUERY_LEN],
            _verdict_label(result.verdict),
            len(result.sub_results),
            allowed_count,
            denied_count,
            _denied_resources(result),
            "",  # OrgAccess Expected — populated by generate_report_with_meta
        ])

    wb.save(str(output_path))
    return output_path


def generate_report_with_meta(
    results: list[QueryResult],
    output_path: str | Path,
    *,
    splits_per_result: list[str],
    meta_per_result: list[dict],
    splits_used: list[str],
    model_used: str,
    random_seed: Optional[int] = None,
) -> Path:
    """Like generate_report() but also writes the OrgAccess expected-response column.

    Args:
        meta_per_result: Parallel list of orgaccess_meta dicts, one per result.
                         Pass an empty dict for any entry without metadata.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.styles import Font
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required to generate reports. "
            "Run: pip install 'openpyxl>=3.1'"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    counts = {Verdict.FULL: 0, Verdict.PARTIAL: 0, Verdict.REJECTED: 0}
    for r in results:
        counts[r.verdict] += 1

    def _pct(n: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "n/a"

    summary_line = (
        f"Total: {total}  |  "
        f"FULL: {counts[Verdict.FULL]} ({_pct(counts[Verdict.FULL])})  |  "
        f"PARTIAL: {counts[Verdict.PARTIAL]} ({_pct(counts[Verdict.PARTIAL])})  |  "
        f"REJECTED: {counts[Verdict.REJECTED]} ({_pct(counts[Verdict.REJECTED])})"
    )

    seed_str = str(random_seed) if random_seed is not None else "none (non-deterministic)"
    meta_line = (
        f"Generated: {datetime.date.today()}  |  "
        f"Model: {model_used}  |  "
        f"Splits: {', '.join(splits_used)}  |  "
        f"Seed: {seed_str}"
    )

    wb = Workbook(write_only=True)
    ws = wb.create_sheet(title="Results")

    bold = Font(bold=True)
    bold_large = Font(bold=True, size=13)

    def _cell(value, font=None):
        c = WriteOnlyCell(ws, value=value)
        if font:
            c.font = font
        return c

    ws.append([_cell("LLM-DB Phase 1 \u2014 OrgAccess Evaluation Report", font=bold_large)])
    ws.append([_cell(meta_line)])
    ws.append([_cell(summary_line, font=bold)])
    ws.append([])
    ws.append([_cell(h, font=bold) for h in _HEADERS])

    for idx, (result, split, orgmeta) in enumerate(
        zip(results, splits_per_result, meta_per_result), start=1
    ):
        allowed_count = sum(
            1 for sr in result.sub_results if sr.decision == Decision.ALLOW
        )
        denied_count = len(result.sub_results) - allowed_count
        ws.append([
            idx,
            split,
            result.user,
            result.role,
            result.original_query[:_MAX_QUERY_LEN],
            _verdict_label(result.verdict),
            len(result.sub_results),
            allowed_count,
            denied_count,
            _denied_resources(result),
            _extract_expected(orgmeta),
        ])

    wb.save(str(output_path))
    return output_path
