"""
Full OrgAccess evaluation run for Phase 2 (ABAC attribute checking).

Mirrors src/run_full_eval.py's proven scaffolding (crash-resumable JSONL
checkpointing, progress/ETA logging, Excel report, LLM-judge scoring via
src/judge.py for the ~39 messy expected_response label variants) but swaps
the per-row check from Phase 1's decompose+PermissionStore path to Phase 2's
attribute_extract+attribute_check path, checking each row directly against
ITS OWN `permissions` dict instead of resolving the dataset's `user_role`
onto one of the 24 flat roles in config/permissions.yaml. src.role_resolution
is intentionally not used here -- see docs/phase2/plan.md for why.

New in this script vs run_full_eval.py: --workers (process multiple rows
concurrently against Ollama) and --models (benchmark several models in one
command, each writing its own checkpoint/report). Both default to the old
fully-sequential, single-model behavior.

Usage:
    python -m src.run_full_eval_v2 --splits easy medium hard --limit 1000 \\
        --checkpoint reports/full_eval_v2_checkpoint.jsonl \\
        --report reports/full_eval_v2_report.xlsx

    # Concurrent rows against one model:
    python -m src.run_full_eval_v2 --limit 200 --workers 4

    # Benchmark multiple models side-by-side (each gets its own checkpoint/report):
    python -m src.run_full_eval_v2 --limit 200 --models qwen2.5:7b-instruct,qwen3.5:latest

    # Just rebuild the Excel report from an existing checkpoint file:
    python -m src.run_full_eval_v2 --report-only \\
        --checkpoint reports/full_eval_v2_checkpoint.jsonl \\
        --report reports/full_eval_v2_report.xlsx
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from src import config as cfg
from src.attribute_check import AttributeCoverageError, AttributeRules, check_attributes
from src.attribute_extract import AttributeExtractionError, extract_attributes, parse_permissions
from src.judge import judge_verdict_match
from src.parallel_runner import LockedCheckpointWriter, run_rows_parallel

ROOT = Path(__file__).resolve().parent.parent


def _extract_expected(meta: dict) -> str:
    for key in ("expected_response", "answer", "label", "response"):
        val = meta.get(key)
        if val and isinstance(val, str):
            return val.strip()
    return ""


def _load_checkpoint(path: Path) -> dict[int, dict]:
    done: dict[int, dict] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["idx"]] = row
    return done


def _fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def _process_one(ex: dict, idx: int, model: str, do_judge: bool, rules: AttributeRules) -> dict:
    meta = ex.get("orgaccess_meta", {})
    dataset_role = meta.get("user_role", "") or ""
    permissions = parse_permissions(meta.get("permissions"))
    expected = _extract_expected(meta)
    query = ex["query"]
    split = ex.get("split", "unknown")

    row: dict = {
        "idx": idx,
        "split": split,
        "dataset_user_role": dataset_role,
        "query": query,
        "orgaccess_expected": expected,
        "verdict": None,
        "field_checks": [],
        "judge_match": "",
        "judge_reasoning": "",
        "error": None,
        "model": model,
    }

    try:
        assessment = extract_attributes(query, permissions, model=model)
        result = check_attributes(
            user_role=dataset_role, original_query=query, permissions=permissions,
            assessment=assessment, rules=rules,
        )
        row["verdict"] = result.verdict.value.upper()
        row["field_checks"] = [
            {
                "field": sr.field_check.field,
                "applicable": sr.field_check.applicable,
                "decision": sr.decision.value,
                "severity": sr.severity,
                "reason": sr.reason,
            }
            for sr in result.sub_results
        ]
        if do_judge:
            match, reasoning = judge_verdict_match(
                our_verdict=row["verdict"], expected_label=expected, model=model, query=query,
            )
            row["judge_match"] = match
            row["judge_reasoning"] = reasoning
    except (AttributeExtractionError, AttributeCoverageError) as e:
        row["error"] = str(e)
    except Exception as e:
        row["error"] = f"unexpected error: {e}"

    return row


def run(
    splits: list[str],
    limit: Optional[int],
    seed: Optional[int],
    model: str,
    checkpoint_path: Path,
    do_judge: bool,
    max_workers: int = 1,
    progress_every: int = 5,
) -> list[dict]:
    from src.load_orgaccess import load_as_pipeline_examples_multi, DatasetFetchError

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(checkpoint_path)
    if done:
        print(f"[{model}] [resume] {len(done)} rows already in checkpoint, will skip them.", file=sys.stderr)

    print(f"[{model}] Fetching splits {splits} (limit={limit}, seed={seed}) ...", file=sys.stderr)
    try:
        examples = load_as_pipeline_examples_multi(
            splits=splits, limit=limit, test_user="__unused__", random_seed=seed,
        )
    except DatasetFetchError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    rules = AttributeRules()

    total = len(examples)
    pending = [(idx, ex) for idx, ex in enumerate(examples, start=1) if idx not in done]
    print(
        f"[{model}] Loaded {total} examples ({len(pending)} remaining, workers={max_workers}).",
        file=sys.stderr,
    )

    ckpt = LockedCheckpointWriter(checkpoint_path)
    start_time = time.time()
    progress_lock_count = {"n": 0}

    def worker(_pos: int, item: tuple[int, dict]) -> dict:
        idx, ex = item
        row = _process_one(ex, idx, model, do_judge, rules)
        ckpt.write_row(row)
        progress_lock_count["n"] += 1
        n = progress_lock_count["n"]
        if n % progress_every == 0 or n == len(pending):
            elapsed = time.time() - start_time
            rate = elapsed / n
            eta = rate * (len(pending) - n)
            print(
                f"[{model}] [progress] {n}/{len(pending)} this run  "
                f"elapsed={_fmt_eta(elapsed)}  avg={rate:.1f}s/row  eta={_fmt_eta(eta)}",
                file=sys.stderr,
            )
        return row

    try:
        new_rows = run_rows_parallel(pending, worker, max_workers=max_workers)
    finally:
        ckpt.close()

    for row in new_rows:
        done[row["idx"]] = row

    return [done[i] for i in sorted(done)]


def write_excel_report(rows: list[dict], output_path: Path) -> Path:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    verdict_counts = {"FULL": 0, "PARTIAL": 0, "REJECTED": 0, "ERROR": 0}
    judge_counts = {"yes": 0, "no": 0, "uncertain": 0, "": 0}
    # Rough error-category tagging (paper's taxonomy), inferred only from a
    # mismatch between our verdict and OrgAccess's expected label -- not a
    # full replication of the paper's own error-annotation methodology.
    error_category_counts: dict[str, int] = {}

    def infer_error_category(row: dict) -> Optional[str]:
        verdict = row.get("verdict") or "ERROR"
        expected = (row.get("orgaccess_expected") or "").strip().lower()
        if row.get("judge_match") == "yes":
            return None
        if verdict == "PARTIAL" and expected in ("full", "rejected"):
            return "False Partial"
        if verdict == "REJECTED" and expected == "partial":
            return "Scope/Constraint (over-rejected)"
        if verdict == "FULL" and expected in ("partial", "rejected"):
            return "Constraint (under-enforced)"
        if verdict != "ERROR" and expected and verdict.lower() != expected and row.get("judge_match") == "no":
            return "Conflict/Other"
        return None

    for r in rows:
        v = r.get("verdict") or "ERROR"
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        j = (r.get("judge_match") or "").lower()
        judge_counts[j] = judge_counts.get(j, 0) + 1
        cat = infer_error_category(r)
        if cat:
            error_category_counts[cat] = error_category_counts.get(cat, 0) + 1

    def pct(n: int) -> str:
        return f"{n/total*100:.1f}%" if total else "n/a"

    judged_total = total - judge_counts.get("", 0)

    def jpct(n: int) -> str:
        return f"{n/judged_total*100:.1f}%" if judged_total else "n/a"

    summary_line = (
        f"Total: {total}  |  "
        f"FULL: {verdict_counts['FULL']} ({pct(verdict_counts['FULL'])})  |  "
        f"PARTIAL: {verdict_counts['PARTIAL']} ({pct(verdict_counts['PARTIAL'])})  |  "
        f"REJECTED: {verdict_counts['REJECTED']} ({pct(verdict_counts['REJECTED'])})  |  "
        f"ERROR: {verdict_counts['ERROR']} ({pct(verdict_counts['ERROR'])})"
    )
    judge_line = (
        f"Judge agreement (of {judged_total} judged): "
        f"YES: {judge_counts['yes']} ({jpct(judge_counts['yes'])})  |  "
        f"NO: {judge_counts['no']} ({jpct(judge_counts['no'])})  |  "
        f"UNCERTAIN: {judge_counts['uncertain']} ({jpct(judge_counts['uncertain'])})"
    )
    error_line = "Inferred error categories: " + (
        "  |  ".join(f"{k}: {v} ({pct(v)})" for k, v in sorted(error_category_counts.items(), key=lambda x: -x[1]))
        or "none"
    )

    wb = Workbook(write_only=True)
    ws = wb.create_sheet(title="Results")
    bold = Font(bold=True)
    bold_large = Font(bold=True, size=13)

    def cell(value, font=None):
        c = WriteOnlyCell(ws, value=value)
        if font:
            c.font = font
        return c

    ws.append([cell("LLM-DB Phase 2 — Full OrgAccess ABAC Evaluation Report", font=bold_large)])
    ws.append([cell(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")])
    ws.append([cell(summary_line, font=bold)])
    ws.append([cell(judge_line, font=bold)])
    ws.append([cell(error_line)])
    ws.append([])

    headers = [
        "#", "Split", "Dataset Role", "Query", "Verdict", "Field Checks",
        "Denied Fields (core)", "Denied Fields (soft)", "OrgAccess Expected",
        "Judge Match", "Judge Reasoning", "Error",
    ]
    col_widths = [6, 9, 22, 55, 10, 12, 30, 30, 22, 11, 45, 30]
    ws.append([cell(h, font=bold) for h in headers])

    for r in rows:
        field_checks = r.get("field_checks") or []
        denied_core = [fc["field"] for fc in field_checks if fc["decision"] == "deny" and fc["severity"] == "core"]
        denied_soft = [fc["field"] for fc in field_checks if fc["decision"] == "deny" and fc["severity"] == "soft"]

        ws.append([
            r["idx"],
            r.get("split", ""),
            r.get("dataset_user_role", ""),
            (r.get("query") or "")[:300],
            r.get("verdict") or "ERROR",
            len(field_checks),
            ", ".join(denied_core[:5]) + (" …" if len(denied_core) > 5 else ""),
            ", ".join(denied_soft[:5]) + (" …" if len(denied_soft) > 5 else ""),
            (r.get("orgaccess_expected") or "")[:60],
            r.get("judge_match", ""),
            (r.get("judge_reasoning") or "")[:300],
            (r.get("error") or "")[:200],
        ])

    wb.save(str(output_path))
    return output_path


_SAFE_MODEL_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _model_suffix(model: str) -> str:
    return _SAFE_MODEL_RE.sub("_", model)


def _run_one_model(
    model: str, splits: list[str], limit: Optional[int], seed: Optional[int],
    checkpoint_path: Path, report_path: Path, do_judge: bool, max_workers: int, progress_every: int,
) -> Path:
    rows = run(
        splits=splits, limit=limit, seed=seed, model=model,
        checkpoint_path=checkpoint_path, do_judge=do_judge,
        max_workers=max_workers, progress_every=progress_every,
    )
    out = write_excel_report(rows, report_path)
    print(f"[{model}] Done. {len(rows)} rows processed. Report written to: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Full OrgAccess evaluation run (Phase 2 - ABAC)")
    parser.add_argument("--splits", nargs="+", choices=("easy", "medium", "hard"),
                         default=["easy", "medium", "hard"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="", help="single model (mutually exclusive with --models)")
    parser.add_argument("--models", default="",
                         help="comma-separated models to benchmark concurrently, "
                              "each writing its own checkpoint/report file")
    parser.add_argument("--workers", type=int, default=1,
                         help="rows to process concurrently against Ollama, per model (default: 1, sequential)")
    parser.add_argument("--checkpoint", default="reports/full_eval_v2_checkpoint.jsonl")
    parser.add_argument("--report", default="reports/full_eval_v2_report.xlsx")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                         help="skip processing; just rebuild the Excel report from the checkpoint file")
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else [args.model or cfg.get_model()]

    checkpoint_path = ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    report_path = ROOT / args.report if not Path(args.report).is_absolute() else Path(args.report)

    if args.report_only:
        done = _load_checkpoint(checkpoint_path)
        rows = [done[i] for i in sorted(done)]
        if not rows:
            print(f"[error] no rows found in {checkpoint_path}", file=sys.stderr)
            sys.exit(1)
        out = write_excel_report(rows, report_path)
        print(f"Report written to: {out}")
        return

    if len(models) == 1:
        _run_one_model(
            models[0], args.splits, args.limit, args.seed, checkpoint_path, report_path,
            not args.no_judge, args.workers, args.progress_every,
        )
        return

    # Multiple models: one full sweep per model, run concurrently, each with
    # its own checkpoint/report file suffixed by a sanitized model name.
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = []
        for model in models:
            suffix = _model_suffix(model)
            ckpt = checkpoint_path.with_name(f"{checkpoint_path.stem}.{suffix}{checkpoint_path.suffix}")
            rpt = report_path.with_name(f"{report_path.stem}.{suffix}{report_path.suffix}")
            futures.append(pool.submit(
                _run_one_model, model, args.splits, args.limit, args.seed, ckpt, rpt,
                not args.no_judge, args.workers, args.progress_every,
            ))
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
