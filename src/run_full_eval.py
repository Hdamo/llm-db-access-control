"""
Full OrgAccess evaluation run: pulls a random sample of queries from the
respai-lab/orgaccess dataset, resolves EACH row to the user in
config/users.yaml whose role matches that row's own `user_role` field
(via src.role_resolution.RoleResolver -- fixes the earlier behavior of
running every query as one fixed persona), runs it through the normal
decompose -> access-check pipeline, then asks the same Ollama model to
judge whether our verdict agrees with OrgAccess's expected_response label
(via src.judge.judge_verdict_match).

Designed for long unattended runs:
  - Every row's result is appended to a JSONL checkpoint file immediately
    (one row = one line, flushed to disk), so a crash or interruption loses
    at most the in-flight row.
  - On restart, rows already present in the checkpoint file (by index) are
    skipped, so the same command resumes a partial run.
  - Progress (elapsed / ETA) is printed periodically to stderr.

The final Excel report is (re)built from the full checkpoint file, so it can
also be regenerated standalone with --report-only.

Usage:
    python -m src.run_full_eval --limit 1000 --seed 42 \\
        --splits easy medium hard \\
        --checkpoint reports/full_eval_checkpoint.jsonl \\
        --report reports/full_eval_report.xlsx

    # Skip the LLM-judge step (faster, decomposition only):
    python -m src.run_full_eval --limit 1000 --no-judge

    # Just rebuild the Excel report from an existing checkpoint file:
    python -m src.run_full_eval --report-only \\
        --checkpoint reports/full_eval_checkpoint.jsonl \\
        --report reports/full_eval_report.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import re
from concurrent.futures import ThreadPoolExecutor

from src import config as cfg
from src.access_check import PermissionStore
from src.decompose import DecompositionError
from src.judge import judge_verdict_match
from src.parallel_runner import LockedCheckpointWriter, run_rows_parallel
from src.pipeline import run_single
from src.role_resolution import RoleResolver

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


def _process_one(ex: dict, idx: int, model: str, do_judge: bool, resolver: RoleResolver, store: PermissionStore) -> dict:
    meta = ex.get("orgaccess_meta", {})
    dataset_role = meta.get("user_role", "") or ""
    user_key, resolved_role, match_method = resolver.resolve(dataset_role)
    expected = _extract_expected(meta)
    query = ex["query"]
    split = ex.get("split", "unknown")

    row: dict = {
        "idx": idx,
        "split": split,
        "dataset_user_role": dataset_role,
        "resolved_user": user_key,
        "resolved_role": resolved_role,
        "match_method": match_method,
        "query": query,
        "orgaccess_expected": expected,
        "verdict": None,
        "sub_queries": [],
        "judge_match": "",
        "judge_reasoning": "",
        "error": None,
    }

    try:
        result = run_single(user=user_key, query=query, model=model, store=store)
        row["verdict"] = result.verdict.value.upper()
        row["sub_queries"] = [
            {
                "text": sr.sub_query.text,
                "resource": sr.sub_query.resource,
                "decision": sr.decision.value,
                "reason": sr.reason,
            }
            for sr in result.sub_results
        ]
        if do_judge:
            match, reasoning = judge_verdict_match(
                our_verdict=row["verdict"], expected_label=expected,
                model=model, query=query,
            )
            row["judge_match"] = match
            row["judge_reasoning"] = reasoning
    except DecompositionError as e:
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
        print(f"[resume] {len(done)} rows already in checkpoint, will skip them.", file=sys.stderr)

    print(f"Fetching splits {splits} (limit={limit}, seed={seed}) ...", file=sys.stderr)
    try:
        examples = load_as_pipeline_examples_multi(
            splits=splits, limit=limit, test_user="__unused__", random_seed=seed,
        )
    except DatasetFetchError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    resolver = RoleResolver()
    store = PermissionStore()

    total = len(examples)
    pending = [(idx, ex) for idx, ex in enumerate(examples, start=1) if idx not in done]
    print(
        f"Loaded {total} examples ({len(pending)} remaining to process). "
        f"Model: {model}  workers={max_workers}",
        file=sys.stderr,
    )

    ckpt = LockedCheckpointWriter(checkpoint_path)
    start_time = time.time()
    progress_count = {"n": 0}

    def worker(_pos: int, item: tuple[int, dict]) -> dict:
        idx, ex = item
        row = _process_one(ex, idx, model, do_judge, resolver, store)
        ckpt.write_row(row)
        progress_count["n"] += 1
        n = progress_count["n"]
        if n % progress_every == 0 or n == len(pending):
            elapsed = time.time() - start_time
            rate = elapsed / n
            eta = rate * (len(pending) - n)
            print(
                f"[progress] {n}/{len(pending)} this run  "
                f"elapsed={_fmt_eta(elapsed)}  avg={rate:.1f}s/row  eta_remaining={_fmt_eta(eta)}",
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
    method_counts: dict[str, int] = {}

    for r in rows:
        v = r.get("verdict") or "ERROR"
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        j = (r.get("judge_match") or "").lower()
        judge_counts[j] = judge_counts.get(j, 0) + 1
        m = r.get("match_method", "unknown").split(":")[0]
        method_counts[m] = method_counts.get(m, 0) + 1

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
    method_line = "Role match method: " + "  |  ".join(
        f"{k}: {v} ({pct(v)})" for k, v in sorted(method_counts.items(), key=lambda x: -x[1])
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

    ws.append([cell("LLM-DB Phase 1 — Full OrgAccess Evaluation Report", font=bold_large)])
    ws.append([cell(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")])
    ws.append([cell(summary_line, font=bold)])
    ws.append([cell(judge_line, font=bold)])
    ws.append([cell(method_line)])
    ws.append([])

    headers = [
        "#", "Split", "Dataset Role", "Resolved User", "Resolved Role", "Match Method",
        "Query", "Verdict", "Sub-queries", "Allowed", "Denied", "Denied Resources",
        "OrgAccess Expected", "Judge Match", "Judge Reasoning", "Error",
    ]
    col_widths = [6, 9, 22, 24, 16, 14, 55, 10, 11, 9, 8, 30, 22, 11, 45, 30]
    ws.append([cell(h, font=bold) for h in headers])

    for r in rows:
        sub_queries = r.get("sub_queries") or []
        allowed = sum(1 for sq in sub_queries if sq["decision"] == "allow")
        denied = len(sub_queries) - allowed
        denied_resources = [sq["resource"] for sq in sub_queries if sq["decision"] == "deny"]
        denied_str = ", ".join(denied_resources[:3]) + (" …" if len(denied_resources) > 3 else "")

        ws.append([
            r["idx"],
            r.get("split", ""),
            r.get("dataset_user_role", ""),
            r.get("resolved_user", ""),
            r.get("resolved_role", ""),
            r.get("match_method", ""),
            (r.get("query") or "")[:300],
            r.get("verdict") or "ERROR",
            len(sub_queries),
            allowed,
            denied,
            denied_str,
            (r.get("orgaccess_expected") or "")[:60],
            r.get("judge_match", ""),
            (r.get("judge_reasoning") or "")[:300],
            (r.get("error") or "")[:200],
        ])

    wb.save(str(output_path))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Full OrgAccess evaluation run")
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
    parser.add_argument("--checkpoint", default="reports/full_eval_checkpoint.jsonl")
    parser.add_argument("--report", default="reports/full_eval_report.xlsx")
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

    def _run_one_model(model: str, ckpt: Path, rpt: Path) -> None:
        rows = run(
            splits=args.splits, limit=args.limit, seed=args.seed, model=model,
            checkpoint_path=ckpt, do_judge=not args.no_judge,
            max_workers=args.workers, progress_every=args.progress_every,
        )
        out = write_excel_report(rows, rpt)
        print(f"[{model}] Done. {len(rows)} rows processed. Report written to: {out}")

    if len(models) == 1:
        _run_one_model(models[0], checkpoint_path, report_path)
        return

    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = []
        for model in models:
            suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
            ckpt = checkpoint_path.with_name(f"{checkpoint_path.stem}.{suffix}{checkpoint_path.suffix}")
            rpt = report_path.with_name(f"{report_path.stem}.{suffix}{report_path.suffix}")
            futures.append(pool.submit(_run_one_model, model, ckpt, rpt))
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
