"""
Phase 1 pipeline: natural-language query -> Ollama decomposition ->
flat role-permission check -> allow/deny/partial per sub-query.

Modes (set via .env or LLMDB_MODE env var, overridable with CLI flags):

  mock      — offline demo using canned decompositions from
              examples/sample_queries.json. No Ollama required.
  live      — call local Ollama for real decompositions, using
              examples/sample_queries.json as the query set.
  orgaccess — pull test queries from HuggingFace respai-lab/orgaccess and
              run each through the flat role-permission pipeline (Ollama
              required). Phase 1.

  orgaccess_attr — pull test queries from respai-lab/orgaccess and check
              each one directly against ITS OWN permissions dict (ABAC-style
              attribute checking, see src/attribute_extract.py +
              src/attribute_check.py + config/attribute_rules.yaml).
              Phase 2. Independent of the flat role config entirely --
              Phase 1 keeps working unmodified alongside this.

  baseline  — pull test queries from respai-lab/orgaccess and ask the LLM
              for ONE holistic FULL/PARTIAL/REJECTED verdict per row, given
              the same permissions dict Phase 2 sees but with no
              decomposition/checklist and no deterministic aggregation step
              (see src/baseline_check.py). The floor Phase 1/2 are measured
              against -- rung zero of the progressive evaluation ladder in
              docs/roadmap.md.

Quick-start examples:
    # Single query (always calls Ollama):
    python -m src.pipeline --user daniel.finance_manager \\
        --query "Show me Sales headcount and this quarter's budget"

    # Use mode from .env (or LLMDB_MODE env var):
    python -m src.pipeline

    # Explicit mode overrides:
    python -m src.pipeline --mock
    python -m src.pipeline --live
    python -m src.pipeline --orgaccess

    # OrgAccess: multiple splits, random sample of 500, reproducible, with Excel report:
    python -m src.pipeline --orgaccess --splits easy medium hard \\
        --limit 500 --seed 42 --report reports/run.xlsx

    # Print effective config (reads .env) and exit:
    python -m src.pipeline --show-config
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src import config as cfg
from src.decompose import decompose_query, DecompositionResult, DecompositionError
from src.access_check import check_query, PermissionStore, QueryResult, Verdict
from src.attribute_extract import extract_attributes, parse_permissions, AttributeExtractionError
from src.attribute_check import check_attributes, AttributeRules, AttributeQueryResult, AttributeCoverageError
from src.baseline_check import get_baseline_verdict, BaselineResult, BaselineExtractionError

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = ROOT / "examples" / "sample_queries.json"


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def run_single(
    user: str,
    query: str,
    model: str = "",
    store: Optional[PermissionStore] = None,
    mock_decomposition: Optional[dict] = None,
) -> QueryResult:
    """Run one query through the full pipeline.

    If mock_decomposition is given, skip Ollama and validate that dict
    directly into a DecompositionResult. Otherwise call the real decomposer.
    """
    if not model:
        model = cfg.get_model()
    if mock_decomposition is not None:
        decomposition = DecompositionResult.model_validate(mock_decomposition)
    else:
        decomposition = decompose_query(query, model=model)
    return check_query(user=user, original_query=query, decomposition=decomposition, store=store)


def print_result(result: QueryResult) -> None:
    verdict_label = {
        Verdict.FULL: "FULL",
        Verdict.PARTIAL: "PARTIAL",
        Verdict.REJECTED: "REJECTED",
    }[result.verdict]

    print(f"\n{'=' * 70}")
    print(f"User:    {result.user}  (role: {result.role})")
    print(f"Query:   {result.original_query}")
    print(f"Verdict: {verdict_label}")
    print("-" * 70)
    for i, sr in enumerate(result.sub_results, 1):
        tag = "[ALLOW]" if sr.decision.value == "allow" else "[DENY] "
        print(f"  {i}. {tag} \"{sr.sub_query.text}\"")
        print(f"       resource: {sr.sub_query.resource}")
        print(f"       reason:   {sr.reason}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Demo mode (examples/sample_queries.json)
# ---------------------------------------------------------------------------

def run_demo(model: str, mock: bool) -> list[QueryResult]:
    """Run every entry in examples/sample_queries.json through the pipeline."""
    with open(EXAMPLES_PATH) as f:
        examples = json.load(f)

    store = PermissionStore()
    results = []
    for ex in examples:
        mock_decomp = ex.get("mock_decomposition") if mock else None
        if mock and mock_decomp is None:
            print(
                f"[skip] no mock_decomposition for: {ex['query']!r}",
                file=sys.stderr,
            )
            continue
        try:
            result = run_single(
                user=ex["user"],
                query=ex["query"],
                model=model,
                store=store,
                mock_decomposition=mock_decomp,
            )
        except DecompositionError as e:
            print(f"[error] decomposition failed for {ex['query']!r}: {e}", file=sys.stderr)
            continue
        print_result(result)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# OrgAccess mode (live HuggingFace dataset, multi-split, random sampling)
# ---------------------------------------------------------------------------

def run_orgaccess(
    splits: list[str],
    limit: Optional[int],
    test_user: str,
    model: str,
    random_seed: Optional[int] = None,
    report_path: Optional[str] = None,
) -> list[QueryResult]:
    """Pull examples from respai-lab/orgaccess on HuggingFace and run each
    through the pipeline.

    Rows are fetched from all requested splits, pooled, shuffled, then capped
    to `limit` (if set). Each query is decomposed by Ollama using our closed
    resource vocabulary. Because OrgAccess's permission model differs from our
    table.field schema, most resources will be tagged UNKNOWN and end up
    REJECTED — that is expected for Phase 1.

    If report_path is given, an Excel report is written after all queries
    are processed (see src/report.py for format details).
    """
    from src.load_orgaccess import load_as_pipeline_examples_multi, DatasetFetchError

    splits_label = ", ".join(splits)
    limit_label = str(limit) if limit is not None else "none (full dataset)"
    seed_label = str(random_seed) if random_seed is not None else "none (non-deterministic)"
    print(
        f"Fetching from splits [{splits_label}]  "
        f"limit={limit_label}  seed={seed_label} ..."
    )

    try:
        examples = load_as_pipeline_examples_multi(
            splits=splits,
            limit=limit,
            test_user=test_user,
            random_seed=random_seed,
        )
    except DatasetFetchError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if not examples:
        print("[warn] No examples loaded — check splits/limit settings.", file=sys.stderr)
        return []

    print(
        f"Loaded {len(examples)} examples. "
        f"Running as user={test_user!r} ...\n"
    )

    store = PermissionStore()
    results: list[QueryResult] = []
    splits_per_result: list[str] = []
    meta_per_result: list[dict] = []

    for ex in examples:
        try:
            result = run_single(
                user=ex["user"],
                query=ex["query"],
                model=model,
                store=store,
            )
        except DecompositionError as e:
            print(f"[error] decomposition failed for {ex['query']!r}: {e}", file=sys.stderr)
            continue

        print_result(result)

        # Print OrgAccess expected response for reference
        orgmeta = ex.get("orgaccess_meta", {})
        expected = (
            orgmeta.get("expected_response")
            or orgmeta.get("answer")
            or orgmeta.get("label")
        )
        if expected:
            print(f"  [orgaccess expected: {str(expected)!r}]")

        results.append(result)
        splits_per_result.append(ex.get("split", "unknown"))
        meta_per_result.append(orgmeta)

    # --- Excel report ---
    if report_path and results:
        try:
            from src.report import generate_report_with_meta
        except ImportError as e:
            print(f"[warn] Could not import report module: {e}", file=sys.stderr)
        else:
            try:
                out = generate_report_with_meta(
                    results,
                    report_path,
                    splits_per_result=splits_per_result,
                    meta_per_result=meta_per_result,
                    splits_used=splits,
                    model_used=model,
                    random_seed=random_seed,
                )
                print(f"\nReport written to: {out}")
            except Exception as e:
                print(f"[error] Failed to write report: {e}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# OrgAccess-attribute mode (Phase 2 — ABAC checking against each row's OWN
# permissions dict; no PermissionStore, no role_resolution involved)
# ---------------------------------------------------------------------------

def print_attribute_result(result: AttributeQueryResult, expected: str = "") -> None:
    verdict_label = {
        Verdict.FULL: "FULL",
        Verdict.PARTIAL: "PARTIAL",
        Verdict.REJECTED: "REJECTED",
    }[result.verdict]

    print(f"\n{'=' * 70}")
    print(f"Role:    {result.user_role}")
    print(f"Query:   {result.original_query}")
    print(f"Verdict: {verdict_label}" + (f"   [orgaccess expected: {expected!r}]" if expected else ""))
    print("-" * 70)
    for i, sr in enumerate(result.sub_results, 1):
        tag = "[ALLOW]" if sr.decision.value == "allow" else "[DENY] "
        applic = "" if sr.field_check.applicable else " (not applicable)"
        print(f"  {i}. {tag} field={sr.field_check.field}{applic}  severity={sr.severity}")
        print(f"       reason: {sr.reason}")
    print("=" * 70)


def run_orgaccess_attr(
    splits: list[str],
    limit: Optional[int],
    model: str,
    random_seed: Optional[int] = None,
    rules: Optional[AttributeRules] = None,
) -> list[AttributeQueryResult]:
    """Pull examples from respai-lab/orgaccess and check each one against
    ITS OWN `permissions` dict (Phase 2, ABAC-style), instead of resolving
    the dataset's `user_role` to one of Phase 1's 24 flat roles. Each row
    already carries the exact permissions ground truth for that role, so
    checking against it directly avoids the lossy role-collapsing that
    src/role_resolution.py has to do for Phase 1's flat checker.
    """
    from src.load_orgaccess import load_as_pipeline_examples_multi, DatasetFetchError

    splits_label = ", ".join(splits)
    limit_label = str(limit) if limit is not None else "none (full dataset)"
    seed_label = str(random_seed) if random_seed is not None else "none (non-deterministic)"
    print(
        f"[orgaccess_attr] Fetching from splits [{splits_label}]  "
        f"limit={limit_label}  seed={seed_label} ..."
    )

    try:
        examples = load_as_pipeline_examples_multi(
            splits=splits, limit=limit, test_user="__unused__", random_seed=random_seed,
        )
    except DatasetFetchError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if not examples:
        print("[warn] No examples loaded — check splits/limit settings.", file=sys.stderr)
        return []

    print(f"Loaded {len(examples)} examples.\n")

    rules = rules or AttributeRules()
    results: list[AttributeQueryResult] = []

    for ex in examples:
        meta = ex.get("orgaccess_meta", {})
        role = meta.get("user_role", "") or "unknown"
        permissions = parse_permissions(meta.get("permissions"))
        expected = meta.get("expected_response", "") or ""

        try:
            assessment = extract_attributes(ex["query"], permissions, model=model)
            result = check_attributes(
                user_role=role, original_query=ex["query"], permissions=permissions,
                assessment=assessment, rules=rules,
            )
        except (AttributeExtractionError, AttributeCoverageError) as e:
            print(f"[error] attribute check failed for {ex['query']!r}: {e}", file=sys.stderr)
            continue

        print_attribute_result(result, expected=expected)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Baseline mode -- rung zero of the progressive ladder: one LLM call, one
# verdict, no decomposition, no deterministic aggregation. See
# src/baseline_check.py and docs/baseline/plan.md.
# ---------------------------------------------------------------------------

def print_baseline_result(result: BaselineResult, expected: str = "") -> None:
    verdict_label = {
        Verdict.FULL: "FULL",
        Verdict.PARTIAL: "PARTIAL",
        Verdict.REJECTED: "REJECTED",
    }[result.verdict]

    print(f"\n{'=' * 70}")
    print(f"Role:    {result.user_role}")
    print(f"Query:   {result.original_query}")
    print(f"Verdict: {verdict_label}" + (f"   [orgaccess expected: {expected!r}]" if expected else ""))
    print("-" * 70)
    print(f"  reasoning: {result.reasoning}")
    print("=" * 70)


def run_baseline(
    splits: list[str],
    limit: Optional[int],
    model: str,
    random_seed: Optional[int] = None,
) -> list[BaselineResult]:
    """Pull examples from respai-lab/orgaccess and ask the LLM for one
    holistic verdict per row, given that row's own `permissions` dict --
    same input data as `run_orgaccess_attr`, but no per-field checklist and
    no separate deterministic aggregation step. Exists to measure how much
    of Phase 2's accuracy actually comes from the checklist+aggregation
    structure versus just handing the model the same information directly.
    """
    from src.load_orgaccess import load_as_pipeline_examples_multi, DatasetFetchError

    splits_label = ", ".join(splits)
    limit_label = str(limit) if limit is not None else "none (full dataset)"
    seed_label = str(random_seed) if random_seed is not None else "none (non-deterministic)"
    print(
        f"[baseline] Fetching from splits [{splits_label}]  "
        f"limit={limit_label}  seed={seed_label} ..."
    )

    try:
        examples = load_as_pipeline_examples_multi(
            splits=splits, limit=limit, test_user="__unused__", random_seed=random_seed,
        )
    except DatasetFetchError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if not examples:
        print("[warn] No examples loaded — check splits/limit settings.", file=sys.stderr)
        return []

    print(f"Loaded {len(examples)} examples.\n")

    results: list[BaselineResult] = []

    for ex in examples:
        meta = ex.get("orgaccess_meta", {})
        role = meta.get("user_role", "") or "unknown"
        permissions = parse_permissions(meta.get("permissions"))
        expected = meta.get("expected_response", "") or ""

        try:
            result = get_baseline_verdict(ex["query"], permissions, user_role=role, model=model)
        except BaselineExtractionError as e:
            print(f"[error] baseline check failed for {ex['query']!r}: {e}", file=sys.stderr)
            continue

        print_baseline_result(result, expected=expected)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-DB Phase 1 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes (in decreasing priority):
  Single query    --user + --query                                   always calls Ollama
  Explicit flags  --mock / --live / --orgaccess / --orgaccess-attr / --baseline
  .env / env var  LLMDB_MODE=mock|live|orgaccess|orgaccess_attr|baseline  (default: mock)

OrgAccess settings (CLI overrides .env):
  --splits, --limit, --seed, --report, --user (also the orgaccess test user)
        """,
    )

    # Single-query args
    parser.add_argument("--user", help="user id from config/users.yaml")
    parser.add_argument("--query", help="natural-language query string")

    # Mode flags (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mock", action="store_true",
        help="run examples/sample_queries.json using canned decompositions (no Ollama)",
    )
    mode_group.add_argument(
        "--live", action="store_true",
        help="run examples/sample_queries.json with real Ollama decompositions",
    )
    mode_group.add_argument(
        "--orgaccess", action="store_true",
        help="pull test queries from respai-lab/orgaccess on HuggingFace (Phase 1, flat role check)",
    )
    mode_group.add_argument(
        "--orgaccess-attr", action="store_true",
        help="pull test queries from respai-lab/orgaccess and check each against its OWN "
             "permissions dict (Phase 2, ABAC attribute check)",
    )
    mode_group.add_argument(
        "--baseline", action="store_true",
        help="pull test queries from respai-lab/orgaccess and ask the LLM for one holistic "
             "verdict per row, no decomposition/checklist, no deterministic aggregation "
             "(rung zero of the ladder, see docs/roadmap.md)",
    )

    # Shared options
    parser.add_argument(
        "--model", default="",
        help="Ollama model name (overrides LLMDB_OLLAMA_MODEL in .env)",
    )

    # OrgAccess-specific options
    parser.add_argument(
        "--splits", nargs="+", choices=("easy", "medium", "hard"),
        metavar="SPLIT",
        help="one or more splits to fetch: easy medium hard "
             "(overrides LLMDB_ORGACCESS_SPLITS in .env)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="total rows to sample across all splits "
             "(overrides LLMDB_ORGACCESS_LIMIT in .env; omit for full dataset)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for cross-split shuffling "
             "(overrides LLMDB_ORGACCESS_RANDOM_SEED in .env)",
    )
    parser.add_argument(
        "--report", default="",
        help="path for Excel report output, e.g. reports/run.xlsx "
             "(overrides LLMDB_REPORT_PATH in .env)",
    )

    parser.add_argument(
        "--show-config", action="store_true",
        help="print the effective configuration (from .env + env vars) and exit",
    )

    args = parser.parse_args()

    # --show-config: dump effective settings and exit
    if args.show_config:
        import pprint
        print("Effective configuration (from .env + environment variables):")
        pprint.pprint(cfg.dump())
        return

    # Resolve model (CLI > .env > hardcoded default)
    model = args.model or cfg.get_model()

    # -----------------------------------------------------------------------
    # Single-query mode (--user + --query always take priority)
    # -----------------------------------------------------------------------
    if args.user and args.query:
        result = run_single(user=args.user, query=args.query, model=model)
        print_result(result)
        return

    # -----------------------------------------------------------------------
    # Determine batch mode: CLI flag > .env LLMDB_MODE > default (mock)
    # -----------------------------------------------------------------------
    if args.mock:
        active_mode = "mock"
    elif args.live:
        active_mode = "live"
    elif args.orgaccess:
        active_mode = "orgaccess"
    elif args.orgaccess_attr:
        active_mode = "orgaccess_attr"
    elif args.baseline:
        active_mode = "baseline"
    else:
        active_mode = cfg.get_mode()

    if active_mode == "mock":
        run_demo(model=model, mock=True)

    elif active_mode == "live":
        run_demo(model=model, mock=False)

    elif active_mode == "orgaccess":
        # Resolve each OrgAccess setting: CLI > .env
        splits = args.splits or cfg.get_orgaccess_splits()
        limit = args.limit if args.limit is not None else cfg.get_orgaccess_limit()
        seed = args.seed if args.seed is not None else cfg.get_orgaccess_random_seed()
        report = args.report or cfg.get_report_path()
        test_user = args.user or cfg.get_orgaccess_user()

        run_orgaccess(
            splits=splits,
            limit=limit,
            test_user=test_user,
            model=model,
            random_seed=seed,
            report_path=report,
        )

    elif active_mode == "orgaccess_attr":
        splits = args.splits or cfg.get_orgaccess_splits()
        limit = args.limit if args.limit is not None else cfg.get_orgaccess_limit()
        seed = args.seed if args.seed is not None else cfg.get_orgaccess_random_seed()

        run_orgaccess_attr(splits=splits, limit=limit, model=model, random_seed=seed)

    elif active_mode == "baseline":
        splits = args.splits or cfg.get_orgaccess_splits()
        limit = args.limit if args.limit is not None else cfg.get_orgaccess_limit()
        seed = args.seed if args.seed is not None else cfg.get_orgaccess_random_seed()

        run_baseline(splits=splits, limit=limit, model=model, random_seed=seed)

    else:
        parser.error(
            f"Unknown mode {active_mode!r}. Set LLMDB_MODE to mock, live, orgaccess, "
            "orgaccess_attr, or baseline."
        )


if __name__ == "__main__":
    main()
