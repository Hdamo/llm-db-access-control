"""
Optimize src/baseline_check.py's system prompt with GEPA (reflective prompt
evolution -- https://github.com/gepa-ai/gepa), then let the user feed the
winning prompt into the real run_full_eval_baseline.py harness via
LLMDB_BASELINE_SYSTEM_PROMPT_FILE to see if it actually moves accuracy.

Why a custom GEPAAdapter instead of GEPA's DefaultAdapter: DefaultAdapter
sends the same literal system-prompt text to every example in a batch, but
our prompt has a per-example {field_list} substitution (the user's
permissions -- see baseline_check._build_system_prompt). The adapter below
does that substitution itself before each call, so GEPA only ever evolves
the fixed instructional wording, never the permissions data.

Scoring is a fast, local, deterministic heuristic (ported from
src/judge.py's messy-label mapping) instead of an LLM-judge call -- GEPA
needs hundreds of scoring calls during optimization, and spending a second
LLM call per score would roughly double the runtime for no benefit at this
stage (the real judge is still used for the final comparison run).

Usage:
    python -m src.gepa_optimize_baseline
    python -m src.gepa_optimize_baseline --train-limit 4 --val-limit 2 --max-metric-calls 10  # smoke test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

from gepa.core.adapter import EvaluationBatch

from src.baseline_check import SYSTEM_PROMPT_INSTRUCTIONS, SYSTEM_PROMPT_JSON_FOOTER
from src.attribute_extract import parse_permissions

ROOT = Path(__file__).resolve().parent.parent

# GEPA's reflective rewrites are free text and could introduce stray `{`/`}`
# characters, so the per-example substitution below uses str.replace, never
# str.format, on this placeholder.
FIELD_LIST_PLACEHOLDER = "{field_list}"

# Appended to the seed candidate (GEPA-side only -- never written into
# baseline_check.py's production default) because a first smoke test showed
# the reflection model will otherwise bake one training example's literal
# permissions/query into the "instructions", treating {field_list} as filler
# to replace rather than a per-request substitution point. evaluate() below
# still guarantees the real per-row permissions reach the model even if this
# gets ignored -- this is a quality nudge, not a correctness requirement.
ANTI_HARDCODE_GUARD = (
    "\n\nIMPORTANT: the permissions block above belongs to the ONE user making "
    "THIS specific request -- it is different on every single request, not a "
    "fixed example. Keep the exact placeholder text \"{field_list}\" in that "
    "spot verbatim. Do not hard-code any specific department, permission "
    "value, example query, or example question/answer pair into these "
    "instructions -- that would make them wrong for every other request."
)


class GepaRow(TypedDict):
    query: str
    field_list: str
    expected: str


def _build_field_list(permissions: dict) -> str:
    """Mirrors baseline_check._build_system_prompt's field_list rendering exactly."""
    if not permissions:
        return "(none -- this user has no listed permissions; deny everything.)"
    return "\n".join(f"- {k}: {json.dumps(v)}" for k, v in permissions.items())


# ---------------------------------------------------------------------------
# Deterministic scorer -- ports src/judge.py's messy-label heuristic so GEPA
# doesn't need a slow LLM-judge call for every one of its scoring rollouts.
# ---------------------------------------------------------------------------

def _heuristic_label(text: str) -> str:
    t = (text or "").strip().lower()
    if t in ("full", "partial", "rejected"):
        return t
    if t == "r" or "reject" in t:
        return "rejected"
    if t == "p" or "partial" in t or t.startswith("approve_") or "_only" in t:
        return "partial"
    if t in ("allow", "accept", "complete") or "full" in t:
        return "full"
    return t  # unrecognized -- will simply fail to match either verdict, i.e. scored wrong


_VERDICT_JSON_RE = re.compile(r'"verdict"\s*:\s*"(full|partial|rejected)"', re.IGNORECASE)


def _extract_verdict(raw: str) -> Optional[str]:
    try:
        obj = json.loads(raw)
        v = str(obj.get("verdict", "")).strip().lower()
        if v in ("full", "partial", "rejected"):
            return v
    except Exception:
        pass
    m = _VERDICT_JSON_RE.search(raw or "")
    if m:
        return m.group(1).lower()
    low = (raw or "").lower()
    for v in ("full", "partial", "rejected"):
        if v in low:
            return v
    return None


class BaselinePromptAdapter:
    """GEPAAdapter for optimizing baseline_check.py's SYSTEM_PROMPT_INSTRUCTIONS.

    Structurally satisfies gepa.core.adapter.GEPAAdapter (a Protocol) via
    evaluate() + make_reflective_dataset() -- no need to subclass it.
    """

    # GEPA's engine reads this attribute directly (not via hasattr/getattr), so
    # it must exist even though we don't override instruction proposal --
    # GEPAAdapter is a Protocol, and its class-level default isn't inherited
    # unless a concrete class sets it itself.
    propose_new_texts = None

    def __init__(self, model: str):
        self.model = model  # litellm model string, e.g. "ollama_chat/qwen2.5:7b-instruct"

    def _call(self, system_text: str, query: str) -> str:
        import litellm

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                num_retries=1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[call error: {e}]"

    def evaluate(
        self,
        batch: list[GepaRow],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        instructions = candidate["system_prompt"]
        outputs: list[str] = []
        scores: list[float] = []
        trajectories: Optional[list[dict]] = [] if capture_traces else None

        for row in batch:
            body = instructions.replace(FIELD_LIST_PLACEHOLDER, row["field_list"])
            if FIELD_LIST_PLACEHOLDER not in instructions:
                # GEPA dropped the placeholder entirely while rewriting -- still
                # give the model the permissions, just appended at the end.
                body = f"{instructions}\n\nTHIS USER'S PERMISSIONS:\n{row['field_list']}"
            system_text = body + SYSTEM_PROMPT_JSON_FOOTER

            raw = self._call(system_text, row["query"])
            predicted = _extract_verdict(raw)
            expected = _heuristic_label(row["expected"])
            score = 1.0 if predicted is not None and predicted == expected else 0.0

            outputs.append(raw)
            scores.append(score)
            if capture_traces:
                trajectories.append(
                    {
                        "query": row["query"],
                        "field_list": row["field_list"],
                        "raw_output": raw,
                        "predicted": predicted,
                        "expected": expected,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        records = []
        for traj, score in zip(eval_batch.trajectories or [], eval_batch.scores):
            correct = score >= 1.0
            records.append(
                {
                    "Inputs": {"query": traj["query"], "permissions": traj["field_list"]},
                    "Generated Outputs": traj["raw_output"],
                    "Feedback": (
                        f"Predicted verdict: {traj['predicted']!r}. Expected: {traj['expected']!r}. "
                        + ("This was correct." if correct else "This was WRONG -- reconsider how the "
                           "permissions and query are weighed against each other.")
                    ),
                }
            )
        return {"system_prompt": records}


def _load_rows(splits: list[str], limit: int, seed: int) -> list[GepaRow]:
    from src.load_orgaccess import load_as_pipeline_examples_multi

    examples = load_as_pipeline_examples_multi(
        splits=splits, limit=limit, test_user="__unused__", random_seed=seed,
    )
    rows: list[GepaRow] = []
    for ex in examples:
        meta = ex.get("orgaccess_meta", {})
        permissions = parse_permissions(meta.get("permissions"))
        expected = ""
        for key in ("expected_response", "answer", "label", "response"):
            val = meta.get(key)
            if val and isinstance(val, str):
                expected = val.strip()
                break
        rows.append(
            GepaRow(query=ex["query"], field_list=_build_field_list(permissions), expected=expected)
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA-optimize the baseline system prompt")
    parser.add_argument("--splits", nargs="+", choices=("easy", "medium", "hard"),
                         default=["easy", "medium", "hard"])
    parser.add_argument("--train-limit", type=int, default=40)
    parser.add_argument("--val-limit", type=int, default=20)
    # Disjoint from the 10k cross-check runs (which used --seed 42), so GEPA's
    # own tuning sample doesn't overlap the rows used for the final judgment.
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="qwen2.5:7b-instruct",
                         help="Ollama model name (used for both the task and, by default, reflection)")
    parser.add_argument("--reflection-model", default="",
                         help="Ollama model name for reflection; defaults to --model")
    parser.add_argument("--max-metric-calls", type=int, default=200)
    parser.add_argument("--out-prompt", default="reports/gepa_best_baseline_prompt.txt")
    parser.add_argument("--out-summary", default="reports/gepa_best_baseline_summary.json")
    args = parser.parse_args()

    import gepa

    reflection_model = args.reflection_model.strip() or args.model
    task_lm = f"ollama_chat/{args.model}"
    reflection_lm = f"ollama_chat/{reflection_model}"

    print(f"Loading {args.train_limit} train + {args.val_limit} val rows "
          f"(splits={args.splits}, seed={args.seed}) ...", file=sys.stderr)
    train_rows = _load_rows(args.splits, args.train_limit, args.seed)
    # Different seed for val so train/val don't sample the exact same pool twice.
    val_rows = _load_rows(args.splits, args.val_limit, args.seed + 1)
    print(f"Loaded {len(train_rows)} train rows, {len(val_rows)} val rows.", file=sys.stderr)

    adapter = BaselinePromptAdapter(model=task_lm)
    seed_candidate = {"system_prompt": SYSTEM_PROMPT_INSTRUCTIONS + ANTI_HARDCODE_GUARD}

    start = time.time()
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_rows,
        valset=val_rows,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        seed=args.seed,
    )
    elapsed = time.time() - start

    best_idx = result.best_idx
    best_prompt = result.candidates[best_idx]["system_prompt"]
    best_score = result.val_aggregate_scores[best_idx]
    seed_score = result.val_aggregate_scores[0]

    out_prompt_path = ROOT / args.out_prompt
    out_summary_path = ROOT / args.out_summary
    out_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    out_prompt_path.write_text(best_prompt, encoding="utf-8")

    summary = {
        "model": args.model,
        "reflection_model": reflection_model,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "max_metric_calls": args.max_metric_calls,
        "total_metric_calls": result.total_metric_calls,
        "num_candidates_explored": result.num_candidates,
        "seed_val_score": seed_score,
        "best_val_score": best_score,
        "best_idx": best_idx,
        "elapsed_seconds": round(elapsed, 1),
        "out_prompt": str(out_prompt_path),
    }
    out_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if FIELD_LIST_PLACEHOLDER not in best_prompt:
        print(
            f"WARNING: the winning prompt dropped the {FIELD_LIST_PLACEHOLDER!r} placeholder -- "
            "real permissions will still be appended at the end of the system prompt for every "
            "row (see evaluate()'s fallback), but the prompt text itself may contain stale "
            "baked-in example data from the training rows. Inspect "
            f"{out_prompt_path} before trusting it at scale.",
            file=sys.stderr,
        )

    print(f"Done in {elapsed:.0f}s. seed val score={seed_score:.3f}  best val score={best_score:.3f}")
    print(f"Best prompt written to: {out_prompt_path}")
    print(f"Summary written to: {out_summary_path}")


if __name__ == "__main__":
    main()
