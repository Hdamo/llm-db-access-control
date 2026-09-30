"""
Phase 3c: self-evolving refinement of config/attribute_rules.yaml.

Adapts the refinement loop from arXiv 2609.09153 ("Procedural Graphs:
Self-Evolving Execution Structures for LLM Agents") -- NOT its literal graph
structure (that paper is about LLM agent execution flow, unrelated to access
control) -- see docs/phase3/self-evolution.md for the full design and why
this is adapted rather than applied directly.

What this does, concretely:
  1. Loads the existing Phase 2 checkpoint (reports/full_eval_v2_checkpoint.jsonl),
     which already carries a `judge_match` field per row -- exactly the
     "success/failure trajectory" signal the paper's refiner needs, already
     computed, already free.
  2. Splits rows by a NEW convention specific to this script: `hard` is held
     out for validation, `easy`+`medium` pooled is the train/refinement-input
     split -- a deliberate departure from every other harness's pooled
     1,000-row sample (see self-evolution.md). Never conflate this script's
     accuracy numbers with the pooled Phase 1/2/Baseline baselines.
  3. Re-aggregates verdicts from each row's already-cached `field_checks`
     under a candidate severity mapping -- no new Ollama extraction calls at
     all, since attribute_check.py's aggregation is purely deterministic.
     Only rows whose verdict actually flips need a fresh judge call.
  4. Proposes a small batch of field severity flips per iteration (an LLM
     call, grounded in a deterministic per-field statistical summary, not
     raw rows) and commits a proposal only if it improves held-out (`hard`)
     accuracy without regressing train accuracy by more than an epsilon.
  5. Writes accepted changes into config/attribute_rules.yaml (only the
     `severity` lines for changed fields -- everything else, including the
     _PROVISIONAL banner, is left untouched) and a JSON report.

Usage:
    # Walking skeleton only -- hand-verify the known biometric_required
    # miscalibration, no LLM calls, no file writes:
    python -m src.refine_attribute_rules --manual-check-only

    # Full automated loop (dry run -- prints what WOULD change, writes nothing):
    python -m src.refine_attribute_rules --dry-run

    # Full automated loop, writes accepted changes to attribute_rules.yaml:
    python -m src.refine_attribute_rules
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ollama
from pydantic import BaseModel, ValidationError, field_validator

from src.attribute_check import AttributeRules
from src.judge import judge_verdict_match

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "reports" / "full_eval_v2_checkpoint.jsonl"
DEFAULT_RULES_PATH = ROOT / "config" / "attribute_rules.yaml"
DEFAULT_REPORT_PATH = ROOT / "reports" / "phase3c_refinement_report.json"

HELD_OUT_SPLIT = "hard"
TRAIN_SPLITS = ("easy", "medium")

REGRESSION_EPSILON = 0.005  # 0.5 percentage points, as a fraction

MAX_PREDICT_TOKENS = int(os.environ.get("LLMDB_REFINER_MAX_TOKENS", "512"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLMDB_OLLAMA_TIMEOUT", "240"))
# See attribute_extract.py's NUM_CTX comment: caps the KV cache the daemon
# preallocates per loaded model, which is the real driver of llama-server's
# oversized memory footprint, not per-request growth.
NUM_CTX = int(os.environ.get("LLMDB_OLLAMA_NUM_CTX", "4096"))
_client = ollama.Client(timeout=REQUEST_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Checkpoint loading / splitting
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_rows(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Returns (train_rows, held_out_rows). Rows with no verdict (errored
    during the original run) are excluded entirely -- there's nothing cached
    to re-aggregate for them."""
    train, held_out = [], []
    for r in rows:
        if r.get("verdict") is None or not r.get("field_checks"):
            continue
        if r.get("split") == HELD_OUT_SPLIT:
            held_out.append(r)
        elif r.get("split") in TRAIN_SPLITS:
            train.append(r)
    return train, held_out


# ---------------------------------------------------------------------------
# Candidate rules -- an in-memory override layer over the real AttributeRules,
# so a candidate severity map can be tested without ever writing to disk.
# ---------------------------------------------------------------------------

class CandidateRules:
    def __init__(self, base: AttributeRules, overrides: Dict[str, str]):
        self.base = base
        self.overrides = overrides

    def severity_for(self, field_name: str) -> str:
        if field_name in self.overrides:
            return self.overrides[field_name]
        return self.base.severity_for(field_name)


def recompute_verdict(field_checks: List[dict], rules) -> str:
    """Mirrors attribute_check.py's _aggregate() exactly, operating directly
    on the checkpoint's plain field_checks dicts instead of FieldCheckResult
    objects -- no re-extraction needed, severity is looked up fresh from
    `rules` for every denied field, same as the original aggregation did."""
    core_violation = False
    soft_violation = False
    for fc in field_checks:
        if fc.get("decision") != "deny":
            continue
        severity = rules.severity_for(fc["field"])
        if severity == "core":
            core_violation = True
        elif severity == "soft":
            soft_violation = True
    if core_violation:
        return "REJECTED"
    if soft_violation:
        return "PARTIAL"
    return "FULL"


# ---------------------------------------------------------------------------
# Evaluation: accuracy of a split under a candidate rules object
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    total: int
    judged: int
    yes: int
    accuracy: float  # yes / judged, 0.0 if judged == 0
    new_verdicts: Dict[int, str] = field(default_factory=dict)     # idx -> new verdict
    new_judge_matches: Dict[int, str] = field(default_factory=dict)  # idx -> judge_match
    flips: int = 0
    judge_calls: int = 0


def evaluate_split(rows: List[dict], rules, model: str, do_judge: bool = True) -> EvalResult:
    yes = 0
    judged = 0
    new_verdicts: Dict[int, str] = {}
    new_judge_matches: Dict[int, str] = {}
    flips = 0
    judge_calls = 0

    for r in rows:
        idx = r["idx"]
        old_verdict = r["verdict"]
        new_verdict = recompute_verdict(r["field_checks"], rules)
        new_verdicts[idx] = new_verdict

        if new_verdict == old_verdict:
            match = (r.get("judge_match") or "").lower()
        else:
            flips += 1
            if do_judge:
                match, _ = judge_verdict_match(
                    our_verdict=new_verdict,
                    expected_label=r.get("orgaccess_expected", ""),
                    model=model,
                    query=r.get("query", ""),
                )
                match = match.lower()
                judge_calls += 1
            else:
                match = ""

        new_judge_matches[idx] = match
        if match in ("yes", "no"):
            judged += 1
            if match == "yes":
                yes += 1

    accuracy = (yes / judged) if judged else 0.0
    return EvalResult(
        total=len(rows), judged=judged, yes=yes, accuracy=accuracy,
        new_verdicts=new_verdicts, new_judge_matches=new_judge_matches,
        flips=flips, judge_calls=judge_calls,
    )


# ---------------------------------------------------------------------------
# Walking skeleton: hand-verify the known biometric_required miscalibration
# (docs/phase2/attribute-schema.md's "Known concrete mismatch") before
# trusting the automated loop below at all.
# ---------------------------------------------------------------------------

def manual_biometric_required_check(held_out_rows: List[dict], base_rules: AttributeRules, model: str) -> dict:
    before = evaluate_split(held_out_rows, base_rules, model, do_judge=True)
    candidate = CandidateRules(base_rules, {"biometric_required": "soft"})
    after = evaluate_split(held_out_rows, candidate, model, do_judge=True)
    return {
        "field": "biometric_required",
        "held_out_n": before.total,
        "before_accuracy": round(before.accuracy * 100, 1),
        "after_accuracy": round(after.accuracy * 100, 1),
        "rows_flipped": after.flips,
        "improved": after.accuracy > before.accuracy,
    }


# ---------------------------------------------------------------------------
# Per-field statistics -- deterministic, auditable evidence fed to the LLM
# refiner (the LLM decides WHICH flips to propose; it never invents the
# underlying counts).
# ---------------------------------------------------------------------------

def build_field_statistics(train_rows: List[dict], rules) -> List[dict]:
    stats: Dict[str, dict] = {}
    for r in train_rows:
        expected = (r.get("orgaccess_expected") or "").strip().lower()
        our_verdict = recompute_verdict(r["field_checks"], rules)
        for fc in r["field_checks"]:
            if fc.get("decision") != "deny":
                continue
            fname = fc["field"]
            severity = rules.severity_for(fname)
            s = stats.setdefault(fname, {
                "field": fname, "current_severity": severity,
                "deny_count": 0, "over_reject_evidence": 0, "under_reject_evidence": 0,
            })
            s["deny_count"] += 1
            if severity == "core" and our_verdict == "REJECTED" and expected in ("full", "partial"):
                s["over_reject_evidence"] += 1
            if severity == "soft" and our_verdict == "PARTIAL" and expected == "rejected":
                s["under_reject_evidence"] += 1

    ranked = sorted(
        stats.values(),
        key=lambda s: max(s["over_reject_evidence"], s["under_reject_evidence"]),
        reverse=True,
    )
    return ranked


# ---------------------------------------------------------------------------
# LLM refiner: given the statistical summary, propose a small batch of
# severity flips with rationale. Reuses existing Ollama infra; same
# reliability guards as attribute_extract.py / baseline_check.py (num_predict
# cap + client timeout) after the real-run incident documented there.
# ---------------------------------------------------------------------------

class SeverityProposal(BaseModel):
    field: str
    proposed_severity: str
    rationale: str

    @field_validator("proposed_severity")
    @classmethod
    def _valid_severity(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in ("core", "soft"):
            raise ValueError(f"proposed_severity must be core or soft, got {v!r}")
        return v


class RefinerOutput(BaseModel):
    proposals: List[SeverityProposal]


REFINER_SYSTEM_PROMPT = """You are refining an access-control severity mapping for an \
attribute-based access control (ABAC) system. Each permission field is classified as \
either "core" (a denied field forces the whole query to be REJECTED) or "soft" (a denied \
field only downgrades the query to PARTIAL, not REJECTED).

Below is evidence from real evaluation data: for each field, how many times a denial of \
that field co-occurred with a MISMATCH against the ground truth, broken down by direction:
- "over_reject_evidence": times this field is core, we said REJECTED, but the true answer \
was actually FULL or PARTIAL -- evidence this field may be classified too strictly (should \
maybe be "soft" instead).
- "under_reject_evidence": times this field is soft, we said PARTIAL, but the true answer \
was actually REJECTED -- evidence this field may be classified too leniently (should maybe \
be "core" instead).

FIELD EVIDENCE:
{stats_json}

Propose at most {max_proposals} severity flips you are genuinely confident about, based on \
this evidence. Only propose a flip when the evidence clearly points one direction. It is \
fine to propose fewer than {max_proposals}, or zero, if the evidence is weak or mixed.

Respond with ONLY valid JSON, no markdown fences, no commentary, matching exactly this \
schema:
{{"proposals": [{{"field": "<field_name>", "proposed_severity": "core" | "soft", \
"rationale": "<one short sentence citing the evidence>"}}]}}
"""


def propose_severity_diffs(
    field_stats: List[dict], model: str, max_proposals: int = 3, top_n_considered: int = 12,
) -> List[SeverityProposal]:
    considered = field_stats[:top_n_considered]
    considered = [s for s in considered if max(s["over_reject_evidence"], s["under_reject_evidence"]) > 0]
    if not considered:
        return []

    prompt = REFINER_SYSTEM_PROMPT.format(
        stats_json=json.dumps(considered, indent=2), max_proposals=max_proposals,
    )
    try:
        response = _client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0, "num_predict": MAX_PREDICT_TOKENS, "num_ctx": NUM_CTX},
        )
        raw = response["message"]["content"]
        parsed = json.loads(raw)
        output = RefinerOutput.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        print(f"[refiner] proposal generation failed, treating as zero proposals: {e}", file=sys.stderr)
        return []

    return output.proposals[:max_proposals]


# ---------------------------------------------------------------------------
# Writing accepted changes back to config/attribute_rules.yaml -- a targeted
# text patch of just the changed fields' `severity:` lines, so the
# _PROVISIONAL banner, comments, and every untouched field are preserved
# byte-for-byte.
# ---------------------------------------------------------------------------

def patch_severity_in_yaml_text(text: str, field_name: str, new_severity: str) -> str:
    pattern = re.compile(
        r"(^  " + re.escape(field_name) + r":\n(?:    .*\n)*?    severity: )(core|soft)(\n)",
        re.MULTILINE,
    )
    new_text, n = pattern.subn(r"\g<1>" + new_severity + r"\g<3>", text)
    if n != 1:
        raise RuntimeError(
            f"Expected exactly one 'severity:' line under field {field_name!r} in "
            f"attribute_rules.yaml, found {n}. Refusing to write a possibly-wrong patch."
        )
    return new_text


def write_accepted_changes(rules_path: Path, accepted: Dict[str, dict]) -> None:
    text = rules_path.read_text(encoding="utf-8")
    for fname, info in accepted.items():
        text = patch_severity_in_yaml_text(text, fname, info["new_severity"])
    note = (
        f"\n# Phase 3c self-evolving refinement (see docs/phase3/self-evolution.md) adjusted "
        f"{len(accepted)} field(s) on {time.strftime('%Y-%m-%d')}, validated against a "
        f"held-out split -- see reports/phase3c_refinement_report.json for the before/after "
        f"numbers and rationale. This is a data-driven ESTIMATE, not the authoritative mapping "
        f"the paper's author has not yet sent; update again when that arrives.\n"
    )
    if "# Phase 3c self-evolving refinement" not in text:
        text = text.replace("default_severity: core\n", "default_severity: core\n" + note, 1)
    rules_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_refinement_loop(
    train_rows: List[dict], held_out_rows: List[dict], base_rules: AttributeRules,
    model: str, max_iterations: int, proposals_per_iteration: int,
) -> Tuple[Dict[str, dict], List[dict], List[dict]]:
    accepted: Dict[str, dict] = {}
    accepted_log: List[dict] = []
    rejected_log: List[dict] = []

    current_held = evaluate_split(held_out_rows, CandidateRules(base_rules, {}), model, do_judge=False)
    current_train = evaluate_split(train_rows, CandidateRules(base_rules, {}), model, do_judge=False)
    current_held_acc = current_held.accuracy
    current_train_acc = current_train.accuracy

    for iteration in range(max_iterations):
        print(f"[refiner] iteration {iteration + 1}/{max_iterations} ...", file=sys.stderr)
        train_candidate = CandidateRules(base_rules, accepted)
        stats = build_field_statistics(train_rows, train_candidate)
        proposals = propose_severity_diffs(stats, model, max_proposals=proposals_per_iteration)

        if not proposals:
            print("[refiner] no proposals this iteration, stopping.", file=sys.stderr)
            break

        any_accepted_this_round = False
        for p in proposals:
            candidate_overrides = {**accepted, p.field: p.proposed_severity}
            candidate = CandidateRules(base_rules, candidate_overrides)

            held_eval = evaluate_split(held_out_rows, candidate, model, do_judge=True)
            train_eval = evaluate_split(train_rows, candidate, model, do_judge=True)

            improves_held = held_eval.accuracy > current_held_acc
            train_ok = train_eval.accuracy >= (current_train_acc - REGRESSION_EPSILON)

            record = {
                "field": p.field,
                "old_severity": base_rules.severity_for(p.field) if p.field not in accepted else accepted[p.field],
                "new_severity": p.proposed_severity,
                "rationale": p.rationale,
                "held_out_accuracy_before": round(current_held_acc * 100, 1),
                "held_out_accuracy_after": round(held_eval.accuracy * 100, 1),
                "train_accuracy_before": round(current_train_acc * 100, 1),
                "train_accuracy_after": round(train_eval.accuracy * 100, 1),
            }

            if improves_held and train_ok:
                accepted[p.field] = p.proposed_severity
                current_held_acc = held_eval.accuracy
                current_train_acc = train_eval.accuracy
                accepted_log.append(record)
                any_accepted_this_round = True
                print(
                    f"[refiner] ACCEPTED {p.field} -> {p.proposed_severity} "
                    f"(held-out {record['held_out_accuracy_before']}% -> {record['held_out_accuracy_after']}%)",
                    file=sys.stderr,
                )
            else:
                record["reason_rejected"] = (
                    "did not improve held-out accuracy" if not improves_held
                    else "regressed train accuracy beyond epsilon"
                )
                rejected_log.append(record)
                print(f"[refiner] rejected {p.field} -> {p.proposed_severity} ({record['reason_rejected']})",
                      file=sys.stderr)

        if not any_accepted_this_round:
            print("[refiner] no proposal accepted this round, stopping.", file=sys.stderr)
            break

    return accepted, accepted_log, rejected_log


def main():
    parser = argparse.ArgumentParser(description="Phase 3c: self-evolving refinement of attribute_rules.yaml")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--proposals-per-iteration", type=int, default=3)
    parser.add_argument("--manual-check-only", action="store_true",
                         help="run only the biometric_required walking-skeleton check, no automated loop")
    parser.add_argument("--dry-run", action="store_true",
                         help="run the full loop but never write config/attribute_rules.yaml")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    rules_path = Path(args.rules)

    print(f"Loading checkpoint: {checkpoint_path}", file=sys.stderr)
    rows = load_checkpoint(checkpoint_path)
    train_rows, held_out_rows = split_rows(rows)
    print(
        f"Split: {len(train_rows)} train (easy+medium), {len(held_out_rows)} held-out (hard). "
        f"[{len(rows) - len(train_rows) - len(held_out_rows)} rows excluded: errored or no field_checks]",
        file=sys.stderr,
    )

    base_rules = AttributeRules(str(rules_path))

    print("\n=== Walking skeleton: biometric_required flip (hand-verified case) ===", file=sys.stderr)
    manual_result = manual_biometric_required_check(held_out_rows, base_rules, args.model)
    print(json.dumps(manual_result, indent=2), file=sys.stderr)

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "split_convention": f"held-out={HELD_OUT_SPLIT}, train={'+'.join(TRAIN_SPLITS)} (NOT comparable to the pooled 1,000-row Phase 1/2/Baseline numbers)",
        "train_n": len(train_rows),
        "held_out_n": len(held_out_rows),
        "manual_walking_skeleton": manual_result,
    }

    if not args.manual_check_only:
        print("\n=== Automated refinement loop ===", file=sys.stderr)
        accepted, accepted_log, rejected_log = run_refinement_loop(
            train_rows, held_out_rows, base_rules, args.model,
            args.max_iterations, args.proposals_per_iteration,
        )

        baseline_held = evaluate_split(held_out_rows, CandidateRules(base_rules, {}), args.model, do_judge=False)
        final_held = evaluate_split(held_out_rows, CandidateRules(base_rules, accepted), args.model, do_judge=False)

        report["accepted_changes"] = accepted_log
        report["rejected_proposals"] = rejected_log
        report["held_out_accuracy_before_pct"] = round(baseline_held.accuracy * 100, 1)
        report["held_out_accuracy_after_pct"] = round(final_held.accuracy * 100, 1)
        report["fields_changed"] = len(accepted)

        print(
            f"\nHeld-out accuracy: {report['held_out_accuracy_before_pct']}% -> "
            f"{report['held_out_accuracy_after_pct']}% ({len(accepted)} field(s) changed)",
            file=sys.stderr,
        )

        if accepted and not args.dry_run:
            write_accepted_changes(rules_path, {f: {"new_severity": s} for f, s in accepted.items()})
            print(f"Wrote {len(accepted)} severity change(s) to {rules_path}", file=sys.stderr)
        elif accepted and args.dry_run:
            print(f"[dry-run] would have written {len(accepted)} severity change(s) to {rules_path}", file=sys.stderr)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
