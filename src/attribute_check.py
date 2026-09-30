"""
Phase 2 - Step 2: deterministic, config-driven verdict aggregation.

No LLM calls here. Takes the AttributeAssessment produced by
attribute_extract.py (per-sub-ask allow/deny decisions, each tagged with the
permission field(s) that governed it) and aggregates them into a single
FULL / PARTIAL / REJECTED verdict.

The only policy decision made in this file is: "does a denied sub-ask sink
the whole query to REJECTED, or only downgrade it to PARTIAL?" -- and that
answer comes entirely from config/attribute_rules.yaml's `severity`
classification (core -> REJECTED, soft -> PARTIAL), NOT from anything
hardcoded here. That file is explicitly a placeholder pending the
authoritative partial-vs-rejected mapping from the OrgAccess paper's author;
when it arrives, only attribute_rules.yaml needs to change.

Reuses the Decision/Verdict enums from access_check.py (Phase 1) so both
phases' results stay uniform for downstream reporting -- Phase 1's own code
is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from src.access_check import Decision, Verdict
from src.attribute_extract import AttributeAssessment, FieldCheck

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_RULES_PATH = CONFIG_DIR / "attribute_rules.yaml"

VALID_SEVERITIES = ("core", "soft")


@dataclass
class FieldCheckResult:
    field_check: FieldCheck
    decision: Decision
    severity: str   # "core" | "soft" -- only meaningful when decision == DENY
    reason: str


@dataclass
class AttributeQueryResult:
    user_role: str
    original_query: str
    permissions: dict
    sub_results: List[FieldCheckResult]
    verdict: Verdict


class AttributeRulesError(RuntimeError):
    pass


class AttributeCoverageError(RuntimeError):
    """Raised when every field_check returned by the model referenced a
    field name that doesn't exist in this row's own permissions -- i.e. the
    model's response gave us zero usable information. Silently treating
    that as "no violations found" -> FULL would be a silent, wrong-by-default
    fallback (this was observed in practice: a garbled response defaulted to
    FULL and happened to match ground truth by luck on one row, which is not
    something to rely on). Mirrors decompose.py's "no silent fallback"
    philosophy -- callers should catch this the same way they catch
    AttributeExtractionError."""


class AttributeRules:
    """Loads config/attribute_rules.yaml once and answers severity lookups."""

    def __init__(self, rules_path: Optional[str] = None):
        rules_path = rules_path or str(DEFAULT_RULES_PATH)
        with open(rules_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}

        self.default_severity: str = doc.get("default_severity", "core")
        if self.default_severity not in VALID_SEVERITIES:
            raise AttributeRulesError(
                f"default_severity in {rules_path} must be one of {VALID_SEVERITIES}, "
                f"got {self.default_severity!r}"
            )
        self.fields: dict = doc.get("fields", {}) or {}

    def severity_for(self, field_name: str) -> str:
        entry = self.fields.get(field_name)
        if not entry:
            return self.default_severity
        severity = entry.get("severity", self.default_severity)
        return severity if severity in VALID_SEVERITIES else self.default_severity


def check_attributes(
    user_role: str,
    original_query: str,
    permissions: dict,
    assessment: AttributeAssessment,
    rules: Optional[AttributeRules] = None,
) -> AttributeQueryResult:
    rules = rules or AttributeRules()

    sub_results: List[FieldCheckResult] = []
    for fc in assessment.field_checks:
        if fc.field not in permissions:
            # The model didn't echo back a real field name from the closed
            # list it was given (observed in practice, e.g. a literal
            # "field" placeholder) -- ignore rather than let a hallucinated
            # name silently drive the verdict via the default severity.
            continue

        if not fc.applicable:
            sub_results.append(
                FieldCheckResult(field_check=fc, decision=Decision.ALLOW, severity="core", reason=fc.reason)
            )
            continue

        decision = Decision.ALLOW if fc.decision == "allow" else Decision.DENY
        severity = rules.severity_for(fc.field)
        sub_results.append(FieldCheckResult(field_check=fc, decision=decision, severity=severity, reason=fc.reason))

    if not sub_results and permissions:
        raise AttributeCoverageError(
            f"All {len(assessment.field_checks)} field_check(s) returned by the model referenced "
            f"field names not present in this row's permissions ({sorted(permissions.keys())}); "
            "cannot compute a verdict from zero usable checks. No silent fallback to FULL."
        )

    verdict = _aggregate(sub_results)
    return AttributeQueryResult(
        user_role=user_role,
        original_query=original_query,
        permissions=permissions,
        sub_results=sub_results,
        verdict=verdict,
    )


def _aggregate(sub_results: List[FieldCheckResult]) -> Verdict:
    core_violation = any(r.decision == Decision.DENY and r.severity == "core" for r in sub_results)
    if core_violation:
        return Verdict.REJECTED
    soft_violation = any(r.decision == Decision.DENY and r.severity == "soft" for r in sub_results)
    if soft_violation:
        return Verdict.PARTIAL
    return Verdict.FULL
