"""
Phase 2 - Step 1: attribute-aware decomposition for OrgAccess-style ABAC.

Phase 1's decompose.py tags each sub-query against a fixed, global
`table.field` vocabulary. OrgAccess rows don't carry a fixed vocabulary --
each row has its own `permissions` dict with a different subset of ~90
possible field names (department, access_level, allowed_actions,
location_restriction, budget_approval_threshold, biometric_required, ...),
and the field VALUES themselves are free natural-language text (e.g.
access_level values range from "read" to "Tier 2" to "Senior Manager" across
different rows). Plain Python string/list matching cannot reliably compare
"requesting to download a legal document" against
allowed_actions=["view_legal_docs"] -- that comparison needs the same kind
of NLU judgment decompose.py already delegates to the LLM, just generalized
to a dynamic per-row vocabulary instead of a static global one.

So this stage evaluates the query against an EXHAUSTIVE CHECKLIST of every
field present in this row's own `permissions` dict -- one output entry per
field, always in the same order the fields were given -- rather than asking
the model to freely decompose the query into sub-asks. An earlier
free-decomposition version of this prompt was tested against a real OrgAccess
row and silently dropped a `biometric_required: true` constraint entirely
(the model only reasoned about `department`/`allowed_actions` and ignored the
boolean precondition), producing FULL where the dataset's own ground truth is
PARTIAL. A fixed per-field checklist closes that gap: the model cannot skip a
field it was never asked about. The core/soft severity classification that
turns per-field deny decisions into a FULL/PARTIAL/REJECTED verdict is NOT
decided here -- see attribute_check.py and config/attribute_rules.yaml, which
is the swappable, non-LLM piece.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Union

import ollama
from pydantic import BaseModel, ValidationError, field_validator

DEFAULT_MODEL = os.environ.get(
    "LLMDB_ORGACCESS_ATTR_MODEL", os.environ.get("LLMDB_OLLAMA_MODEL", "qwen2.5:7b-instruct")
)

# Real per-row `permissions` dicts observed in data/orgaccess_raw/*.jsonl are
# small (4-8 fields, mean 5.7), so a correct field_checks response never
# needs more than a few hundred tokens. Without a cap, a degenerate/looping
# generation has no natural stopping point short of the model's full context
# window -- observed in practice during a real run: one row's request ran
# past 8000 generated tokens at ~5-6 tokens/sec (Ollama's own server log)
# without terminating, stalling the whole sequential pipeline indefinitely.
# num_predict bounds the worst case; the client-side timeout is a second,
# independent backstop against the same failure mode running slow instead
# of long.
MAX_PREDICT_TOKENS = int(os.environ.get("LLMDB_ORGACCESS_ATTR_MAX_TOKENS", "1024"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLMDB_OLLAMA_TIMEOUT", "240"))
# The daemon's default context length (OLLAMA_CONTEXT_LENGTH, seen at 131072
# on this machine, clamped to the model's own 32768 max) is preallocated as
# KV cache the instant the model loads -- real prompts here run 700-900
# tokens, so an uncapped num_ctx wastes several GB of RAM per loaded model
# for headroom that's never used. Confirmed via Ollama's server log
# (n_ctx_slot = 32768 while task.n_tokens ~ 800) and process memory
# (llama-server resident at ~6.4GB for a ~2GB Q4_K_M 3B model). Capping this
# explicitly, per request, is the fix -- not model-unload cycling.
NUM_CTX = int(os.environ.get("LLMDB_OLLAMA_NUM_CTX", "4096"))
_client = ollama.Client(timeout=REQUEST_TIMEOUT_SECONDS)


class FieldCheck(BaseModel):
    field: str                     # exact permission field name this entry evaluates
    applicable: bool                # does this field actually constrain what the query is asking?
    decision: str                  # "allow" | "deny" -- meaningful only when applicable is true
    severity_hint: str             # "core" | "soft" | "none" -- informational only, not authoritative
    reason: str                    # one-line justification quoting the field's actual value

    @field_validator("decision")
    @classmethod
    def _valid_decision(cls, v: str) -> str:
        v = (v or "allow").strip().lower()
        return v if v in ("allow", "deny") else "allow"

    @field_validator("severity_hint")
    @classmethod
    def _valid_severity(cls, v: str) -> str:
        v = (v or "none").strip().lower()
        return v if v in ("core", "soft", "none") else "none"


class AttributeAssessment(BaseModel):
    field_checks: List[FieldCheck]

    @field_validator("field_checks")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("field_checks must contain at least one item")
        return v


class AttributeExtractionError(RuntimeError):
    pass


SYSTEM_PROMPT_TEMPLATE = """You are an attribute-based access-control (ABAC) reasoning \
engine. A user with the permissions listed below has submitted a query. Evaluate the \
query against EVERY SINGLE field listed below, one checklist entry per field, in the \
SAME ORDER they are listed. Do not skip any field, even ones that seem minor.

THIS USER'S PERMISSIONS (evaluate every one of these fields -- do not invent field \
names that aren't listed here):
{field_list}

For each field, produce one checklist entry:
- field: the exact field name, copied verbatim from the list above.
- applicable: true if this field actually constrains or is implicated by what the query \
  is asking (e.g. the query touches the department/action/data this field governs, or \
  this field is a standing precondition -- like a required boolean flag, a rate/budget/\
  session limit, a location/date restriction, a delegation or quorum requirement -- that \
  applies to any request this user makes). false only if the field is genuinely \
  unrelated to this query.
- decision: (only meaningful if applicable is true) "allow" if the field's ACTUAL value \
  shown above permits the query's request, "deny" if it doesn't. For a required boolean \
  precondition (e.g. biometric_required, device_compliance, log_required) that is true, \
  treat it as "deny" UNLESS the query text itself states or implies the precondition is \
  satisfied -- the dataset never gives the benefit of the doubt for unstated \
  preconditions. For numeric thresholds, deny if the query's request would exceed the \
  threshold or the query doesn't establish it stays within it when the field mandates a \
  hard cap. If applicable is false, set decision to "allow" (it's a no-op).
- severity_hint: "soft" if this field is a location/geography/region/zone constraint, a \
  cross-department collaboration constraint, or an ethical-guideline constraint. "core" \
  for every other kind of field (department, access_level, allowed_actions, security \
  preconditions, numeric/budget/rate/session limits, temporal/deadline constraints, \
  delegation/quorum requirements, etc.). "none" if applicable is false.
- reason: one short sentence citing the specific field value that drove the decision.

Respond with ONLY valid JSON, no markdown fences, no commentary, matching exactly this \
schema:
{{
  "field_checks": [
    {{"field": "<field_name>", "applicable": true | false, "decision": "allow" | "deny", \
"severity_hint": "core" | "soft" | "none", "reason": "<one line>"}}
  ]
}}
"""


def parse_permissions(raw_permissions: Union[str, dict, None]) -> dict:
    """OrgAccess rows store `permissions` as a JSON-encoded STRING, not a
    native dict, in the raw HuggingFace data (confirmed against real fetched
    rows). Handle both shapes defensively."""
    if raw_permissions is None:
        return {}
    if isinstance(raw_permissions, dict):
        return raw_permissions
    if isinstance(raw_permissions, str):
        try:
            parsed = json.loads(raw_permissions)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _build_system_prompt(permissions: dict) -> str:
    if not permissions:
        field_list = "(none -- this user has no listed permissions; deny everything.)"
    else:
        field_list = "\n".join(f"- {k}: {json.dumps(v)}" for k, v in permissions.items())
    return SYSTEM_PROMPT_TEMPLATE.format(field_list=field_list)


def _call_ollama(model: str, system_prompt: str, user_query: str) -> str:
    try:
        response = _client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            format="json",
            options={"temperature": 0.0, "num_predict": MAX_PREDICT_TOKENS, "num_ctx": NUM_CTX},
        )
    except Exception as e:
        raise AttributeExtractionError(
            f"Ollama request to '{model}' failed or exceeded the "
            f"{REQUEST_TIMEOUT_SECONDS:.0f}s timeout: {e}"
        ) from e
    return response["message"]["content"]


def extract_attributes(
    user_query: str,
    permissions: Union[str, dict, None],
    model: str = DEFAULT_MODEL,
    max_retries: int = 1,
) -> AttributeAssessment:
    """Decompose a query into governing-field-tagged sub-asks for ONE row's
    own permissions dict.

    Raises AttributeExtractionError if the model can't produce valid,
    schema-conforming JSON within max_retries + 1 attempts (no silent
    regex/free-text fallback, matching decompose.py's philosophy).
    """
    perms = parse_permissions(permissions)
    system_prompt = _build_system_prompt(perms)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        prompt = user_query
        if attempt > 0:
            prompt = (
                f"{user_query}\n\n"
                "REMINDER: respond with ONLY the JSON object described in the "
                "system prompt. No markdown, no extra text. Only cite field names "
                "that were listed in the permissions above."
            )
        try:
            raw = _call_ollama(model, system_prompt, prompt)
            parsed = json.loads(raw)
            return AttributeAssessment.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            continue

    raise AttributeExtractionError(
        f"Model '{model}' failed to produce valid attribute-assessment JSON after "
        f"{max_retries + 1} attempt(s). Last error: {last_error}"
    )
