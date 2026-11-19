"""
Baseline mode -- rung zero of the progressive evaluation ladder: raw LLM,
one call, no decomposition, no deterministic aggregation step.

Phase 1 (src/decompose.py + src/access_check.py) and Phase 2
(src/attribute_extract.py + src/attribute_check.py) both split the decision
into "LLM tags/decomposes" -> "deterministic code aggregates". That's useful
for isolating exactly which piece drives accuracy, but it also means neither
phase answers a simpler question: how well does the LLM do if you just hand
it the user's permissions and the query and ask for one verdict directly,
with no structure imposed at all? This module is that floor -- it exists so
the accuracy gain from Phase 1/2's decomposition step can be measured against
something, instead of asserted.

Deliberately NOT built as free-text/no-context guessing: the model is given
the exact same permissions dict Phase 2 sees (so the comparison isolates
"does structuring the reasoning into a checklist + separate deterministic
aggregation help" -- not "does giving the model more information help",
which would be a different, less useful experiment). One LLM call in, one
verdict out, no field-by-field breakdown, no config-driven severity lookup.

Reuses Verdict from access_check.py (Phase 1) and parse_permissions from
attribute_extract.py (Phase 2) so results stay uniform with both other
phases for downstream reporting -- neither of those files is modified.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import ollama
from pydantic import BaseModel, ValidationError, field_validator

from src.access_check import Verdict
from src.attribute_extract import parse_permissions

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = os.environ.get(
    "LLMDB_BASELINE_MODEL", os.environ.get("LLMDB_OLLAMA_MODEL", "qwen2.5:7b-instruct")
)

# A single holistic verdict + one-sentence reasoning never needs more than a
# couple hundred tokens -- same "no silent runaway generation" reasoning as
# attribute_extract.py's MAX_PREDICT_TOKENS (see that file's comment for the
# real-run incident that motivated capping this).
MAX_PREDICT_TOKENS = int(os.environ.get("LLMDB_BASELINE_MAX_TOKENS", "256"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLMDB_OLLAMA_TIMEOUT", "240"))
# See attribute_extract.py's NUM_CTX comment: caps the KV cache the daemon
# preallocates per loaded model, which is the real driver of llama-server's
# oversized memory footprint, not per-request growth.
NUM_CTX = int(os.environ.get("LLMDB_OLLAMA_NUM_CTX", "4096"))
_client = ollama.Client(timeout=REQUEST_TIMEOUT_SECONDS)


class BaselineVerdict(BaseModel):
    verdict: str
    reasoning: str

    @field_validator("verdict")
    @classmethod
    def _valid_verdict(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in ("full", "partial", "rejected"):
            raise ValueError(f"verdict must be one of full/partial/rejected, got {v!r}")
        return v


class BaselineExtractionError(RuntimeError):
    pass


@dataclass
class BaselineResult:
    user_role: str
    original_query: str
    permissions: dict
    verdict: Verdict
    reasoning: str


# Split into an evolvable "instructions" block (the part
# src/gepa_optimize_baseline.py is allowed to rewrite -- see
# LLMDB_BASELINE_SYSTEM_PROMPT_FILE below) and a fixed JSON-schema footer that's
# always appended verbatim, so an evolved prompt can never accidentally drop the
# field names BaselineVerdict.model_validate() depends on.
SYSTEM_PROMPT_INSTRUCTIONS = """You are an access-control decision engine. A user with the \
permissions listed below has submitted a query. Decide, in ONE holistic judgment, whether \
the query should be FULL access (fully allowed), PARTIAL access (part of what's asked is \
allowed, part isn't or comes with an unmet restriction), or REJECTED (fully denied).

THIS USER'S PERMISSIONS:
{field_list}

Weigh everything shown above together and pick the single outcome that best fits. Do NOT \
break the query into pieces or list field-by-field reasoning -- one overall decision only."""

SYSTEM_PROMPT_JSON_FOOTER = """

Respond with ONLY valid JSON, no markdown fences, no commentary, matching exactly this \
schema:
{"verdict": "full" | "partial" | "rejected", "reasoning": "<one short sentence>"}
"""

SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_INSTRUCTIONS + SYSTEM_PROMPT_JSON_FOOTER

# Optional override: point this at a file (e.g. the prompt
# src/gepa_optimize_baseline.py writes to reports/gepa_best_baseline_prompt.txt) to
# swap in a different instructions block without editing this file. Unset ->
# SYSTEM_PROMPT_INSTRUCTIONS above, byte-identical to before this option existed.
_SYSTEM_PROMPT_OVERRIDE_ENV = "LLMDB_BASELINE_SYSTEM_PROMPT_FILE"


def _load_instructions_template() -> str:
    path_str = os.environ.get(_SYSTEM_PROMPT_OVERRIDE_ENV, "").strip()
    if not path_str:
        return SYSTEM_PROMPT_INSTRUCTIONS
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    text = path.read_text(encoding="utf-8").strip()
    if "{field_list}" not in text:
        text += "\n\nTHIS USER'S PERMISSIONS:\n{field_list}"
    return text


INSTRUCTIONS_TEMPLATE = _load_instructions_template()


def _build_system_prompt(permissions: dict) -> str:
    if not permissions:
        field_list = "(none -- this user has no listed permissions; deny everything.)"
    else:
        field_list = "\n".join(f"- {k}: {json.dumps(v)}" for k, v in permissions.items())
    body = INSTRUCTIONS_TEMPLATE.replace("{field_list}", field_list)
    return body + SYSTEM_PROMPT_JSON_FOOTER


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
        raise BaselineExtractionError(
            f"Ollama request to '{model}' failed or exceeded the "
            f"{REQUEST_TIMEOUT_SECONDS:.0f}s timeout: {e}"
        ) from e
    return response["message"]["content"]


def get_baseline_verdict(
    user_query: str,
    permissions: Union[str, dict, None],
    user_role: str = "",
    model: str = DEFAULT_MODEL,
    max_retries: int = 1,
) -> BaselineResult:
    """One LLM call, one verdict -- no decomposition, no checklist, no
    deterministic aggregation. See module docstring and docs/baseline/plan.md.

    Raises BaselineExtractionError if the model can't produce valid,
    schema-conforming JSON within max_retries + 1 attempts (no silent
    regex/free-text fallback -- same philosophy as decompose.py and
    attribute_extract.py).
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
                "system prompt. No markdown, no extra text."
            )
        try:
            raw = _call_ollama(model, system_prompt, prompt)
            parsed = json.loads(raw)
            assessment = BaselineVerdict.model_validate(parsed)
            return BaselineResult(
                user_role=user_role,
                original_query=user_query,
                permissions=perms,
                verdict=Verdict(assessment.verdict),
                reasoning=assessment.reasoning,
            )
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            continue

    raise BaselineExtractionError(
        f"Model '{model}' failed to produce valid baseline-verdict JSON after "
        f"{max_retries + 1} attempt(s). Last error: {last_error}"
    )
