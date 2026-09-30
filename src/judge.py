"""
LLM-judge comparison: asks the configured Ollama model whether our system's
verdict (FULL / PARTIAL / REJECTED) semantically agrees with OrgAccess's
expected_response label for the same query.

Needed because expected_response isn't a clean three-way label: alongside
"full"/"partial"/"rejected" (~98.8% of rows) there are ~35 messy free-text
variants ("approve_budget_only", "r", "partial_reject",
"edit_budget_reject_generate_report_allow", ...) that don't string-match our
verdict directly. Using the same local model already configured for
decomposition keeps this offline and consistent with the rest of Phase 1.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import ollama

# See attribute_extract.py's NUM_CTX comment: caps the KV cache the daemon
# preallocates per loaded model, which is the real driver of llama-server's
# oversized memory footprint, not per-request growth.
NUM_CTX = int(os.environ.get("LLMDB_OLLAMA_NUM_CTX", "4096"))

JUDGE_SYSTEM_PROMPT = """You are comparing two access-control verdicts for the \
same user query.

SYSTEM VERDICT: our access-control system evaluated the query and decided \
FULL (fully allowed), PARTIAL (some parts allowed, some denied), or REJECTED \
(fully denied).

BENCHMARK LABEL: a free-text label from a reference dataset. It is usually a \
clean word (full / partial / rejected) but is sometimes a messier phrase \
(e.g. "approve_budget_only", "r", "partial_reject", "edit_x_reject_y").

Decide whether the SYSTEM VERDICT and the BENCHMARK LABEL describe the same \
underlying access outcome (fully allowed vs. partially allowed vs. fully \
denied). Treat labels like "r"/"reject" as REJECTED, "p"/"approve_x_only" as \
PARTIAL, "allow"/"accept"/"complete" as FULL, unless the phrase clearly says \
otherwise. Minor wording differences are not disagreement -- only judge the \
access outcome itself.

Respond with ONLY valid JSON, no markdown fences, no commentary, matching \
exactly this schema:
{"match": "yes" | "no" | "uncertain", "reasoning": "<one short sentence>"}
"""


def judge_verdict_match(
    our_verdict: str,
    expected_label: str,
    model: str,
    query: str = "",
) -> Tuple[str, str]:
    """Ask the LLM whether our_verdict and expected_label agree.

    Returns (match, reasoning) where match is 'yes' | 'no' | 'uncertain'.
    Never raises -- falls back to 'uncertain' with an error note so a single
    bad call doesn't abort a long batch run.
    """
    if not expected_label:
        return "uncertain", "no benchmark label present for this row"

    user_content = (
        f"QUERY: {query}\n"
        f"SYSTEM VERDICT: {our_verdict}\n"
        f"BENCHMARK LABEL: {expected_label!r}\n"
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            format="json",
            options={"temperature": 0.0, "num_ctx": NUM_CTX},
        )
        raw = response["message"]["content"]
        parsed = json.loads(raw)
        match = str(parsed.get("match", "uncertain")).lower().strip()
        if match not in ("yes", "no", "uncertain"):
            match = "uncertain"
        reasoning = str(parsed.get("reasoning", ""))[:200]
        return match, reasoning
    except Exception as e:
        return "uncertain", f"[judge error: {e}]"
