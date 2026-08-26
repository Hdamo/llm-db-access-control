"""
Phase 1 - Step 1: Query decomposition.

Sends the user's natural-language query to a local Ollama model and asks it
to split the query into sub-queries, each tagged with the resource ID it
touches (drawn from a closed vocabulary -- see resource_vocab.py). Output is
validated with pydantic; malformed responses trigger one retry with a
stricter reminder before failing loudly (we do NOT silently fall back to
regex/free-text parsing).

The model name is read from an env var / constructor arg, not hardcoded, so
you can swap between qwen2.5:7b-instruct, llama3.1:8b-instruct-q4_K_M,
mistral:7b-instruct, etc. without touching this file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import ollama
from pydantic import BaseModel, ValidationError, field_validator

from src.resource_vocab import load_resource_vocab

DEFAULT_MODEL = os.environ.get("LLMDB_OLLAMA_MODEL", "qwen2.5:7b-instruct")
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class SubQuery(BaseModel):
    text: str            # the isolated sub-question, in natural language
    resource: str        # resource ID from the closed vocabulary, or "UNKNOWN"
    rationale: str        # one-line justification for the resource tag


class DecompositionResult(BaseModel):
    sub_queries: List[SubQuery]

    @field_validator("sub_queries")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("sub_queries must contain at least one item")
        return v


SYSTEM_PROMPT_TEMPLATE = """You are a query decomposition engine for an access-controlled \
database system. Given a single user question, split it into its constituent \
independent sub-questions (a sub-question is the smallest unit that maps to ONE \
resource). If the question already only touches one resource, return a single \
sub-query.

For each sub-question, tag it with exactly one resource ID chosen from this \
CLOSED VOCABULARY (do not invent new resource IDs, do not modify their spelling):
{vocab_list}

If a sub-question does not clearly match any resource in the vocabulary, set \
resource to "UNKNOWN".

Respond with ONLY valid JSON, no markdown fences, no commentary, matching \
exactly this schema:
{{
  "sub_queries": [
    {{"text": "<isolated sub-question>", "resource": "<resource.id or UNKNOWN>", "rationale": "<one line>"}}
  ]
}}
"""


class DecompositionError(RuntimeError):
    pass


def _build_system_prompt(vocab: List[str]) -> str:
    vocab_list = "\n".join(f"- {r}" for r in vocab)
    return SYSTEM_PROMPT_TEMPLATE.format(vocab_list=vocab_list)


def _call_ollama(model: str, system_prompt: str, user_query: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        format="json",     # ask Ollama to constrain output to valid JSON
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


def decompose_query(
    user_query: str,
    model: str = DEFAULT_MODEL,
    permissions_path: Optional[str] = None,
    max_retries: int = 1,
) -> DecompositionResult:
    """Decompose a single natural-language query into tagged sub-queries.

    Raises DecompositionError if the model can't produce valid, schema-
    conforming JSON within max_retries + 1 attempts.
    """
    permissions_path = permissions_path or str(CONFIG_DIR / "permissions.yaml")
    vocab = load_resource_vocab(permissions_path)
    system_prompt = _build_system_prompt(vocab)

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
            return DecompositionResult.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            continue

    raise DecompositionError(
        f"Model '{model}' failed to produce valid decomposition JSON after "
        f"{max_retries + 1} attempt(s). Last error: {last_error}"
    )


class MockDecomposer:
    """Deterministic stand-in for the Ollama call, used in tests / demos when
    no local model is available. NOT used in the real pipeline path -- swap
    decompose_query for this only in test harnesses."""

    def __init__(self, canned: dict):
        # canned: {query_text: DecompositionResult-like dict}
        self.canned = canned

    def __call__(self, user_query: str, **_) -> DecompositionResult:
        if user_query not in self.canned:
            raise DecompositionError(f"No canned response for: {user_query!r}")
        return DecompositionResult.model_validate(self.canned[user_query])
