"""
Runtime configuration for LLM-DB Phase 1.

Reads from a .env file (if present in the project root) and then from
environment variables. Env vars already set in the shell always take
precedence over what is in .env.

All values can be overridden at runtime by CLI flags in pipeline.py —
this module just supplies the defaults so you don't have to repeat them
on every invocation.

Supported env vars:
    LLMDB_MODE                   mock | live | orgaccess | orgaccess_attr | baseline   (default: mock)
    LLMDB_OLLAMA_MODEL           Ollama model name            (default: qwen2.5:7b-instruct)
    LLMDB_ORGACCESS_SPLITS       comma-sep splits             (default: easy)
    LLMDB_ORGACCESS_SPLIT        deprecated singular alias    (fallback if SPLITS not set)
    LLMDB_ORGACCESS_LIMIT        integer or empty             (default: unset = no cap)
    LLMDB_ORGACCESS_USER         user id from users.yaml      (default: daniel.finance_manager)
    LLMDB_ORGACCESS_RANDOM_SEED  integer or empty             (default: unset = non-deterministic)
    LLMDB_REPORT_PATH            file path for Excel report   (default: unset = no report)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = ROOT / ".env"

VALID_MODES = ("mock", "live", "orgaccess", "orgaccess_attr", "baseline")
VALID_SPLITS = ("easy", "medium", "hard")

Mode = Literal["mock", "live", "orgaccess", "orgaccess_attr", "baseline"]
OrgAccessSplit = Literal["easy", "medium", "hard"]


def _load_dotenv(env_path: Path = _DOTENV_PATH) -> None:
    """Minimal .env parser — sets env vars not already present in the environment.

    Handles:
      - blank lines and # comments
      - KEY=VALUE pairs (leading/trailing whitespace stripped)
      - Quoted values: 'value' or "value" have their outer quotes stripped
      - Does NOT expand shell variables or handle multi-line values
    """
    if not env_path.exists():
        return
    with open(env_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip matched surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


# Load .env once at import time — safe to call multiple times (idempotent due
# to the "not already in os.environ" guard inside _load_dotenv).
_load_dotenv()


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_mode() -> Mode:
    """Return the active run mode (mock / live / orgaccess)."""
    raw = os.environ.get("LLMDB_MODE", "mock").lower()
    if raw not in VALID_MODES:
        raise ValueError(
            f"LLMDB_MODE={raw!r} is not valid. Choose one of: {VALID_MODES}"
        )
    return raw  # type: ignore[return-value]


def get_model() -> str:
    """Return the Ollama model name to use for decomposition."""
    return os.environ.get("LLMDB_OLLAMA_MODEL", "qwen2.5:7b-instruct")


def get_orgaccess_split() -> OrgAccessSplit:
    """Return the OrgAccess split to pull (easy / medium / hard)."""
    raw = os.environ.get("LLMDB_ORGACCESS_SPLIT", "easy").lower()
    if raw not in VALID_SPLITS:
        raise ValueError(
            f"LLMDB_ORGACCESS_SPLIT={raw!r} is not valid. Choose one of: {VALID_SPLITS}"
        )
    return raw  # type: ignore[return-value]


def get_orgaccess_limit() -> Optional[int]:
    """Return the row cap per split, or None for no cap."""
    val = os.environ.get("LLMDB_ORGACCESS_LIMIT", "").strip()
    if not val:
        return None
    try:
        limit = int(val)
    except ValueError:
        raise ValueError(
            f"LLMDB_ORGACCESS_LIMIT={val!r} is not a valid integer."
        )
    if limit <= 0:
        raise ValueError("LLMDB_ORGACCESS_LIMIT must be a positive integer.")
    return limit


def get_orgaccess_user() -> str:
    """Return the user id (from users.yaml) used when running orgaccess examples."""
    return os.environ.get("LLMDB_ORGACCESS_USER", "daniel.finance_manager")


def get_orgaccess_splits() -> list[str]:
    """Return the list of OrgAccess splits to fetch.

    Reads LLMDB_ORGACCESS_SPLITS (comma-separated, e.g. "easy,medium,hard").
    Falls back to LLMDB_ORGACCESS_SPLIT (singular) for backward compatibility.
    Defaults to ["easy"] if neither is set.
    """
    plural = os.environ.get("LLMDB_ORGACCESS_SPLITS", "").strip()
    if plural:
        splits = [s.strip().lower() for s in plural.split(",") if s.strip()]
    else:
        # backward-compat: singular LLMDB_ORGACCESS_SPLIT
        singular = os.environ.get("LLMDB_ORGACCESS_SPLIT", "easy").strip().lower()
        splits = [singular] if singular else ["easy"]

    for s in splits:
        if s not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split {s!r} in LLMDB_ORGACCESS_SPLITS. "
                f"Valid values: {VALID_SPLITS}"
            )
    return splits


def get_orgaccess_random_seed() -> Optional[int]:
    """Return the random seed for cross-split shuffling, or None for non-deterministic runs."""
    val = os.environ.get("LLMDB_ORGACCESS_RANDOM_SEED", "").strip()
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        raise ValueError(
            f"LLMDB_ORGACCESS_RANDOM_SEED={val!r} is not a valid integer."
        )


def get_report_path() -> Optional[str]:
    """Return the output path for the Excel report, or None if not configured."""
    val = os.environ.get("LLMDB_REPORT_PATH", "").strip()
    return val if val else None


def dump() -> dict:
    """Return a snapshot of the current effective config (for --show-config use)."""
    return {
        "mode": get_mode(),
        "model": get_model(),
        "orgaccess_splits": get_orgaccess_splits(),
        "orgaccess_limit": get_orgaccess_limit(),
        "orgaccess_user": get_orgaccess_user(),
        "orgaccess_random_seed": get_orgaccess_random_seed(),
        "report_path": get_report_path(),
        "dotenv_loaded": _DOTENV_PATH.exists(),
    }
