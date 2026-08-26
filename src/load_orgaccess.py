"""
Downloads the OrgAccess benchmark (Sanyal et al., 2025) from Hugging Face
and saves each split locally as JSONL under data/orgaccess_raw/.

This is kept SEPARATE from examples/sample_queries.json on purpose:
OrgAccess's permission schema is a free-form dict per example (department,
access_level, allowed_actions, location_restriction, etc. -- 40 different
permission types across the dataset) rather than the fixed table.field
resource vocabulary our access_check.py currently understands. Dropping
OrgAccess items straight into sample_queries.json would just default-deny
everything, since none of its resource strings exist in our permissions.yaml.

For Phase 1 testing, load_as_pipeline_examples() maps each OrgAccess row
to the {user, query, orgaccess_meta} format that pipeline.run_orgaccess()
consumes. The Ollama decomposer will tag resources from our closed vocabulary;
most will come back UNKNOWN/REJECTED since the schemas don't yet align --
that is intentional and expected for Phase 1. The raw OrgAccess metadata is
preserved alongside each result so it can be used for alignment work later.

Usage (standalone fetch-and-save):
    python -m src.load_orgaccess                 # pulls all 3 splits
    python -m src.load_orgaccess --split easy     # just one split
    python -m src.load_orgaccess --limit 50       # cap rows per split

Usage (programmatic, from pipeline):
    from src.load_orgaccess import load_as_pipeline_examples
    examples = load_as_pipeline_examples(split="easy", limit=10,
                                         test_user="daniel.finance_manager")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "orgaccess_raw"

SPLITS = ("easy", "medium", "hard")


class DatasetFetchError(RuntimeError):
    pass


def fetch_split(split: str, limit: Optional[int] = None) -> list[dict]:
    """Pull one split of respai-lab/orgaccess via the HF datasets library."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DatasetFetchError(
            "The 'datasets' package isn't installed. Run: "
            "pip install datasets huggingface_hub"
        ) from e

    try:
        ds = load_dataset("respai-lab/orgaccess", split=split)
    except Exception as e:
        raise DatasetFetchError(
            f"Failed to download split '{split}' from respai-lab/orgaccess on "
            f"Hugging Face. Check your internet connection and that the "
            f"dataset name/split still exists. Original error: {e}"
        ) from e

    rows = ds.to_list()
    if limit is not None:
        rows = rows[:limit]
    return rows


def save_split(split: str, rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path


def _extract_query(row: dict) -> Optional[str]:
    """Pull the natural-language query string out of an OrgAccess row.

    The dataset uses the field name 'query'; fall back to 'question' and
    'input' for resilience against minor schema changes.
    """
    for key in ("query", "question", "input"):
        val = row.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def load_as_pipeline_examples_multi(
    splits: list[str],
    limit: Optional[int] = None,
    test_user: str = "daniel.finance_manager",
    random_seed: Optional[int] = None,
) -> list[dict]:
    """Return OrgAccess rows from multiple splits as pipeline-ready example dicts.

    Rows from all requested splits are pooled and shuffled together before the
    limit is applied, giving a uniform random sample across splits.

    Each returned dict has:
        "user"          — the test_user passed in (a valid users.yaml id)
        "query"         — the natural-language query from the OrgAccess row
        "split"         — which split the row came from ("easy", "medium", "hard")
        "orgaccess_meta"— the full raw row (for later schema-alignment work)

    Args:
        splits:      list of splits to fetch, e.g. ["easy", "medium", "hard"]
        limit:       total rows to return across all splits (None = no cap).
                     Sampling is done AFTER pooling all splits, so with limit=100
                     and splits=["easy","hard"] you still fetch both full splits
                     first — the only way to guarantee uniform cross-split sampling.
        test_user:   user id from config/users.yaml to run each query as
        random_seed: integer seed for reproducibility (None = different each run)
    """
    import random
    import sys

    if not splits:
        raise ValueError("splits must be a non-empty list")
    for s in splits:
        if s not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {s!r}")

    # Fetch and tag all rows from every requested split
    all_raw: list[dict] = []
    for split in splits:
        print(f"  Fetching '{split}' split ...", file=sys.stderr)
        rows = fetch_split(split, limit=None)
        for row in rows:
            row["_split"] = split
        all_raw.extend(rows)

    # Shuffle for random sampling across splits
    rng = random.Random(random_seed)
    rng.shuffle(all_raw)

    if limit is not None:
        all_raw = all_raw[:limit]

    # Convert to pipeline dicts
    examples: list[dict] = []
    skipped = 0
    for row in all_raw:
        split_tag = row.pop("_split", "unknown")
        query = _extract_query(row)
        if query is None:
            skipped += 1
            continue
        examples.append({
            "user": test_user,
            "query": query,
            "split": split_tag,
            "orgaccess_meta": row,
        })

    if skipped:
        print(
            f"[load_orgaccess] skipped {skipped} row(s) with no extractable query field.",
            file=sys.stderr,
        )
    return examples


def load_as_pipeline_examples(
    split: str,
    limit: Optional[int] = None,
    test_user: str = "daniel.finance_manager",
) -> list[dict]:
    """Return OrgAccess rows as pipeline-ready example dicts.

    Each returned dict has:
        "user"          — the test_user passed in (a valid users.yaml id)
        "query"         — the natural-language query from the OrgAccess row
        "orgaccess_meta"— the full raw row (for later schema-alignment work)

    Rows where no query string can be extracted are silently skipped with a
    warning printed to stderr.

    Args:
        split:      "easy", "medium", or "hard"
        limit:      cap on number of rows returned (None = no cap)
        test_user:  user id from config/users.yaml to run each query as
    """
    import sys
    rows = fetch_split(split, limit=limit)
    examples: list[dict] = []
    skipped = 0
    for row in rows:
        query = _extract_query(row)
        if query is None:
            skipped += 1
            continue
        examples.append({
            "user": test_user,
            "query": query,
            "split": split,
            "orgaccess_meta": row,
        })
    if skipped:
        print(
            f"[load_orgaccess] skipped {skipped} row(s) with no extractable query field.",
            file=sys.stderr,
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Fetch the OrgAccess benchmark from Hugging Face")
    parser.add_argument("--split", choices=SPLITS, help="only fetch this split (default: all three)")
    parser.add_argument("--limit", type=int, default=None, help="cap number of rows per split")
    args = parser.parse_args()

    splits_to_fetch = [args.split] if args.split else list(SPLITS)

    for split in splits_to_fetch:
        print(f"Fetching '{split}' split from respai-lab/orgaccess ...")
        try:
            rows = fetch_split(split, limit=args.limit)
        except DatasetFetchError as e:
            print(f"  [error] {e}")
            continue
        out_path = save_split(split, rows)
        print(f"  saved {len(rows)} rows -> {out_path.relative_to(ROOT)}")

    print("\nDone. Raw data saved to data/orgaccess_raw/. "
          "Use load_as_pipeline_examples() to feed these into the pipeline "
          "or --orgaccess in pipeline.py for end-to-end testing.")


if __name__ == "__main__":
    main()