# LLM-DB Phase 1: Query Decomposition + Flat Access Check

Phase 1 of the graph-based authorization project for agentic LLM–database
access. This phase deliberately does **not** implement the relationship
graph, session memory, or FragFuse-style attack simulation — those are
future phases. This is just the foundation: split a query into sub-queries,
check each one against a flat role → resource permission map.

## What this does

```
user query
   │
   ▼
[decompose.py]  -- local Ollama model, forced JSON output, tags each
   │               sub-query with a resource ID from a CLOSED vocabulary
   ▼
[access_check.py] -- looks up the user's role in a flat YAML permission
   │                   map, checks each tagged resource: ALLOW / DENY
   ▼
aggregate verdict: FULL / PARTIAL / REJECTED
```

The FULL/PARTIAL/REJECTED verdict naming matches the OrgAccess benchmark
(Sanyal et al., 2025) response taxonomy on purpose, since that benchmark
is the natural evaluation set for this line of work later on.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# pull a model if you haven't already (any of these work — swappable):
ollama pull qwen2.5:7b-instruct
# or: ollama pull llama3.1:8b-instruct-q4_K_M
# or: ollama pull mistral:7b-instruct
```

---

## Configuration (`.env` file)

Copy `.env.example` to `.env` and edit it to control the run mode, model,
OrgAccess dataset settings, and report output. Shell environment variables
always override what is in `.env`.

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `LLMDB_MODE` | `mock` | Run mode: `mock` \| `live` \| `orgaccess` |
| `LLMDB_OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Ollama model to use for decomposition |
| `LLMDB_ORGACCESS_SPLITS` | `easy` | Comma-separated splits: `easy,medium,hard` (any combination) |
| `LLMDB_ORGACCESS_LIMIT` | *(unset)* | Total rows to sample across all splits (leave blank for full dataset) |
| `LLMDB_ORGACCESS_RANDOM_SEED` | *(unset)* | Integer seed for reproducible sampling; blank = random each run |
| `LLMDB_ORGACCESS_USER` | `daniel.finance_manager` | User id (from `config/users.yaml`) used when running OrgAccess queries |
| `LLMDB_REPORT_PATH` | *(unset)* | Path for Excel report output, e.g. `reports/run.xlsx`; blank = no report |

Print the effective config (merges `.env` + env vars) with:

```bash
python -m src.pipeline --show-config
```

---

## Running it

CLI flags always override `.env`. When `--user` + `--query` are both given,
a single query is run regardless of `LLMDB_MODE`.

### Single query (always calls Ollama)

```bash
python -m src.pipeline --user daniel.finance_manager \
  --query "Show me Sales headcount and this quarter's budget"
```

### Batch mode — controlled by `.env` or CLI flag

```bash
# Use whatever mode LLMDB_MODE is set to in .env (default: mock)
python -m src.pipeline

# Explicit mode flags (override .env):
python -m src.pipeline --mock        # offline, no Ollama, uses sample_queries.json
python -m src.pipeline --live        # Ollama on, uses sample_queries.json
python -m src.pipeline --orgaccess   # Ollama on, pulls from HuggingFace

# OrgAccess with multiple splits + random sample of 500 + reproducible seed:
python -m src.pipeline --orgaccess --splits easy medium hard \
    --limit 500 --seed 42

# OrgAccess with Excel report output:
python -m src.pipeline --orgaccess --splits easy hard --limit 200 \
    --seed 1 --report reports/run_easy_hard.xlsx

# Full dataset run (no --limit = all ~75k rows):
python -m src.pipeline --orgaccess --splits easy medium hard \
    --seed 42 --report reports/full_run.xlsx

# Swap the Ollama model without touching config:
python -m src.pipeline --live --model llama3.1:8b-instruct-q4_K_M
```

### Modes explained

| Mode | Ollama? | Query source | Use when |
|---|---|---|---|
| `mock` | No | `examples/sample_queries.json` (canned) | Verifying access-check logic offline; CI; demo before Ollama is set up |
| `live` | Yes | `examples/sample_queries.json` | Testing decomposer + access-check together on known queries |
| `orgaccess` | Yes | `respai-lab/orgaccess` on HuggingFace | Stress-testing the pipeline on real benchmark queries |

---

## OrgAccess test mode detail

When `LLMDB_MODE=orgaccess` (or `--orgaccess`), the pipeline:

1. Fetches rows from all requested splits of `respai-lab/orgaccess` on HuggingFace.
2. Pools all rows, shuffles them (using `--seed` if set), then takes up to `--limit` rows.
3. Maps each row's `query` field → Ollama decomposition → access check.
4. Prints verdict + the OrgAccess `expected_response` field for later comparison.
5. Writes an Excel report if `--report` / `LLMDB_REPORT_PATH` is set.

Because OrgAccess uses a free-form permission schema that doesn't map 1:1
to our `table.field` vocabulary, most resources will be tagged `UNKNOWN`
and verdicts will be `REJECTED`. This is **expected for Phase 1** — the
mismatch is what the Phase 2 schema-alignment layer will address.

```bash
# .env settings for a multi-split orgaccess run with report:
LLMDB_MODE=orgaccess
LLMDB_ORGACCESS_SPLITS=easy,medium,hard
LLMDB_ORGACCESS_LIMIT=20
LLMDB_ORGACCESS_RANDOM_SEED=42
LLMDB_ORGACCESS_USER=daniel.finance_manager
LLMDB_REPORT_PATH=reports/orgaccess_report.xlsx

# Then just run:
python -m src.pipeline
```

### Excel report format

The report has a single sheet with a summary block at the top followed by one
row per query:

```
Row 1  Title
Row 2  Run metadata (date, model, splits, seed)
Row 3  Summary: Total N | FULL X (Y%) | PARTIAL X (Y%) | REJECTED X (Y%)
Row 4  (blank)
Row 5  Column headers
Row 6+ One data row per query
```

Columns: `#`, `Split`, `User`, `Role`, `Query` (120 chars), `Verdict`,
`Sub-queries`, `Allowed`, `Denied`, `Denied Resources` (up to 3), `OrgAccess Expected`.

The report uses openpyxl's write-only streaming mode, so memory usage stays
flat even at 75k rows — it never holds the full sheet in memory.

---

## Running tests

```bash
python tests/test_access_check.py
```

Tests the access-check logic directly (no Ollama dependency) — allow,
deny, partial, wildcard resolution, explicit-deny-overrides-allow, unknown
resource default-deny, and unknown user handling.

---

## Files

| File | Purpose |
|---|---|
| `.env.example` | Template for the `.env` config file — copy to `.env` and edit. |
| `src/config.py` | Reads `.env` + env vars; exposes typed getters for all settings. Loaded automatically on import. |
| `config/permissions.yaml` | Flat role → allowed/denied resource map. Resources are `table.field` strings; `table.*` and bare `*` are wildcards. Explicit deny always wins over allow. |
| `config/users.yaml` | User → role mapping. |
| `src/resource_vocab.py` | Extracts the closed resource-ID vocabulary from `permissions.yaml` so the LLM has a fixed set of tags to choose from instead of inventing free text. |
| `src/decompose.py` | Calls the local Ollama model, forces JSON-mode output, validates against a pydantic schema, retries once on malformed output. Includes `MockDecomposer` for offline testing. |
| `src/access_check.py` | Resolves each sub-query's resource against the requesting user's role. Default-deny (least privilege) if nothing matches. |
| `src/pipeline.py` | CLI orchestration: query → decompose → check → print. Supports `--mock`, `--live`, `--orgaccess`, `--show-config`. Reads defaults from `.env` via `src/config.py`. |
| `src/load_orgaccess.py` | Fetches `respai-lab/orgaccess` from HuggingFace. `load_as_pipeline_examples_multi(splits, limit, test_user, random_seed)` pools multiple splits, shuffles, samples, and returns pipeline-ready dicts. |
| `src/report.py` | Generates the Excel report. Uses openpyxl write-only (streaming) mode — memory stays flat at 75k rows. `generate_report_with_meta()` is the main entry point from `run_orgaccess()`. |
| `examples/sample_queries.json` | 5 example queries across all 4 roles, each with a canned mock decomposition for offline demoing, covering FULL, PARTIAL, and REJECTED outcomes. |
| `tests/test_access_check.py` | Unit tests for the permission logic. |

---

## Design notes / why these choices

- **Closed resource vocabulary, not free text.** The decomposer is given
  the exact list of resource IDs from `permissions.yaml` and told to pick
  from it (or say `UNKNOWN`). This is what makes the access-check step
  purely programmatic — no regex/NLP matching of arbitrary LLM phrasing
  against permission names.
- **Default-deny.** If a resource doesn't match any `allowed_resources`
  rule for the role, it's denied — not silently ignored. Same for
  `UNKNOWN` resources the decomposer couldn't classify.
- **Explicit deny beats wildcard allow.** e.g. `cfo` has `financials.*`
  allowed but `employees.performance_reviews` denied even if some future
  rule broadened employee access — deny rules are checked first.
- **Model swappability.** `LLMDB_OLLAMA_MODEL` env var (or `.env`) or
  `--model` flag; nothing in `decompose.py` hardcodes a specific model.
- **Config layering.** `.env` sets project defaults; shell env vars
  override those; CLI flags override everything. `--show-config` prints
  the merged result.

---

## Explicitly NOT in this phase

- No relationship graph / ReBAC join-sensitivity edges
- No cross-query or session-level memory
- No FragFuse-style attack simulation or leakage-rate benchmarking
- No RAG/vector DB integration
- No schema-alignment layer between OrgAccess permissions and `table.field` vocab

These are the next phases, once this foundation is validated with your
professor.
