# LLM-DB Phase 1 — Codebase Breakdown

This document explains what every file does, how they connect, what
arguments/defaults exist, and where to hook in future phases.

---

## 1. The big picture

```
                    ┌─────────────────────────────┐
                    │   config/permissions.yaml    │  flat role -> resource rules
                    │   config/users.yaml          │  user -> role mapping
                    └──────────────┬───────────────┘
                                   │ read by
                                   ▼
   user query   ┌──────────────────────────────────┐
   ───────────► │  src/pipeline.py  (orchestrator)  │
                └───────┬───────────────────┬────────┘
                        │                   │
                        ▼                   ▼
          ┌─────────────────────┐  ┌──────────────────────┐
          │  src/decompose.py    │  │  src/access_check.py  │
          │  (query -> sub-      │  │  (sub-query resource  │
          │   queries via Ollama)│  │   -> allow/deny)      │
          └──────────┬───────────┘  └───────────┬───────────┘
                     │ uses                      │ uses
                     ▼                            ▼
          ┌─────────────────────┐      config/permissions.yaml
          │ src/resource_vocab.py│      config/users.yaml
          │ (closed vocabulary   │
          │  from permissions.   │
          │  yaml)                │
          └───────────────────────┘

   src/load_orgaccess.py  — standalone, NOT part of the pipeline above.
   Pulls the OrgAccess benchmark from Hugging Face into data/orgaccess_raw/
   for future use as an eval set. Nothing else imports it yet.
```

Two-stage pipeline, one config source of truth:

1. **Decompose** — split one natural-language query into sub-queries, each
   tagged with a resource ID.
2. **Check** — look up the requesting user's role, check each tagged
   resource against that role's allow/deny rules.
3. **Aggregate** — combine the per-sub-query decisions into one verdict:
   `FULL` (everything allowed), `PARTIAL` (some allowed, some not), or
   `REJECTED` (nothing allowed).

---

## 2. File-by-file

### `config/permissions.yaml` — the permission source of truth
Flat map: `role -> {allowed_resources: [...], denied_resources: [...]}`.

- Resource IDs are `table.field` strings, e.g. `budget.department_total`.
- Wildcards: `table.*` (any field in that table), or bare `*` (everything).
- Roles currently defined: `hr_intern`, `finance_manager`, `cfo`, `sales_rep`.

**This file is the single source of truth two different ways:**
- `access_check.py` reads it directly to decide allow/deny.
- `resource_vocab.py` reads it to build the *list of valid resource names*
  that the LLM decomposer is allowed to tag things with. So adding a new
  resource here automatically makes it available to both the decomposer's
  prompt and the checker's rules — no need to update it in two places.

### `config/users.yaml` — who has which role
Simple `user_id -> {role, display_name}` map. Read only by
`access_check.py` (`PermissionStore.role_for_user`).

### `src/resource_vocab.py` — permissions.yaml → closed vocabulary
One function: `load_resource_vocab(permissions_path) -> list[str]`.
Scans every role's `allowed_resources` + `denied_resources`, strips out
wildcard entries (`table.*`, `*`), returns a sorted deduplicated list of
concrete resource IDs.

**Why it exists:** without a closed vocabulary, the LLM in the decompose
step would invent its own resource names in free text, and the access-check
step would have no reliable way to match those names against the
permission rules. This file is what makes the two stages speak the same
language.

Called by: `decompose.py` (`decompose_query`, inside `_build_system_prompt`).

### `src/decompose.py` — Stage 1: query → tagged sub-queries

**Public function:**
```python
decompose_query(
    user_query: str,
    model: str = DEFAULT_MODEL,          # default: "qwen2.5:7b-instruct"
    permissions_path: Optional[str] = None,  # default: config/permissions.yaml
    max_retries: int = 1,
) -> DecompositionResult
```

`DEFAULT_MODEL` is read from the env var `LLMDB_OLLAMA_MODEL` if set,
otherwise falls back to `"qwen2.5:7b-instruct"`. This is how you swap
models without touching code — either export the env var, or pass
`--model` on the CLI (see `pipeline.py` below).

**What it does, step by step:**
1. Loads the closed resource vocabulary via `resource_vocab.load_resource_vocab`.
2. Builds a system prompt (`_build_system_prompt`) that lists every valid
   resource ID and instructs the model to output ONLY JSON matching a
   fixed schema — a list of `{text, resource, rationale}` objects.
3. Calls `ollama.chat(...)` with `format="json"` (Ollama's JSON-mode
   constraint) and `temperature=0.0` for determinism.
4. Parses and validates the response against the `DecompositionResult`
   pydantic model (`sub_queries: List[SubQuery]`, each `SubQuery` has
   `text`, `resource`, `rationale`).
5. If parsing/validation fails, retries once (`max_retries=1` by default)
   with a stricter reminder appended to the prompt. If it still fails,
   raises `DecompositionError` — **it does not fall back to regex
   parsing**, by design (see project notes: "don't rely on free-text
   parsing").
6. If the model can't confidently map a sub-question to any resource in
   the vocabulary, it's expected to tag it `"UNKNOWN"` — which
   `access_check.py` treats as an automatic deny.

**`MockDecomposer` class:** a deterministic stand-in used only for
offline testing/demoing (`--mock` flag). Takes a dict of
`{query_text: canned_response_dict}` and returns validated
`DecompositionResult` objects without touching Ollama at all. Not used in
the live pipeline path — only wired in via `pipeline.py`'s `--demo --mock`.

### `src/access_check.py` — Stage 2: tagged sub-queries → allow/deny → verdict

**Key classes:**
- `PermissionStore` — loads `permissions.yaml` + `users.yaml` once (in
  `__init__`), then answers two kinds of questions:
  - `role_for_user(user) -> str` — raises `AccessCheckError` if the user
    isn't in `users.yaml`.
  - `check_resource(role, resource) -> (Decision, reason_string)` — the
    actual rule resolution.
- `Decision` enum: `ALLOW` / `DENY` (per sub-query).
- `Verdict` enum: `FULL` / `PARTIAL` / `REJECTED` (per whole query, after
  aggregation).

**Rule resolution order in `check_resource`** (this is the core logic —
important to know when adding new roles or resources):
1. If `resource == "UNKNOWN"` → deny immediately (decomposer couldn't
   classify it).
2. Check `denied_resources` first — if any pattern matches → **DENY**.
   Explicit deny always wins, even if a broader wildcard would allow it.
3. Check `allowed_resources` — if any pattern matches → **ALLOW**.
4. If nothing matched either list → **DENY** (default-deny / least
   privilege — an unlisted resource is never implicitly allowed).

Pattern matching (`_matches`, static method): exact string match, or
`table.*` matches anything starting with `table.`, or bare `*` matches
everything.

**Public function:**
```python
check_query(
    user: str,
    original_query: str,
    decomposition: DecompositionResult,   # from decompose.py
    store: Optional[PermissionStore] = None,  # created fresh if not passed in
) -> QueryResult
```
Loops over every sub-query in the decomposition, calls `check_resource`
for each, collects `SubQueryResult` objects, then aggregates via
`_aggregate()`:
- all ALLOW → `FULL`
- all DENY → `REJECTED`
- mixed → `PARTIAL`

Note the verdict names (`full`/`partial`/`rejected`) deliberately match
the OrgAccess benchmark's response taxonomy, since that's the natural
future eval target.

### `src/pipeline.py` — orchestration + CLI entry point

This is the file you actually run. It imports from both `decompose.py`
and `access_check.py` and wires them together.

**`run_single(user, query, model=DEFAULT_MODEL, store=None, mock_decomposition=None) -> QueryResult`**
The core function other code should call if using this as a library. If
`mock_decomposition` (a dict) is given, it skips Ollama entirely and
validates that dict directly into a `DecompositionResult`. Otherwise it
calls `decompose_query(...)` for real, then `check_query(...)`.

**`run_demo(model, mock) -> list[QueryResult]`**
Reads `examples/sample_queries.json`, runs `run_single` on every entry.
If `mock=True`, requires each example to carry a `mock_decomposition` key
(skips + warns on stderr if missing). If `mock=False`, calls the real
Ollama pipeline for each example query.

**`print_result(result)`** — formats a `QueryResult` to stdout: user,
role, original query, overall verdict, then each sub-query with its
ALLOW/DENY tag, resource, and reason.

**CLI arguments** (via `argparse` in `main()`):

| Flag | Required? | Default | Meaning |
|---|---|---|---|
| `--user` | required unless `--demo` | — | user id, must exist in `config/users.yaml` |
| `--query` | required unless `--demo` | — | the natural-language query string |
| `--model` | optional | `DEFAULT_MODEL` (`qwen2.5:7b-instruct`, or `$LLMDB_OLLAMA_MODEL` if set) | Ollama model name to use for decomposition |
| `--demo` | optional flag | `False` | run every entry in `examples/sample_queries.json` instead of a single query |
| `--mock` | optional flag | `False` | only valid together with `--demo` — skip Ollama, use canned decompositions from the examples file |

**Invocation examples:**
```bash
# single live query (hits Ollama)
python -m src.pipeline --user daniel.finance_manager --query "Show me Sales headcount and this quarter's budget"

# all 5 example queries, live
python -m src.pipeline --demo

# all 5 example queries, offline (no Ollama needed)
python -m src.pipeline --demo --mock

# swap model
python -m src.pipeline --demo --model llama3.1:8b-instruct-q4_K_M
```

If you call `--mock` without `--demo`, `main()` exits with an argparse
error (`--mock` needs a canned response source, which only the demo file
provides right now).

### `examples/sample_queries.json` — demo/test fixtures
A JSON array of 5 objects, each:
```json
{
  "user": "daniel.finance_manager",
  "query": "Show me Sales headcount and this quarter's budget",
  "mock_decomposition": { "sub_queries": [ {"text": ..., "resource": ..., "rationale": ...}, ... ] }
}
```
`user` must exist in `config/users.yaml`. `mock_decomposition` is only
read when `--mock` is passed; it's ignored (and a real Ollama call is
made instead) when running `--demo` without `--mock`. The 5 examples were
chosen to hit all three verdicts at least once (FULL, PARTIAL, REJECTED)
across all 4 roles.

### `src/load_orgaccess.py` — standalone dataset fetch (not wired into the pipeline)
Pulls `respai-lab/orgaccess` from Hugging Face via the `datasets` library
and writes each split (`easy`/`medium`/`hard`) to
`data/orgaccess_raw/{split}.jsonl`.

**CLI arguments:**

| Flag | Default | Meaning |
|---|---|---|
| `--split` | all three (`easy`, `medium`, `hard`) | fetch just one split |
| `--limit` | `None` (no cap) | cap rows per split, useful for a quick sample |

Nothing else in the codebase imports this file. It's disconnected from
`pipeline.py` on purpose — see the docstring at the top of the file for
why (schema mismatch between OrgAccess's free-form permission dicts and
our fixed `table.field` resource vocabulary). It exists purely so the raw
data is sitting in the repo, ready for whoever builds the eval-harness
phase.

### `tests/test_access_check.py` — unit tests for Stage 2 only
Deliberately does **not** touch `decompose.py` or Ollama — it builds
`DecompositionResult` objects directly with `make_decomp(*resources)` and
feeds them straight into `check_query`. Covers: full-allow, full-reject,
partial, wildcard-allow, unknown-resource-default-deny,
unknown-user-raises. Run with `python3 tests/test_access_check.py`
(plain assertions, no pytest dependency — prints PASS/FAIL per test).

### `.gitignore`, `requirements.txt`, `README.md`
Standard housekeeping. `requirements.txt` splits into two logical
groups: the three actually used by the live pipeline
(`ollama`, `pyyaml`, `pydantic`), plus `datasets` + `huggingface_hub`
which are only needed if you run `load_orgaccess.py`.

---

## 3. How a request flows through the code, concretely

Example: `python -m src.pipeline --user daniel.finance_manager --query "Show me Sales headcount and this quarter's budget"`

1. `pipeline.main()` parses args, calls `run_single(user="daniel.finance_manager", query="...", model=DEFAULT_MODEL)`.
2. `mock_decomposition` is `None` → `run_single` calls `decompose_query(query, model=DEFAULT_MODEL)`.
3. Inside `decompose_query`:
   - `load_resource_vocab("config/permissions.yaml")` returns something like `["budget.department_total", "employees.headcount", ...]`.
   - `_build_system_prompt(vocab)` embeds that list into the system prompt template.
   - `_call_ollama(model, system_prompt, query)` sends the request to the local Ollama server, JSON-mode, temperature 0.
   - Response is JSON-parsed and validated into a `DecompositionResult` with two `SubQuery` objects (headcount, budget).
4. Back in `run_single`: `check_query(user, query, decomposition, store=None)` is called.
5. Inside `check_query`: a fresh `PermissionStore()` loads both YAML configs. `role_for_user("daniel.finance_manager")` returns `"finance_manager"`.
6. For each sub-query, `store.check_resource("finance_manager", "employees.headcount")` and `store.check_resource("finance_manager", "budget.department_total")` are called — both match `allowed_resources` for that role → both `ALLOW`.
7. `_aggregate([ALLOW, ALLOW])` → `Verdict.FULL`.
8. `run_single` returns a `QueryResult`; `pipeline.main()` calls `print_result(result)` to print it.

---

## 4. Planning future changes — where things plug in

| If you want to... | Touch this file | Notes |
|---|---|---|
| Add a new role or resource | `config/permissions.yaml` only | Automatically flows into both the LLM vocabulary and the access rules — no code changes needed. |
| Add/remove a user | `config/users.yaml` only | |
| Change how sub-queries are tagged (e.g. multi-resource tags, confidence scores) | `src/decompose.py` (`SubQuery` model + prompt) | Will also require updating `access_check.py`'s `check_resource` signature if the tag shape changes. |
| Change the allow/deny/default-deny logic | `src/access_check.py` (`check_resource`, `_matches`) | Unit tests in `tests/test_access_check.py` should be extended alongside any change here. |
| Add the relationship graph / ReBAC layer (Phase 2) | New module, e.g. `src/graph_check.py`, called from `pipeline.py` alongside or instead of `access_check.py` | `QueryResult`/`SubQueryResult` dataclasses will likely need a session/history field added. |
| Add cross-query / session memory (Phase 2) | `src/pipeline.py` — needs a session object threaded through `run_single` calls | Currently every `run_single` call is fully independent/stateless; this is the biggest structural change coming. |
| Wire in the OrgAccess dataset as an eval harness (later phase) | New module, e.g. `src/eval_orgaccess.py`, reading `data/orgaccess_raw/*.jsonl` | Needs a schema-translation layer since OrgAccess's permission dicts don't map 1:1 to `table.field` resource IDs yet — that translation logic doesn't exist anywhere currently. |
| Swap/add Ollama models | No code change | `--model` flag or `LLMDB_OLLAMA_MODEL` env var. |

---

## 5. Known gaps (intentional, Phase 1 scope)

- Every `run_single` / `check_query` call is **stateless** — no memory of
  prior queries in the same session. This is exactly the gap FragFuse-style
  attacks exploit, and it's the whole point of the next phase.
- `access_check.py` only ever sees resources one at a time — it has no
  concept of "these two resources together reveal something neither does
  alone."
- `load_orgaccess.py` is disconnected from everything else — data sits on
  disk, unused, until the eval-harness phase translates its schema.
