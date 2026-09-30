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

### `src/baseline_check.py` — Baseline mode: one LLM call, no decomposition
**Public function:**
```python
get_baseline_verdict(
    user_query: str,
    permissions: Union[str, dict, None],
    user_role: str = "",
    model: str = DEFAULT_MODEL,
    max_retries: int = 1,
) -> BaselineResult
```
Same input as Phase 2's `extract_attributes` (a row's own `permissions`
dict, via the same `parse_permissions()` from `attribute_extract.py`), but
one Ollama call producing one holistic `{verdict, reasoning}` JSON object —
no per-field checklist, no separate deterministic aggregation step. This is
rung zero of the progressive evaluation ladder: the floor Phase 1 and Phase
2's decomposition+aggregation structure is measured against. See
`docs/baseline/plan.md` for why it exists and `docs/roadmap.md` for how it
fits with the still-unbuilt graph-based rungs.

Carries the same reliability guards Phase 2 added after a real incident (see
`docs/phase2/parallel-execution.md`): a `num_predict` cap
(`MAX_PREDICT_TOKENS`) and a client-side timeout (`REQUEST_TIMEOUT_SECONDS`,
shared env var `LLMDB_OLLAMA_TIMEOUT` with `attribute_extract.py`).

### `src/run_full_eval_baseline.py` — eval harness for Baseline mode
Same crash-resumable JSONL checkpointing / progress-ETA logging / Excel
report / `--workers`+`--models` shape as `run_full_eval_v2.py`, calling
`get_baseline_verdict` per row instead of `attribute_extract`+
`attribute_check`. Default sample (`easy medium hard`, limit 1000, seed 42)
matches both other harnesses exactly for direct comparison.

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
| Wire in the OrgAccess dataset as an ABAC eval harness (**Phase 2 — done**, see `docs/phase2/plan.md`) | `src/attribute_extract.py`, `src/attribute_check.py`, `config/attribute_rules.yaml`, `src/run_full_eval_v2.py`, new `orgaccess_attr` pipeline mode | Checks each OrgAccess row against its OWN `permissions` dict (not a fixed `table.field` vocabulary or a role table) — no relationship graph turned out to be needed; the real gap was multi-field ABAC reasoning, not org-hierarchy traversal. See `docs/phase2/plan.md` for what the paper's dataset actually contains vs. what was originally guessed here. |
| Measure the floor decomposition is compared against (**Baseline — done**, see `docs/baseline/plan.md`) | `src/baseline_check.py`, new `baseline` pipeline mode, `src/run_full_eval_baseline.py` | One LLM call, same permissions input as Phase 2, no checklist, no deterministic aggregation. Rung zero of the ladder in `docs/roadmap.md`. |
| Add role inheritance (**Phase 3a, hierarchical role/permission graph — not started**, see `docs/roadmap.md`) | `config/permissions.yaml` (or a new `config/role_hierarchy.yaml`), `src/role_resolution.py`, `src/access_check.py`'s `check_resource` | Flat role table today has no parent/child relationship between roles at all. |
| Add cross-resource / ReBAC checking (**Phase 3b, relationship graph — not started**, see `docs/roadmap.md`) | New module (e.g. `src/relationship_graph.py`), new config (e.g. `config/resource_graph.yaml`), `src/pipeline.py` for the session object this needs | This is the FragFuse-style combinatorial-disclosure gap in section 5 below — needs cross-query/session memory to be meaningful, not just a graph. |
| Add cross-query / session memory (needed by Phase 3b above) | `src/pipeline.py` — needs a session object threaded through `run_single`/`run_baseline`/`run_orgaccess_attr` calls | Currently every call is fully independent/stateless; still true after Phase 2 and Baseline, which are also per-query. |
| Swap/add Ollama models | No code change | `--model` flag or `LLMDB_OLLAMA_MODEL` env var (Baseline: `LLMDB_BASELINE_MODEL` if set independently). |

---

## 5. Known gaps (intentional, Phase 1 scope)

- Every `run_single` / `check_query` call is **stateless** — no memory of
  prior queries in the same session. This is exactly the gap FragFuse-style
  attacks exploit, and it's still true after Phase 2 (see below) — cross-
  query/session memory is still not implemented.
- `access_check.py` only ever sees resources one at a time — it has no
  concept of "these two resources together reveal something neither does
  alone."
- `load_orgaccess.py` is now used by both Phase 1 (`orgaccess` mode) and
  Phase 2 (`orgaccess_attr` mode) via the same
  `load_as_pipeline_examples_multi()` — it is no longer disconnected.

## 6. Phase 2 (added; see `docs/phase2/plan.md` for full detail)

Phase 1's `table.field` closed vocabulary and flat role table cannot
represent OrgAccess's real permission schema (each row has its own
multi-field ABAC-style `permissions` dict — department, access_level,
allowed_actions, location_restriction, numeric thresholds, booleans,
delegation/quorum, ~90 distinct field names total). Phase 2 adds a parallel
path that checks each row against **its own** permissions dict instead of a
shared config:

```
   OrgAccess row  ┌──────────────────────────────┐
   (query +        │ src/attribute_extract.py      │  per-field checklist:
    permissions) ─►│ (Ollama, one call per row)     │  applicable? allow/deny?
                    └──────────────┬─────────────────┘
                                   ▼
                    ┌──────────────────────────────┐   config/attribute_rules.yaml
                    │ src/attribute_check.py         │◄─ (severity: core -> REJECTED,
                    │ (deterministic aggregation)     │   soft -> PARTIAL; PROVISIONAL,
                    └──────────────┬─────────────────┘   pending author's mapping)
                                   ▼
                    FULL / PARTIAL / REJECTED
```

Entry points: `python -m src.pipeline --orgaccess-attr` (single sweep,
prints results) and `python -m src.run_full_eval_v2` (checkpointed bulk eval
with Excel report, mirroring `run_full_eval.py`'s shape). Neither touches
any Phase 1 file. See `docs/phase2/plan.md`, `docs/phase2/attribute-schema.md`,
and `docs/phase2/parallel-execution.md` for the full design rationale,
including a documented real-data case where the current placeholder
severity mapping produces the wrong verdict (`biometric_required`), left
as-is pending the authoritative mapping.

## 7. Baseline mode (added; see `docs/baseline/plan.md` for full detail)

Phase 1 and Phase 2 both decompose the query before deciding — they differ
in *what* they decompose against, not *whether* they decompose. Baseline
removes the decomposition/checklist and the separate deterministic
aggregation step entirely, holding the input information constant (the same
`permissions` dict Phase 2 reads) so the comparison isolates the structure's
contribution, not the information's:

```
   OrgAccess row  ┌──────────────────────────────┐
   (query +        │ src/baseline_check.py         │  ONE call, ONE verdict:
    permissions) ─►│ (Ollama, one call per row,     │  no checklist, no
                    │  no decomposition)             │  separate aggregation
                    └──────────────┬─────────────────┘
                                   ▼
                    FULL / PARTIAL / REJECTED  (+ one-line reasoning)
```

Entry points: `python -m src.pipeline --baseline` (single sweep, prints
results) and `python -m src.run_full_eval_baseline` (checkpointed bulk eval
with Excel report, same default 1,000-query sample as the other two
harnesses). Neither touches any Phase 1 or Phase 2 file.

## 8. Roadmap — the graph-based rungs (designed, not yet built; see `docs/roadmap.md` and `docs/phase3/`)

The full progressive ladder is: Baseline -> + Query decomposition (Phase 1
& 2, both done; Baseline done too — 40.8% agreement, zero FULL verdicts) ->
+ Hierarchical role/permission graph -> + Role-permission mapping -> +
Conflict-resolution mechanism -> Full framework. The graph rung itself
splits into three sub-phases, each with a full architecture design written
in `docs/phase3/` (informed by web research on hierarchical RBAC/Zanzibar-
style ReBAC/FAVA's SMT-verified permission graphs, an empirical check of
whether OrgAccess's own role-name strings support a mined hierarchy — they
only cleanly do for ~20% of roles — and arXiv 2609.09153's self-evolving
"procedural graphs" concept, adapted for access-control severity tuning
rather than agent execution flow):

- **Phase 3a — hierarchical role/permission graph** (role inheritance, a
  small hand-authored tree over the 24 canonical Phase 1 roles, not mined
  from the dataset's raw role strings). See `docs/phase3/role-hierarchy.md`
  for the concrete spine, `config/role_hierarchy.yaml` shape, and the
  precedence policy (own rules always resolve before any inherited rule).
- **Phase 3b — relationship graph / ReBAC** (cross-resource
  combinatorial-disclosure detection — the FragFuse-style gap named in
  section 5 above). See `docs/phase3/relationship-graph.md` — explicitly
  scoped as a stretch goal, not a benchmarked rung, since it needs both new
  session-memory infrastructure and new ground truth OrgAccess can't
  supply.
- **Phase 3c — self-evolving refinement** (new addition to the ladder, not
  originally planned): an LLM refiner proposes edits to
  `config/attribute_rules.yaml`'s severity mapping, validated on a held-out
  split before committing — see `docs/phase3/self-evolution.md`. Reuses
  100% existing infrastructure (checkpoints, judge scores, splits), so it's
  the recommended **first** build, ahead of 3a.

`docs/roadmap.md` has the ladder-level status view; `docs/phase3/plan.md`
has the full comparison table (3a vs. 3b vs. FAVA vs. 3c) and the
recommended build order with reasoning.
