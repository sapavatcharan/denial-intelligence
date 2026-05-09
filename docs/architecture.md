# Architecture

## High-level flow

```
+-----------------+        +----------------------+        +-----------------+
|  837 + 835 raw  |  -->   |   ingest.py builds   |  -->   |   Claim object  |
|     records     |        |  Claim from flat in  |        |  (Pydantic v2)  |
+-----------------+        +----------------------+        +--------+--------+
                                                                    |
                  +-------------------------------------------------+
                  |
                  v
+----------------------------------+      +--------------------------------+
| Deterministic Evidence Layer     |      | Hybrid Retrieval               |
| - timely_filing                  |      | - structured filter            |
| - prior_auth                     |      |   (payer, ins, proc family)    |
| - coding (modifiers)             |      | - TF-IDF cosine on claim       |
| - medical_necessity (dx/proc)    |      |   "signature"                  |
| - duplicate                      |      | - top-K similar paid claims    |
| - missing_info (RARC-driven)     |      | - per (payer x proc x carc)    |
+---------------+------------------+      |   pattern aggregator           |
                |                         +----------------+---------------+
                v                                          |
       +------------------+                                |
       | EvidenceItem[]   |                                |
       +--------+---------+                                |
                |                                          |
                v                                          v
        +-------------------------------+      +-------------------------+
        |      Root-Cause Agent         | <--- | LLM Router              |
        |  - assemble grounded prompt   |      |  triage  -> 4o-mini     |
        |  - call LLM (or mock)         |      |  strong  -> 4o          |
        |  - grounding gate (re-prompt) |      +-------------------------+
        |  - confidence calibration     |
        +---------------+---------------+
                        |
                        v
                +---------------+      +--------------------------+
                |    Verdict    | ---> | Eval Harness             |
                | (json+md+ui)  |      | - accuracy, F1, Brier    |
                +-------+-------+      | - grounding rate         |
                        |              | - keyword recall         |
                        v              | - per-scenario, per-cost |
              +-------------------+    +--------------------------+
              | Clustering layer  |
              | - by (payer,      |
              |   carc_family,    |    +-----------------------+
              |   proc_family)    |--> | Batch Brief (md+json) |
              | - $ ranked        |    +-----------------------+
              | - LLM narratives  |
              +-------------------+
```

## Module layout and responsibilities

### `src/gabeo/schemas.py`
The single source of truth for typed data. `Claim` joins `ClaimSubmission`
(837) and `Remittance` (835), and exposes derived helpers (`is_denied`,
`primary_carc`, `all_diagnoses`, `primary_procedure`). `Verdict` is the public
output: recoverability enum, calibrated confidence + components, recommended
action, LLM citations (`SupportingEvidenceCitation`), full deterministic
evidence trail, and cost / latency / model telemetry.

### `src/gabeo/ingest.py`
Parses flat dictionaries into typed objects. Handles two non-trivial cases:

* `pcl_ProcedureModifier1`–`pcl_ProcedureModifier4` are flattened on
  remittance lines but must populate `RemittanceLine.modifiers`.
* `pcl_RemarkCodes` is a comma-delimited string but must populate
  `RemittanceLine.remark_codes` as a list.

These are extracted explicitly (`_gather_modifiers`, `_gather_remark_codes`)
because Pydantic's `model_validate` won't trigger validators on aliased fields
that arrive flat.

### `src/gabeo/reference.py`
Loads four curated JSON tables and caches them with `lru_cache`:

* `carc_codes.json` — CARC code → family / description / appealability /
  common resolution.
* `rarc_codes.json` — RARC code → description / `missing_field_hint`.
* `payer_filing_limits.json` — payer × insurance type → days from service
  (or from secondary EOB) and notes about secondary anchoring.
* `dx_procedure_pairings.json` — diagnosis → procedure compatibility,
  prior-auth requirement, common modifiers.

### `src/gabeo/evidence/`
Six independent extractors, each producing zero or more `EvidenceItem`
objects with a fixed shape (`check_name`, `passed`, `severity`, `message`,
`fields_referenced`). They run for every claim regardless of CARC because the
evidence trail is useful even when the LLM is silent.

### `src/gabeo/llm/`
* `client.py` — thin OpenAI wrapper. Returns text + parsed Pydantic + token
  counts + USD cost + latency. Retries are gated by `_should_retry` so
  terminal errors (`AuthenticationError`, `BadRequestError`,
  `insufficient_quota`) fail immediately without burning the
  exponential-backoff budget.
* `router.py` — picks `gpt-4o-mini` for low-complexity evidence packs and
  `gpt-4o` when the deterministic layer surfaces nuance (mixed signals,
  secondary diagnoses, secondary EOB anchoring).
* `mock_client.py` — deterministic, grounded mock that produces verdicts that
  pass the grounding gate. Used in tests, in CI, and as the runtime fallback
  whenever the real client raises.

### `src/gabeo/agents/`
* `grounding.py` — closed list of `_FIELD_ACCESSORS` mapping cite-able field
  names to lambdas that read the actual claim. Returns a list of human-readable
  violation strings; empty list = grounded.
* `root_cause_agent.py` — orchestrates the per-claim pipeline:
    1. Run deterministic evidence.
    2. Pick model via the router.
    3. Build a structured user prompt that includes CARC/RARC context, the
       claim's flat field table, the evidence trail, optional historical
       context, and an explicit `<allowed_field_names>` list.
    4. Call the LLM, parse to `_LLMVerdict`.
    5. Run the grounding gate; on violations, re-prompt once with a
       corrective hint. If still violated, downgrade recoverability to
       `needs_review`.
    6. Calibrate confidence (50/50 blend of LLM self-report and evidence
       pass rate).
    7. Return a `Verdict` with full telemetry.

### `src/gabeo/retrieval/`
* `claim_signature` packs payer, insurance, procedure, procedure family,
  place of service, specialty, amount bucket, and dx tokens into a short
  whitespace-separated string.
* `SimilarityIndex` does (a) hard structured filtering on payer,
  insurance type, and CPT family, then (b) TF-IDF cosine ranking inside the
  filtered set.
* `aggregate_payer_procedure_carc_stats` produces per-(payer × procedure ×
  CARC) denial rate, denied $, and a recovery proxy (paid $ / (paid + denied)
  $) for the (payer × procedure) pair, used by both the agent and the
  clustering layer.

### `src/gabeo/clustering/`
Groups denials by `(payer, carc_family, procedure_family)` and ranks by total
denied $. Each cluster carries top procedures, top diagnoses, claim IDs,
historical denial rate, and recovery proxy. The top-N clusters get an
LLM-narrated paragraph that a manager can read in 15 seconds; the tail uses a
deterministic templated narrative with the same structure.

### `src/gabeo/eval/`
Pure metrics in `metrics.py`; the harness in `harness.py` runs the agent on
every gold-labeled denial, validates each citation against the grounding gate,
and reports accuracy, macro-F1 + per-class precision/recall, confusion matrix,
grounding rate, Brier score, root-cause keyword recall, per-scenario accuracy,
total cost, and average latency. Outputs are both Markdown (for review) and
JSON (for diffs / CI).

### `src/gabeo/cli.py` / `demo/streamlit_app.py`
Typer CLI and Streamlit demo. Both share the same agent, retrieval index,
clustering layer, and eval harness; they are pure UI.

## Failure modes the system explicitly handles

* **No OpenAI key, dead key, or quota exhausted** → `RootCauseAgent` and
  `build_batch_brief` fall back to the deterministic mock; no exception
  reaches the user.
* **LLM hallucinates a field name or value** → grounding gate catches it,
  triggers a corrective re-prompt; persistent violations downgrade
  recoverability to `needs_review`.
* **CARC code outside the curated catalog** → `carc()` returns `None`; the
  agent still emits a verdict, the deterministic evidence trail is unaffected,
  and the prompt explicitly notes the code is unknown to the local catalog.
* **Empty corpus / no similar paid claims** → similarity returns `[]`, the
  prompt simply omits the historical context block.
* **Service-line modifiers and remark codes flattened in the wire format** →
  `ingest._gather_modifiers` / `_gather_remark_codes` rebuild the lists before
  Pydantic validation runs.
