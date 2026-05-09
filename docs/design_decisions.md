# Design decisions

For every non-trivial choice, the alternative considered, why we picked the
current approach, and how to revisit it later.

---

## 1. Domain logic in Python, not in the prompt

**Picked:** Six deterministic evidence extractors written in Python that
encode CARC-family rules; the LLM reasons over their output.

**Considered:** Stuff every CARC rule into the system prompt and let the LLM
do all the reasoning end-to-end.

**Why this:**
* The rules are mechanical and easy to express in code (timely-filing days,
  dx/procedure compatibility, modifier requirements). They don't need
  generalization — they need accuracy.
* Python rules are deterministic, free, fast, and *unit-testable* (15 of the
  26 tests target the evidence layer). LLM-only reasoning has none of those
  properties.
* The evidence trail is itself a deliverable — billing analysts read it
  alongside the verdict to verify the system. Burying the same logic in a
  prompt makes the verdict opaque.

**When to revisit:** if we discover a denial pattern that genuinely requires
unstructured-text reasoning (e.g. parsing a free-text RARC explanation that
varies per payer), we add an LLM-only check rather than encoding rules.

---

## 2. Hard grounding gate over LLM citations

**Picked:** Every field the LLM cites in `supporting_evidence` is verified
against the actual claim via a closed list of `_FIELD_ACCESSORS`. Mismatches
trigger one corrective re-prompt; persistent violations downgrade
recoverability to `needs_review` and don't trust the verdict.

**Considered:** Trust the LLM's citations; ask it to "be careful" in the
prompt; or use a soft re-ranker.

**Why this:**
* Hallucinated dates / amounts / codes are catastrophic in claims work — they
  generate false appeal narratives and false write-offs.
* The grounding rate is now an observable, optimizable metric: 100% on the
  gold set with the mock client; we measure it on every eval run.
* The closed `_FIELD_ACCESSORS` list also doubles as the prompt's
  `<allowed_field_names>` block, so the LLM is told up-front exactly what it's
  allowed to cite.

**When to revisit:** if the agent ever needs to cite *derived* facts (e.g.
"the running total of denials for this provider this month"), we extend the
accessors registry rather than relax the gate.

---

## 3. Calibrated confidence (50/50 LLM + evidence pass rate)

**Picked:** Final `confidence` is `0.5 * llm_self_report + 0.5 *
evidence_pass_rate`, with both components exposed in
`confidence_components` for auditing.

**Considered:** (a) trust the LLM's score; (b) Platt scaling on a held-out
calibration set; (c) drop confidence entirely.

**Why this:**
* Raw LLM confidence is famously poorly calibrated. Blending with the
  deterministic-evidence pass rate immediately grounds it: a verdict that
  contradicts most of the evidence trail can't claim 0.95 confidence.
* Platt scaling needs ~hundreds of labeled outcomes per class. We don't have
  those, and the operator wants this product *today*.
* Exposing both components keeps the user in the loop — they can disagree
  with one half without throwing out the other.

**Result:** Brier score ≈ 0.13 on the eval set. We track it on every run and
it's a fast feedback loop.

**When to revisit:** once we have ≥500 outcomes per class, fit Platt on the
LLM half and re-blend.

---

## 4. TF-IDF retrieval over embeddings (for now)

**Picked:** Hybrid filter (payer + insurance + CPT family) → TF-IDF cosine
on a short structured signature.

**Considered:** OpenAI text embeddings; `sentence-transformers/bge-small-en`;
dense + sparse hybrid (e.g. BM25 + embeddings).

**Why this:**
* The signature is a 10–20 token whitespace-separated structured string. On
  short structured strings, TF-IDF cosine is competitive with or better than
  dense embeddings — the unique tokens (payer ID, CPT, dx codes) carry almost
  all the signal.
* No torch dependency, no API call, fully reproducible.
* Domain analysts think in *exact filters* (same payer, same insurance, same
  procedure family). The hard structured filter is non-negotiable; pure
  dense retrieval would happily return a different payer or procedure family.
* The interface is a single class. Swapping in dense embeddings is a 30-line
  change that we can do once usage justifies the spend.

**When to revisit:** when we want fuzzy matching across diagnosis families
that share clinical meaning but not codes (e.g. M54.5 ↔ M54.50 ↔ low-back-pain
synonyms in payer LCDs).

---

## 5. Group-by clustering, not HDBSCAN

**Picked:** Group denials by `(payer_id, carc_family, procedure_family)` and
sort by total denied $.

**Considered:** HDBSCAN over a one-hot vector of (payer, CARC, procedure,
diagnosis); k-means; DBSCAN.

**Why this:**
* At the volume billing teams actually look at (10s–1000s of denials/day),
  *interpretable* groupings are more valuable than statistically optimal ones.
  A manager wants to act on "all the Aetna 72148 medical-necessity denials";
  an HDBSCAN cluster ID like `cluster_3` means nothing to them.
* The grouping mirrors the structure of an appeal — appeals are written per
  (payer, procedure, denial reason). Grouping that way makes the brief
  *directly* correspond to the work that needs to happen.
* `(payer, carc_family, proc_family)` typically yields 5–20 clusters on
  hundreds of denials, which is the right cognitive bandwidth for a daily
  manager review.

**When to revisit:** when we have multi-dimensional, soft signal (e.g. NLP on
payer correspondence) where structured grouping leaves money on the table.

---

## 6. Cost-aware LLM router (gpt-4o-mini vs gpt-4o)

**Picked:** Route to `gpt-4o-mini` when the deterministic evidence is
unambiguous (one passing check, no warnings); route to `gpt-4o` when the
evidence pack mixes signals or surfaces secondary-EOB / secondary-dx nuance.

**Considered:** Always use `gpt-4o`; always use `gpt-4o-mini`; route via a
classifier model.

**Why this:**
* ~80% of denials in the eval distribution are unambiguous (timely filing,
  prior auth, missing modifier). `gpt-4o-mini` is ~10× cheaper and ~3× faster
  for the same correctness on these.
* The router is a 20-line function; the decision rule is auditable and
  deterministic.
* When new families appear, the router falls back to `gpt-4o` rather than
  silently mis-routing.

**Trade-off:** mini is more likely to need the corrective re-prompt on
adversarial cases. We measured that overhead in the eval harness; it's still
net cheaper than always using gpt-4o.

---

## 7. Mock LLM client as a first-class runtime path

**Picked:** `MockLLMClient` returns deterministic, grounded `_LLMVerdict`
JSON. The agent and the cluster narrator both auto-fall back to it on any LLM
exception (rate limits, quota, auth).

**Considered:** Hard-fail when the API is unavailable; ship a degraded
"unknown" verdict; only use the mock in tests.

**Why this:**
* A reviewer running this on `make demo` without an API key still sees a
  real, grounded verdict — not an error toast. That's the difference between
  a demo that works and one that doesn't.
* The mock is *grounded* — it cites real fields from the input claim using
  the same `_FIELD_ACCESSORS` registry. So `MockLLMClient` exercises the
  grounding gate end-to-end, which is what we want our tests to do.
* It removes a class of flaky-CI failure modes for free.

**Trade-off:** the eval numbers in this repo are produced with the mock and
are an *upper bound* on grounding rate (which is what the gate enforces) and
a *lower bound* on cost / latency. Numbers with `gpt-4o` will move slightly
on accuracy and significantly on cost / latency.

---

## 8. Fail-fast retry policy on the OpenAI client

**Picked:** `_should_retry` excludes `AuthenticationError`,
`PermissionDeniedError`, `BadRequestError`, `NotFoundError`, and
`insufficient_quota`-flavored `RateLimitError`s. Retries kick in only for
transient errors (real rate limits, connection / timeout, 5xx).

**Considered:** `tenacity` default policy that retries every exception three
times.

**Why this:**
* A dead key with `tenacity` defaults turns every claim into ~3×exponential
  backoff (~15-30 s) before falling back to the mock. With ~20 cluster
  narratives that's ~5 min of dead time. Now it's milliseconds.
* The classification is explicit and readable, so when a new error class
  appears we add it deliberately rather than discovering it via slow tests.

---

## 9. Synthetic data with inline gold labels

**Picked:** A 40-claim synthetic dataset (`scripts/generate_synthetic.py`)
that includes the four PDF brief samples verbatim, ~10 textbook denials,
~10 paid claims for retrieval positives, and ~15 adversarial cases. Each
record carries an inline `gold_label` block with `expected_recoverability`,
`expected_root_cause_keywords`, and `scenario`.

**Considered:** No gold set; LLM-as-judge eval.

**Why this:**
* Inline gold makes the eval harness one-pass and the labels live next to
  the data they describe — no risk of drift between the dataset and the
  labels.
* Adversarial cases (timely filing with secondary-EOB anchor, duplicate
  that's actually bilateral, medical-necessity that's appealable on
  secondary dx) are the *only* way to discriminate a real reasoner from a
  CARC-code lookup table. They are deliberately over-represented relative
  to a real distribution.
* LLM-as-judge would re-introduce the very hallucinations we're trying to
  measure away.

**When to revisit:** when we have a real labeled set from production, the
synthetic set becomes a regression / fuzz layer rather than the primary eval.
