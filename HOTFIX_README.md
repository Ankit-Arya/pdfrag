# pdfrag Smart RAG Hotfix - Acronyms + Amendment-aware Compensation Retrieval

Base expected state:
- Repository: Ankit-Arya/pdfrag
- Original inspected commit: 89162d856c568141c5863def73377342be3c786c
- Apply this ZIP **on top of** the previously supplied `pdfrag-smart-rag-patch-89162d8.zip` overlay.

This is an incremental backend-only hotfix. It contains no frontend files and does not replace the repository tree.
Extract it over an intact pdfrag checkout that already has the Smart RAG patch.

## Fix 1 - acronym / full-form questions

Problems fixed:
- `BIC full form` could retrieve BIC usage but miss the actual definition.
- Bare `BIC` was intentionally treated as corpus-reference mode and dumped occurrences.
- A small legacy occurrence scan could exhaust its limit on strings such as `BIC isolation` before reaching `Brake Isolating cock(BIC)`.

New behavior:
- Bare uppercase acronyms and explicit definition wording (`BIC`, `BIC full form`, `what is BIC`, `what does BIC mean`, `expand BIC`) use answer mode.
- Explicit navigation (`find BIC`, `show references to BIC`) remains reference mode.
- The terminology table is checked first.
- The original PDF definition chunk is force-retrieved and retained for citation.
- If the terminology backfill is missing/stale, a definition-shaped corpus fallback prioritizes `Long Form (ABC)`, `ABC (Long Form)`, `ABC - Long Form`, and glossary-table patterns rather than arbitrary acronym occurrences.
- Procedure questions such as `what is BIC procedure` are not reclassified as definition requests.

Expected result for your corpus:
- `BIC` -> `BIC stands for Brake Isolating Cock` with a PDF citation.
- `BIC full form` -> same.

## Fix 2 - compensation retrieval and old-vs-amended schedule

Problems fixed:
- Broad OR FTS allowed common words such as accident/injury to swamp the discriminative term compensation.
- `compensation amount for different injuries` was handled as a single fact rather than a category/value schedule lookup.
- Similar accident-response prose could make retrieval appear confident even though it contained no monetary compensation evidence.
- The supplied Claims PDF contains both the appended 2017 schedule and the 2025 replacement schedule, so conventional relevance can return the clean old value instead of the current amended value.

New behavior:
- Adds a high-signal AND FTS arm for discriminative query terms.
- Adds monetary/table answer-shape retrieval for compensation/claim questions.
- Monetary queries are considered answerable only when retrieved evidence actually contains compensation/claim wording plus a monetary-looking number; related accident prose alone is insufficient.
- Multi-category compensation requests are routed as list/structured lookups.
- For version-sensitive lookups, candidates near a preceding explicit amendment/substitution chunk receive an authority-context marker.
- The actual amendment/substitution chunk is force-added to final evidence so the answer model can cite the replacement instruction.
- The final-answer prompt is instructed to use amended/substituted text only when the PDF explicitly establishes the replacement relationship. Proximity alone is not treated as legal authority.

For the supplied Claims Rules PDF this is designed to keep the 2025 replacement schedule and the explicit `Second Schedule ... shall be substituted` anchor ahead of the appended 2017 schedule.

Regression expectations:
- `what is compensation amount in case of different injuries and accidents` -> structured current schedule/list, not `not specified`.
- `how much compensation ... loss of one eye ...` -> current amended value `3,20,000`, not old `1,60,000`.
- `loss of vision of one eye ...` remains a distinct row from physical loss of one eye.

## Files replaced

- `backend/app/rag/terminology.py`
- `backend/app/rag/smart_retrieval.py`
- `backend/app/rag/smart_runtime.py`
- `backend/tests/test_smart_terminology_extract.py`

New test file:
- `backend/tests/test_smart_query_shapes.py`

No changes are made to frontend, PDF viewer, admin reprocess UI, chat streaming, database source chunks, OCR, or raw PDF storage.

## Apply

From the repository root, extract the ZIP over the existing checkout. Do not delete your repository first.

Then rebuild only the backend:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.smart-rag.yml \
  build backend

docker compose \
  -f docker-compose.yml \
  -f docker-compose.smart-rag.yml \
  up -d backend
```

No schema migration is added by this hotfix.

If Smart RAG terminology was never backfilled, diagnostics show a suspiciously low terminology count, or `BIC` still has no terminology entry, refresh the derived indexes:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.smart-rag.yml \
  exec backend python -m app.rag.smart_backfill
```

This uses the existing document chunks; it does not re-OCR all PDFs.

Then run diagnostics:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.smart-rag.yml \
  exec backend python -m app.rag.smart_diagnostics
```

## Optional tuning

These new variables have safe defaults and do not need to be added to `.env` initially:

```env
SMART_RAG_PRIORITY_FTS_K=80
SMART_RAG_AMOUNT_TABLE_K=80
SMART_RAG_AMENDMENT_WINDOW_CHUNKS=12
SMART_RAG_CURRENT_AMENDMENT_CONTEXT_CHUNKS=12
```

Keep the defaults for the first test run.

## Mandatory smoke tests

1. `BIC`
   - Should synthesize the full form, not dump references.

2. `BIC full form`
   - Should answer `Brake Isolating Cock` from original PDF evidence.

3. `find BIC`
   - Should remain reference/document-occurrence mode.

4. `what is BIC procedure`
   - Must not be mistaken for a full-form query.

5. `what is compensation amount in case of different injuries and accidents`
   - Must retrieve compensation schedule evidence rather than accident-response SOPs.

6. `how much compensation is to be given to a person who has loss of one eye in an accident in metro system?`
   - Expected current value: `3,20,000` from the amended schedule.
   - Must not return old `1,60,000`.

7. `how much compensation is payable for loss of vision of one eye without disfigurement, other eye normal?`
   - Test separately from physical loss of one eye.

## QA performed

Focused local hotfix suite:
- 32 tests passed.
- Python compileall passed.
- Includes existing scenario/rule tests plus new acronym/query-shape/amendment-marker regression tests.

The complete repository test suite could not be run in this build environment because outbound GitHub cloning is unavailable. Run your normal CI/staging suite before production deployment.
