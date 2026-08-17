# pdfrag Smart Scenario RAG Patch

Target repository: `Ankit-Arya/pdfrag`

Patch base: `main` commit `89162d856c568141c5863def73377342be3c786c` (Filter docs and search).

This patch is an additive runtime upgrade for the active `backend/app` implementation. It does not replace PDF/OCR/chunking/citation code and it does not require the existing ~500 PDFs to be OCRed again.

## What this patch changes

It addresses the issues discussed together:

1. **Latency at ~50k chunks**
   - Adds pgvector HNSW index for chunk cosine search.
   - Adds PostgreSQL GIN indexes for `simple` and `english` FTS.
   - Removes the every-document candidate requirement from the normal path by monkey-patching the service's imported retrieval functions.
   - Caps normal query variants to 3.
   - Uses bounded global vector + FTS search, procedure-card routing and a bounded indexed lexical fallback.
   - Caps primary/reference document reads and final evidence context.
   - Existing broad retrieval remains available only as a low-confidence fallback.

2. **Situation-based questions**
   - Compiles the user's wording into a lightweight scenario state (numeric facts, equipment states and a few safe derived states such as `unable_to_proceed`).
   - Builds section/procedure cards from the existing chunk context headers and uses them as a routing layer above raw chunks.
   - Keeps original PDF chunks as final factual evidence.

3. **Logical threshold questions**
   - Extracts deterministic numeric conditions such as `2 or more`, `at least 2`, `above 25`, `below 4.5`, etc.
   - Stores those rules in `rag_rules` and uses the rule index for retrieval.
   - Example: user reports **4 brakes failed**, source says **2 or more brakes failed** -> deterministic evaluation `4 >= 2` -> that source rule is routed as applicable.
   - A mismatch such as 1 brake against `>=2` is penalized rather than treated as applicable.
   - The final LLM sees the comparison clearly marked as **system-derived applicability, not PDF text**, and still cites the underlying PDF rule.

4. **Organisation-wide abbreviations**
   - Builds a global terminology index from definitions found anywhere in the corpus, including forms such as `Station Controller (SC)`, `SC - Station Controller`, `SC: Station Controller`, and glossary/table rows.
   - A question using `SC` can retrieve a procedure that only says `Station Controller`.
   - Expansion is bidirectional in retrieval because the canonical name is added to the search representation while the original question is retained.
   - Ambiguous abbreviations are not silently forced to one meaning when multiple corpus-backed meanings are close.
   - Document codes such as `SC-06` remain document identifiers and are not expanded as `Station Controller-06`.

## Files in the patch

Replace/add these files by extracting this ZIP over the repository root:

- `backend/app/main.py` - small startup hook only; otherwise based on current main.
- `backend/app/rag/scenario_reasoning.py` - scenario facts + deterministic condition evaluation.
- `backend/app/rag/terminology.py` - global abbreviation resolver.
- `backend/app/rag/smart_schema.py` - additive tables and HNSW/GIN indexes.
- `backend/app/rag/smart_index.py` - backfills terminology, procedure cards and rules from existing chunks.
- `backend/app/rag/smart_retrieval.py` - bounded hybrid retrieval + procedure/rule routing.
- `backend/app/rag/smart_runtime.py` - installs the upgrade into the existing `RagService` without rewriting the large service module.
- `backend/app/rag/smart_backfill.py` - one-time backfill command for existing documents.
- `backend/app/rag/smart_diagnostics.py` - diagnostics command.
- `backend/tests/test_smart_reasoning.py` - deterministic reasoning tests.
- `backend/tests/test_smart_terminology_extract.py` - corpus terminology extraction tests.
- `.env.smart-rag.example` - optional tuning values.
- `docker-compose.smart-rag.yml` - Compose override that passes `SMART_RAG_*` settings into the backend without replacing your existing Compose file.

## Install

### 1. Backup your current repo/database

The schema change is additive, but take your normal DB backup before production deployment.

### 2. Extract the ZIP over the pdfrag repo root

The only existing application file overwritten is `backend/app/main.py`. All `smart_*` files are new.

If your repository has moved beyond commit `89162d856c568141c5863def73377342be3c786c`, compare `backend/app/main.py` before overwriting it. The intended manual change is only:

```python
from app.rag.smart_runtime import install_smart_rag_patch
```

and inside lifespan immediately after `initialize_database()`:

```python
install_smart_rag_patch()
```

### 3. Rebuild and start backend

```bash
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml build backend
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml up -d backend
```

On the first startup, the application creates additive tables and attempts the HNSW/GIN indexes. With ~50k chunks this is a one-time database operation. If an optional index cannot be built, the service logs the failure rather than refusing to start.

### 4. Backfill the existing 500-document corpus

Run this **once after deploying the patch**:

```bash
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml exec backend python -m app.rag.smart_backfill
```

This does **not** OCR or rechunk the PDFs. It reads the already-stored `document_chunks` and creates:

- organisation terminology rows,
- section/procedure cards,
- deterministic numeric rules,
- procedure-card embeddings.

Newly processed documents are smart-indexed automatically after the normal document-processing path completes.

### 5. Verify indexes/data

```bash
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml exec backend python -m app.rag.smart_diagnostics
```

You should see non-zero counts for `procedure_cards` and normally for `terminology_rows` / `rules` depending on the actual corpus. You should also see HNSW and FTS index names.

## Test the core behaviors

Run backend tests if your normal dev/test image is available:

```bash
cd backend
pytest -q tests/test_smart_reasoning.py tests/test_smart_terminology_extract.py
```

The supplied tests include:

- 4 failed brakes matches `2 or more failed brakes`.
- 1 failed brake does not match `2 or more failed brakes`.
- failed brakes do **not** automatically satisfy a rule specifically about isolated brakes.
- `SC = Station Controller` is added to the canonical scenario.
- parenthetical/table terminology extraction avoids swallowing unrelated leading words.
- relaxed scenario queries remove the exact user number while retaining the underlying condition concept.

Then test with real corpus questions, for example:

### Terminology

> What should SC do if the train cannot proceed between stations?

Expected: `SC` is resolved from the organisation terminology index to the corpus-backed canonical role (for example Station Controller if that is what your PDFs define), even if the answer document only uses the full form.

### Threshold logic

Use a real rule from your PDFs. For example, if a source genuinely says `2 or more brakes ...`, compare queries for 1, 2 and 4 affected brakes. 2 and 4 should route the threshold rule; 1 should not be promoted by that rule.

### Situation routing

Describe a state rather than naming a procedure, e.g. train unable to proceed, location, indications, VCB/voltage/traction states and known line/rolling stock. Inspect whether the answer comes from the relevant procedure section instead of hundreds of broad chunks.

## Measure latency

The patch logs stage timings such as:

```text
smart_rag stage=terminology ...
smart_rag stage=planner ...
smart_rag stage=hybrid_retrieval ...
smart_rag stage=indexed_corpus_fallback ...
```

The API already returns `X-Process-Time-Ms`, so compare the same test set before/after the patch.

For a useful evaluation, record at least:

- total response latency (p50 / p95),
- retrieval time,
- LLM time,
- number of candidate chunks,
- number of final evidence chunks,
- correct document/procedure,
- correct applicability (line/stock/role),
- correct threshold decision,
- citation correctness.

## Recommended production embedding setting

Your current application can fall back to hashing embeddings. For production-quality semantic scenario routing, use the same real transformer model that was used to generate the stored chunk embeddings.

Once your model is reliably mounted/preloaded, strongly consider:

```env
EMBEDDING_FALLBACK_MODE=disabled
REQUIRE_EMBEDDING_AT_STARTUP=true
```

Do not build procedure-card embeddings with one embedding backend and query existing chunk embeddings with a different backend.

## Tuning

The patch works without adding environment variables. Defaults are in `.env.smart-rag.example`.

Useful first knobs:

```env
SMART_RAG_QUERY_VARIANTS=3
SMART_RAG_VECTOR_K=90
SMART_RAG_FTS_K=90
SMART_RAG_SCOPED_K=70
SMART_RAG_MAX_CANDIDATES=180
SMART_RAG_FINAL_CONTEXT_CHUNKS=48
```

If recall is poor, raise the retrieval values gradually rather than returning to thousands of chunks. If latency is still high but retrieval timings are low, the remaining bottleneck is likely model/rate-limit/synthesis behavior rather than PostgreSQL.

## Safety/logic boundary

This patch deliberately distinguishes:

- **deterministic inference**: `4 >= 2`, `28 > 25`;
- **semantic routing**: user phrasing such as “won't take traction” vs a formal procedure title;
- **source evidence**: the actual PDF text.

Only explicit deterministic comparisons are presented as derived applicability. The patch does not allow the LLM to invent threshold rules. Complex AND/OR/exception/state-machine rules can be added to `scenario_reasoning.py` after validating them against your organisation's real document patterns.

## Rollback

Fastest runtime rollback: set this in the repository's normal `.env` file:

```env
SMART_RAG_ENABLED=0
```

Then restart with the supplied Compose override:

```bash
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml up -d backend
```

The new tables can remain; they are not used by the legacy path.

For complete code rollback, restore the old `backend/app/main.py` and remove the new `smart_*` modules. The `rag_terminology`, `rag_procedure_cards` and `rag_rules` tables are additive and may be dropped later if desired.

## What this patch intentionally does not do

- It does not auto-decide which document revision/amendment supersedes another because your current schema does not yet have authoritative revision metadata.
- It does not replace OCR/table extraction.
- It does not claim that every natural-language operational situation can be converted into a deterministic rule. Numeric thresholds are handled deterministically; broader situation interpretation uses terminology + scenario features + procedure-card semantic routing and then original PDF evidence.
- It does not change the frontend.

Those choices make this patch suitable for controlled testing against your current corpus before a larger document-governance/versioning migration.
