# Smart RAG Reliability Hotfix v2

This is an incremental overlay for an intact `pdfrag` repository that already contains the Smart RAG patch / first hotfix. It is deliberately **not** a standalone repository and contains **no frontend or nginx application files**.

## What this fixes

The implementation addresses failure classes rather than hard-coding the reported examples:

1. **Multi-entity definitions and terminology**
   - Parses multiple requested aliases/acronyms in one question.
   - Retrieves source-backed definitions independently for each target.
   - Uses exact source-derived long forms for pure definition/full-form answers instead of allowing the LLM to paraphrase them.
   - Keeps explicit `find/search/references` requests in reference/navigation mode.
   - Falls back safely when a requested target has no explicit source definition or has multiple corpus meanings.

2. **General structured/measurable-value retrieval**
   - Applies to amounts, compensation, speeds, limits, pressures, voltages, times, counts, thresholds, distances, capacities and other measurable values.
   - Retrieves numeric/table rows semantically inside likely documents, then joins nearby header/unit/context chunks.
   - Does not require a user phrase and a formal manual phrase to use the same words.
   - Treats a table row plus its adjacent heading/header as one local evidence unit for answerability.

3. **Explicit amendment and supersession authority**
   - Creates a derived `rag_authority_directives` table from explicit source wording such as `shall be substituted`, replacement wording and omission/deletion directives.
   - Tracks replacement spans so an amended schedule/section can outrank an older copy appended later in the same combined PDF.
   - Quarantines old competing values only when explicit source authority and matching current evidence justify doing so.
   - Derived authority metadata is navigation/ranking metadata only; final answers still cite original PDF chunks.

4. **Evidence-set answerability**
   - Definition questions require explicit definition coverage for all requested targets.
   - Value questions can be answered from row + adjacent header rather than requiring every cue in one chunk.
   - Broad fallback is triggered by missing answer-shaped evidence, not merely by a low top similarity score.

5. **Cold-start resilience**
   - Smart RAG compose override gives the backend healthcheck a 300-second start period, 10-second timeout and 5 retries.
   - Production embedding preload guidance remains in `.env.smart-rag.example`.

## Important installation rule

Extract this ZIP **over the root of your existing intact pdfrag checkout**. Do not create a new project directory using only this ZIP.

The overlay does not replace `frontend/`, `nginx/`, `backend/app/api.py`, or the original streaming/progress service files.

## PowerShell install

From the pdfrag repository root:

```powershell
Expand-Archive -Path .\pdfrag-smart-rag-reliability-hotfix-v2-20260817.zip -DestinationPath . -Force

docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml down
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml build backend
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml up -d backend
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml exec backend python -m app.rag.smart_backfill
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml exec backend python -m app.rag.smart_diagnostics
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml up -d
```

**Do not use `docker compose down -v`.** The volume contains your existing PostgreSQL data.

## Mandatory one-time backfill

Run `python -m app.rag.smart_backfill` once after installing this v2 hotfix. It builds the new authority index and refreshes terminology/procedure/rule derived indexes from the **existing stored chunks**. It does not rerun OCR or reprocess all PDFs.

Newly processed documents are indexed automatically by the Smart RAG runtime.

## Diagnostics

`smart_diagnostics` now reports:

- ready documents and chunks
- terminology rows
- procedure cards
- deterministic rules
- authority directives, including counts by directive type
- terminology aliases that have multiple corpus meanings

If your corpus contains amendment/substitution documents but `authority_directives` is zero, do not trust version-sensitive answers until the backfill/index issue is resolved.

## Recommended regression families

Do not test only the examples that motivated this patch. Exercise classes such as:

- one and several acronym/full-form questions
- acronyms with multiple corpus meanings
- explicit `find references to ...` navigation queries
- colloquial descriptions versus formal manual terminology
- amount/speed/pressure/time/limit lookups from tables
- rows whose heading/unit is in an adjacent chunk
- base rule + later amendment in the same PDF
- base document and separate amendment PDFs
- replacement wording (`for the words X, Y shall be substituted`)
- whole schedule/section substitution
- numerical thresholds and exception branches
- procedure questions containing acronyms (must remain procedure questions)

## Rollback

The runtime kill switch remains:

```env
SMART_RAG_ENABLED=0
```

This disables the runtime Smart RAG monkeypatch after a backend restart. The additive derived tables/indexes may remain in PostgreSQL; they do not alter the original PDF chunks.
