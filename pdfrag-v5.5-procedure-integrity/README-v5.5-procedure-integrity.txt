IMS RAG v5.5 - Procedure Integrity / Balanced Routed Retrieval
==============================================================

Target
------
Repository: Ankit-Arya/pdfrag
Required runtime baseline: IMS RAG v5.4 smart-completeness already applied.

Why this patch exists
---------------------
v5.4 correctly separated entity/list completeness from conditional procedure coverage,
but a cross-scope procedure query can still fail in four important ways:

1. A routed document can be inspected but its actual governing chunk can be crowded out
   by stronger chunks from another routed document because the existing scoped vector/FTS
   arm uses a global top-K over all routed document IDs.

2. `ensure_routed_document_evidence()` can preserve one weak chunk from every route in the
   final context. v5.3/v5.4 diagnostics then treat `final_evidence=True` as sufficient to
   make an RS/Line scope mandatory, even after the coverage critic says the document is
   not a contributor. This produces empty/no-evidence answer headings.

3. A conditional table can be answered from disconnected rows. The model can therefore
   pair the threshold/condition from one row with a speed/action from a neighboring row.

4. Scope identity is inferred mainly from a numeric RS/Line token in the filename. Manuals
   whose real identity is on the cover/subject page, source-defined RS codes, named lines,
   or manuals with more than one scope can be under-promoted or mislabeled.

v5.5 fixes the pipeline boundaries rather than adding case-specific prompts or a larger
single top-K. There are no hard-coded brake speeds, rolling-stock numbers, line numbers,
document names, or answers in runtime logic.

Core behavioral contract
------------------------

DISCOVERY ROUTE
    -> balanced deep search inside every routed document
    -> governing evidence found?
       YES -> expand complete governing table/section
              -> coverage contributor validation
              -> eligible answer evidence
       NO  -> review/diagnostics only

A ROUTED DOCUMENT IS NOT AN ANSWER DOCUMENT.
A FINAL/REVIEW CHUNK IS NOT AUTOMATICALLY A REQUIRED SCOPE.
A REQUIRED SCOPE MUST BE VALIDATED CONTRIBUTING GOVERNING EVIDENCE.

What changes
------------
1. Balanced per-routed-document deep retrieval
   - Adds partitioned vector and FTS ranking by document_id.
   - Every routed document gets its own bounded candidate slots.
   - SQL remains O(query variants), not O(routed documents x query variants).
   - Existing global scoped retrieval remains as an independent signal.

2. Review evidence is separated from answer evidence
   - Cross-scope procedure retrieval no longer forces one arbitrary candidate per route
     into the answer set.
   - A separate `review_results` pool gives the coverage critic one candidate per route.
   - The coverage critic sees every route before any route receives a second review slot.

3. Contributor-gated required scopes
   - `required_answer_documents` and `required_scope_labels` are produced only from
     contributor-validated final evidence with a strong role: governing, definition,
     applicability, exception, restriction, authority, or conflict.
   - Rejected routes remain visible in diagnostics as `REVIEWED_NON_CONTRIBUTOR`.
   - They cannot force "No applicable evidence" headings into the answer.

4. Complete procedure structure expansion
   - A governing table-row seed expands by the exact database `table_id`.
   - A governing prose seed expands by `parent_key`.
   - A local adjacent-page table fallback handles legacy bad parent headings.
   - Full source rows are also grouped into source-grounded procedure-structure chunks.
   - Every row/branch stays explicitly separated and in source order.

5. Condition/action and measurement-basis integrity
   - Reranking, coverage review, and answer generation are instructed never to attach an
     action/value from one table branch to another threshold.
   - Percentages/counts of different source-defined units are not silently converted or
     treated as equivalent unless the supplied evidence defines that equivalence.

6. Content-aware, multi-label scope identity
   - Scope profiling uses filename + the first few cover/subject pages in one batched query.
   - Supports numeric RS/Line labels, source-defined uppercase RS codes, and named lines.
   - One document can carry multiple labels.
   - Only identity pages are scanned, avoiding false relabeling from later body references.

7. Existing v5.4 protections remain
   - Definition enumeration remains intact.
   - Entity/list structure reconstruction remains intact.
   - Headerless first-row recovery/ingestion protection remains intact.
   - Direct lookups do not pay for procedure structure expansion.

Files modified by installer
---------------------------
backend/app/rag/v5/retrieval_completeness.py
backend/app/rag/v5/synthesis_retrieval.py
backend/app/rag/v5/service.py
backend/tests/test_v55_procedure_integrity.py   (created/copied)

Optional payload mirrors are patched only if they already carry the v5.4 runtime marker.
An old/stale payload template cannot block the active backend patch.

No database migration is required.
No embedding rebuild is required.
No full PDF reprocessing is required for the first test.

Why reprocessing is not required for the first test
---------------------------------------------------
The fixes are query-time changes over the existing v5 schema/chunks:
- per-document ranking uses existing chunk embeddings and FTS indexes;
- structure expansion uses existing `table_id`, `parent_key`, and table-row chunks;
- scope profiling reads existing first-page chunks;
- contributor gating is retrieval/answer policy.

If a particular source table was already extracted incorrectly at ingestion, reprocessing
that document may still improve it. That is a separate source-quality issue; v5.5 does not
require rebuilding hundreds of healthy PDFs just to test this retrieval fix.

Installation - PowerShell
-------------------------
From the pdfrag repository root after v5.4 is already applied:

1) Extract this ZIP inside or next to the repository.

2) Preflight only:

python .\pdfrag-v5.5-procedure-integrity\apply_ims_v55_procedure_integrity_patch.py --repo . --check

Expected:
[preflight] all transformed Python files compile in memory
[check only] no repository files were changed

3) Apply:

python .\pdfrag-v5.5-procedure-integrity\apply_ims_v55_procedure_integrity_patch.py --repo .

4) Focused tests:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  run --rm backend pytest -q `
  backend/tests/test_v55_procedure_integrity.py `
  backend/tests/test_v54_smart_completeness.py `
  backend/tests/test_v53_coverage_first.py

5) Build backend:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  build backend

6) Restart backend:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  up -d --force-recreate backend

7) Check logs:

docker compose logs --tail=250 backend

Linux/server equivalents
------------------------
python ./pdfrag-v5.5-procedure-integrity/apply_ims_v55_procedure_integrity_patch.py --repo . --check
python ./pdfrag-v5.5-procedure-integrity/apply_ims_v55_procedure_integrity_patch.py --repo .

docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml \
  run --rm backend pytest -q \
  backend/tests/test_v55_procedure_integrity.py \
  backend/tests/test_v54_smart_completeness.py \
  backend/tests/test_v53_coverage_first.py

docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build backend
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml up -d --force-recreate backend

Recommended acceptance sequence
-------------------------------
A. Run the exact brake-isolation query that exposed the current failure.
B. Verify that genuinely governing RS/manual sources are represented and routed-but-
   rejected scopes are diagnostics-only rather than empty answer headings.
C. Verify threshold/action branches against the PDFs, especially adjacent rows.
D. Run v5.4 entity/list and definition tests to ensure no regression.
E. Run semantic/noisy queries such as wake-up process, track work and absence examples.
F. Inspect diagnostics for:
   - routed/deep_searched
   - contributing
   - rerank_role
   - answer_eligible
   - required_scope_labels

Important expected behavior for the 25% brake-isolation test
-------------------------------------------------------------
The patch intentionally does NOT hard-code what "25% brakes" means or a speed. It must:
- retrieve the governing brake-isolation structures independently from each relevant
  manual;
- preserve each manual's own measurement basis (bogies/BIC/cars/etc. as written);
- preserve each threshold-to-action row association;
- include only validated contributing manuals in the answer;
- leave unrelated routed scopes in diagnostics only;
- state a measurement-basis mismatch rather than inventing a conversion when the source
  does not establish one.

Tuning knobs (defaults are conservative)
----------------------------------------
RAG_V55_SCOPE_PROFILE_PAGES=4
RAG_V55_BALANCED_QUERY_COUNT=4
RAG_V55_BALANCED_PER_DOCUMENT=6
RAG_V55_PROCEDURE_SEEDS=36
RAG_V55_PROCEDURE_SEEDS_PER_DOCUMENT=3
RAG_V55_PROCEDURE_TABLE_ROWS=200
RAG_V55_PROCEDURE_SECTION_CHUNKS=48
RAG_V55_PROCEDURE_PAGE_ROWS=48
RAG_V55_PROCEDURE_AGGREGATE_CHARS=18000
RAG_V55_PROCEDURE_EVIDENCE=180
RAG_V55_REVIEW_CANDIDATES=120

Do not increase these blindly. The v5.5 design obtains completeness by per-document
partitioning and structure expansion, not by making one global top-K enormous.

Rollback
--------
The installer creates one-time backups with suffix:
.bak-before-ims-v55-procedure-integrity

Restore those three runtime files and rebuild/restart the backend if rollback is needed.
