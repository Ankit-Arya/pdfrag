IMS RAG v5.3 - Coverage-First Deployment Patch
===============================================

Purpose
-------
This patch is the deployment version for the retrieval/answer behavior agreed after
RS-3 chunk-level diagnostics. It is generalized: no RS number, Line number, BIC,
door procedure, speed value, or document filename is hard-coded.

It expects the current IMS v5.2.1 completeness/diagnostics layer to already be present.
It can be applied after the PDF Chunk Explorer changes and before the next Docker build.

Behavior contract
-----------------
1. Definition enumeration
   Queries such as "What is XYZ?", "full form of XYZ", "meaning of XYZ", and
   "what does XYZ stand for" use exact source-grounded alias enumeration first.
   - Every distinct explicit meaning is retained.
   - The same meaning in multiple PDFs is not source-collapsed.
   - Every available definition source location (PDF/page range) is inventoried.
   - Lowercase short aliases such as "what is bic?" are supported.
   - Table/OCR case degradation such as BIC -> Bic is tolerated only when the
     expansion remains source-grounded and initials-consistent.
   - A second corpus alias scan is not run merely to build diagnostics; the definition
     inventory is derived from the already-retrieved definition chunks.

2. Cross-scope conditional/procedure coverage
   Queries such as "what happens if", "what to do if", "what should ... if/when",
   "how to", procedures, troubleshooting, requirements, and similar operational
   questions enter coverage-first discovery.
   - Relevant RS and Line scopes are retained independently.
   - A relevant scope cannot be removed only because the ordinary global route cap
     was reached.
   - If a user names RS-3 / Line-7, that scope is pinned and deep-searched even if its
     initial global route rank is outside the ordinary cap.
   - Unless the user says only/just/solely/specifically, other applicable RS/Line
     scopes can still be returned.
   - If several RS/Lines state the same procedure, each source remains separately
     visible and separately citable.
   - One evidence seed per routed document is preserved through final candidate
     selection so a routed scope cannot disappear only because another PDF is longer.

3. Answer-first presentation
   - Start with the useful supported action/value/procedure, not a missing-information
     caveat, when materially applicable evidence exists.
   - If the user's wording differs from the formal source wording, answer the closest
     supported source condition first and place one concise clarification Note at the
     end where useful.
   - Definition answers must retain all distinct meanings and source locations.
   - Cross-scope procedure answers must use source/scope-specific headings and cite
     each required PDF.
   - A deterministic policy check triggers a repair only when the generated answer
     omits required source documents, required definition meanings/locations, required
     RS/Line labels, or starts with an inappropriate negative caveat.
   - Citation repair is followed by one final policy check so citation cleanup cannot
     silently collapse required source coverage.

4. Diagnostics
   The existing Diagnostics tab is upgraded to show:
   - scope type / scope label
   - scope-pinned state
   - explicit-scope state
   - route rank versus ordinary global route cap
   - RS/Line scope counts
   - required source documents for answer coverage
   - definition inventory with each meaning and PDF/page source location

Retrieval architecture
----------------------
Synthesis/conditional questions use one initial corpus-discovery stage:
  semantic/vector signal + per-document FTS signal
    -> explicit scope pinning + RS/Line coverage promotion
    -> deep search inside routed documents
    -> cross-document rerank + section expansion
    -> coverage review
    -> targeted retry only when coverage is incomplete
    -> grounded answer + policy verification

The ordinary route cap remains useful for unrelated/common extra documents, but it no
longer acts as a blanket cap that can remove a relevant RS/Line scope.

Default runtime limits (environment-overridable)
-------------------------------------------------
RAG_V53_DISCOVERY_QUERY_COUNT=8
RAG_V53_DISCOVERY_VECTOR_PER_QUERY=180
RAG_V53_DISCOVERY_FTS_PER_DOCUMENT=2
RAG_V53_DISCOVERY_FTS_MAX_ROWS=2200
RAG_V53_MAX_RELEVANT_DOCUMENTS=24          # ordinary/common route cap
RAG_V53_MAX_COVERAGE_DOCUMENTS=48          # hard safety cap after scope promotion
RAG_V53_DOCS_PER_SCOPE=2
RAG_V53_SCOPE_MIN_SCORE=0.14
RAG_V53_SCOPE_MIN_SIGNAL=0.10
RAG_V53_ENUMERATION_EVIDENCE=64
RAG_V53_DIAGNOSTIC_MAX_DOCS=300
RAG_V53_ALIAS_ROWS_PER_DOCUMENT=80
RAG_V53_ALIAS_SCAN_ROWS=7000

No migration/reprocessing
-------------------------
No DB migration is required.
No PDF reprocessing is required.
No OCR rerun is required.
No chunk rebuild is required.
No embedding rebuild is required.

Install from repository root (PowerShell)
-----------------------------------------
Expand-Archive `
  -Path .\pdfrag-ims-rag-v5.3-coverage-first-deployment.zip `
  -DestinationPath . `
  -Force

python `
  .\pdfrag-rag-v5.3-coverage-first-deployment\apply_ims_v53_coverage_first_patch.py `
  --repo .

The installer preflights transformed backend Python before writing repository files.
Modified files receive .bak-before-ims-v53-coverage-first backups.
It is idempotent and can be run twice safely.

Compile
-------
python -m py_compile `
  backend\app\rag\v5\service.py `
  backend\app\rag\v5\synthesis_retrieval.py `
  backend\app\rag\v5\retrieval_completeness.py `
  backend\tests\test_v53_coverage_first.py

Focused test
------------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  run --rm backend pytest -q backend/tests/test_v53_coverage_first.py

Build/deploy
------------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  build frontend

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  build backend

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  up -d --force-recreate frontend backend

Acceptance queries
------------------
A. What is BIC?
   Expected: all distinct source-grounded meanings available in the corpus, grouped by
   meaning, with every supplied definition PDF/page location represented. Diagnostics
   should show Definition inventory.

B. If 50% brakes got isolated in RS-3 train, what should be its speed?
   Expected: RS-3 is explicitly pinned rather than rejected by ROUTE_LIMIT. The answer
   should lead with the applicable RS-3 rule found under the BECU Failure/BIC Isolation
   procedure, then place any wording clarification at the end. Other applicable RS/Line
   scopes can be shown separately unless the question says only RS-3.

C. What to do if a train door fails to close?
   Expected: every materially applicable RS/Line scope found by coverage discovery is
   shown separately with PDF/page citations, even when several procedures are identical.

D. Only RS-3: what should be done if brakes fail to apply?
   Expected: RS-3 is pinned and other RS/Line scopes are excluded by explicit-only scope.

Testing note
------------
Keep the existing git stash/rollback point until frontend build, backend build, startup,
Chunk Explorer, Diagnostics, and these acceptance queries have all passed.
