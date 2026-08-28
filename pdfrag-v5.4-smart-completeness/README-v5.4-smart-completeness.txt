IMS RAG v5.4 - Smart Completeness / Contributor Validation
=========================================================

Target
------
Repository: Ankit-Arya/pdfrag
Base expected on the server: IMS RAG v5.3 coverage-first already applied.

Why this patch exists
---------------------
v5.3 correctly solved an important class of misses: a genuine RS/Line procedure should not disappear only because another document ranks higher. However, at large corpus scale the same coverage behavior can over-expand ordinary list/entity queries. A query such as "members of X" can be classified as a list, and v5.3's scope-preservation logic can then retain weak RS/Line routes and make them visible to the answer layer even when those PDFs only mention the subject.

v5.4 separates QUERY INTENT from COMPLETENESS POLICY. It does not hard-code DMT, any RS/Line, doors, brakes, speeds, document names or values.

Completeness policies
---------------------
1. definition_enumeration
   For full-form/meaning/definition questions. Preserve all explicit source-grounded meanings and all available PDF/page definition locations.

2. entity_enumeration
   For members/types/categories/components/roles/duties/list-style questions. Completeness comes from reconstructing the governing list/table/section, not from selecting one document from every RS/Line filename family.

3. cross_scope_procedure
   For what-happens-if / what-to-do-if / procedure / troubleshooting / operational requirement questions. Preserve every materially applicable RS/Line governing source, even when procedures are identical.

4. direct_lookup
   Focused fact/value/navigation questions.

What changes
------------
- `list` is removed from blanket RS/Line coverage promotion.
- Conditional semantics override a generic `list` classification, so "list steps if X fails" still receives cross-scope procedure coverage.
- One-evidence-per-routed-document preservation is used only for genuine cross-scope procedures.
- Multi-document synthesis again has a contributor gate: routing means "inspect", not "must appear in answer".
- Strong governing/definition/applicability/exception/restriction/authority evidence remains protected even if the coverage critic omits a filename.
- Entity/list queries expand complete local table/section siblings from relevant seeds.
- Page-local table fallback repairs legacy cases where a table was associated with a poor preceding heading.
- Query-time conservative recovery can restore the first row of a legacy headerless table when source metadata proves a serial sequence such as 1 then 2.
- `layout.py` stops blindly treating the first row of every pdfplumber table as a header when a numbered row sequence proves the table is headerless.
- Existing v5.3 answer-first, definition enumeration and explicit scope pinning behavior is retained.

Files modified by the installer
-------------------------------
backend/app/rag/v5/retrieval_completeness.py
backend/app/rag/v5/synthesis_retrieval.py
backend/app/rag/v5/service.py
backend/app/rag/v5/layout.py
backend/tests/test_v54_smart_completeness.py

The installer also creates:
pdfrag-v5.4-replacement-files/

That folder contains the exact POST-PATCH copies of the files above. After applying on one validated host, those files can be copied directly to another host running the same code baseline.

Safety / rollback
-----------------
Each changed source file gets a one-time backup suffix:
.bak-before-ims-v54-smart-completeness

The installer pre-compiles every transformed Python source BEFORE writing.
It is idempotent and can be run again safely.

Installation - PowerShell
-------------------------
From the pdfrag repository root:

1) Extract this package somewhere inside or next to the repo.

2) Preflight only:

python .\pdfrag-v5.4-smart-completeness\apply_ims_v54_smart_completeness_patch.py --repo . --check

Expected:
[preflight] all transformed Python files compile in memory
[check only] no repository files were changed

3) Apply:

python .\pdfrag-v5.4-smart-completeness\apply_ims_v54_smart_completeness_patch.py --repo .

4) Focused tests:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  run --rm backend pytest -q backend/tests/test_v54_smart_completeness.py backend/tests/test_v53_coverage_first.py

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

docker compose logs --tail=200 backend

Database / reprocessing
-----------------------
No database migration is required.
No embedding rebuild is required.
No full reprocessing is required for the FIRST test of query behavior.

Why no immediate 700-PDF reprocess?
- v5.4's policy separation, contributor validation, local table/section expansion and conservative legacy first-row recovery work against the CURRENT v5 chunks.
- `layout.py` improves future ingestion and reprocessed documents.

Recommended rollout for a 700+ PDF corpus:
A. Apply v5.4 and test current chunks first.
B. Run a representative acceptance set across definitions, lists, procedures, explicit scopes, values and noisy wording.
C. Only after retrieval behavior is accepted, decide whether to reprocess documents whose stored table structure is known to be damaged. Reprocessing can be staged instead of immediately rebuilding the whole corpus.

If you later choose to reprocess all v5 PDFs using your existing migration worker:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  --profile v5-migration run --rm v5-worker

Use your existing deployment command if it differs; do not run a second worker concurrently against the same document set.

Acceptance expectations
-----------------------
See scripts/acceptance_queries.txt.

Most important behavioral contract:

ROUTED DOCUMENT
    -> inspect/deep-search
    -> governing evidence found?
       YES -> eligible contributor -> answer
       NO  -> diagnostics only

For entity enumeration:
relevant seed -> complete governing table/section -> contributor review -> answer

For conditional procedures:
corpus discovery -> applicable RS/Line scopes -> deep search -> governing evidence per scope -> answer with separate source headings

This is intentionally not a prompt-only fix and not a top-K increase. It changes the completeness policy and answer eligibility boundary while keeping the v5.3 coverage guarantees where they actually belong.
