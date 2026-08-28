IMS RAG v5.5.1 - Virtual/Synthetic Chunk ID UUID Guard
======================================================

Symptom fixed
-------------
After v5.5 procedure-integrity retrieval creates source-grounded virtual evidence such as:

    v55-procedure:<physical-chunk-uuid>:table:1

a later section-expansion pass can receive that virtual chunk again (especially on a
coverage retry/search round). The legacy v5.1 helper `_section_keys_for_chunks()` assumes
every candidate chunk_id is a physical `rag_v5_chunks.id` UUID and sends the complete
list to PostgreSQL as `uuid[]`.

PostgreSQL then fails with:

    invalid input syntax for type uuid: "v55-procedure:...:table:1"

Root cause
----------
This is an interface-boundary bug between two valid pipeline concepts:

1. physical database chunks: chunk_id is a UUID stored in rag_v5_chunks.id
2. virtual evidence chunks: chunk_id is a stable synthetic identifier and is not stored
   in rag_v5_chunks

Virtual evidence must remain in reranking, coverage review and final synthesis, but a
DB-backed section lookup must use physical UUIDs only.

Fix
---
`backend/app/rag/v5/assistant_retrieval.py` now:

- parses candidate IDs as UUIDs before any `uuid[]` query;
- silently excludes virtual/synthetic IDs from the physical DB section lookup;
- preserves order and de-duplicates canonical physical UUIDs;
- skips the DB call completely when every candidate is virtual;
- does NOT special-case only `v55-` or `v54-` prefixes, so future virtual evidence ID
  formats are protected too.

The synthetic evidence itself is NOT discarded. Only the physical-table lookup ignores
its synthetic identifier. The evidence continues through rerank/coverage/answer logic.

Files changed
-------------
backend/app/rag/v5/assistant_retrieval.py
backend/tests/test_v551_virtual_chunk_ids.py

The installer also creates post-patch replacement copies under:

pdfrag-v5.5.1-replacement-files/

Installation - PowerShell
-------------------------
From the pdfrag repository root:

1) Preflight only:

python .\pdfrag-v5.5.1-virtual-chunk-id-hotfix\apply_ims_v551_virtual_chunk_id_hotfix.py --repo . --check

Expected:

[preflight] all transformed Python files compile in memory
[check only] no repository files were changed

2) Apply:

python .\pdfrag-v5.5.1-virtual-chunk-id-hotfix\apply_ims_v551_virtual_chunk_id_hotfix.py --repo .

3) Focused regression tests:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  run --rm backend pytest -q `
  backend/tests/test_v551_virtual_chunk_ids.py `
  backend/tests/test_v55_procedure_integrity.py `
  backend/tests/test_v54_smart_completeness.py `
  backend/tests/test_v53_coverage_first.py

4) Build backend:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  build backend

5) Restart backend:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  up -d --force-recreate backend

6) Re-run the exact query that produced the UUID error and check logs:

docker compose logs --tail=250 backend postgres

Expected after the fix
----------------------
- no `invalid input syntax for type uuid: "v55-procedure:..."` error;
- virtual procedure aggregates may still appear in internal retrieval/coverage state;
- section expansion runs only for physical UUID chunks;
- request completes instead of failing in `_section_keys_for_chunks`.

Database / reprocessing
-----------------------
No database migration is required.
No embedding rebuild is required.
No PDF reprocessing is required.

Rollback
--------
The first apply creates:

backend/app/rag/v5/assistant_retrieval.py.bak-before-ims-v551-virtual-chunk-id-guard

Restore that file only if you need to roll back this hotfix.
