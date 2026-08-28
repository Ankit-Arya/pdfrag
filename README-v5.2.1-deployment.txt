IMS RAG v5.2.1 - Deployment + Testing Patch
==============================================

This package supersedes the earlier v5.2.1 retrieval-completeness draft.
Do not apply the older pdfrag-ims-v5.2.1-retrieval-completeness.zip first.

Prerequisite
------------
IMS UI/RAG v5.2 synthesis must already be applied. The installer preflights the v5.2
files and aborts before writing if required anchors are not present.

What changes
------------
1. Synthesis retrieval now has one corpus-discovery stage:
   - global semantic/vector candidates
   - per-document lexical best-evidence candidates
   - one combined document score and route decision
   - then deep search only inside routed documents
   - targeted retry is still allowed only when coverage is incomplete

2. Exact acronym/full-form requests use exact source-grounded alias discovery first.
   A successful alias lookup does not run the broad semantic synthesis path.

3. Acronym extraction is case tolerant only when initials validate the long form.
   Example: Bic (Brake Isolation Cock) is accepted for BIC; Bic (Brake Control Unit) is rejected.
   Multiple meanings remain distinct and source-scoped.

4. Incomplete coverage review is non-destructive. It cannot hard-delete otherwise
   strong routed evidence merely because its contributing-document list is incomplete.

5. Retrieval diagnostics UI:
   Sources | Query Plan | Details | Diagnostics

   The Diagnostics tab shows:
   - corpus eligible PDFs
   - documents with retrieval signal
   - routed PDFs
   - deep-searched PDFs
   - final-evidence/contributing/cited state
   - vector and lexical scores
   - number of query dimensions matched
   - deterministic decision codes and reasons
   - best page/heading when available
   - no-signal PDFs by comparing the trace with the normal document list

   It does NOT expose private chain-of-thought, raw prompts, SQL or hidden reasoning.

No migration/reprocessing
-------------------------
No database migration.
No PDF reprocessing.
No OCR rerun.
No chunk rebuild.
No embedding rebuild.

Apply
-----
python .\ims-v521-deployment\apply_ims_v521_deployment_patch.py --repo .

The installer preflights and compiles all transformed backend Python BEFORE it writes
any repository file. Existing files receive .bak-before-ims-v521-deployment backups.

Compile
-------
python -m py_compile `
  backend\app\models.py `
  backend\app\api.py `
  backend\app\rag\smart_index.py `
  backend\app\rag\v5\ingestion.py `
  backend\app\rag\v5\retrieval_completeness.py `
  backend\app\rag\v5\synthesis_retrieval.py `
  backend\app\rag\v5\service.py `
  backend\app\rag\v5\query_trace.py

Focused backend test
--------------------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  run --rm backend pytest -q backend/tests/test_v521_retrieval_completeness.py

Build frontend first
--------------------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  build frontend

Build backend
-------------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  build backend

Deploy
------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  up -d --force-recreate backend frontend

CLI retrieval trace
-------------------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  exec backend python -m app.rag.v5.query_trace --question "What is full form of BIC" --limit 100

Repeat for:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  exec backend python -m app.rag.v5.query_trace --question "What should be the speed of train if 25% of its brakes are isolated" --limit 100

UI validation
-------------
For each answer open Diagnostics in the right Answer Inspector.

BIC expectation:
- exact alias path
- every distinct explicit PDF-grounded BIC expansion can surface
- RS-3 Brake Isolation Cock should no longer be lost because the extracted alias became Bic

Brake-isolation expectation:
- one corpus discovery stage
- relevant OTM families get a per-document lexical opportunity
- only routed PDFs receive the deep heading/scoped search
- generic speed documents without brake-isolation scenario linkage should be incidental/rejected
- a missing OTM can be located in Diagnostics and its exact rejection stage/reason inspected

Policy fingerprint
------------------
rag-v5.2.1-completeness
