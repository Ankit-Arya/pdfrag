IMS RAG v6 - Evidence Workspace + Premium Drafting Pilot
========================================================

Target
------
Repository: Ankit-Arya/pdfrag
Required active baseline on the test host:
- IMS RAG v5.5 procedure-integrity applied
- IMS RAG v5.5.1 virtual/synthetic chunk-id UUID guard applied

This is intentionally a PILOT architecture for complex-query testing. It does not require a DB
migration or reprocessing the existing PDF corpus.

Why this patch exists
---------------------
The v5.3-v5.5 chain improved corpus coverage, route preservation and table reconstruction, but the
final answer layer still receives retrieval artifacts and must simultaneously decide applicability,
reconcile units/scenarios, understand table branches, honor coverage metadata and draft prose.
That causes a false-positive route/seed to influence the final answer too strongly and makes the
writer sound like a retrieval audit rather than an expert assistant.

v6 changes the boundary:

  retrieval -> applicability gate -> structural reconstruction -> EVIDENCE WORKSPACE
            -> targeted gap retry if needed -> premium answer writer -> verifier

The final writer does NOT receive required-scope lists, routing diagnostics or contributor-enforcement
instructions. It receives a compact internal map of supported claims plus the exact source blocks that
support those claims.

What changes
------------
1. Applicability BEFORE structure expansion
   For cross-scope procedures, reranked candidates are checked for scenario/scope/condition applicability
   before they are allowed to seed a whole table/section expansion. The original candidate is NOT deleted;
   only structural expansion is gated. This prevents one loosely related rescue/emergency/test/etc. excerpt
   from expanding into a large irrelevant procedure structure.

2. Evidence Workspace
   Retrieved chunks remain storage/retrieval units. Before writing, a dedicated compiler converts them into
   atomic evidence claims with:
   - source IDs
   - scope/applicability
   - scenario
   - condition/threshold
   - action/value
   - authority/exception role
   - conflicts/ambiguities
   - rejected evidence + reason
   - missing facets

   A document being routed no longer means its content must appear in the answer.

3. Workspace-guided retrieval retry
   If the compiler can identify a material missing facet, it can issue up to four targeted semantic retry
   queries using only the user's/source terminology. One additional search round is enabled by default.
   This is designed for cases where the first retrieval finds a related branch but misses the exact threshold,
   wording variant or governing section.

4. Premium final drafting
   The writer behaves as if it already understands the relevant material:
   - direct answer first
   - compact comparison tables when useful
   - operational numbered steps for procedures
   - short descriptive headings only for materially different/applicable scopes
   - no empty/no-evidence headings
   - no retrieval/coverage/required-scope language
   - no repetitive document-name prose
   - precise final Note only when a source-defined ambiguity actually matters
   - citations at the end of each factual sentence/step/row

5. Negative-claim safety
   Failed retrieval is not allowed to become "document X does not contain Y". The evidence workspace marks
   whether a negative conclusion is safe; otherwise the writer must state only the unresolved distinction.

6. Verification is a checker, not a free rewrite
   A dedicated verifier checks:
   - unsupported claims
   - missing primary evidence points
   - citations
   - condition/action pairing
   - unit/measurement basis
   - scenario/scope applicability
   - formatting/diagnostic leakage

   Only the exact reported issues are sent to a repair pass.

Files modified/added
--------------------
Modified:
- backend/app/rag/v5/service.py
- backend/app/rag/v5/synthesis_retrieval.py

Added:
- backend/app/rag/v6/__init__.py
- backend/app/rag/v6/evidence_workspace.py
- backend/tests/test_v6_evidence_workspace.py

The installer also creates after successful application:
- pdfrag-v6-replacement-files/

That snapshot contains the exact post-patch runtime files for inspection/copying to another host with the
same baseline.

Important behavior change
-------------------------
The v6 _ask_impl intentionally bypasses the accumulated v5.3-v5.5 required-scope answer-repair chain.
Those mechanisms may remain in the source file for compatibility/diagnostics, but they are not used by the
v6 writer. This prevents internal "required answer scopes" state from leaking into user-visible replies.

Installation - PowerShell
-------------------------
From the pdfrag repository root:

1. Extract this folder inside or next to the repository.

2. Preflight only (NO files changed):

python .\pdfrag-v6-evidence-workspace-pilot\apply_ims_v6_evidence_workspace_pilot.py --repo . --check

Expected:
[preflight] v6 transformed/copied Python compiles in memory
[check only] no repository files were changed

If the preflight says the v5.5.1 UUID guard is missing, apply the v5.5.1 hotfix first. Do not force v6 over
an older baseline.

3. Apply:

python .\pdfrag-v6-evidence-workspace-pilot\apply_ims_v6_evidence_workspace_pilot.py --repo .

4. Tests:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  run --rm backend pytest -q `
  backend/tests/test_v6_evidence_workspace.py `
  backend/tests/test_v551_virtual_chunk_ids.py `
  backend/tests/test_v55_procedure_integrity.py `
  backend/tests/test_v54_smart_completeness.py `
  backend/tests/test_v53_coverage_first.py

5. Build backend:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  build backend

6. Restart backend:

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  up -d --force-recreate backend

7. Logs:

docker compose logs --tail=300 backend postgres

Linux equivalents
-----------------
python ./pdfrag-v6-evidence-workspace-pilot/apply_ims_v6_evidence_workspace_pilot.py --repo . --check
python ./pdfrag-v6-evidence-workspace-pilot/apply_ims_v6_evidence_workspace_pilot.py --repo .

docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml \
  run --rm backend pytest -q \
  backend/tests/test_v6_evidence_workspace.py \
  backend/tests/test_v551_virtual_chunk_ids.py \
  backend/tests/test_v55_procedure_integrity.py \
  backend/tests/test_v54_smart_completeness.py \
  backend/tests/test_v53_coverage_first.py

docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build backend
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml up -d --force-recreate backend

Recommended complex-query acceptance sequence
---------------------------------------------
Use scripts/acceptance_queries.txt. For each answer verify BOTH factual correctness and drafting quality.

Expected qualitative contract:
- Answer the actual question in the first paragraph/line.
- Do not show irrelevant routed documents.
- Do not produce "Required answer scopes" or internal pipeline terminology.
- Do not state that a document lacks a rule unless exhaustive negative evidence is actually established.
- Do not equate percentage/count units that the source treats differently.
- Do not pair a condition from one table row with the action from another row.
- When multiple materially applicable source scopes differ, show only those scopes and make comparison easy.
- When the same rule repeats across many sources, cite/group naturally unless separate scope-specific provenance
  materially matters to the question.
- Procedure answers should read like an expert operational answer, not a dump of retrieved fragments.

Runtime tuning knobs
--------------------
Defaults are intended for quality-first pilot testing:

RAG_V6_APPLICABILITY_GATE=1
RAG_V6_APPLICABILITY_CANDIDATES=48
RAG_V6_APPLICABILITY_EXCERPT_CHARS=700
RAG_V6_APPLICABILITY_MAX_OUTPUT_TOKENS=1100

RAG_V6_WORKSPACE_EVIDENCE=64
RAG_V6_WORKSPACE_PER_DOCUMENT=10
RAG_V6_COMPILER_EXCERPT_CHARS=3600
RAG_V6_COMPILER_MAX_OUTPUT_TOKENS=3200
RAG_V6_MAX_WORKSPACE_ROUNDS=2
RAG_V6_MAX_RETRY_QUERIES=4

RAG_V6_WRITER_EXCERPT_CHARS=5200
RAG_V6_VERIFY_MAX_OUTPUT_TOKENS=1400

Do not increase these blindly. More raw context is not automatically better. v6 is designed to compress broad
retrieval into a smaller set of applicable evidence claims before writing.

Cost/latency expectation
------------------------
Compared with v5.5, complex multi-document questions can add:
- one applicability-gate query-model call for cross-scope procedures;
- one evidence-compiler query-model call per workspace round (normally 1, maximum 2 by default);
- one final verification query-model call;
- a repair call only if verification finds a specific issue.

The extra calls are deliberate for this pilot: quality is prioritized over minimum latency. After acceptance,
we can measure which stages can be cached, made deterministic, or skipped for simple fact queries.

Database / reprocessing
-----------------------
No database migration is required.
No embedding rebuild is required.
No OCR rerun is required.
Do NOT reprocess the 700+ PDFs just to test this patch.

The patch works over the current v5 chunks/embeddings/tables/parent keys. If a particular PDF has genuinely
bad extraction, that remains a document-specific ingestion-quality issue and can be reprocessed separately.

Rollback
--------
Changed runtime files receive one-time backups with suffix:
.bak-before-ims-v6-evidence-workspace-pilot

Restore those backups, remove/ignore backend/app/rag/v6, rebuild and restart the backend to return to v5.5.1
behavior.
