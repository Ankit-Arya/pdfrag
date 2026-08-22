IMS Live Activity / Working Panel v1
====================================

Purpose
-------
Adds a ChatGPT-style live "Working" experience to IMS Assistant v5.1 while preserving the existing SSE chat stream and closed-book PDF grounding.

What it adds
------------
1. Rich structured progress schema:
   actor, phase, status, operation_id, sequence, elapsed time, prompt/task summary, AI summary,
   document/page/heading and safe metrics.
2. Actor labels in the UI: AI / Backend / Search / Verification.
3. Real counts from the live pipeline: routed documents, baseline candidates, heading matches,
   scoped candidates, rerank pool, AI-ranked candidates, section expansion, final evidence, cited sources.
4. Strong structural matches show actual filename, page and heading/rule/section metadata.
5. Search rounds are shown separately (1/N, 2/N, 3/N) and persisted.
6. Evidence-completeness review shows sufficient / more evidence needed and a concise safe summary.
7. A collapsible Working panel stays open while answering and auto-collapses after the answer.
8. Per-operation duration and total working time are shown.
9. The safe activity trace is persisted in ChatMessage metadata, so it can be reopened after chat reload.
10. AI transparency uses concise task/reasoning summaries only.

Important transparency boundary
-------------------------------
The patch intentionally DOES NOT expose private chain-of-thought, hidden model reasoning, raw system/developer prompts,
credentials, SQL text, API keys or private scratch work. Instead it exposes safe summaries such as:
- "Resolved question: ..."
- "Rank candidate excerpts for governing relevance; do not answer."
- "Evidence incomplete: missing applicability condition."
- "Top evidence is <file>, page <n>, section <heading>."

This gives users useful live transparency without leaking protected reasoning or security-sensitive internals.

Prerequisite
------------
IMS Assistant Retrieval v5.1 must already be applied to backend/app/rag/v5/service.py.
The installer checks for retrieve_assistant_v51 and will stop safely if v5.1 is not active.

No data rebuild
---------------
- No database migration.
- No PDF reprocessing.
- Existing rag_v5_* generations are reused.
- No embedding rebuild.

Files
-----
Replaced/added:
  backend/app/rag/progress.py
  frontend/src/components/ActivityTrace.vue
  backend/tests/test_live_activity_progress.py

Modified:
  backend/app/models.py
  backend/app/api.py
  backend/app/rag/v5/service.py
  backend/app/rag/v5/assistant_retrieval.py
  frontend/src/services/api.ts
  frontend/src/App.vue
  frontend/src/components/ChatPanel.vue

Apply
-----
From the pdfrag repository root:

  python .\apply_ims_live_activity_patch.py --repo .

Inspect:

  git --no-pager diff -- backend/app/rag/progress.py backend/app/models.py backend/app/api.py backend/app/rag/v5/service.py backend/app/rag/v5/assistant_retrieval.py frontend/src/services/api.ts frontend/src/App.vue frontend/src/components/ChatPanel.vue frontend/src/components/ActivityTrace.vue backend/tests/test_live_activity_progress.py

Compile backend:

  python -m py_compile backend\app\rag\progress.py backend\app\models.py backend\app\api.py backend\app\rag\v5\service.py backend\app\rag\v5\assistant_retrieval.py

Optional focused test:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml run --rm backend pytest -q backend/tests/test_live_activity_progress.py

Build/recreate
--------------
Both backend and frontend changed:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build backend frontend

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml up -d --force-recreate backend frontend

Do NOT run the v5 reprocessor.

Expected live UI
----------------
Example:

  Working · 11.8s

  AI            Question interpreted
                Resolved question: duties of Station Controller ...

  Search        Document routing complete
                8 routed documents

  Search        Strong structural match found
                Responsibilities of Station Controller
                02. MRGR 2020.pdf · p. 99

  AI            AI evidence ranking complete
                48 candidate pool · 12 AI-ranked

  Verification  More evidence needed
                AI summary: missing applicability/exception evidence

  Search        Search round 2 of 3

  Verification  Evidence sufficient

  AI            Writing the grounded answer

  Verification  Grounding verified

After the answer arrives the panel becomes:

  Worked for 14.2s · 13 steps

and is collapsed by default.

Rollback
--------
The installer creates .bak-before-ims-live-activity-v1 backups for modified existing files.
Restore those backups, delete ActivityTrace.vue and test_live_activity_progress.py if desired, then rebuild backend/frontend.
No database rollback or PDF reprocessing is required.

v1.1 installer compatibility fix
--------------------------------
The live-activity installer now accepts both ChatPanel evidence guards:
  message.response?.evidence?.length
  message.response?.evidence.length
This fixes the "completed Working panel: expected exactly one anchor, found 0" error
and is safe to rerun after a partial v1 installation; already-marked files are left intact.
