IMS UI v2 + Multi-Document Synthesis v5.2
=========================================

Purpose
-------
This patch upgrades both the IMS information architecture and the query-time RAG policy.
It assumes the local repository already has:
- IMS Assistant Retrieval v5.1
- Live Activity / Working panel
- Separate Search Chunks workspace

It does NOT require a database migration, PDF reprocessing, or an embedding rebuild.
Existing active rag_v5_* generations are reused.

Backend behavior
----------------
Every question is assigned one of two answer strategies:

1. direct_lookup
   Used for definitions, page/rule navigation, and simple factual lookups where a directly
   governing source can be sufficient.

2. multi_document_synthesis
   Used for duties/roles/responsibilities, procedures, requirements, "what should/must X do",
   "when does X apply", conditions, exceptions, comparisons, summaries and other questions whose
   complete answer may be distributed across documents.

Synthesis mode performs:
- corpus-wide evidence discovery;
- dynamic relevant-document selection (threshold based, not merely top-one-document);
- evidence-dimension search;
- document-balanced candidate selection;
- AI evidence reranking that labels governing/supporting/applicability/exception/authority/etc.;
- section expansion;
- cross-document coverage review;
- targeted retries for missing dimensions;
- conflict/scope/authority review;
- filtering to materially contributing documents before final answer generation;
- no forced minimum document count: broad synthesis can still conclude that only one document materially contributes;
- a larger synthesis answer-evidence budget (48 by default, capped at 80) so cross-document retrieval is not squeezed back to the normal 32-chunk context;
- per-point citations and explicit unresolved-conflict handling.

Important guardrails
--------------------
- Multi-document synthesis does NOT force multiple documents into an answer.
- Incidental lexical matches are excluded whenever the contribution reviewer can establish that.
- Scope-specific differences are preserved.
- Current/amended authority remains preferred when source evidence establishes precedence.
- Genuine unresolved conflicts are surfaced instead of silently choosing a source.
- The answer model remains closed-book for internal metro facts.

UI v2
-----
Desktop chat becomes a three-column information architecture:

  left navigation | main conversation | answer inspector

The answer inspector contains:
- Sources
- Query Plan
- Details

It exposes:
- cited sources and relevance scores;
- interpreted request;
- direct lookup vs multi-document synthesis;
- synthesis dimensions;
- relevant and contributing documents;
- search queries;
- candidate/final evidence counts;
- search rounds;
- evidence coverage status;
- cross-document conflicts/differences;
- grounding/policy version;
- basic system readiness.

Other UI additions:
- compact Key References under each assistant answer;
- View details action to select an older answer in the inspector;
- separate Documents workspace for ordinary users;
- Ctrl/Cmd+K global search palette using the existing direct chunk-search API;
- existing Search Chunks, Live Working, Admin and Account functionality are retained.

Default v5.2 accuracy settings
------------------------------
RAG_V52_MAX_QUERY_VARIANTS=12
RAG_V52_DISCOVERY_PER_QUERY=110
RAG_V52_MAX_RELEVANT_DOCUMENTS=24
RAG_V52_DOCUMENT_RELATIVE_THRESHOLD=0.18
RAG_V52_RERANK_CANDIDATES=72
RAG_V52_FINAL_CANDIDATES=96
RAG_V52_SYNTHESIS_EVIDENCE=48
RAG_V52_COVERAGE_CANDIDATES=48

These are upper budgets, not quotas. The system does not force 24 documents or 48 evidence chunks into an answer.

Not included in this patch
--------------------------
The following were deliberately left for later because they change additional product behavior or
need separate persistence/analytics work:
- user-created Collections;
- Reports/analytics dashboards;
- a user-controlled Deep Search request mode;
- numeric "answer confidence" probabilities;
- an inline PDF renderer;
- theme/dark-mode redesign.

Apply
-----
From the pdfrag repository root:

  python .\apply_ims_ui_v2_synthesis_v52_patch.py --repo .

Inspect:

  git status --short

  git --no-pager diff -- backend/app/models.py backend/app/api.py backend/app/rag/v5/service.py backend/app/rag/v5/synthesis_retrieval.py frontend/src/services/api.ts frontend/src/App.vue frontend/src/style.css frontend/src/components/UploadPanel.vue frontend/src/components/ChatPanel.vue frontend/src/components/AnswerInspector.vue frontend/src/components/DocumentsPanel.vue frontend/src/components/CommandPalette.vue

Refresh v5 environment defaults (keeps query mode enabled):

  powershell -ExecutionPolicy Bypass -File .\merge-v5-env.ps1 -EnableQuery

Compile backend:

  python -m py_compile backend\app\models.py backend\app\api.py backend\app\rag\v5\service.py backend\app\rag\v5\synthesis_retrieval.py

Focused backend test:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml run --rm backend pytest -q backend/tests/test_v52_synthesis_policy.py

Build frontend first so any Vue/TypeScript issue is isolated:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build frontend

Then build backend:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build backend

Recreate both:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml up -d --force-recreate backend frontend

Do NOT run the v5 reprocessor.

Suggested validation queries
----------------------------
Direct lookup:
- What is ATP?
- On which page is Rule 39?

Synthesis:
- What are the duties of Station Controller?
- When must a speed of 25 km/h be followed?
- What should the Train Operator do during a mid-section evacuation?
- What precautions and responsibilities apply before engineering possession work starts?

For synthesis questions, check the Query Plan tab. It should show:
- Answer strategy: multi document synthesis
- Search scope: broad relevant corpus
- multiple evidence dimensions
- relevant/contributing documents
- coverage status and search rounds

Rollback
--------
The installer creates *.bak-before-ims-ui-v2-synthesis-v52 backups for modified existing files.
Restore those backups and remove the added v5.2/UI component files, then rebuild backend/frontend.
No database rollback or PDF reprocessing is required.
