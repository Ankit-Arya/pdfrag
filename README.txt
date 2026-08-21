IMS Assistant Retrieval v5.1
============================

Goal
----
Make IMS behave more like a capable document assistant instead of a top-K chunk search engine,
while remaining closed-book for metro facts.

This is a QUERY-TIME patch. Existing successfully processed rag_v5_* generations are reused.
No PDF reprocessing and no database migration are required.

Architecture added
------------------
1. Grounded acronym recovery
   - Uses existing rag_v5_terminology first.
   - If an uppercase internal abbreviation is missing, searches active PDF chunks and extracts only
     explicit source patterns such as "Station Controller (SC)". It never invents an expansion.

2. Semantic query planning remains AI-driven
   - Existing smart-understanding layer already fixes typos/paraphrases.
   - Patch strengthens it to produce corrected natural wording plus likely formal heading wording,
     actor/object wording and requested output dimensions.
   - User does not need to know the source document.

3. Document-first routing
   - Combines user filename/document hints, broad v5 results, corpus-wide vector evidence and strict FTS.
   - A document hint is a preference, not a hard constraint. If the hint is wrong/incomplete, search can recover.

4. Structural navigation
   - Searches headings/section paths inside routed documents.
   - Promotes governing/defining sections over incidental mentions.

5. Scoped hybrid retrieval
   - Re-runs vector + strict lexical search inside the strongest candidate documents.
   - Keeps the original v5 broad retrieval as a safety net.

6. AI evidence reranking
   - Query model ranks candidate sections/chunks for directness and governing relevance.
   - It is explicitly forbidden from answering or inventing facts.

7. Section-complete expansion
   - Expands top governing parent sections after reranking.
   - Lists/procedures/navigation can therefore include multiple related rule sections rather than one lucky chunk.

8. Bounded evidence search loop
   - Evidence reviewer can trigger up to 3 targeted search rounds.
   - Previous good evidence is accumulated rather than replaced.

9. Better diagnostics
   - Answer metadata uses answer_policy_version=rag-v5.1-assistant.
   - primary_documents now reports routed document candidates.
   - assistant_debug CLI shows interpretation, grounded terminology, routed docs and top evidence.

Safety/grounding retained
-------------------------
The final answer model still receives only retrieved PDF evidence and may not invent metro facts.
The patch improves what evidence reaches the model; it does not relax grounding.

Files
-----
Installer:
  apply_ims_assistant_v51_patch.py

Added:
  backend/app/rag/v5/assistant_retrieval.py
  backend/app/rag/v5/assistant_debug.py
  backend/tests/test_v5_assistant_retrieval.py

Modified by installer:
  backend/app/rag/v5/service.py
  backend/app/rag/smart_understanding.py
  docker-compose.v5.yml
  merge-v5-env.ps1

Apply
-----
From repository root:

  python .\apply_ims_assistant_v51_patch.py --repo .

Inspect:

  git --no-pager diff -- backend/app/rag/v5/service.py backend/app/rag/smart_understanding.py docker-compose.v5.yml merge-v5-env.ps1 backend/app/rag/v5/assistant_retrieval.py backend/app/rag/v5/assistant_debug.py backend/tests/test_v5_assistant_retrieval.py

Compile/test:

  python -m py_compile backend\app\rag\v5\assistant_retrieval.py backend\app\rag\v5\assistant_debug.py backend\app\rag\v5\service.py backend\app\rag\smart_understanding.py
  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml run --rm backend pytest -q backend/tests/test_v5_assistant_retrieval.py

Enable/refresh environment
--------------------------
If v5 query mode is already enabled, rerun the helper so new v5.1 tuning keys are written to .env:

  powershell -ExecutionPolicy Bypass -File .\merge-v5-env.ps1 -EnableQuery

Build and recreate backend only:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build backend
  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml up -d --force-recreate backend

No v5-worker/reprocess command is needed.

Debug before UI testing
-----------------------
Examples:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml exec backend python -m app.rag.v5.assistant_debug --question "on which page number of MRGR, duties of SC are defined"

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml exec backend python -m app.rag.v5.assistant_debug --question "duties of train operator as per MRGR 2020"

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml exec backend python -m app.rag.v5.assistant_debug --question "explain provision of hand signals in DMRC"

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml exec backend python -m app.rag.v5.assistant_debug --question "use of alcohal in dmrc"

For the first query, the debug output should route MRGR strongly and should put a Station Controller
responsibility/duty heading above incidental SC mentions. Exact rule/page depends on what the active
processed PDF actually contains.

Default tuning
--------------
RAG_V51_ASSISTANT_ENABLED=1
RAG_V51_AI_RERANK_ENABLED=1
RAG_V51_MAX_SEARCH_ROUNDS=3
RAG_V51_ROUTE_DOCUMENTS=8
RAG_V51_ROUTE_PER_QUERY=80
RAG_V51_SCOPED_PER_ARM=56
RAG_V51_RERANK_CANDIDATES=48
RAG_V51_FINAL_CANDIDATES=64
RAG_V51_SECTION_EXPANSION_ENABLED=1
RAG_V51_SECTION_SEEDS=10
RAG_V51_MAX_SECTION_CHUNKS=12

Cost/latency
------------
Compared with v5.0, a normal query usually adds one low-effort query-model rerank call.
If evidence is incomplete, the existing evidence critic may trigger additional bounded search rounds.
This intentionally spends more retrieval/reasoning effort for accuracy. If API/latency pressure is high,
reduce RAG_V51_MAX_SEARCH_ROUNDS to 2 and/or RAG_V51_RERANK_CANDIDATES to 36 before disabling reranking.

Rollback / A-B switch
-------------------
For a quick retrieval fallback without touching data, set RAG_V51_ASSISTANT_ENABLED=0 and recreate backend.
That makes the v5.1 wrapper use the existing v5 broad retrieval as its candidate source.
For a full code rollback, the installer creates .bak-before-ims-assistant-v51 backups for modified existing files.
Restore those four backups and delete the three added files, then rebuild/recreate backend.
Do not delete rag_v5_* data and do not reprocess PDFs for rollback.

Important expectation
---------------------
No RAG system can guarantee ChatGPT-perfect answers across arbitrary noisy PDFs. This patch addresses the
specific architectural gap shown by IMS: correct evidence is often present but loses to incidental chunks.
It upgrades retrieval from one broad top-K search to document routing + structural navigation + scoped
hybrid retrieval + AI reranking + completeness expansion + bounded evidence search, while keeping the
closed-book grounding constraint.
