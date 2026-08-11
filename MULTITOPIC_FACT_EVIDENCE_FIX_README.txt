PDFRAG MULTI-TOPIC CHAT + FACT LOOKUP + EVIDENCE DISPLAY FIX
=============================================================

What this cumulative patch fixes
--------------------------------
1. A self-contained question is now isolated inside QueryPlanner itself. Asking a
   completely different subject in the same chat must produce the same retrieval
   plan as asking it in a fresh chat. Prior routing hints such as SC-06 and prior
   abbreviations are discarded unless the current wording genuinely needs context.

2. "What is ..." is no longer automatically treated as a definition. Questions
   containing measurable fact dimensions such as speed, date, pressure, limit,
   duration, number, etc. are fact lookups. Example:
       what is pilot speed in AEL -> fact_lookup
       what is AEL                -> definition

3. For fact/definition questions, evidence remains strongest-first for the answer
   model instead of being re-sorted by filename/page and potentially burying the
   best table hundreds of sources deep.

4. Hierarchical synthesis still processes the full selected evidence set, but also
   gives the final answer model a bounded high-priority direct-evidence section for
   fact questions so a compressor cannot accidentally drop an obvious number.

5. If a direct fact lookup nevertheless returns the configured NO_ANSWER while
   evidence exists, one small rescue call checks only the strongest original
   excerpts. It uses the original S-number mapping. This retry happens only on the
   false-no-answer path, not on normal successful questions.

6. Evidence expansion is restored to the earlier source-card presentation:
       S# | filename | page
       expandable readable excerpt/table
   The raw [PDF CHUNK CONTEXT] envelope is removed for display. Both new and older
   saved messages are cleaned client-side. The AI still receives the original raw
   evidence internally.

7. Cumulative defaults are aligned with the lower-cost model setup requested:
       LLM_MODEL=gpt-5.6-terra
       QUERY_MODEL=gpt-5.6-luna
       SUMMARY_MODEL=gpt-5.6-luna
       LLM_REASONING_EFFORT=medium
       QUERY_REASONING_EFFORT=low
       SUMMARY_REASONING_EFFORT=low
       LLM_TIMEOUT_SECONDS=60
       MAX_OUTPUT_TOKENS=2500
       SUMMARY_MAX_OUTPUT_TOKENS=2500
   Your real .env overrides these defaults. Update the real .env separately if it
   still contains the older Sol/high/6000 settings.

Deployment
----------
Replace the files from this ZIP, then run:
    docker compose up -d --build

No PDF reprocessing is required for these changes.

Recommended regression sequence in ONE chat
-------------------------------------------
1. speed of train in high winds
2. what is pilot speed in AEL
3. provide the high wind procedure
4. what is pilot speed in AEL
5. what about underground?   (after a high-wind turn, this should follow high winds)

Expected behavior
-----------------
- #1 answers high winds.
- #2 is a new independent topic and answers AEL pilot speed from the PDF evidence.
- #3 is explicitly about high winds and retrieves that procedure.
- #4 again switches cleanly back to AEL and should match a fresh-chat result.
- #5 only inherits the immediately preceding topic when its wording lacks a subject.

Validation performed
--------------------
- Python py_compile on included backend Python files.
- Deterministic query classification tests.
- Same-question fresh-chat vs unrelated-history plan equality test.
- Fact evidence priority test.
- Evidence display-envelope stripping/table preservation test.
- TypeScript syntax transpilation for App.vue, ChatPanel.vue and api.ts.

A live PostgreSQL/pgvector/OpenAI integration test was not available in this
container, so validate the regression sequence above against your deployed corpus.
