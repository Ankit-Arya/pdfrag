PDFrag adaptive-answer + expandable-evidence fix
================================================

This patch is cumulative with pdfrag_contextual_research_fix.zip.
It does NOT replace backend/Dockerfile, so any separate PyTorch Docker build fix remains intact.

Behavior
--------
1. Normal questions always go through AI answer synthesis.
   - Simple fact/figure/date/speed/limit: normally one direct cited sentence.
   - Multi-part questions/procedures: answer expands only as much as necessary.
2. Bare concept/reference lookups remain reference searches.
   - Example: "alcohol" -> document/reference discovery.
   - Example: "alcohol rules" -> synthesized answer.
3. "speed of pilot train on AEL" is explicitly classified as answer mode because "speed"
   is a fact dimension and the phrase contains a subject/context.
4. API response now separates:
   - sources: only chunks cited in the final answer.
   - evidence: every chunk reviewed by the answer/summarization pipeline.
   Both use the same S# numbering.
5. Chat UI adds:
   - Copy answer button (Clipboard API plus HTTP/LAN-compatible fallback).
   - Collapsed "Evidence reviewed by AI" containing ALL reviewed chunks.
   - Existing collapsed "Retrieved evidence" containing cited chunks only.
   - Full AI evidence is lazy-rendered only when expanded.
6. New answer evidence is stored in saved chat message metadata, so reopening a new chat keeps
   the evidence expansion. Chats created before this patch do not retroactively contain it.

Install
-------
Copy the files in this ZIP over the same repository paths, then rebuild:

    docker compose up -d --build

No PDF reprocessing is required for this change because it changes query classification,
answer synthesis, API response metadata, and frontend presentation only.

Validation performed
--------------------
- Python py_compile: all included backend Python files.
- TypeScript strict compile: frontend/src/services/api.ts.
- TypeScript syntax/transpile validation: App.vue and ChatPanel.vue script blocks.
- Search-mode regression checks including:
    speed of pilot train on AEL -> answer
    alcohol -> references
    alcohol rules -> answer
    speed -> references
    what speed? -> answer
