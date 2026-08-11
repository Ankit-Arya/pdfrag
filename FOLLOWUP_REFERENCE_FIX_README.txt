PDFRAG follow-up context + referenced-procedure retrieval fix
============================================================

This ZIP is cumulative with pdfrag_adaptive_answer_evidence_fix.zip.
Replace files using the paths in this archive. It intentionally does NOT contain
backend/Dockerfile, so keep the separate PyTorch/Docker build fix if you applied it.

What this fixes
---------------
1. Complete new questions start a clean topic. An older AEL/Line/mode/abbreviation
   cannot leak into a new self-contained question such as "speed of train in high winds".
2. Subject-less directives such as "provide applicable procedure then", "give more",
   "show the procedure", and "provide this" are treated as follow-ups.
3. Follow-ups use only the immediately preceding user topic plus the previous turn's
   validated self-contained question and PDF-derived routing codes. Earlier unrelated
   chat topics are not scanned for abbreviation/context resolution.
4. Index/catalog/cross-reference hits are treated as pointers. If a relevant index row
   identifies SC-06 (or another supported document code), the backend resolves the
   corresponding ready PDF and retrieves from that actual procedure before answering.
5. Explicit codes typed by the user (for example SC-06) also trigger the document hop.
6. Expanded evidence is rendered as document -> page(s) -> section -> readable
   excerpt/table. Synthetic [PDF CHUNK CONTEXT] envelopes and retrieval scores are hidden.
   All evidence chunks reviewed by the AI remain available in the expansion panel.

New optional environment settings
---------------------------------
REFERENCE_HOP_ENABLED=1
REFERENCE_HOP_MAX_DOCUMENTS=6
REFERENCE_HOP_CHUNKS_PER_DOCUMENT=400

The docker-compose defaults already enable these, so an existing .env does not need
changes unless you want to override them.

Deployment
----------
docker compose up -d --build

No PDF reprocessing is required for this fix. It operates on existing ready documents,
chunks, embeddings, filenames and chat metadata.

Recommended regression test
---------------------------
Ask these in one chat, in order:
1. what is pilot speed in AEL
2. speed of train in high winds
3. speed of train in high winds in general
4. provide applicable procedure then
5. what about underground?

Expected behavior:
- #2 is a clean new topic and must not inherit AEL from #1.
- If an index result identifies SC-06, the backend should retrieve the actual SC-06 PDF.
- #4 remains about high-wind train movement and retains SC-06 only as a routing hint.
- #5 remains in the high-wind/SC-06 topic unless the user's wording clearly starts a new topic.
- Factual answers remain grounded in freshly retrieved PDF chunks, never prior assistant prose.
