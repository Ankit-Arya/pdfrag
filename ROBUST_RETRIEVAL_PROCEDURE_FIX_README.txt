PDFrag robust retrieval + procedure formatting fix
=================================================

Purpose
-------
This cumulative patch fixes cases where an uploaded dedicated SOP exists but the answer
was vague, mixed with generic evidence, or failed to follow the correct document. It also
makes answer/evidence presentation consistent.

Retrieval changes
-----------------
1. Two-stage document routing: questions are scored against ready PDF filenames and the
   opening subject text. Strong dedicated SOP/instruction matches are inspected directly.
2. PostgreSQL English-stem FTS runs beside the existing exact/simple FTS and pgvector path,
   improving word-family recall (obstruct/obstruction/obstructing, etc.).
3. SM-xx references are now recognized by the document-reference hop, alongside SC/SOP/JPO.
4. Index reference discovery is ranked instead of requiring a brittle fixed two-term overlap.
5. Query-model search variants may add a few ordinary semantic terms while still being
   forbidden from inventing numeric/internal identifiers or dropping protected acronyms.
6. Procedure questions such as 'what to do if ...' are deterministically classified as
   procedures even if the query model is unavailable.
7. A strongly matched dedicated procedure document is preferred during reranking; unrelated
   PDFs are limited to strong, directly applicable supplementary evidence.
8. Empty/low-information extracted table shells are heavily down-ranked.

Answer behavior
---------------
- Fact lookup: normally one direct sentence.
- Procedure/safety question: short applicability/scope line when material, then numbered
  actionable steps preserving roles, notifications, restrictions, modes and restoration criteria.
- A dedicated SOP becomes the answer backbone; generic nearby safety rules cannot replace it.
- Semantic negative answers ('does not specify/define/provide...') now trigger the fact rescue
  path just like the exact configured NO_ANSWER sentence.

Evidence UI
-----------
- Primary answer first, then Copy answer.
- Answer-mode responses consistently expose 'Evidence reviewed by AI' (all reviewed evidence)
  and 'Retrieved evidence' (the cited subset) when available.
- Evidence is grouped by document/page/section instead of raw database chunk cards.
- Synthetic [PDF CHUNK CONTEXT] metadata is removed from display.
- Blank markdown table shells are replaced by a readable verification message instead of an
  empty grid.

Dependencies
------------
Adds rapidfuzz>=3.9,<4.0 for fast document-title/subject fuzzy routing. No model download or
corpus is required. PostgreSQL's built-in English text-search configuration provides stemming.

Install
-------
Replace the files in this ZIP, then rebuild:

  docker compose up -d --build

No PDF reprocessing is required. Existing indexed chunks are reused.

Recommended regression questions
--------------------------------
- what to do if someone obstruct train movement
  Expected: SM-41 should be a primary document and the answer should be an actionable,
  role-aware procedure rather than generic OHE/adjacent-track guidance.
- what is pilot speed in AEL
  Expected: SC-04/table evidence should yield 50 km/h.
- speed of train in high winds
  Expected: SC-06 should remain the dominant high-wind procedure.
- Switch between the above topics in one chat. Each complete question must remain standalone.

Notes
-----
The patch does not change embeddings or database schema and does not require reprocessing.
The real .env, if present, overrides .env.example.
