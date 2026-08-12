PDFrag live backend progress upgrade
====================================

Purpose
-------
Show users truthful, real-time RAG workflow activity while a question is being processed,
then replace that temporary work panel with the normal final grounded answer.

This is operational progress only. It does NOT expose chain-of-thought, hidden reasoning,
model prompts, or private scratch work.

Transport
---------
- New POST /api/chat/stream endpoint using Server-Sent Events framing over fetch().
- POST is retained because chat questions need JSON payloads and Authorization headers.
- Existing POST /api/chat remains available for compatibility.
- Nginx disables buffering and gzip specifically for /api/chat/stream.
- 15-second SSE comment heartbeats keep long model calls/proxy connections alive.

Progress shown when applicable
------------------------------
- Understanding the current question and conversation context
- Planning search variants / identifying intent
- Routing to likely dedicated SOPs or instructions
- Primary document identification
- Semantic + exact + stemmed lexical retrieval progress
- Full ready-corpus lexical scan
- Reading primary document sections
- Following SC/SM/SOP/JPO/etc. document references
- Candidate evidence and selected-evidence counts
- Reading neighboring sections/pages
- Direct answer generation or hierarchical summarization
- Evidence summary batch X/Y
- Recursive evidence consolidation when required
- Waiting for model/API token capacity when rate-limit pacing is active
- Citation/applicability validation
- Fact re-check / citation repair / procedure repair only when those safeguards actually run
- Saving the completed grounded answer

Frontend behavior
-----------------
- While busy, a DMRC Q&A work card shows the current backend stage and the recent stage history.
- Stages with current/total values show a real progress meter and X/Y count.
- The panel auto-scrolls as backend stages arrive.
- Once the persisted final answer arrives, the work card disappears and the normal answer is shown.
- Existing Copy answer, Evidence reviewed by AI, and Retrieved evidence formatting is unchanged.
- Progress events are temporary and are not stored as chat messages.

Files in this cumulative patch
------------------------------
This ZIP is based on pdfrag_rate_limit_safe_rag_fix.zip and adds/updates:
- backend/app/api.py
- backend/app/rag/progress.py (new)
- backend/app/rag/service.py
- backend/app/rag/synthesis.py
- backend/app/rag/llm.py
- frontend/src/services/api.ts
- frontend/src/App.vue
- frontend/src/components/ChatPanel.vue
- nginx/nginx.conf

The ZIP also retains the files from the previous cumulative rate-limit/retrieval patch.

Install
-------
Replace the ZIP paths in the repository, preserving directories, then rebuild:

  docker compose up -d --build

No PDF reprocessing, re-embedding, or database migration is required.

Expected UI example
-------------------
DMRC Q&A
  Understanding your question
  Question interpreted
  Primary document identified: 41. SM-41 ...pdf
  Searching semantic and lexical indexes  4/4
  Scanning exact matches across the PDF corpus
  Relevant evidence selected: 86 excerpts
  Summarizing relevant evidence  3/6
  Checking citations and applicability
  Saving the grounded answer

Then the temporary work card is replaced by the final answer plus the existing evidence panels.

Validation performed
--------------------
- Python compilation passed for every Python file included under backend/app in this patch.
- Progress callback isolation/basic payload test passed.
- frontend/src/services/api.ts passed TypeScript --strict syntax/type checking in isolation.
- App.vue and ChatPanel.vue <script setup lang="ts"> sections transpiled successfully with TypeScript 5.8.3.
- Nginx configuration syntax passed nginx -t after substituting local test upstream/user values.
- The full Vue template compiler/live PostgreSQL/OpenAI end-to-end test was not available in this build environment.
