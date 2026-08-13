PDFrag answer-first summary fix - 2026-08-13
================================================

Observed regression
-------------------
Question:
    types of signals for train operation

The backend retrieved the authoritative MRGR 2020 Rule 16 evidence that explicitly lists:
    - cab signals
    - fixed signals and equipment
    - hand signals
    - virtual signals

But the response started with "Information found in the documents" and dumped broad excerpts
instead of synthesizing the straightforward four-item answer.

Root cause
----------
1. query.py recognized the word "types" as intent=list, but _infer_search_mode() separately
   treated a short noun phrase without a question mark as search_mode=references.
2. service.py branches on search_mode BEFORE synthesize_answer(). Therefore the request went
   directly to _reference_answer().
3. _reference_answer() intentionally uses build_evidence_answer(), whose job is broad readable
   evidence display, not answer synthesis. Retrieval was successful; synthesis was bypassed.
4. The previous list detector focused on composition words (items/contents/equipment) and did
   not recognize taxonomy/enumeration phrases such as types/kinds/categories/classes/modes.

Important network conclusion
----------------------------
For identical request text hitting the same backend build, this mode decision is deterministic.
The office network itself cannot turn answer mode into reference mode. If two offices still show
different behavior after this patch, they are almost certainly hitting different backend/frontend
builds, different host/DNS targets, or one browser is serving an older SPA shell.

This patch therefore also exposes the answer policy fingerprint in /api/health and in saved answer
metadata, and prevents index.html from being cached across deployments.

Changes
-------
backend/app/rag/query.py
- Added ANSWER_POLICY_VERSION=answer-first-v3-2026-08-13.
- Added enumeration cues for plural taxonomy wording such as types, kinds, categories, classes,
  forms, modes, methods and ways.
- Natural taxonomy questions such as "types of signals for train operation" are list+answer.
- Explicit intent cues in the original user wording now outrank LLM rewrite wording, so a rewrite
  cannot randomly turn the same request from list into fact_lookup or vice versa.
- Multi-word informational fragments default to answer synthesis.
- Reference mode is deliberately narrow:
    * explicit corpus navigation: find/search/mentions/references; or
    * genuinely bare one/two-term lookups such as "alcohol", "kit bag", "SC-06".
- Added search expansion for enumeration questions. Example:
    types of signals for train operation
      -> signals for train operation
      -> types of signals for train operation types kinds categories classification list
- Enumeration words are removed from list focus terms so retrieval focuses on the subject.

backend/app/rag/service.py
- Added a defense-in-depth mode invariant after QueryPlanner returns.
- The service recomputes the deterministic answer/reference contract from the validated plan.
- If a future planner change accidentally returns references for an answer request, service.py
  corrects it before retrieval/synthesis branches.
- Fuzzy title/subject routing no longer becomes a dominant primary-document backbone for ordinary
  fact/list/taxonomy questions. It remains dominant for procedures, requirements, troubleshooting,
  summaries, or when the user explicitly names a document code. This prevents a topical title from
  overshadowing a cleaner authoritative rule in another PDF.
- Logs the policy fingerprint, mode, intent and question for deployment diagnosis.

backend/app/rag/relevance.py
- Added generic list-definition cues such as following/namely/category/kind/class/mode.
- Added a bounded enumeration evidence bonus. A source saying "the following ... namely" is
  preferred over a document that merely discusses the same topic broadly.
- Normal subject/focus/applicability gates still apply; unrelated enumerations cannot win just
  because they contain "following" or "namely".

backend/app/rag/synthesis.py
- Straightforward list answers are quality-checked for evidence-dump formatting and excessive
  verbosity.
- Non-exhaustive list requests with >20 bullets, >5000 characters, or raw PDF/document headings
  are repaired into a concise supported list.
- The list repair prompt now prefers the most direct authoritative enumeration and asks for the
  smallest complete supported set, not background material.
- Explicit exhaustive requests (all/complete/comprehensive/every/exhaustive/full/entire) are not
  artificially truncated by this compactness gate.

backend/app/models.py + backend/app/api.py
- /api/health now returns:
    answer_policy_version: answer-first-v3-2026-08-13
- Each completed answer stores the same policy version in chat metadata and audit metadata.

frontend/nginx.conf
- index.html is no-store/no-cache so another office/browser cannot remain pinned to an older SPA
  entrypoint after deployment.
- Vite hashed /assets/ remain immutable and cacheable.

frontend/src/services/api.ts + frontend/src/App.vue
- Preserve the optional answer_policy_version returned/stored by the backend.

Regression expectations
-----------------------
These must synthesize answers:
    types of signals for train operation        -> list / answer
    types of signals                            -> list / answer
    signal types                                -> list / answer
    signals for train operation                 -> answer
    items to be kept in kit bag                 -> list / answer
    speed of train in high winds                -> answer
    what is pilot speed in AEL                  -> answer
    provide applicable procedure then           -> procedure / answer

These deliberately remain reference lookups:
    alcohol
    kit bag
    SC-06
    find mentions of SC-06

Expected answer shape for the reported case
-------------------------------------------
A short grounded answer such as:
    The signal types used for controlling train movements are:
    - Cab signals. [S#]
    - Fixed signals and equipment. [S#]
    - Hand signals. [S#]
    - Virtual signals. [S#]

Then the existing UI contract remains:
    Copy answer
    > Evidence reviewed by AI
    > Retrieved evidence

Deployment
----------
Replace the files from this ZIP and rebuild:
    docker compose up -d --build

No PDF reprocessing, re-embedding, or database migration is required.

After deployment, compare both offices:
    GET /api/health

Both must report:
    "answer_policy_version": "answer-first-v3-2026-08-13"

Also check backend logs for:
    RAG policy=answer-first-v3-2026-08-13 mode=answer intent=list question='types of signals for train operation'

If two offices report different policy versions, they are not reaching the same deployed build.
