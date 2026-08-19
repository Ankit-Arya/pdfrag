# pdfrag Smart RAG AI-First Reliability Hotfix v3

Target: current Smart RAG v2 tree / repository commit `caac7d44af721d6b1cc1c127d064917658f32650`.

This is an **incremental overlay**. Extract it over the existing pdfrag repository. It does not contain or replace the frontend.

## What changes

v3 replaces the normal "keyword/intent first" philosophy with an AI-first language-understanding layer while keeping the PDF corpus as the only factual authority.

For each question the query model now produces a structured interpretation containing:

- corrected/self-contained intended question;
- conversational act (question, request, navigation, pasted evidence/correction);
- semantic concepts;
- evidence requirements that must be satisfied;
- 2-5 diversified search queries;
- explicit applicability scope;
- material ambiguity;
- whether recent USER history is actually needed;
- whether current revision/amendment authority matters;
- routing strategy (dedicated procedure, authoritative rule, structured lookup, broad corpus).

The model may use ordinary language knowledge to understand spelling mistakes, grammar, shorthand and synonyms. It is explicitly prohibited from supplying metro facts, internal acronym expansions, document numbers, procedures, amounts, speeds, applicability or role assignments from memory.

## Retrieval flow

1. AI interpretation converts noisy natural language into the intended question and evidence needs.
2. Existing Smart RAG HNSW/GIN/procedure-card/rule/authority retrieval runs using the resolved semantic search plan.
3. For complex/high-risk requests, an AI evidence critic checks whether the retrieved excerpts actually satisfy the requested evidence needs. Similarity alone is not considered enough.
4. If evidence is insufficient, at most a small bounded set of targeted retry queries is generated and searched once.
5. Existing deterministic authority/supersession, applicability and numeric-rule logic remains in force.
6. The normal grounded answer model answers the resolved question from PDF evidence.
7. For negative, authority-sensitive, ambiguous, procedure/requirement/troubleshooting answers, a bounded final grounding verifier checks the draft against the supplied PDF sources before the existing citation validator runs.

There is no open-ended agent loop.

## Generalization

The production logic is not keyed to one example such as passenger counting, a specific injury, train division, BIC, or compensation. Those examples are regression cases only. The normal path reasons in terms of semantic concepts and evidence requirements such as:

- who is responsible;
- whether an action is mandatory/permitted;
- what procedure/branch applies;
- which formal category corresponds to colloquial wording;
- which current authority/version governs;
- which measurable value belongs to which category;
- whether retrieved evidence actually answers the question.

## Changed production files

- `backend/app/rag/smart_understanding.py` (new)
- `backend/app/rag/smart_runtime.py`
- `backend/app/rag/smart_diagnostics.py`
- `docker-compose.smart-rag.yml`
- `.env.smart-rag.example`

No frontend files are included.

## Database / PDF processing

v3 adds no database table and does not require PDF OCR/reprocessing.

If Smart RAG v2 was already backfilled successfully, no new backfill is required. If diagnostics show `authority_directives: 0` or your v2 derived indexes were never built, run the existing `python -m app.rag.smart_backfill` once.

## Runtime model-call budget

Default behavior is deliberately bounded:

- 1 query-model call: interpretation/search plan (normal questions)
- 0 or 1 query-model call: evidence coverage review (complex/high-risk requests)
- 0 or 1 targeted retrieval retry; no extra answer-model call
- normal grounded answer generation
- 0 or 1 query-model call: final grounding verification (negative/high-risk/ambiguous/procedure-style answers)

All v3 AI control calls use the configured `QUERY_MODEL` and `QUERY_REASONING_EFFORT`, so the inexpensive query model can be used for these stages.

## Important environment switches

Defaults are already provided by the Smart RAG compose override:

```
SMART_RAG_AI_INTERPRETATION=1
SMART_RAG_AI_SEARCH_QUERIES=4
SMART_RAG_AI_EVIDENCE_REVIEW=1
SMART_RAG_AI_RETRY_QUERIES=2
SMART_RAG_AI_RETRY_CANDIDATES=120
SMART_RAG_AI_ANSWER_VERIFY=1
SMART_RAG_AI_VERIFY_SOURCES=24
```

Emergency rollback of the AI-first behavior without removing files:

```
SMART_RAG_AI_INTERPRETATION=0
SMART_RAG_AI_EVIDENCE_REVIEW=0
SMART_RAG_AI_ANSWER_VERIFY=0
```

The existing Smart RAG layer continues to work through deterministic fallbacks.

## Installation (PowerShell)

From the pdfrag repository directory:

```
Expand-Archive -Path .\pdfrag-smart-rag-ai-first-hotfix-v3-20260819.zip -DestinationPath . -Force
powershell -ExecutionPolicy Bypass -File .\verify-ai-first-hotfix.ps1

docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml down
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml build backend
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml up -d
```

Do **not** run `down -v`.

Do **not** use Reprocess All merely to install v3.

Run diagnostics after the backend is healthy:

```
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml exec backend python -m app.rag.smart_diagnostics
```

The diagnostics output now includes the active AI-first settings.

## Acceptance testing philosophy

Do not test only one exact wording. For each known rule/scenario, ask several variants: correct English, poor English, spelling errors, shorthand, role-first wording, yes/no wording, indirect wording and conversational follow-ups. Paraphrases that mean the same thing should route to the same governing document family/current authority and produce the same core facts.

Negative answers deserve special attention. A weak first retrieval must not become a statement that the corpus lacks the answer.
