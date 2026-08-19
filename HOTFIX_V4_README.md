# pdfrag Smart RAG v4 - Context Isolation + Post-Merge Evidence Coverage

Target: pdfrag Smart RAG v3 installed over repository commit `caac7d44af721d6b1cc1c127d064917658f32650` (or an equivalent tree containing the v2 + v3 overlays).

This is an **incremental backend-only overlay**. It does not replace frontend files, does not change the database schema, and does not require PDF OCR/reprocessing or re-embedding.

## Why v4 exists

Two architecture regressions remained after v3:

1. **Context bleed**: v3 deliberately made recent USER turns available to the AI interpreter for every request. The model was expected to set `uses_history=false` for a self-contained new question, but a probabilistic model could still treat a topically-related new question as a follow-up. Example regression: a new question about compensation for death/injuries inherited the previous monkey-bite subject.

2. **Evidence review happened too early**: v3 ran its evidence critic inside the first per-query `smart_search()` call. That is before the baseline service finishes merging the other AI search variants, corpus scan, primary procedure reads, reference hops and neighbor chunks. The turn was then marked reviewed, so the final assembled evidence set was never coverage-checked. This can produce a confident false-negative even when the governing PDF contains an explicit answer.

## v4 behavior

### 1. Standalone-first conversation interpretation

Every message is first interpreted **without conversation history**.

- If the message is self-contained, that interpretation is final and history is never exposed to the interpreter.
- Only when the standalone interpretation explicitly says the message contains an unresolved referent/ellipsis does a second history-assisted interpretation run.
- History is allowed to fill omitted context only; it must not overwrite an explicit subject/scope in the current message.
- A shared word such as `compensation`, `evacuation`, `brake`, etc. is not itself evidence that the new message is a follow-up.

This preserves ChatGPT-like follow-ups while strongly isolating complete new questions.

### 2. Preserve scenario structure

The interpretation prompt now explicitly preserves origin, intermediate route and destination as distinct concepts. This is important for scenarios such as a stalled train where passengers detrain onto a track/walkway **en route to a station/platform**. The route should not be mistaken for the final destination.

### 3. Evidence critic moved after full candidate assembly

The v3 critic was removed from individual `smart_search()` calls. v4 reviews coverage in `smart_select()` after the normal service has assembled candidates from all normal retrieval arms.

If coverage is insufficient, v4 performs at most one bounded retry using:

- semantic/indexed Smart RAG retrieval;
- corpus-wide lexical retrieval;
- a small neighbor expansion around retry hits.

If an actual retry occurs, a second small coverage check evaluates the merged set. It cannot trigger another retry. There is still no open-ended agent loop.

### 4. More robust current-authority retrieval

When the AI marks a question authority-sensitive, v4 forces one generic current/amendment/replacement search formulation before final selection. This does not invent a document or value; it exists to make current amendment/substitution evidence compete with older cleanly-OCR'd base provisions.

Existing v2 authority/supersession filtering remains in force.

### 5. Answer-first user style

Normal answers should no longer begin with retrieval-engine language such as:

- `The supplied excerpts do not...`
- `The retrieved excerpts do not...`
- `The evidence does not...`

The answer model is instructed to lead with the supported answer: Yes/No, role, current amount, action or procedure. A genuine limitation comes afterward.

A retrieval miss is explicitly not treated as proof that the corpus lacks the answer. When evidence coverage remains unverified, the system may say it could not verify a point from the retrieved governing evidence, but it must not make a corpus-wide absence claim.

### 6. Evidence review now covers normal factual questions too

The v3 evidence critic was mostly triggered for procedure/requirement/complex questions. v4 coverage-checks every normal non-definition answer request that has evidence requirements. This catches narrow factual questions such as `who counts passengers?` or `what amount applies?` before weak related evidence becomes a negative answer.

## Regression examples (not production hard-coding)

These scenarios are used as acceptance/regression examples only. Production logic is generic.

- A complete question about death/injury compensation asked after a monkey-bite question must not inherit the monkey-bite subject.
- A genuine follow-up such as `what amount for that case?` may use the previous USER turn.
- A mid-section evacuation question that explicitly mentions passengers reaching a platform must preserve `stalled train -> track/walkway -> station/platform` rather than collapse it to a trackside-only scenario.
- A responsibility question must retrieve evidence that explicitly assigns responsibility.
- A current compensation/value question must retrieve current amendment/schedule authority rather than silently use a superseded base provision.

## Environment

**No new environment variable is required for v4.** Keep the consolidated v3 `.env` already provided, including:

```env
SMART_RAG_ENABLED=1
SMART_RAG_QUERY_VARIANTS=4
SMART_RAG_AI_INTERPRETATION=1
SMART_RAG_AI_EVIDENCE_REVIEW=1
SMART_RAG_AI_RETRY_QUERIES=2
SMART_RAG_AI_RETRY_CANDIDATES=120
SMART_RAG_AI_ANSWER_VERIFY=1
```

Changing `.env` again will not fix the v3 context/evidence-review bugs addressed by this overlay.

## Installation - PowerShell

From the pdfrag repository directory:

```powershell
Expand-Archive -Path .\pdfrag-smart-rag-context-evidence-hotfix-v4-20260819.zip -DestinationPath . -Force
powershell -ExecutionPolicy Bypass -File .\verify-v4-hotfix.ps1

docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml down
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml build --no-cache backend
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml up -d
```

Do **not** use `down -v`.

No `smart_backfill` is required just for v4 if the v2 authority/derived indexes were already built successfully. No PDF reprocessing is required.

## Runtime confirmation

After startup:

```powershell
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml logs backend --since 10m | Select-String "AI-first understanding v4|ai_postmerge|stage=ai_interpret"
```

Expected startup marker:

```text
Smart RAG runtime patch installed (AI-first understanding v4; post-merge evidence coverage)
```

For a normal complex question, logs may contain:

```text
smart_rag stage=ai_interpret ...
smart_rag stage=ai_postmerge_evidence_review ...
```

and, only when the first assembled set was incomplete:

```text
smart_rag stage=ai_postmerge_final_review ...
```

## Validation performed

Overlay simulation was built from:

1. original Smart RAG patch;
2. reliability v2 overlay;
3. AI-first v3 overlay;
4. this v4 overlay.

Focused regression suite result: **52 passed**.

Python `compileall`: PASS.

This is not a substitute for running the full repository CI and acceptance tests against the production PostgreSQL corpus. After deployment, test multiple paraphrases for each known scenario and compare the governing PDF family/current authority, not merely answer wording.
