# Scenario / Applicability Scope Retrieval Fix

Built against the current `main` branch of `Ankit-Arya/pdfrag` (answer-first-v3 baseline).

## Problem reproduced

A general VCB/NSCZ question could retrieve SC-30 and answer correctly, while adding the explicit
constraint `Line 1` caused the relevant continuation chunks to be rejected and the fallback to use an
unrelated Airport/BHS procedure. The source procedure places applicability (`A. In Line - 1, 2, 3 & 4`)
in one part of section 6.2 and the operational actions in following paragraphs/chunks. Requiring every
individual chunk to repeat every context token made the more-specific question less reliable.

## Fix

1. Conditional action questions such as `if ... what should TO do` are deterministically classified as
   procedures. They no longer depend on the query LLM to choose the procedure path.
2. `Line 1` is kept as one applicability constraint; the standalone `1` is no longer duplicated as a
   second hard anchor.
3. Line-scope headings such as `In Line - 1, 2, 3 & 4` are inherited by nearby continuation chunks at
   retrieval/reranking time. No re-embedding is needed.
4. Local neighboring chunks can supply context (for example TO/line heading) to an action chunk without
   artificially increasing the action chunk's topical score.
5. Explicit line/equipment/document constraints are mandatory. A preferred document cannot bypass a
   wrong-line or wrong-equipment constraint.
6. A second-stage scenario-body router examines small consecutive windows of retrieved chunks. This can
   promote a dedicated SOP even when its filename/opening title is generic and the exact scenario appears
   deep inside section 6.x.
7. Strong candidate neighbors are fetched before the first relevance gate, so parent applicability text is
   available before filtering instead of only after filtering.
8. Best-supported fallback is hard-context guarded. A BHS/undershoot procedure cannot be substituted for a
   Line-1 + VCB + NSCZ question merely because both mention a stopped train.
9. If a routed primary procedure exists but the first synthesis returns a negative/no-answer result, the
   primary SOP receives a targeted procedure re-check before generic best-supported recovery.
10. Named lines (for example `Red Line`) are not hard-coded. The backend only maps a named line to a
    canonical `Line N` when a ready PDF explicitly contains that mapping. The mapping chunk is retained as
    evidence and must be cited when the answer relies on it.
11. Answer policy version is now `answer-first-v4-scenario-scope-2026-08-13` for deployment verification.

## New settings (safe defaults)

```env
SCENARIO_DOCUMENT_ROUTING_ENABLED=1
SCENARIO_DOCUMENT_MAX_DOCUMENTS=3
SCENARIO_DOCUMENT_WINDOW_CHUNKS=6
APPLICABILITY_INHERIT_CHUNK_WINDOW=8
LOCAL_ANCHOR_CONTEXT_WINDOW=2
PRESELECTION_NEIGHBOR_SEED_LIMIT=180
PRESELECTION_NEIGHBOR_WINDOW=2
LINE_ALIAS_SCAN_CHUNKS=60
```

They are also supplied through `docker-compose.yml`, so an existing real `.env` does not need edits
unless you want to override the defaults.

## Deployment

Replace the paths from this ZIP in the repository and run:

```bash
docker compose up -d --build
```

No PDF reprocessing, re-embedding, or database migration is required. The change operates over the
existing stored chunks and their chunk indexes.

## Regression sequence

Run these independently and in the same chat:

```text
if train stops after passing VCB open board but before NSCZ board and both VCBs are in open condition, what should TO do to bring the train to next station

In Line 1 if train stops after passing VCB open board but before NSCZ board and both VCBs are in open condition, what should TO do to bring the train to next station

In Red Line if train stops after passing VCB open board but before NSCZ board and both VCBs are in open condition, what should TO do to bring the train to next station
```

Expected:
- the Line-1 query routes to the same Line-1/2/3/4 branch in SC-30 as the broad query;
- adding `Line 1` increases applicability precision rather than suppressing the procedure;
- wrong-Line-7 or Airport/BHS evidence is not eligible as a fallback answer;
- `Red Line` is mapped only if the indexed PDFs explicitly establish `Red Line = Line 1`; otherwise the
  system does not guess the alias;
- the final procedure remains numbered and cited, with Evidence reviewed by AI and Retrieved evidence
  unchanged from the current UI contract.

## Validation performed

- Python compilation passed for all modified backend files and tests.
- Docker Compose YAML parsed successfully.
- Focused runtime test: conditional query -> `procedure`, `answer` mode.
- Focused runtime test: context contains `Line 1` but not duplicate standalone `1`.
- Scenario-body router selected simulated SC-30 over unrelated BHS/Line-7 documents.
- Hard-context fallback retained only SC-30.
- Relevance selection retained the SC-30 continuation chunks even when they did not repeat `Line 1`.
- A preferred wrong-Line-7 candidate was rejected when the user explicitly requested Line 1.
- Named-line parser test accepted explicit `Red Line (Line 1)` mapping and does not manufacture a mapping.

A live PostgreSQL/OpenAI end-to-end request against the deployed corpus was not available in this build
environment, so the above regression sequence should be run after deployment.
