BEST-SUPPORTED ANSWER + CONSISTENT EVIDENCE FIX
================================================

Problem demonstrated by the supplied transcript
------------------------------------------------
The query:
  items to be kept in kit bag
was treated as a short reference lookup instead of a synthesized list request.
That caused the old reference evidence builder to dump a very large set of corpus
matches under "Information found in the documents" instead of answering the
question. The useful TO/RA/ETO kit-bag evidence was present near the top, but was
not synthesized into the requested answer.

What this patch changes
-----------------------
1. Natural shorthand list/composition requests are deterministic answer-mode:
   - items to be kept in kit bag
   - kit bag contents
   - kit bag items
   - equipment carried by TO
   - documents required for manual point operation
   - what is in kit bag

   Bare concepts still remain reference-mode:
   - kit bag
   - alcohol
   - document SC-06

   Explicit corpus-navigation requests still remain reference-mode:
   - find kit bag
   - mentions of kit bag
   - references to SC-06

2. List queries receive retrieval expansions that look for list/items/contents/
   equipment/documents/carried/kept/possession/required language, while generic
   words such as "items" and "kept" are removed from list focus terms so the
   actual subject (for example, kit bag) drives relevance.

3. List evidence reranking recognizes composition language such as carry,
   contents, equipment, items, kept, possession and required.

4. Best-supported synthesis fallback:
   If normal synthesis returns no answer or a negative answer despite relevant
   evidence, the answer model re-checks a bounded strongest-evidence set.
   It may organize, deduplicate and reconcile explicit facts, but must not invent
   a missing rule, item, number, condition, role or procedure step.

   If there is no single definitive/complete answer but useful parts are directly
   supported, the response briefly states that limitation and gives the supported
   facts with citations.

5. Strict-selection salvage:
   If deterministic relevance selection returns zero chunks but retrieval produced
   candidates, the AI reviews a small strongest candidate set before giving up.
   If it still cannot produce a supported answer, those reviewed excerpts are
   returned in the evidence payload instead of being hidden.

6. Consistent evidence UI:
   Answer mode:
     Answer
     Copy answer
     > Evidence reviewed by AI   (all excerpts actually reviewed by synthesis)
     > Retrieved evidence        (only sources cited in the final answer)

   Explicit reference mode:
     Reference result
     > Matching evidence

   Reference mode no longer duplicates the same evidence under two expanders.

7. List formatting guard:
   A list/composition answer that comes back as prose is repaired into compact
   cited bullets. Role/context-specific additions are separated only when the PDFs
   distinguish them.

Safety behavior
---------------
"Best-supported" does NOT mean guessed. For critical operational documents:
- every factual sentence/bullet must cite supplied PDF evidence;
- exact numerical/factual questions never substitute a merely related number;
- partial procedures are explicitly described as partial and are not presented as
  a complete SOP;
- unrelated keyword hits are omitted from the answer even though relevant reviewed
  evidence remains available in the expandable evidence panel.

New optional settings
---------------------
BEST_SUPPORTED_ANSWER_ENABLED=1
BEST_SUPPORTED_SOURCE_LIMIT=64
BEST_SUPPORTED_CANDIDATE_REVIEW_LIMIT=48

These are bounded to avoid turning the fallback into another huge prompt.

Deployment
----------
Replace the files in this ZIP and run:
  docker compose up -d --build

No PDF reprocessing, re-embedding, or database migration is required.

Regression checks
-----------------
Expected deterministic classification:
  items to be kept in kit bag                   -> answer / list
  kit bag contents                              -> answer / list
  kit bag items                                 -> answer / list
  equipment carried by TO                       -> answer / list
  documents required for manual point operation -> answer / list
  what is in kit bag                            -> answer / list
  kit bag                                       -> references
  alcohol                                       -> references
  document SC-06                                -> references
  find kit bag                                  -> references
  mentions of kit bag                           -> references
  speed of pilot train on AEL                   -> answer / fact_lookup

The live backend progress panel from the previous cumulative patch remains enabled.
During fallback it can show stages such as:
  Reviewing related evidence before giving up
  Building the best-supported answer
  Formatting the supported list
