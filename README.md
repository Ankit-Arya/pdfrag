# RAG PDF QA patch

Copy the files in `app/rag/` over your existing backend `app/rag/` files.

Modified files:

- `app/rag/service.py`
- `app/rag/store.py`
- `app/rag/prompts.py`
- `app/rag/guardrails.py`
- `app/rag/types.py`

What changed:

1. Adds hybrid retrieval: FAISS vector search + dependency-free keyword search.
2. Retrieves wider context by default: at least 12 chunks and more candidates internally.
3. Caps the effective similarity threshold at `0.08` so useful chunks are not discarded too early.
4. Falls back to best available retrieved chunks instead of returning no-answer before the LLM sees PDF text.
5. Improves the prompt so partial PDF-supported answers are allowed.
6. Softens citation validation so headings and short lead-ins do not invalidate otherwise cited answers.
7. Adds one citation-repair retry when the LLM gives a useful answer but citation formatting fails.
8. Returns sources with `grounded=false` when citation validation still fails, making debugging possible instead of hiding everything behind the generic no-answer message.

Recommended environment values after applying this patch:

```env
TOP_K=12
MIN_SIMILARITY=0.05
MAX_CONTEXT_CHARS=30000
CHUNK_SIZE_CHARS=900
CHUNK_OVERLAP_CHARS=250
```

Then rebuild:

```cmd
docker compose down
docker compose up -d --build
```
