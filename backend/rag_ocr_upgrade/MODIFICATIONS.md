# Modification map

## Existing files changed

- `app/api.py` — total upload limit, OCR/table capability health fields, collection warnings, and per-request query rewrite switch.
- `app/config.py` — OCR, table, query rewrite, fuzzy retrieval, timeout, output, and resource-limit settings with cross-field validation.
- `app/main.py` — version updated to 1.1.0; existing startup and request tracing retained.
- `app/models.py` — OCR/table statistics, query interpretation details, retrieval metadata, warnings, and grounding status.
- `app/rag/chunking.py` — structure-aware text splitting and row-safe Markdown table splitting.
- `app/rag/embeddings.py` — existing normalized embedding behavior retained.
- `app/rag/guardrails.py` — citation validation now understands Markdown answer tables.
- `app/rag/llm.py` — generic generation method, explicit timeout/output limits, transient retry handling, and query-rewrite support.
- `app/rag/pdf.py` — layout-preserving extraction, OCR fallback, repeated margin cleanup, and table-to-Markdown conversion.
- `app/rag/prompts.py` — query rewrite prompts, original-intent protection, table reasoning instructions, and exact prompt excerpt tracking.
- `app/rag/service.py` — extraction statistics, query planning, multiple query embeddings, strict thresholds, exact sources, and safe grounding fallback.
- `app/rag/store.py` — BM25-style keyword retrieval, fuzzy typo correction, multi-query semantic search, score fusion, diversity filtering, and NumPy fallback when FAISS is unavailable.
- `app/rag/types.py` — content types, query plans, prompt sources, and richer retrieval scores.

## New files

- `app/rag/query.py` — AI-assisted spelling correction, semantic query expansion, JSON parsing, and fail-open fallback.
- `.env.example` — configuration examples for OCR, tables, and query rewriting.
- `README.md` — installation and operation guide.
- `requirements.txt` — application dependencies.
- `requirements-dev.txt` — test dependencies.
- `tests/` — OCR, table extraction/chunking, fuzzy retrieval, hybrid retrieval, query parsing, and prompt-source tests.
