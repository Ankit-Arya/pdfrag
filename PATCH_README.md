# pdfrag metro context patch

This patch replaces the current page/character chunking path with heading-aware, context-preserving chunking for metro internal PDF procedures.

## Why this patch exists

The current backend stores chunks mostly as page number + content type + text. For metro operations, the same question can have different answers depending on rolling stock, equipment variant, procedure, heading/subheading, mode, warning, exception, and table row. This patch makes the stored chunk text include a structured context header before embedding and indexing:

```text
[PDF CHUNK CONTEXT]
File: ...
Pages: ...
Section path: Main heading > Subheading > Procedure
Content type: text/table
Rolling stock / train context: ...
Procedure context: ...
Important tags: ...
[/PDF CHUNK CONTEXT]
```

Because the header is stored in `document_chunks.text`, both pgvector and PostgreSQL full-text search can retrieve by procedure context, rolling stock context, acronyms, headings, warnings, and table metadata without requiring a database migration.

## Files replaced

Copy these files over the existing repository files:

- `backend/app/config.py`
- `backend/app/rag/types.py`
- `backend/app/rag/chunking.py`
- `backend/app/rag/postgres_store.py`
- `backend/app/rag/service.py`
- `backend/app/rag/prompts.py`

New helper file:

- `backend/app/reprocess_documents.py`

## What changed

1. Heading/subheading-aware chunking.
   - Detects numbered headings such as `1`, `1.1`, `1.1.1`.
   - Detects uppercase manual-style headings.
   - Detects important colon subheadings such as `Warning:`, `Procedure:`, `Isolation:`.
   - Carries heading path across page boundaries.

2. Context-safe chunks.
   - Adds file, page range, section path, content type, rolling stock/procedure hints, and important tags into every stored chunk.
   - Keeps table headers with table row chunks.
   - Preserves warning/caution/mandatory/prohibited language in retrieval tags.

3. Hybrid retrieval strengthened for operational manuals.
   - Retrieves both vector candidates and full-text candidates.
   - Adds lexical overlap scoring for exact acronyms, equipment identifiers, rolling-stock names, procedure names, and fault codes.
   - Expands high-confidence hits with neighboring chunks so prerequisites/warnings/verification steps are not lost at boundaries.

4. Safer metro-specific prompting.
   - Requires the answer to state applicable context first.
   - Prohibits blending instructions across rolling stocks/procedures/sections unless explicitly supported.
   - If the question is ambiguous and retrieved sources show multiple possible contexts, the model should ask for clarification instead of giving a blended procedure.

5. No database migration required.
   - The existing `document_chunks.text` column stores the new context-enriched chunk text.
   - Existing PDFs must be reprocessed so their stored chunks get the new context header.

## Apply

From the repository root, copy the patch files over the existing files. On Linux/macOS:

```bash
cp -R backend/app ./backend/
```

If you extract this ZIP into a separate folder, run:

```bash
./scripts/apply_patch.sh /path/to/pdfrag
```

On Windows PowerShell:

```powershell
.\scripts\apply_patch.ps1 C:\path\to\pdfrag
```

## Recommended `.env` values

```env
TOP_K=12
MIN_SIMILARITY=0.05
MAX_CONTEXT_CHARS=30000
CHUNK_SIZE_CHARS=900
CHUNK_OVERLAP_CHARS=220
MAX_OUTPUT_TOKENS=1800
QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_MAX_VARIANTS=4
EXTRACT_TABLES=true
OCR_MODE=auto
```

## Rebuild and reprocess

Rebuild the backend:

```bash
docker compose down
docker compose up -d --build
```

Then reprocess existing uploaded PDFs so stored chunks are rebuilt with heading context:

```bash
docker compose exec backend python -m app.reprocess_documents
```

If you do not reprocess, old chunks will remain searchable, but they will not contain the new heading/rolling-stock/procedure context header.

## Important operational note

This patch improves retrieval and answer grounding, but it does not replace formal operational validation. Before internal rollout, test with representative PDFs for each rolling stock/procedure and verify that answers remain correctly scoped when two documents contain similar fault/procedure names but different steps.
