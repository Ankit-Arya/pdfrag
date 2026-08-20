# pdfrag RAG v5 structured-ingestion migration

Validated repository: `Ankit-Arya/pdfrag`

Validated base commit: `665655ecb1011bad2f08497f879e07403c508f56` (`AI patch`)

## Why this is a migration, not another hotfix

The existing repository has strong OCR/retrieval components, but its persistent knowledge unit is still essentially flattened chunk text. Rich layout information calculated during extraction/chunking is largely not persisted as first-class structure. A retrieval or prompt patch cannot reliably recover row/column relationships, reading order, heading hierarchy or content that was discarded during ingestion.

RAG v5 therefore creates an additive, structure-preserving generation in new `rag_v5_*` tables. The current v4 query path remains live while the new generation is built and audited. Query cutover is an explicit environment switch and rollback does not require deleting the v5 data.

## What v5 changes

### 1. Layout-preserving PDF extraction

Each page is reconstructed from positioned words rather than only one flattened page string. V5 stores ordered elements with page number, bounding box, extraction source and confidence. Elements include headings, paragraphs, list items, tables/table rows and figure/image regions.

Native PyMuPDF text is preferred where it is complete. Tesseract OCR is used for scan/image-heavy pages and stores word boxes/confidence. The image installs English and Hindi Tesseract packs by default (`eng+hin`). Add other language packs if the corpus requires them.

### 2. Multiple table recovery paths

V5 does not assume that `pdfplumber.extract_tables()` finding zero tables means a page has no table. It combines:

- pdfplumber line/border extraction;
- pdfplumber text/whitespace extraction with the repository's false-table quality filter;
- aligned word-coordinate reconstruction from PyMuPDF/OCR words;
- multi-page table continuation merging;
- nearby-table schema inheritance when continuation pages omit column headers.

Numbered procedure paragraphs are explicitly rejected as fake tables so rule headings are not swallowed by table detection.

Every accepted table row becomes its own semantic retrieval child with inherited table title, section path and column names.

### 3. Layout-aware headings and section trees

Heading detection uses font size, boldness, capitalization, numbering and geometry. Standalone rule numbers printed beside a heading are joined (for example `52.` + `Unusual occurrences --`). Heading state is carried forward to subsequent prose and table rows.

### 4. Semantic evidence chunks

V5 does not blindly split everything every N characters:

- prose is grouped by section and split on element/sentence boundaries;
- numbered/list content remains tied to the governing section;
- one table row is one retrieval child plus inherited schema/context;
- table rows never mix with unrelated prose;
- figures remain searchable from captions/nearby extracted labels and are flagged when visual semantics have not been interpreted.

### 5. Persistent structure

New additive tables preserve processing runs, pages, layout elements, tables, table rows, terminology, semantic chunks and authority directives. The existing `documents` and v4 corpus are not deleted during migration.

### 6. Authority/current-version metadata

Explicit PDF wording such as `shall be substituted`, `shall be replaced` and explicit word substitutions is indexed. Replacement sections use inherited v5 section paths rather than a fixed chunk window, so a multi-page amended schedule remains attached to the amendment that introduced it. An appended older/base instrument is marked historical only after a strong new-instrument boundary.

Current evidence is preferred by default for ordinary questions; historical/superseded material remains searchable rather than being deleted.

### 7. Grounded terminology

Strong acronym definitions are indexed from original v5 chunks. Table-style definitions are accepted only when acronym initials support the candidate expansion, reducing OCR/index poisoning. These hints may help the AI understand the question, but the final definition still requires a cited PDF source.

### 8. ChatGPT-like language understanding, closed-book facts

The existing AI-first interpreter is retained as the language-understanding layer. It can correct spelling/grammar, understand colloquial language, infer the likely request and create evidence needs/search formulations. It is not factual evidence.

V5 retrieval operates per interpreted evidence requirement using vector search, exact/simple FTS, English-stemmed FTS, structured table-row search and reciprocal-rank fusion. Parent evidence is expanded only after strong child matches are found.

The answer layer is simpler than v4: one direct grounded answer, one evidence verifier, then citation repair only if citation validation fails. It explicitly avoids search-engine phrases such as `the supplied excerpts...` in ordinary answers.

## Important limitation

No local parser can truthfully guarantee perfect extraction of every possible PDF layout. V5's policy is therefore: **recover structure where possible and expose extraction uncertainty instead of silently pretending the page was completely understood**. Figure/image regions are retained and reported; their visual meaning is not inferred by this package. Low-confidence pages and rejected table candidates are visible in diagnostics.

## Safe deployment model

### Stage A — install code, keep v4 serving

Set `RAG_V5_QUERY_ENABLED=0`. V5 schema exists, but user questions continue through the existing v4 pipeline.

### Stage B — reprocess the corpus into v5

Run `python -m app.rag.v5.reprocess` from the dedicated `v5-worker` Compose service. It reads the existing PDF bytes from PostgreSQL and writes only `rag_v5_*` generation tables. It does **not** replace the live v4 `document_chunks` during migration.

Processing is intentionally CPU-heavy because difficult pages may require OCR and coordinate reconstruction. Run it as a migration worker rather than in the API process.

### Stage C — audit

Run:

```text
python -m app.rag.v5.diagnostics --strict
```

Every ready document must have an active v5 generation before cutover. Review low-confidence pages/table warnings as well.

Inspect individual documents with:

```text
python -m app.rag.v5.audit_document --filename "Claims" --find "Fracture of Major Bone-Femur"
python -m app.rag.v5.audit_document --filename "MRGR" --find "Train divided"
```

### Stage D — query cutover

Only after diagnostics pass, set:

```text
RAG_V5_QUERY_ENABLED=1
```

and recreate the backend. `main.py` then skips the v4 runtime monkeypatch chain and the repository singleton explicitly instantiates `V5RagService`.

For PDFs uploaded/reprocessed after cutover, v5 also mirrors the already-computed semantic chunks/embeddings into legacy `document_chunks` by default (`RAG_V5_LEGACY_CHUNK_MIRROR=1`). This does not affect the one-time migration, and gives a safer rollback path for later documents without parsing/embedding the PDF twice.

## Rollback

Set `RAG_V5_QUERY_ENABLED=0` and recreate the backend. The pre-existing v4 corpus remains present because the migration never deletes it. If documents were newly uploaded/reprocessed after v5 cutover, the legacy mirror contains their v5 semantic chunks; run the existing Smart-RAG backfill after rollback if you need to rebuild v4 derived indexes for those documents.

Do **not** run `docker compose down -v` during installation or rollback.

## Validation performed for this package

The focused v5 unit suite covers geometry tables, multi-page tables, false-table rejection for numbered procedure text, rule-number/heading reconstruction, semantic chunking, authority replacement spans and terminology poisoning safeguards.

The supplied Claims PDF was also processed end-to-end through the v5 layout/chunk/authority path. The current 2025 row was recovered as a structured table-row chunk carrying the `Amount of Compensation` column and marked `current_replacement`; the later old 2017 copy remained separate and was marked `historical_appended`.

See `QA_RESULTS_V5.txt` for exact local validation results and limitations.
