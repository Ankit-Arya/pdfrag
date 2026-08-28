PDF Chunk Explorer v1
=====================

Purpose
-------
Adds a read-only PDF-wise chunk explorer to the IMS Documents workspace so retrieval
failures can be diagnosed against the exact active RAG v5 chunks.

What is visible
---------------
- Select any ready PDF from Documents -> Inspect chunks.
- Browse every active v5 chunk in chunk_index order.
- Pagination: 50 / 100 / 200 / 500 chunks per page.
- Filter by text/heading/section, PDF page, content type and authority status.
- View chunk index, page range, content type, authority status, extraction confidence,
  section path, heading, parent key, table ID/row, chunk UUID and raw metadata.
- Toggle clean text vs the raw [PDF STRUCTURE] envelope stored in rag_v5_chunks.text.
- Copy raw text or full chunk JSON.
- Export the currently visible chunk page as JSON.
- Download the source PDF from the same screen.

Security / scope
----------------
The endpoint is authenticated and read-only. It does not expose embedding vectors and
it performs no mutations. Users already have authenticated access to the source PDFs.

Backend endpoint
----------------
GET /api/documents/{document_id}/v5-chunks

Query parameters:
  offset=0
  limit=100              max 500
  q=<text>
  page=<pdf page>
  content_type=<type>
  authority_status=<status>

Only chunks belonging to the document's active, ready rag_v5_processing_runs row are
returned. Old/inactive processing runs are deliberately excluded.

Apply
-----
python .\pdfrag-pdf-chunk-explorer-v1\apply_pdfrag_pdf_chunk_explorer_v1.py --repo .

Validate backend
----------------
python -m py_compile `
  backend\app\models.py `
  backend\app\api.py

docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  run --rm backend pytest -q backend/tests/test_pdf_chunk_explorer_contract.py

Build frontend first
--------------------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  build frontend

Then backend
------------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  build backend

Deploy
------
docker compose `
  -f docker-compose.yml `
  -f docker-compose.smart-rag.yml `
  -f docker-compose.v5.yml `
  up -d --force-recreate frontend backend

No reprocessing
---------------
No DB migration, PDF reprocessing, OCR rerun, chunk rebuild or embedding rebuild is required.

Suggested diagnosis for the current RS-3 case
----------------------------------------------
1. Open Documents.
2. Search "RS 3".
3. Click Inspect chunks.
4. Search "BIC".
5. Search "Brake Isolation Cock".
6. Search "50%" and "25 km/h" separately.
7. Filter/page-jump to the relevant PDF pages.
8. Toggle raw [PDF STRUCTURE] to see the exact section/heading envelope available to retrieval.
9. Copy chunk JSON for any governing chunk that retrieval diagnostics failed to route.
