RAG v5 table-row + parallel reprocess hotfix

Purpose
-------
1. Fix duplicate rag_v5_table_rows(table_id,row_index) failures caused by gapped source row indexes plus multi-page table merging.
2. Canonicalize every final table to row indexes 1..N before chunking/persistence.
3. Add a pre-chunking invariant check so malformed row indexes fail explicitly before PostgreSQL insertion.
4. Add --workers N process-level document concurrency.
5. Use one PostgreSQL advisory lock per document so concurrent workers/CLI runs cannot process the same document simultaneously.
6. Avoid repeated ensure_v5_schema()/ANALYZE work inside each reprocess child; the parent does it once.

Apply
-----
From the pdfrag repository root:

python .\apply_v5_table_parallel_hotfix.py --repo .
python -m py_compile backend\app\rag\v5\layout.py backend\app\rag\v5\ingestion.py backend\app\rag\v5\reprocess.py

git --no-pager diff -- backend/app/rag/v5/layout.py backend/app/rag/v5/ingestion.py backend/app/rag/v5/reprocess.py

Rebuild
-------
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build backend
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml up -d --force-recreate backend

Two-document regression + concurrency test
------------------------------------------
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml --profile v5-migration run --rm v5-worker python -m app.rag.v5.reprocess --workers 2 --document-id 101f86f9-7129-48ef-8f15-79d4d9ed031f --document-id b9101850-2474-4e59-8aa5-a71139f7e2d6

Expected final summary: attempted=2 succeeded=2 failed=0 workers=2

Resume all remaining documents
------------------------------
docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml --profile v5-migration run --rm v5-worker python -m app.rag.v5.reprocess --workers 2

Do not add --force. Completed active/current v5 generations will still be skipped.

Parallelism guidance
--------------------
Start with --workers 2. If CPU and memory remain comfortable, try 3 then 4. Each worker is a separate process and can load its own OCR/PDF/embedding state, so throughput does not scale linearly and memory use increases with worker count.

Rollback
--------
The installer creates these backups if they do not already exist:
backend/app/rag/v5/layout.py.bak-v5-table-parallel
backend/app/rag/v5/ingestion.py.bak-v5-table-parallel
backend/app/rag/v5/reprocess.py.bak-v5-table-parallel

The processing version is intentionally not changed. This bugfix changes internal row numbering/persistence behavior but does not require rebuilding already successful active generations.
