IMS Separate Chunk Search v1
============================

Purpose
-------
Adds a separate authenticated "Search chunks" workspace for ordinary users and admins.
It is outside chat history and does not generate an AI answer.

Search modes
------------
Hybrid: local embedding similarity + PostgreSQL full-text search.
Keyword: PostgreSQL websearch full-text search; useful for rules, codes and quoted phrases.
Semantic: local sentence-transformer search; useful when wording differs from the PDF.

Filters
-------
- all ready PDFs or one document
- text / list / table row / figure-caption chunks
- top 20 / 30 / 50 / 100 results

Result metadata
---------------
- PDF filename
- page/page range
- section/heading
- content type
- chunk index
- authority status when available
- search method and relevance score
- actual stored chunk text, with the internal [PDF STRUCTURE] envelope removed

Apply
-----
From repository root:

  python .\apply_ims_chunk_search_patch.py --repo .

Inspect:

  git --no-pager diff -- backend/app/models.py backend/app/api.py frontend/src/services/api.ts frontend/src/App.vue frontend/src/components/UploadPanel.vue frontend/src/components/ChunkSearchPanel.vue

Compile backend:

  python -m py_compile backend\app\models.py backend\app\api.py

Build/recreate backend + frontend:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build backend frontend

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml up -d --force-recreate backend frontend

No PDF reprocessing is required.
