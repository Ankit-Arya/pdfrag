PDFrag smart document grouping patch
====================================

Purpose
-------
Makes the Admin Console document inventory easier to audit while uploading a large corpus.

Features
--------
- Groups uploaded documents by filename family (SWO, OTM, Safety Circular, TI, SM Instructions,
  ATP, MRGR, handbooks, addenda, procedure orders, manuals).
- Learns repeated uppercase acronym families for documents not covered by explicit generic rules.
- Natural numeric filename sorting within each family.
- Shows status counts per family.
- Shows conservative "Possible gaps" only when an observed numeric naming range is dense enough
  to behave like a real sequence.
- Search + status filters.
- Groups the selected upload queue using the same rules.
- Marks queued files whose exact normalized filename is already present in the uploaded inventory.
- If the admin selects the source/master folder, the queue immediately shows how many filenames
  are not currently listed versus how many names already exist.

Important limitation
--------------------
Without an expected/master folder or manifest, the UI cannot know documents beyond the highest
number already observed. Therefore sequence warnings are deliberately labelled "Possible gaps".
Selecting the master folder gives the admin an exact filename-level comparison against uploaded docs.

Apply
-----
From repository root:

  python .\apply_smart_document_grouping_patch.py --repo .

Then inspect:

  git --no-pager diff -- frontend/src/components/AdminPanel.vue frontend/src/styles.css frontend/src/utils/documentGrouping.ts

Build frontend:

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build frontend

Recreate frontend (and keep backend/query settings unchanged):

  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml up -d --force-recreate frontend

No database migration is required.
