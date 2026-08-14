PDFRAG CHAT DOCUMENT LIBRARY UI PATCH
====================================

Purpose
-------
The chat page previously rendered the complete shared PDF library immediately after login.
This patch keeps the chat focused and moves the PDF library behind an on-demand modal.

Changed file
------------
frontend/src/components/ChatPanel.vue

New behavior
------------
- The chat page no longer renders all PDFs above the conversation.
- A compact "Documents · N" button appears in the top-right chat header.
- The PDF list is lazy-loaded only when the user opens the Documents dialog.
- The dialog includes live filename search.
- The dialog includes status filtering: All / Ready / Processing / Uploaded / Failed.
- Users can download individual PDFs without blocking the entire dialog.
- Users can refresh the PDF list from the dialog.
- The dialog closes with the X button, backdrop click, or Escape.
- Page scrolling is locked while the modal is open and restored on close/unmount.
- Mobile layout becomes a bottom-sheet style dialog.
- Existing chat, evidence, streaming progress, and answer behavior are unchanged.

Deployment
----------
Replace frontend/src/components/ChatPanel.vue with the file in this ZIP, then rebuild:

    docker compose up -d --build frontend

or rebuild the full stack if that is your normal deployment command:

    docker compose up -d --build

No backend changes, database migration, PDF reprocessing, or re-embedding are required.

Validation performed
--------------------
- TypeScript script block transpiled successfully with TypeScript 5.8.3.
- Verified that the old document-library-card no longer exists in the template.
- Verified there is no listDocuments() call on component mount.
- Verified the Documents dialog is lazy-loaded on first open.
- Verified search/status filtering paths and per-document download state are present.
- Verified Escape listener and body-scroll cleanup are removed on component unmount.

Notes
-----
DOCUMENT_LIBRARY_MODAL.patch is included for review. The replacement Vue file is the easiest
way to deploy the change.
