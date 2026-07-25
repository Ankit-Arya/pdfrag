# pdfrag intelligent formatting and relevance fix

This package supersedes `pdfrag_complete_replacement.zip`.

Extract it at the project root and allow replacement of existing files.

## What this fixes

### 1. Raw Markdown in the browser

The original Vue component prints assistant text with interpolation:

```vue
<div class="message-text">{{ message.text }}</div>
```

That displays `##`, `**`, numbered lists, and Markdown tables literally.

The replacement adds a dependency-free, HTML-escaping Markdown renderer supporting:

- headings;
- paragraphs;
- numbered lists;
- bullet lists;
- nested condition bullets;
- bold text;
- citations such as `[S1]`;
- inline code;
- Markdown tables.

### 2. Wrong response structure

The backend now chooses the response layout according to the request:

- operating procedure / workflow / how-to -> numbered steps;
- comparison -> table only when common comparison fields exist;
- summary -> grouped bullets;
- direct fact or definition -> concise paragraphs;
- troubleshooting -> supported sequence or symptom/cause/action table.

### 3. Unrelated retrieved content

A focused context-selection stage now runs after hybrid retrieval.

For acronym-led questions such as `FDS operating procedures`:

- chunks containing `FDS` become anchors;
- same-page or adjacent-page continuations may be included;
- unrelated manual sections such as door isolation and degraded movement are removed;
- short requests are limited to at most four context chunks.

## Replacement files

The ZIP contains the previous complete OCR, acronym, and grounding fixes, plus:

```text
backend/app/rag/prompts.py
backend/app/rag/service.py
backend/tests/test_context_selection.py

frontend/src/components/ChatPanel.vue
frontend/src/utils/markdown.ts
frontend/src/main.ts
frontend/src/markdown.css
```

No new npm package is required.

## Apply and rebuild

From the project root:

```powershell
docker compose down --remove-orphans
docker compose build --no-cache --pull --progress=plain backend frontend
docker compose up -d --force-recreate
```

A full rebuild of both `backend` and `frontend` is required. Rebuilding only the
backend will leave the raw Markdown display unchanged.

## Retest

Ask:

```text
FDS operating procedures
```

Expected behavior:

- only FDS-related operating steps appear;
- door isolation and generic degraded-mode procedures are excluded;
- procedures render as an actual numbered list;
- headings and bold text render visually instead of showing Markdown symbols;
- tables are used only for genuine comparisons;
- citations render as small source badges.
