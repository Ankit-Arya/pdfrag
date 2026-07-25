# pdfrag complete replacement package

Copy the included `backend` directory over your existing project-level
`backend` directory.

Included replacements:

- `backend/Dockerfile`
- `backend/requirements.txt`
- `backend/app/rag/query.py`
- `backend/app/rag/prompts.py`
- `backend/app/rag/guardrails.py`
- `backend/app/rag/service.py`
- `backend/tests/test_query.py`
- `backend/tests/test_guardrails.py`

## Changes included

- Installs Tesseract and English OCR language data inside the backend image.
- Installs PyMuPDF, Pillow, pytesseract, and pdfplumber.
- Prevents `FDS` from being rewritten as `Fire Dynamics Simulator`.
- Preserves domain acronyms in the interpreted question and search variants.
- Treats short topic phrases as explain/summarize requests.
- Retries an initial false no-answer response using the retrieved evidence.
- Accepts Markdown headings and bold headings without heading citations.
- Logs exact grounding failure reasons and rejected drafts.
- Returns a generated answer as explicitly unverified when only citation
  validation failed, instead of replacing it with a false insufficient-evidence
  message.

## Replace files in PowerShell

From your project root:

```powershell
Copy-Item .\backend\Dockerfile .\backend\Dockerfile -Force
```

When extracting the ZIP directly into your project root, allow Windows to
replace the existing files.

## Rebuild

```powershell
docker compose down --remove-orphans
docker compose build --no-cache --pull --progress=plain backend
docker compose up -d --force-recreate
```

## Verify OCR

```powershell
docker compose exec backend tesseract --version
docker compose exec backend python -c "from app.rag.pdf import ocr_available; print('OCR available:', ocr_available())"
```

Expected:

```text
OCR available: True
```

## Run tests

The production backend image may not contain pytest. Run tests using your
development environment or install `requirements-dev.txt`.

Example from an activated local virtual environment:

```powershell
cd backend
pytest -q
```

## Retest the question

```text
FDS operating procedure
```

Expected behavior:

- `interpreted_question` should be null or remain `FDS operating procedure`.
- Search queries should not contain `Fire Dynamics Simulator`.
- A properly cited answer returns `grounded=true`.
- A useful answer that still fails citation formatting remains visible with
  `grounded=false` and `grounding_status="citation_validation_failed"`.
