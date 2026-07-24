# Grounded PDF Q&A — OCR, tables and query-rewrite upgrade

This package is a drop-in upgrade of the supplied FastAPI RAG backend.

## Added functionality

- **OCR fallback:** pages with little or no native PDF text are rendered with PyMuPDF and read with Tesseract.
- **Better layout cleanup:** layout-preserving extraction, de-hyphenation, repeated header/footer removal, and stable line breaks.
- **Table-aware extraction:** `pdfplumber` detects bordered and borderless tables and converts them to Markdown so the LLM can reason across rows and columns.
- **Table-aware chunking:** large tables are split only on row boundaries, and their header is repeated in each chunk.
- **AI query interpretation:** a small LLM call corrects likely spelling errors, expands clear abbreviations, and generates semantic search variants without answering or changing intent.
- **Local typo tolerance:** BM25-style keyword retrieval uses fuzzy vocabulary matching even when the AI rewrite call fails.
- **Hybrid retrieval:** semantic and keyword results are fused, deduplicated, and diversified across pages.
- **Transparent responses:** `/api/chat` returns `interpreted_question`, `search_queries`, `grounding_status`, source content type, and retrieval method.
- **Safer grounding:** answers that still fail citation validation after one repair attempt are suppressed instead of returning unvalidated text.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Tesseract is a system dependency:

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# Add language packs as needed, for example:
sudo apt-get install -y tesseract-ocr-hin
```

Set multiple OCR languages with `OCR_LANGUAGES=eng+hin` after installing the corresponding packs.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open API docs at `http://localhost:8000/api/docs`.

## Chat request

```json
{
  "collection_id": "...",
  "question": "wat is the reveneu in qarter 3?",
  "top_k": 8,
  "rewrite_question": true
}
```

The response includes the corrected/interpreted question and the actual search variants. Set `rewrite_question` to `false` per request to disable AI rewriting while retaining fuzzy local retrieval.

## OCR behavior

- `OCR_MODE=auto`: OCR only pages whose native text is below `OCR_MIN_NATIVE_CHARS`.
- `OCR_MODE=always`: OCR every page and select the stronger result.
- `OCR_MODE=never`: native extraction only.

Digital tables are structurally extracted. For scanned tables, OCR makes the text searchable, but highly complex visual tables may still benefit from a specialized document-AI/table-recognition service.
