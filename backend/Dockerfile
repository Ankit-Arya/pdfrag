FROM python:3.12-slim AS runtime

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    EMBEDDING_MODEL=${EMBEDDING_MODEL}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Fail the image build immediately if any OCR dependency is unavailable.
RUN tesseract --version \
    && tesseract --list-langs | grep -qx eng \
    && python -c "import fitz, pytesseract; from PIL import Image; print('OCR dependencies OK'); print(pytesseract.get_tesseract_version())"

# Cache the embedding model so runtime does not require network access.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/huggingface

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
