FROM python:3.12-slim AS runtime

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG PRELOAD_EMBEDDING_MODEL=0
ARG REQUIRE_EMBEDDING_PRELOAD=0
ARG ALLOW_INSECURE_HF_DOWNLOAD=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    HF_HOME=/opt/huggingface \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    PRELOAD_EMBEDDING_MODEL=${PRELOAD_EMBEDDING_MODEL} \
    REQUIRE_EMBEDDING_PRELOAD=${REQUIRE_EMBEDDING_PRELOAD} \
    ALLOW_INSECURE_HF_DOWNLOAD=${ALLOW_INSECURE_HF_DOWNLOAD} \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        curl \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Put organization/private CA certificates in backend/certs/*.crt before building.
# update-ca-certificates safely ignores README and other non-certificate files.
COPY certs/ /usr/local/share/ca-certificates/pdfrag/
RUN update-ca-certificates \
    && mkdir -p /opt/huggingface \
    && chmod 0755 /opt/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN tesseract --version \
    && tesseract --list-langs | grep -qx eng \
    && python -c "import fitz, pytesseract; from PIL import Image; print('OCR dependencies OK'); print(pytesseract.get_tesseract_version())"

COPY docker/preload_embedding.py /usr/local/bin/preload_embedding.py

# Model preload is disabled by default. Enable PRELOAD_EMBEDDING_MODEL=1 only
# when the build environment can reach Hugging Face. Runtime otherwise uses a
# mounted/cached transformer model or the built-in local hashing fallback.
RUN python /usr/local/bin/preload_embedding.py

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/huggingface

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
