# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
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

# Install the CPU-only PyTorch wheel separately. Do not upgrade pip here: the
# Python base image already ships a compatible pip, and keeping the installer
# unchanged removes one variable from large-wheel installation failures.
# --no-cache-dir also avoids reusing a potentially bad BuildKit/pip wheel cache.
# --no-compile reduces temporary CPU/memory/disk pressure while unpacking torch.
ARG TORCH_VERSION=2.7.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install \
        --no-cache-dir \
        --no-compile \
        "torch==${TORCH_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
    && python -c "import torch; print('PyTorch OK:', torch.__version__)"

# Keep a cache for the smaller normal application dependencies. sentence-transformers
# sees the already-installed CPU torch wheel and does not need to replace it.
RUN --mount=type=cache,id=pdfrag-pip-app,target=/root/.cache/pip \
    python -m pip install --no-compile -r requirements.txt

RUN tesseract --version \
    && tesseract --list-langs | grep -qx eng \
    && python -c "import fitz, pytesseract; from PIL import Image; print('OCR dependencies OK'); print(pytesseract.get_tesseract_version())"

COPY docker/preload_embedding.py /usr/local/bin/preload_embedding.py

# Model preload is disabled by default. Enable PRELOAD_EMBEDDING_MODEL=1 only
# when the build environment can reach Hugging Face. Runtime otherwise uses a
# mounted/cached transformer model or the built-in local hashing fallback.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG PRELOAD_EMBEDDING_MODEL=0
ARG REQUIRE_EMBEDDING_PRELOAD=0
ARG ALLOW_INSECURE_HF_DOWNLOAD=0
ENV HF_HOME=/opt/huggingface \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    PRELOAD_EMBEDDING_MODEL=${PRELOAD_EMBEDDING_MODEL} \
    REQUIRE_EMBEDDING_PRELOAD=${REQUIRE_EMBEDDING_PRELOAD} \
    ALLOW_INSECURE_HF_DOWNLOAD=${ALLOW_INSECURE_HF_DOWNLOAD}
RUN python /usr/local/bin/preload_embedding.py

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/huggingface

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
