from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"(?u)\b[\w][\w./:#-]*\b")


class EmbeddingUnavailableError(RuntimeError):
    """Raised when no configured embedding backend can be loaded."""


class _HashingEmbeddingModel:
    """Deterministic, dependency-free embedding fallback.

    This is not a replacement for a transformer embedding model. It provides
    normalized 384-dimensional lexical vectors so PostgreSQL/pgvector ingestion
    and retrieval remain usable when Hugging Face is blocked. PostgreSQL full-text
    search continues to complement these vectors in the existing hybrid search.
    """

    def __init__(self, dimensions: int) -> None:
        if dimensions < 64:
            raise ValueError("Hashing embeddings require at least 64 dimensions")
        self.dimensions = dimensions

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimensions

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype="float32")
        for row, text in enumerate(texts):
            tokens = [token.casefold() for token in _TOKEN_PATTERN.findall(text)]
            if not tokens:
                tokens = ["__empty__"]

            for token in tokens:
                self._add_feature(matrix[row], f"w:{token}", 1.0)
                padded = f"^{token}$"
                for size, weight in ((3, 0.35), (4, 0.2)):
                    for index in range(max(0, len(padded) - size + 1)):
                        self._add_feature(
                            matrix[row],
                            f"c{size}:{padded[index:index + size]}",
                            weight,
                        )

            for left, right in zip(tokens, tokens[1:]):
                self._add_feature(matrix[row], f"b:{left}|{right}", 0.7)

            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm

        return matrix

    def _add_feature(self, vector: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
        first = int.from_bytes(digest[:8], "little") % self.dimensions
        second = int.from_bytes(digest[8:], "little") % self.dimensions
        first_sign = 1.0 if digest[0] & 1 else -1.0
        second_sign = 1.0 if digest[8] & 1 else -1.0
        vector[first] += weight * first_sign
        vector[second] += weight * 0.5 * second_sign


class EmbeddingService:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._backend: str | None = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> str | None:
        return self._backend

    @property
    def using_fallback(self) -> bool:
        return self._backend == "local-hashing"

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @staticmethod
    def _configure_insecure_hf_client() -> None:
        settings = get_settings()
        if not settings.allow_insecure_hf_download:
            return

        import httpx
        import huggingface_hub

        setter = getattr(huggingface_hub, "set_client_factory", None)
        if setter is None:
            raise EmbeddingUnavailableError(
                "ALLOW_INSECURE_HF_DOWNLOAD is enabled, but this huggingface_hub "
                "version cannot configure its HTTP client."
            )

        logger.warning(
            "TLS verification is disabled for Hugging Face downloads. "
            "This is intended only for temporary local development."
        )
        setter(
            lambda: httpx.Client(
                verify=False,
                follow_redirects=True,
                timeout=httpx.Timeout(120.0),
            )
        )

    @staticmethod
    def _validate_dimensions(model: Any) -> None:
        expected = get_settings().embedding_dimensions
        getter = getattr(model, "get_sentence_embedding_dimension", None)
        actual = getter() if callable(getter) else None
        if actual is not None and int(actual) != expected:
            raise EmbeddingUnavailableError(
                f"Embedding dimension mismatch: model returns {actual}, but "
                f"EMBEDDING_DIMENSIONS is {expected}."
            )

    def _fallback_or_raise(self, errors: list[str]) -> tuple[Any, str, str | None]:
        settings = get_settings()
        detail = "; ".join(errors) or "configured model is unavailable"
        if settings.embedding_fallback_mode == "hashing":
            logger.warning(
                "Transformer embedding model is unavailable; using the built-in "
                "local hashing fallback (%s dimensions). Cause: %s",
                settings.embedding_dimensions,
                detail,
            )
            return (
                _HashingEmbeddingModel(settings.embedding_dimensions),
                "local-hashing",
                detail,
            )
        raise EmbeddingUnavailableError(detail)

    def _load_model(self) -> tuple[Any, str, str | None]:
        settings = get_settings()
        model_name = settings.embedding_model
        errors: list[str] = []

        # Prefer a mounted path or cache. This never makes a network request.
        try:
            model = SentenceTransformer(model_name, local_files_only=True)
            self._validate_dimensions(model)
            return model, "sentence-transformer-local", None
        except TypeError:
            errors.append(
                "installed sentence-transformers does not support local_files_only"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"local load: {type(exc).__name__}: {exc}")

        if settings.embedding_local_files_only or not settings.embedding_download_enabled:
            return self._fallback_or_raise(errors)

        try:
            self._configure_insecure_hf_client()
            model = SentenceTransformer(model_name)
            self._validate_dimensions(model)
            return model, "sentence-transformer-huggingface", None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Hugging Face load: {type(exc).__name__}: {exc}")
            return self._fallback_or_raise(errors)

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        model, backend, warning = self._load_model()
                        self._model = model
                        self._backend = backend
                        self._last_error = warning
                    except EmbeddingUnavailableError as exc:
                        self._last_error = str(exc)
                        raise
        return self._model

    def warmup(self) -> bool:
        try:
            self.encode(["warmup"])
        except EmbeddingUnavailableError:
            logger.exception("Embedding model warmup failed")
            return False
        return True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one text is required for embedding")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        matrix = np.asarray(embeddings, dtype="float32")
        expected = get_settings().embedding_dimensions
        if matrix.ndim != 2 or matrix.shape != (len(texts), expected):
            raise EmbeddingUnavailableError(
                f"Embedding backend returned shape {matrix.shape}; expected "
                f"({len(texts)}, {expected})."
            )

        # Normalize all backends before pgvector cosine-distance search.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        return np.ascontiguousarray(matrix, dtype="float32")


embedding_service = EmbeddingService()
