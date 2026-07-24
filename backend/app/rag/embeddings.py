import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    settings = get_settings()
                    self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def warmup(self) -> None:
        self.encode(["warmup"])

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
        return np.asarray(embeddings, dtype="float32")


embedding_service = EmbeddingService()
