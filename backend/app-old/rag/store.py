import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

import faiss
import numpy as np

from app.config import get_settings
from app.models import FileSummary
from app.rag.types import RetrievedChunk, TextChunk


class CollectionNotFoundError(KeyError):
    pass


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/#-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "pdf",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]


@dataclass(slots=True)
class Collection:
    collection_id: str
    chunks: list[TextChunk]
    index: faiss.Index
    files: list[FileSummary]
    total_pages: int
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        self.last_accessed_at = time.time()
        limit = min(max(top_k, 1), len(self.chunks))
        scores, indexes = self.index.search(query_vector.reshape(1, -1), limit)
        results: list[RetrievedChunk] = []
        for score, index in zip(scores[0], indexes[0], strict=True):
            if index < 0:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[int(index)],
                    score=float(score),
                    method="vector",
                )
            )
        return results

    def keyword_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Small dependency-free keyword retriever for exact terms, names, dates, IDs.

        Dense embeddings can miss exact strings common in PDFs. This scorer is not
        intended to replace vector search; it complements it and helps with clauses,
        section numbers, acronyms, invoice IDs, and named entities.
        """
        self.last_accessed_at = time.time()
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        query_set = set(query_tokens)
        query_lower = query.lower().strip()
        scored: list[RetrievedChunk] = []

        for chunk in self.chunks:
            chunk_lower = chunk.text.lower()
            chunk_tokens = _tokens(chunk.text)
            if not chunk_tokens:
                continue

            chunk_counts = Counter(chunk_tokens)
            overlap = query_set.intersection(chunk_counts)
            if not overlap and query_lower not in chunk_lower:
                continue

            matched_weight = sum(min(query_counts[token], chunk_counts[token]) for token in overlap)
            coverage = matched_weight / max(sum(query_counts.values()), 1)
            density = matched_weight / max(len(chunk_tokens), 1)
            phrase_bonus = 0.35 if query_lower and query_lower in chunk_lower else 0.0
            score = min(1.0, coverage + min(density * 8, 0.25) + phrase_bonus)

            scored.append(RetrievedChunk(chunk=chunk, score=score, method="keyword"))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: min(max(top_k, 1), len(scored))]

    def hybrid_search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int,
        vector_k: int | None = None,
        keyword_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Merge vector and keyword results, de-duplicated by chunk id."""
        limit = min(max(top_k, 1), len(self.chunks))
        vector_limit = min(max(vector_k or top_k * 3, limit), len(self.chunks))
        keyword_limit = min(max(keyword_k or top_k * 3, limit), len(self.chunks))

        vector_results = self.search(query_vector, vector_limit)
        keyword_results = self.keyword_search(query, keyword_limit)

        merged: dict[str, tuple[TextChunk, float, float, set[str]]] = {}

        for result in vector_results:
            existing = merged.get(result.chunk.chunk_id)
            vector_score = max(float(result.score), 0.0)
            if existing is None:
                merged[result.chunk.chunk_id] = (result.chunk, vector_score, 0.0, {"vector"})
            else:
                chunk, old_vector, keyword_score, methods = existing
                methods.add("vector")
                merged[result.chunk.chunk_id] = (
                    chunk,
                    max(old_vector, vector_score),
                    keyword_score,
                    methods,
                )

        for result in keyword_results:
            existing = merged.get(result.chunk.chunk_id)
            keyword_score = max(float(result.score), 0.0)
            if existing is None:
                merged[result.chunk.chunk_id] = (result.chunk, 0.0, keyword_score, {"keyword"})
            else:
                chunk, vector_score, old_keyword, methods = existing
                methods.add("keyword")
                merged[result.chunk.chunk_id] = (
                    chunk,
                    vector_score,
                    max(old_keyword, keyword_score),
                    methods,
                )

        combined: list[RetrievedChunk] = []
        for chunk, vector_score, keyword_score, methods in merged.values():
            # Keep scores comparable with the existing MIN_SIMILARITY setting while
            # still giving exact keyword matches enough influence to surface.
            score = max(vector_score, keyword_score, (vector_score * 0.75) + (keyword_score * 0.35))
            method = "+".join(sorted(methods))
            combined.append(RetrievedChunk(chunk=chunk, score=score, method=method))

        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[:limit]

    @property
    def created_at_iso(self) -> str:
        return datetime.fromtimestamp(self.created_at, tz=UTC).isoformat()

    @property
    def last_accessed_at_iso(self) -> str:
        return datetime.fromtimestamp(self.last_accessed_at, tz=UTC).isoformat()


class CollectionStore:
    def __init__(self) -> None:
        self._collections: dict[str, Collection] = {}
        self._lock = threading.RLock()

    def create(
        self,
        chunks: list[TextChunk],
        vectors: np.ndarray,
        files: list[FileSummary],
        total_pages: int,
    ) -> Collection:
        if not chunks or vectors.shape[0] != len(chunks):
            raise ValueError("Chunks and vectors must be non-empty and aligned")
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        collection = Collection(
            collection_id=uuid.uuid4().hex,
            chunks=chunks,
            index=index,
            files=files,
            total_pages=total_pages,
        )
        with self._lock:
            self._cleanup_locked()
            self._evict_if_needed_locked()
            self._collections[collection.collection_id] = collection
        return collection

    def get(self, collection_id: str) -> Collection:
        with self._lock:
            self._cleanup_locked()
            collection = self._collections.get(collection_id)
            if collection is None:
                raise CollectionNotFoundError(collection_id)
            collection.last_accessed_at = time.time()
            return collection

    def delete(self, collection_id: str) -> bool:
        with self._lock:
            return self._collections.pop(collection_id, None) is not None

    def _cleanup_locked(self) -> None:
        ttl_seconds = get_settings().collection_ttl_minutes * 60
        cutoff = time.time() - ttl_seconds
        expired = [
            key
            for key, collection in self._collections.items()
            if collection.last_accessed_at < cutoff
        ]
        for key in expired:
            del self._collections[key]

    def _evict_if_needed_locked(self) -> None:
        max_collections = get_settings().max_collections
        while len(self._collections) >= max_collections:
            oldest_id = min(
                self._collections,
                key=lambda key: self._collections[key].last_accessed_at,
            )
            del self._collections[oldest_id]


collection_store = CollectionStore()
