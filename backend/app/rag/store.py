from __future__ import annotations

import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher, get_close_matches

import numpy as np

try:
    import faiss
except ImportError:  # Small deployments can use the NumPy fallback.
    faiss = None  # type: ignore[assignment]

from app.config import get_settings
from app.models import FileSummary
from app.rag.types import QueryPlan, RetrievedChunk, TextChunk


class CollectionNotFoundError(KeyError):
    pass


_TOKEN_PATTERN = re.compile(r"(?u)\b[\w][\w./:#-]*\b")
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
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    ]


@dataclass(slots=True)
class Collection:
    collection_id: str
    chunks: list[TextChunk]
    vectors: np.ndarray
    index: object
    files: list[FileSummary]
    total_pages: int
    warnings: list[str]
    chunk_token_counts: list[Counter[str]]
    document_frequencies: Counter[str]
    average_document_length: float
    vocabulary_buckets: dict[tuple[str, int], tuple[str, ...]]
    chunk_positions: dict[str, int]
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)

    def search_many(self, query_vectors: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        self.last_accessed_at = time.time()
        limit = min(max(top_k, 1), len(self.chunks))
        scores, indexes = self.index.search(  # type: ignore[attr-defined]
            np.ascontiguousarray(query_vectors, dtype="float32"), limit
        )
        merged: dict[str, tuple[TextChunk, float, int]] = {}
        for query_scores, query_indexes in zip(scores, indexes, strict=True):
            for rank, (score, index) in enumerate(
                zip(query_scores, query_indexes, strict=True), start=1
            ):
                if index < 0:
                    continue
                chunk = self.chunks[int(index)]
                current = merged.get(chunk.chunk_id)
                candidate = (chunk, float(score), rank)
                if current is None or score > current[1] or (score == current[1] and rank < current[2]):
                    merged[chunk.chunk_id] = candidate

        results = [
            RetrievedChunk(
                chunk=chunk,
                score=max(0.0, min(1.0, score)),
                method="vector",
                vector_score=max(0.0, min(1.0, score)),
            )
            for chunk, score, _ in merged.values()
        ]
        results.sort(key=lambda item: item.vector_score, reverse=True)
        return results[:limit]

    def keyword_search(self, plan: QueryPlan, top_k: int) -> list[RetrievedChunk]:
        """BM25-style exact/fuzzy retrieval for names, IDs, dates and misspellings."""
        self.last_accessed_at = time.time()
        settings = get_settings()
        term_weights: Counter[str] = Counter()
        original_terms: set[str] = set()

        for query_index, query in enumerate(plan.search_queries):
            query_weight = 1.0 / (1.0 + query_index * 0.2)
            for token in _tokens(query)[: settings.max_query_terms]:
                original_terms.add(token)
                term_weights[token] += query_weight

        for keyword in plan.keywords:
            for token in _tokens(keyword):
                original_terms.add(token)
                term_weights[token] += 1.4

        if not term_weights:
            return []

        expanded_weights = Counter(term_weights)
        if settings.fuzzy_keyword_enabled:
            for token, weight in list(term_weights.items()):
                if token in self.document_frequencies or len(token) < 4:
                    continue
                for match, ratio in self._fuzzy_matches(token, settings.fuzzy_match_cutoff):
                    expanded_weights[match] = max(expanded_weights[match], weight * ratio * 0.8)

        total_query_weight = max(sum(term_weights.values()), 1.0)
        document_count = len(self.chunks)
        k1 = 1.5
        b = 0.75
        scored: list[tuple[int, float, float]] = []

        lowered_queries = [query.casefold() for query in plan.search_queries if query.strip()]
        for index, counts in enumerate(self.chunk_token_counts):
            document_length = max(sum(counts.values()), 1)
            bm25 = 0.0
            matched_original_weight = 0.0
            for term, query_weight in expanded_weights.items():
                frequency = counts.get(term, 0)
                if frequency <= 0:
                    continue
                document_frequency = self.document_frequencies.get(term, 0)
                inverse_document_frequency = np.log(
                    1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0 - b + b * document_length / max(self.average_document_length, 1.0)
                )
                bm25 += query_weight * inverse_document_frequency * (
                    frequency * (k1 + 1.0) / denominator
                )
                if term in original_terms:
                    matched_original_weight += term_weights.get(term, 0.0)

            chunk_lower = self.chunks[index].text.casefold()
            phrase_bonus = 0.0
            for query in lowered_queries:
                if len(query) >= 4 and query in chunk_lower:
                    phrase_bonus = max(phrase_bonus, 1.0)
            if bm25 <= 0 and phrase_bonus <= 0:
                continue

            coverage = min(1.0, matched_original_weight / total_query_weight)
            scored.append((index, bm25 + phrase_bonus, coverage))

        if not scored:
            return []

        max_raw = max(score for _, score, _ in scored) or 1.0
        results: list[RetrievedChunk] = []
        for index, raw_score, coverage in scored:
            normalized_rank_score = raw_score / max_raw
            quality = min(1.0, normalized_rank_score * 0.65 + coverage * 0.35)
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[index],
                    score=quality,
                    method="keyword",
                    keyword_score=quality,
                )
            )
        results.sort(key=lambda item: item.keyword_score, reverse=True)
        return results[: min(max(top_k, 1), len(results))]

    def hybrid_search(
        self,
        plan: QueryPlan,
        query_vectors: np.ndarray,
        top_k: int,
        vector_k: int | None = None,
        keyword_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse semantic and typo-tolerant keyword retrieval with calibrated scores."""
        limit = min(max(top_k, 1), len(self.chunks))
        vector_limit = min(max(vector_k or top_k * 4, limit), len(self.chunks))
        keyword_limit = min(max(keyword_k or top_k * 4, limit), len(self.chunks))

        vector_results = self.search_many(query_vectors, vector_limit)
        keyword_results = self.keyword_search(plan, keyword_limit)
        vector_rank = {item.chunk.chunk_id: rank for rank, item in enumerate(vector_results, start=1)}
        keyword_rank = {item.chunk.chunk_id: rank for rank, item in enumerate(keyword_results, start=1)}

        merged: dict[str, dict[str, object]] = {}
        for item in vector_results:
            merged[item.chunk.chunk_id] = {
                "chunk": item.chunk,
                "vector": item.vector_score,
                "keyword": 0.0,
                "methods": {"vector"},
            }
        for item in keyword_results:
            current = merged.setdefault(
                item.chunk.chunk_id,
                {"chunk": item.chunk, "vector": 0.0, "keyword": 0.0, "methods": set()},
            )
            current["keyword"] = max(float(current["keyword"]), item.keyword_score)
            methods = current["methods"]
            assert isinstance(methods, set)
            methods.add("keyword")

        rrf_constant = 60.0
        combined: list[RetrievedChunk] = []
        for chunk_id, values in merged.items():
            chunk = values["chunk"]
            assert isinstance(chunk, TextChunk)
            vector_score = float(values["vector"])
            keyword_score = float(values["keyword"])
            methods = values["methods"]
            assert isinstance(methods, set)

            rank_terms: list[float] = []
            if chunk_id in vector_rank:
                rank_terms.append(1.0 / (rrf_constant + vector_rank[chunk_id]))
            if chunk_id in keyword_rank:
                rank_terms.append(1.0 / (rrf_constant + keyword_rank[chunk_id]))
            max_rrf = len(rank_terms) / (rrf_constant + 1.0) if rank_terms else 1.0
            rank_score = min(1.0, sum(rank_terms) / max_rrf) if rank_terms else 0.0

            if vector_score and keyword_score:
                score = vector_score * 0.55 + keyword_score * 0.40 + rank_score * 0.05
            elif vector_score:
                score = vector_score * 0.95 + rank_score * 0.05
            else:
                score = keyword_score * 0.95 + rank_score * 0.05

            combined.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=min(1.0, max(0.0, score)),
                    method="+".join(sorted(methods)),
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                )
            )

        combined.sort(key=lambda item: item.score, reverse=True)
        return self._diversify(combined, limit)

    def _fuzzy_matches(self, token: str, cutoff: float) -> list[tuple[str, float]]:
        candidates: list[str] = []
        for length in range(max(2, len(token) - 2), len(token) + 3):
            candidates.extend(self.vocabulary_buckets.get((token[0], length), ()))
        matches = get_close_matches(token, candidates, n=2, cutoff=cutoff)
        return [(match, _similarity_ratio(token, match)) for match in matches]

    @staticmethod
    def _diversify(results: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
        settings = get_settings()
        selected: list[RetrievedChunk] = []
        page_counts: Counter[tuple[str, int]] = Counter()
        selected_tokens: list[set[str]] = []

        for result in results:
            page_key = (result.chunk.filename, result.chunk.page_number)
            if page_counts[page_key] >= settings.max_chunks_per_page:
                continue
            token_set = set(_tokens(result.chunk.text))
            if token_set and any(_jaccard(token_set, existing) >= 0.88 for existing in selected_tokens):
                continue
            selected.append(result)
            selected_tokens.append(token_set)
            page_counts[page_key] += 1
            if len(selected) >= limit:
                break
        return selected

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
        warnings: list[str] | None = None,
    ) -> Collection:
        if not chunks or vectors.shape[0] != len(chunks):
            raise ValueError("Chunks and vectors must be non-empty and aligned")

        contiguous_vectors = np.ascontiguousarray(vectors, dtype="float32")
        if faiss is not None:
            index = faiss.IndexFlatIP(int(contiguous_vectors.shape[1]))
        else:
            index = _NumpyFlatIPIndex(int(contiguous_vectors.shape[1]))
        index.add(contiguous_vectors)

        token_counts = [Counter(_tokens(chunk.text)) for chunk in chunks]
        document_frequencies: Counter[str] = Counter()
        vocabulary_buckets_mutable: dict[tuple[str, int], set[str]] = defaultdict(set)
        for counts in token_counts:
            document_frequencies.update(counts.keys())
            for token in counts:
                vocabulary_buckets_mutable[(token[0], len(token))].add(token)
        average_length = sum(sum(counts.values()) for counts in token_counts) / max(len(chunks), 1)
        vocabulary_buckets = {
            key: tuple(sorted(values)) for key, values in vocabulary_buckets_mutable.items()
        }

        collection = Collection(
            collection_id=uuid.uuid4().hex,
            chunks=chunks,
            vectors=contiguous_vectors,
            index=index,
            files=files,
            total_pages=total_pages,
            warnings=warnings or [],
            chunk_token_counts=token_counts,
            document_frequencies=document_frequencies,
            average_document_length=average_length,
            vocabulary_buckets=vocabulary_buckets,
            chunk_positions={chunk.chunk_id: index for index, chunk in enumerate(chunks)},
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


class _NumpyFlatIPIndex:
    """Minimal FAISS-compatible exact inner-product index for development."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self._vectors = np.empty((0, dimensions), dtype="float32")

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimensions:
            raise ValueError("Vector dimensions do not match the index")
        self._vectors = np.ascontiguousarray(vectors, dtype="float32")

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if not len(self._vectors):
            return (
                np.full((len(queries), top_k), -np.inf, dtype="float32"),
                np.full((len(queries), top_k), -1, dtype="int64"),
            )
        similarities = np.asarray(queries, dtype="float32") @ self._vectors.T
        limit = min(top_k, self._vectors.shape[0])
        indexes = np.argsort(-similarities, axis=1)[:, :limit]
        scores = np.take_along_axis(similarities, indexes, axis=1)
        if limit < top_k:
            pad = top_k - limit
            indexes = np.pad(indexes, ((0, 0), (0, pad)), constant_values=-1)
            scores = np.pad(scores, ((0, 0), (0, pad)), constant_values=-np.inf)
        return scores.astype("float32"), indexes.astype("int64")


def _similarity_ratio(left: str, right: str) -> float:
    return SequenceMatcher(a=left, b=right).ratio()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


collection_store = CollectionStore()
