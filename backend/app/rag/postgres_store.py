from __future__ import annotations

import re
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.rag.types import RetrievedChunk, TextChunk

_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_/-]{1,}")


def search_chunks(
    db: Session,
    query_vector: list[float],
    query_text: str,
    limit: int,
) -> list[RetrievedChunk]:
    """Hybrid pgvector + full-text retrieval.

    This intentionally retrieves candidates from both vector similarity and full
    text search. Metro procedure questions often contain exact train/procedure
    identifiers; the exact identifier must be able to pull a chunk even when the
    semantic vector score is not high.
    """
    candidate_limit = max(limit * 4, 40)
    sql = text(
        """
        WITH q AS (
          SELECT CAST(:embedding AS vector) AS embedding,
                 plainto_tsquery('simple', :query) AS tsq
        ),
        vector_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 ts_rank_cd(to_tsvector('simple', c.text), q.tsq) AS keyword_score
          FROM document_chunks c
          JOIN documents d ON d.id = c.document_id
          CROSS JOIN q
          WHERE d.status = 'ready'
          ORDER BY c.embedding <=> q.embedding
          LIMIT :candidate_limit
        ),
        keyword_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 ts_rank_cd(to_tsvector('simple', c.text), q.tsq) AS keyword_score
          FROM document_chunks c
          JOIN documents d ON d.id = c.document_id
          CROSS JOIN q
          WHERE d.status = 'ready'
            AND to_tsvector('simple', c.text) @@ q.tsq
          ORDER BY ts_rank_cd(to_tsvector('simple', c.text), q.tsq) DESC
          LIMIT :candidate_limit
        ),
        candidates AS (
          SELECT * FROM vector_candidates
          UNION
          SELECT * FROM keyword_candidates
        )
        SELECT * FROM candidates
        """
    )
    rows = db.execute(
        sql,
        {
            "embedding": str(query_vector),
            "query": query_text,
            "candidate_limit": candidate_limit,
        },
    ).mappings()

    query_terms = _terms(query_text)
    results: list[RetrievedChunk] = []
    for row in rows:
        vector_score = float(row["vector_score"] or 0.0)
        keyword_score = min(float(row["keyword_score"] or 0.0) * 4.0, 1.0)
        lexical_score = _lexical_overlap(query_terms, row["text"])
        score = min(
            1.0,
            max(0.0, vector_score * 0.50 + keyword_score * 0.35 + lexical_score * 0.15),
        )
        method = _method(vector_score, keyword_score, lexical_score)
        results.append(
            RetrievedChunk(
                chunk=TextChunk(
                    chunk_id=str(row["id"]),
                    filename=str(row["filename"]),
                    page_number=int(row["page_number"]),
                    text=str(row["text"]),
                    content_type=str(row["content_type"]),
                    document_id=str(row["document_id"]),
                    chunk_index=int(row["chunk_index"]),
                ),
                score=score,
                method=method,
                vector_score=vector_score,
                keyword_score=keyword_score,
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def fetch_neighbor_chunks(
    db: Session,
    seeds: list[RetrievedChunk],
    window: int = 1,
) -> list[RetrievedChunk]:
    """Fetch adjacent chunks around high-confidence hits.

    Heading-aware chunks reduce boundary problems, but some procedures still span
    multiple chunks. Neighbor expansion gives the LLM prerequisites, warnings,
    branches and verification steps that may be adjacent to the directly matched
    chunk.
    """
    if window <= 0 or not seeds:
        return []

    pairs = [
        (seed.chunk.document_id, seed.chunk.chunk_index)
        for seed in seeds
        if seed.chunk.document_id and seed.chunk.chunk_index is not None
    ]
    if not pairs:
        return []

    # Build a compact VALUES list because SQLAlchemy text() cannot bind tuple
    # arrays portably for all psycopg/pgvector combinations.
    values_sql = ",".join(
        f"(:doc_{index}, :idx_{index})" for index, _ in enumerate(pairs)
    )
    params: dict[str, object] = {"window": window}
    for index, (document_id, chunk_index) in enumerate(pairs):
        params[f"doc_{index}"] = document_id
        params[f"idx_{index}"] = chunk_index

    sql = text(
        f"""
        WITH seeds(document_id, chunk_index) AS (VALUES {values_sql})
        SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
               c.text, d.filename,
               MIN(ABS(c.chunk_index - seeds.chunk_index)) AS distance
        FROM seeds
        JOIN document_chunks c
          ON c.document_id::text = seeds.document_id
         AND c.chunk_index BETWEEN seeds.chunk_index - :window AND seeds.chunk_index + :window
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'ready'
        GROUP BY c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename
        ORDER BY c.document_id, c.chunk_index
        """
    )
    rows = db.execute(sql, params).mappings()
    neighbors: list[RetrievedChunk] = []
    for row in rows:
        distance = int(row["distance"] or 0)
        score = max(0.01, 0.70 - distance * 0.15)
        neighbors.append(
            RetrievedChunk(
                chunk=TextChunk(
                    chunk_id=str(row["id"]),
                    filename=str(row["filename"]),
                    page_number=int(row["page_number"]),
                    text=str(row["text"]),
                    content_type=str(row["content_type"]),
                    document_id=str(row["document_id"]),
                    chunk_index=int(row["chunk_index"]),
                ),
                score=score,
                method="neighbor-context" if distance else "direct-context",
                vector_score=0.0,
                keyword_score=0.0,
            )
        )
    return neighbors


def _method(vector_score: float, keyword_score: float, lexical_score: float) -> str:
    if keyword_score >= 0.08 and keyword_score >= vector_score * 0.5:
        return "pgvector+fts"
    if lexical_score >= 0.35:
        return "pgvector+lexical"
    return "pgvector"


def _terms(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _TERM_RE.findall(value)
        if len(token) >= 2 and token.casefold() not in _STOPWORDS
    }


def _lexical_overlap(query_terms: set[str], text_value: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = _terms(text_value[:5000])
    if not text_terms:
        return 0.0
    return min(1.0, len(query_terms & text_terms) / max(len(query_terms), 1))


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "when",
    "where",
    "which",
    "how",
    "are",
    "is",
    "in",
    "on",
    "of",
    "to",
    "a",
    "an",
    "or",
    "by",
    "as",
    "be",
    "do",
    "does",
}
