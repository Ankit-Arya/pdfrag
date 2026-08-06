from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.rag.normalization import canonical_phrase, search_terms
from app.rag.structure import section_match_score, section_path_from_text
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
    settings = get_settings()
    candidate_limit = max(limit * 2, 40)
    per_document_limit = settings.retrieval_chunks_per_document
    sql = text(
        """
        WITH q AS (
          SELECT CAST(:embedding AS vector) AS embedding,
                 plainto_tsquery('simple', :query) AS tsq,
                 to_tsquery('simple', :any_query) AS tsq_any,
                 CAST(:normalized_section_query AS text) AS normalized_section_query
        ),
        ready_documents AS (
          SELECT id, filename
          FROM documents
          WHERE status = 'ready'
        ),
        global_vector_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM document_chunks c
          JOIN ready_documents d ON d.id = c.document_id
          CROSS JOIN q
          ORDER BY c.embedding <=> q.embedding
          LIMIT :candidate_limit
        ),
        global_keyword_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM document_chunks c
          JOIN ready_documents d ON d.id = c.document_id
          CROSS JOIN q
          WHERE to_tsvector('simple', c.text) @@ q.tsq_any
          ORDER BY ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) DESC
          LIMIT :candidate_limit
        ),
        per_document_vector_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM ready_documents d
          CROSS JOIN q
          CROSS JOIN LATERAL (
            SELECT candidate.*
            FROM document_chunks candidate
            WHERE candidate.document_id = d.id
            ORDER BY candidate.embedding <=> q.embedding
            LIMIT :per_document_limit
          ) c
        ),
        per_document_keyword_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM ready_documents d
          CROSS JOIN q
          CROSS JOIN LATERAL (
            SELECT candidate.*
            FROM document_chunks candidate
            WHERE candidate.document_id = d.id
              AND to_tsvector('simple', candidate.text) @@ q.tsq_any
            ORDER BY ts_rank_cd(to_tsvector('simple', candidate.text), q.tsq_any) DESC
            LIMIT :per_document_limit
          ) c
        ),
        section_path_candidates AS (
          SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                 c.text, d.filename,
                 GREATEST(0.0, 1 - (c.embedding <=> q.embedding)) AS vector_score,
                 GREATEST(
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq),
                   ts_rank_cd(to_tsvector('simple', c.text), q.tsq_any) * 0.65
                 ) AS keyword_score
          FROM document_chunks c
          JOIN ready_documents d ON d.id = c.document_id
          CROSS JOIN q
          WHERE length(q.normalized_section_query) >= 3
            AND btrim(regexp_replace(
                  lower(split_part(split_part(c.text, 'Section path:', 2), E'\n', 1)),
                  '[^a-z0-9]+', ' ', 'g'
                )) LIKE '%' || q.normalized_section_query || '%'
          ORDER BY c.document_id, c.chunk_index
          LIMIT :candidate_limit
        ),
        candidates AS (
          SELECT * FROM global_vector_candidates
          UNION
          SELECT * FROM global_keyword_candidates
          UNION
          SELECT * FROM per_document_vector_candidates
          UNION
          SELECT * FROM per_document_keyword_candidates
          UNION
          SELECT * FROM section_path_candidates
        )
        SELECT * FROM candidates
        """
    )
    rows = db.execute(
        sql,
        {
            "embedding": str(query_vector),
            "query": query_text,
            "any_query": _keyword_or_query(query_text),
            "normalized_section_query": _normalize_phrase(query_text),
            "candidate_limit": candidate_limit,
            "per_document_limit": per_document_limit,
        },
    ).mappings()

    query_terms = _terms(query_text)
    results: list[RetrievedChunk] = []
    for row in rows:
        vector_score = float(row["vector_score"] or 0.0)
        keyword_score = min(float(row["keyword_score"] or 0.0) * 4.0, 1.0)
        lexical_score = _lexical_overlap(query_terms, row["text"])
        phrase_score = _phrase_match(query_text, row["text"])
        section_score = _section_phrase_match(query_text, row["text"])
        blended_score = min(
            1.0,
            max(
                0.0,
                vector_score * 0.36
                + keyword_score * 0.24
                + lexical_score * 0.18
                + phrase_score * 0.10
                + section_score * 0.12,
            ),
        )
        score = max(blended_score, 0.78 + section_score * 0.18) if section_score else blended_score
        method = _method(
            vector_score,
            keyword_score,
            lexical_score,
            phrase_score,
            section_score,
        )
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
    return _document_diverse_results(results, settings.max_retrieval_candidates)


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
    values_sql = ",".join(f"(:doc_{index}, :idx_{index})" for index, _ in enumerate(pairs))
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
        GROUP BY c.id, c.document_id, c.chunk_index, c.page_number,
                 c.content_type, c.text, d.filename
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


def _method(
    vector_score: float,
    keyword_score: float,
    lexical_score: float,
    phrase_score: float,
    section_score: float,
) -> str:
    if section_score >= 0.88:
        return "section-heading"
    if phrase_score:
        return "pgvector+exact-phrase"
    if keyword_score >= 0.08 and keyword_score >= vector_score * 0.5:
        return "pgvector+fts"
    if lexical_score >= 0.35:
        return "pgvector+lexical"
    return "pgvector"


def _terms(value: str) -> set[str]:
    return {
        term
        for term in search_terms(value, keep_single=True)
        if term not in _STOPWORDS and (len(term) > 1 or term.isdigit())
    }


def _lexical_overlap(query_terms: set[str], text_value: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = _terms(text_value[:5000])
    if not text_terms:
        return 0.0
    return min(1.0, len(query_terms & text_terms) / max(len(query_terms), 1))


def _phrase_match(query: str, text_value: str) -> float:
    normalized_query = _normalize_phrase(query)
    if len(normalized_query) < 4:
        return 0.0
    return 1.0 if normalized_query in _normalize_phrase(text_value[:10000]) else 0.0


def _section_phrase_match(query: str, text_value: str) -> float:
    return section_match_score(query, section_path_from_text(text_value))


def _normalize_phrase(value: str) -> str:
    return canonical_phrase(value)


def _keyword_or_query(value: str) -> str:
    terms = sorted(term for term in _terms(value) if re.fullmatch(r"[a-z0-9]+", term))
    return " | ".join(f"{term}:*" for term in terms) or "pdfrag_no_match"


def _document_diverse_results(
    results: list[RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    """Keep each probed document represented before adding extra chunks."""
    if limit <= 0:
        return []

    deduplicated: list[RetrievedChunk] = []
    seen_chunks: set[str] = set()
    for item in results:
        if item.chunk.chunk_id in seen_chunks:
            continue
        seen_chunks.add(item.chunk.chunk_id)
        deduplicated.append(item)

    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    seen_documents: set[str] = set()
    for item in deduplicated:
        document_key = item.chunk.document_id or item.chunk.filename.casefold()
        if document_key in seen_documents:
            continue
        selected.append(item)
        selected_ids.add(item.chunk.chunk_id)
        seen_documents.add(document_key)
        if len(selected) >= limit:
            return selected

    for item in deduplicated:
        if item.chunk.chunk_id in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


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
