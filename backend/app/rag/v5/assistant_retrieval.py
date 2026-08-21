from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.rag.embeddings import embedding_service
from app.rag.llm import llm_service
from app.rag.smart_understanding import SmartInterpretation
from app.rag.types import RetrievedChunk, TextChunk
from app.rag.v5.retrieval import retrieve_v5
from app.rag.v5.terminology import terminology_hints

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:#-]*")
_UPPER_ALIAS_RE = re.compile(r"\b[A-Z][A-Z0-9/-]{1,12}\b")
_COMMON = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "if", "in", "is", "it", "of", "on", "or", "should", "that",
    "the", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "would", "please", "tell", "give", "explain", "show", "find", "page", "number",
    "pdf", "document", "documents", "rev", "revision", "version", "final", "dmrc", "metro",
}

_RERANK_SYSTEM = """You are the evidence-ranking layer for a CLOSED-BOOK assistant over official PDF documents.
You do NOT answer the user's question. Rank only the supplied candidate evidence.

Ranking principles:
- Prefer the governing/defining rule, heading, section, table, procedure or responsibility statement over an incidental mention.
- Judge semantic meaning, not literal word overlap. Treat spelling mistakes, paraphrases, singular/plural changes and ordinary synonyms as equivalent when the candidate clearly expresses the same concept.
- A candidate that merely mentions the actor/object is weaker than one that directly defines duties, requirements, prohibitions, procedure, limits, meaning, applicability or the requested value.
- Respect explicit scope such as document hint, line, location, equipment, person type, mode and revision, but a document hint is a routing preference rather than proof that the answer must be there.
- For list/procedure/summary questions, retain multiple candidates when they are needed for completeness.
- For page/rule/section/navigation questions, strongly prefer candidates whose heading/section is the direct subject of the request.
- Do not invent internal acronym expansions or factual relationships that are not visible in the candidates or grounded hints.
- Penalize unrelated but lexically similar material.

Return JSON only:
{"ranking":[{"id":"candidate-id","score":0-100,"role":"primary|support|incidental"}]}
Include only candidate IDs that are materially useful, ordered best first.
"""


@dataclass(frozen=True, slots=True)
class DocumentRoute:
    document_id: str
    filename: str
    score: float


@dataclass(slots=True)
class AssistantRetrievalBundle:
    results: list[RetrievedChunk]
    search_queries: list[str]
    candidate_count: int
    routed_documents: list[str]


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 500) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _clean(value).casefold()))


def _terms(value: object) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(_clean(value).casefold()):
        token = re.sub(r"[^a-z0-9]+", "", raw)
        if not token or token in _COMMON:
            continue
        if len(token) > 6 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 5 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        if token and token not in seen:
            seen.add(token)
            output.append(token)
    return output[:48]


def _unique(values: Iterable[str], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _clean(value)
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        output.append(clean)
        if len(output) >= limit:
            break
    return output


def _query_variants(
    interpretation: SmartInterpretation,
    extra_queries: Iterable[str] = (),
) -> list[str]:
    values: list[str] = [
        interpretation.resolved_question,
        *interpretation.search_queries,
    ]
    values.extend(
        f"{interpretation.resolved_question} {need}"
        for need in interpretation.evidence_needs
    )
    values.extend(extra_queries)
    return _unique(values, _int_env("RAG_V51_MAX_QUERY_VARIANTS", 10, 3, 16))


def _row_to_result(
    row: object,
    *,
    score: float,
    method: str,
    vector: float = 0.0,
    keyword: float = 0.0,
) -> RetrievedChunk:
    section_path = row["section_path"] or []  # type: ignore[index]
    if not isinstance(section_path, list):
        section_path = []
    page_number = int(row["page_number"])  # type: ignore[index]
    page_end = int(row["page_end"] or page_number)  # type: ignore[index]
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=str(row["id"]),  # type: ignore[index]
            filename=str(row["filename"]),  # type: ignore[index]
            page_number=page_number,
            page_end=page_end,
            text=str(row["text"]),  # type: ignore[index]
            content_type=str(row["content_type"]),  # type: ignore[index]
            section_path=tuple(str(item) for item in section_path),
            heading=str(row["heading"] or ""),  # type: ignore[index]
            document_id=str(row["document_id"]),  # type: ignore[index]
            chunk_index=int(row["chunk_index"]),  # type: ignore[index]
        ),
        score=max(0.0, min(1.0, float(score))),
        method=method,
        vector_score=max(0.0, min(1.0, float(vector))),
        keyword_score=max(0.0, min(1.0, float(keyword))),
    )


def _copy_result(item: RetrievedChunk, *, score: float, method: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=item.chunk,
        score=max(0.0, min(1.0, score)),
        method=method,
        vector_score=item.vector_score,
        keyword_score=item.keyword_score,
    )


def _merge_results(items: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    methods: dict[str, list[str]] = defaultdict(list)
    for item in items:
        chunk_id = item.chunk.chunk_id
        if item.method and item.method not in methods[chunk_id]:
            methods[chunk_id].append(item.method)
        current = merged.get(chunk_id)
        if current is None:
            merged[chunk_id] = item
            continue
        best = max(float(current.score), float(item.score))
        # Independent retrieval paths agreeing on the same chunk are a useful signal,
        # but keep the bonus small so repeated noisy matches cannot dominate.
        agreement_bonus = min(0.06, 0.015 * max(0, len(methods[chunk_id]) - 1))
        winner = item if item.score > current.score else current
        merged[chunk_id] = _copy_result(
            winner,
            score=min(1.0, best + agreement_bonus),
            method=winner.method,
        )
    output: list[RetrievedChunk] = []
    for chunk_id, item in merged.items():
        method = "+".join(methods[chunk_id][:6]) or item.method
        output.append(_copy_result(item, score=item.score, method=method))
    output.sort(
        key=lambda item: (
            -float(item.score),
            item.chunk.filename.casefold(),
            item.chunk.page_number,
            item.chunk.chunk_index or -1,
        )
    )
    return output


def merge_assistant_results(*groups: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    return _merge_results(item for group in groups for item in group)


def _active_documents(db: Session) -> list[object]:
    return list(
        db.execute(
            text(
                """
                SELECT DISTINCT d.id, d.filename
                FROM documents d
                JOIN rag_v5_processing_runs r
                  ON r.document_id=d.id AND r.is_active=true AND r.status='ready'
                WHERE d.status='ready'
                ORDER BY lower(d.filename), d.id
                """
            )
        ).mappings()
    )


def _filename_affinity(question: str, filename: str, queries: Sequence[str]) -> float:
    filename_norm = _norm(re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE))
    filename_terms = set(_terms(filename_norm))
    if not filename_norm or not filename_terms:
        return 0.0

    originals = [question, *queries[:5]]
    best = 0.0
    raw_question_folded = _clean(question).casefold()
    uppercase_filename_tokens = {
        token.casefold()
        for token in re.findall(r"\b[A-Z][A-Z0-9/-]{2,12}\b", filename)
        if token.casefold() not in _COMMON
    }
    if any(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", raw_question_folded) for token in uppercase_filename_tokens):
        best = max(best, 0.94)

    for value in originals:
        query_norm = _norm(value)
        query_terms = set(_terms(query_norm))
        if not query_norm or not query_terms:
            continue
        intersection = filename_terms & query_terms
        coverage = len(intersection) / max(1, min(len(filename_terms), len(query_terms)))
        sequence = SequenceMatcher(None, filename_norm[:220], query_norm[:220]).ratio()
        substring = 1.0 if query_norm in filename_norm or filename_norm in query_norm else 0.0
        best = max(best, min(1.0, 0.62 * coverage + 0.26 * sequence + 0.12 * substring))
    return best


def _vector_rows(
    db: Session,
    query_vector: list[float],
    limit: int,
    *,
    document_ids: Sequence[str] = (),
) -> list[object]:
    doc_filter = ""
    params: dict[str, object] = {"embedding": str(query_vector), "limit": limit}
    if document_ids:
        doc_filter = "AND c.document_id = ANY(CAST(:document_ids AS uuid[]))"
        params["document_ids"] = list(document_ids)
    return list(
        db.execute(
            text(
                f"""
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                       c.content_type, c.parent_key, c.section_path, c.heading, c.authority_status,
                       c.text, d.filename,
                       GREATEST(0.0, 1 - (c.embedding <=> CAST(:embedding AS vector))) AS vector_score
                FROM rag_v5_chunks c
                JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
                JOIN documents d ON d.id=c.document_id AND d.status='ready'
                WHERE 1=1 {doc_filter}
                ORDER BY c.embedding <=> CAST(:embedding AS vector), c.id
                LIMIT :limit
                """
            ),
            params,
        ).mappings()
    )


def _strict_fts_rows(
    db: Session,
    query: str,
    limit: int,
    *,
    document_ids: Sequence[str] = (),
) -> list[object]:
    doc_filter = ""
    params: dict[str, object] = {"query": query, "limit": limit}
    if document_ids:
        doc_filter = "AND c.document_id = ANY(CAST(:document_ids AS uuid[]))"
        params["document_ids"] = list(document_ids)
    return list(
        db.execute(
            text(
                f"""
                WITH q AS (
                    SELECT websearch_to_tsquery('simple', :query) AS simple_q,
                           websearch_to_tsquery('english', :query) AS english_q
                )
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                       c.content_type, c.parent_key, c.section_path, c.heading, c.authority_status,
                       c.text, d.filename,
                       GREATEST(
                         ts_rank_cd(to_tsvector('simple', c.text), q.simple_q),
                         ts_rank_cd(to_tsvector('english', c.text), q.english_q) * 0.9
                       ) AS keyword_score
                FROM rag_v5_chunks c
                JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
                JOIN documents d ON d.id=c.document_id AND d.status='ready'
                CROSS JOIN q
                WHERE (
                    to_tsvector('simple', c.text) @@ q.simple_q
                    OR to_tsvector('english', c.text) @@ q.english_q
                ) {doc_filter}
                ORDER BY keyword_score DESC, lower(d.filename), c.page_number, c.chunk_index
                LIMIT :limit
                """
            ),
            params,
        ).mappings()
    )


def _route_documents(
    db: Session,
    *,
    question: str,
    queries: Sequence[str],
    baseline: Sequence[RetrievedChunk],
) -> list[DocumentRoute]:
    documents = _active_documents(db)
    if not documents:
        return []

    scores: defaultdict[str, float] = defaultdict(float)
    filenames: dict[str, str] = {str(row["id"]): str(row["filename"]) for row in documents}

    # User-supplied document names/codes are preferences, not hard constraints.
    for document_id, filename in filenames.items():
        affinity = _filename_affinity(question, filename, queries)
        if affinity:
            scores[document_id] += 0.90 * affinity

    # Existing broad v5 retrieval is useful as one independent document signal.
    for rank, item in enumerate(baseline[:48], 1):
        if not item.chunk.document_id:
            continue
        scores[item.chunk.document_id] += 0.20 / (4.0 + rank) + 0.08 * float(item.score)

    route_queries = list(queries[: _int_env("RAG_V51_ROUTE_QUERY_COUNT", 5, 2, 8)])
    if route_queries:
        try:
            vectors = embedding_service.encode(route_queries)
            per_query = _int_env("RAG_V51_ROUTE_PER_QUERY", 80, 24, 180)
            for query, vector in zip(route_queries, vectors, strict=True):
                for rank, row in enumerate(_vector_rows(db, vector.tolist(), per_query), 1):
                    document_id = str(row["document_id"])
                    vector_score = float(row["vector_score"] or 0.0)
                    scores[document_id] += 0.24 / (6.0 + rank) + 0.035 * vector_score
                for rank, row in enumerate(_strict_fts_rows(db, query, per_query), 1):
                    document_id = str(row["document_id"])
                    keyword = min(1.0, float(row["keyword_score"] or 0.0) * 4.0)
                    scores[document_id] += 0.22 / (6.0 + rank) + 0.03 * keyword
        except Exception:
            logger.exception("Assistant document routing vector/FTS arm failed; continuing with remaining routes")

    max_score = max(scores.values(), default=1.0) or 1.0
    routes = [
        DocumentRoute(
            document_id=document_id,
            filename=filename,
            score=max(0.0, min(1.0, scores.get(document_id, 0.0) / max_score)),
        )
        for document_id, filename in filenames.items()
        if scores.get(document_id, 0.0) > 0
    ]
    routes.sort(key=lambda item: (-item.score, item.filename.casefold(), item.document_id))
    return routes[: _int_env("RAG_V51_ROUTE_DOCUMENTS", 8, 3, 16)]


def _heading_catalog(db: Session, document_ids: Sequence[str]) -> list[object]:
    if not document_ids:
        return []
    return list(
        db.execute(
            text(
                """
                SELECT DISTINCT ON (c.document_id, c.parent_key)
                       c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                       c.content_type, c.parent_key, c.section_path, c.heading,
                       c.authority_status, c.text, d.filename
                FROM rag_v5_chunks c
                JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
                JOIN documents d ON d.id=c.document_id AND d.status='ready'
                WHERE c.document_id = ANY(CAST(:document_ids AS uuid[]))
                  AND COALESCE(c.heading, '') <> ''
                ORDER BY c.document_id, c.parent_key, c.page_number, c.chunk_index
                LIMIT :limit
                """
            ),
            {
                "document_ids": list(document_ids),
                "limit": _int_env("RAG_V51_HEADING_CATALOG_LIMIT", 4000, 200, 10000),
            },
        ).mappings()
    )


def _text_affinity(candidate: str, queries: Sequence[str], concepts: Sequence[str]) -> float:
    candidate_norm = _norm(candidate)
    candidate_terms = set(_terms(candidate_norm))
    if not candidate_norm or not candidate_terms:
        return 0.0
    best = 0.0
    values = [*queries[:8], *concepts[:12]]
    for value in values:
        query_norm = _norm(value)
        query_terms = set(_terms(query_norm))
        if not query_norm or not query_terms:
            continue
        overlap = len(candidate_terms & query_terms)
        coverage = overlap / max(1, min(len(candidate_terms), len(query_terms)))
        jaccard = overlap / max(1, len(candidate_terms | query_terms))
        sequence = SequenceMatcher(None, candidate_norm[:220], query_norm[:220]).ratio()
        substring = 1.0 if query_norm in candidate_norm or candidate_norm in query_norm else 0.0
        best = max(best, min(1.0, 0.48 * coverage + 0.18 * jaccard + 0.24 * sequence + 0.10 * substring))
    return best


def _heading_candidates(
    db: Session,
    routes: Sequence[DocumentRoute],
    queries: Sequence[str],
    interpretation: SmartInterpretation,
) -> list[RetrievedChunk]:
    route_scores = {route.document_id: route.score for route in routes}
    rows = _heading_catalog(db, [route.document_id for route in routes])
    output: list[RetrievedChunk] = []
    for row in rows:
        heading = str(row["heading"] or "")
        section_path = row["section_path"] or []
        section_text = " > ".join(str(item) for item in section_path) if isinstance(section_path, list) else ""
        affinity = _text_affinity(
            f"{heading} {section_text}",
            queries,
            list(interpretation.concepts),
        )
        if affinity < 0.26:
            continue
        route_score = route_scores.get(str(row["document_id"]), 0.0)
        score = min(1.0, 0.50 + 0.37 * affinity + 0.13 * route_score)
        output.append(_row_to_result(row, score=score, method="v5.1-heading-navigation"))
    output.sort(key=lambda item: (-item.score, item.chunk.filename.casefold(), item.chunk.page_number))
    return output[: _int_env("RAG_V51_HEADING_CANDIDATES", 48, 12, 120)]


def _scoped_candidates(
    db: Session,
    routes: Sequence[DocumentRoute],
    queries: Sequence[str],
) -> list[RetrievedChunk]:
    document_ids = [route.document_id for route in routes]
    if not document_ids or not queries:
        return []
    route_scores = {route.document_id: route.score for route in routes}
    per_arm = _int_env("RAG_V51_SCOPED_PER_ARM", 56, 16, 140)
    query_subset = list(queries[: _int_env("RAG_V51_SCOPED_QUERY_COUNT", 6, 2, 10)])
    output: list[RetrievedChunk] = []
    try:
        vectors = embedding_service.encode(query_subset)
        for query, vector in zip(query_subset, vectors, strict=True):
            for rank, row in enumerate(_vector_rows(db, vector.tolist(), per_arm, document_ids=document_ids), 1):
                vector_score = float(row["vector_score"] or 0.0)
                route_score = route_scores.get(str(row["document_id"]), 0.0)
                score = min(1.0, 0.54 * vector_score + 0.24 * route_score + 0.22 / (1.0 + rank / 8.0))
                output.append(_row_to_result(row, score=score, method="v5.1-scoped-vector", vector=vector_score))
            for rank, row in enumerate(_strict_fts_rows(db, query, per_arm, document_ids=document_ids), 1):
                keyword = min(1.0, float(row["keyword_score"] or 0.0) * 4.5)
                route_score = route_scores.get(str(row["document_id"]), 0.0)
                score = min(1.0, 0.52 * keyword + 0.28 * route_score + 0.20 / (1.0 + rank / 8.0))
                output.append(_row_to_result(row, score=score, method="v5.1-scoped-strict-fts", keyword=keyword))
    except Exception:
        logger.exception("Assistant scoped retrieval arm failed; preserving baseline/structural candidates")
    return _merge_results(output)


def _rerank_prompt(
    interpretation: SmartInterpretation,
    candidates: Sequence[RetrievedChunk],
    routed_documents: Sequence[str],
) -> str:
    blocks: list[str] = []
    excerpt_chars = _int_env("RAG_V51_RERANK_EXCERPT_CHARS", 520, 180, 1000)
    for item in candidates:
        chunk = item.chunk
        pages = str(chunk.page_number) if not chunk.page_end or chunk.page_end == chunk.page_number else f"{chunk.page_number}-{chunk.page_end}"
        section = " > ".join(chunk.section_path) if chunk.section_path else (chunk.heading or "")
        body = re.sub(r"\s+", " ", chunk.text).strip()[:excerpt_chars]
        blocks.append(
            f"ID: {chunk.chunk_id}\n"
            f"File: {chunk.filename}\n"
            f"Pages: {pages}\n"
            f"Section: {section or 'Unsectioned'}\n"
            f"Type: {chunk.content_type}\n"
            f"Retrieval score: {item.score:.4f}\n"
            f"Excerpt: {body}"
        )
    return f"""RESOLVED QUESTION:
{interpretation.resolved_question}

INTENT:
{interpretation.intent}

EVIDENCE NEEDS:
{chr(10).join(f'- {item}' for item in interpretation.evidence_needs) or '- Direct evidence answering the request'}

SCOPE:
{interpretation.scope or 'None specified'}

ROUTED DOCUMENT CANDIDATES (routing hints, not factual proof):
{chr(10).join(f'- {name}' for name in routed_documents[:12]) or 'None'}

CANDIDATES:
{chr(10).join(chr(10) + block for block in blocks)}

Rank candidates for directness and completeness. Do not answer the question.
"""


def _json_object(raw: str) -> dict[str, object]:
    value = raw.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("reranker did not return JSON")
    payload = json.loads(value[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("reranker payload is not an object")
    return payload


def _ai_rerank(
    interpretation: SmartInterpretation,
    candidates: Sequence[RetrievedChunk],
    routed_documents: Sequence[str],
) -> list[RetrievedChunk]:
    if not candidates or not _bool_env("RAG_V51_AI_RERANK_ENABLED", True):
        return list(candidates)
    limit = _int_env("RAG_V51_RERANK_CANDIDATES", 48, 12, 80)
    pool = list(candidates[:limit])
    settings = get_settings()
    try:
        raw = llm_service.generate(
            _RERANK_SYSTEM,
            _rerank_prompt(interpretation, pool, routed_documents),
            max_output_tokens=_int_env("RAG_V51_RERANK_MAX_OUTPUT_TOKENS", 1400, 400, 3000),
            model=settings.query_model,
            reasoning_effort=settings.query_reasoning_effort,
        )
        payload = _json_object(raw)
        ranking = payload.get("ranking")
        if not isinstance(ranking, list):
            raise ValueError("reranker ranking is not a list")
    except Exception:
        logger.exception("Assistant AI reranker failed; using deterministic candidate order")
        return list(candidates)

    by_id = {item.chunk.chunk_id: item for item in pool}
    ranked: list[RetrievedChunk] = []
    seen: set[str] = set()
    for entry in ranking:
        if not isinstance(entry, dict):
            continue
        chunk_id = str(entry.get("id") or "")
        item = by_id.get(chunk_id)
        if item is None or chunk_id in seen:
            continue
        try:
            relevance = max(0.0, min(100.0, float(entry.get("score", 0.0)))) / 100.0
        except (TypeError, ValueError):
            relevance = 0.0
        role = str(entry.get("role") or "support").casefold()
        role_bonus = 0.08 if role == "primary" else (0.025 if role == "support" else -0.18)
        score = max(0.0, min(1.0, 0.32 * float(item.score) + 0.68 * relevance + role_bonus))
        ranked.append(_copy_result(item, score=score, method=f"{item.method}+v5.1-ai-rerank:{role}"))
        seen.add(chunk_id)

    # Keep deterministic fallbacks after AI-ranked items. This avoids a malformed/over-selective
    # reranker silently deleting potentially useful evidence.
    for item in candidates:
        if item.chunk.chunk_id in seen:
            continue
        ranked.append(_copy_result(item, score=max(0.0, item.score - 0.12), method=item.method))
    ranked.sort(key=lambda item: (-item.score, item.chunk.filename.casefold(), item.chunk.page_number))
    return ranked


def _section_keys_for_chunks(db: Session, chunk_ids: Sequence[str]) -> list[tuple[str, str]]:
    if not chunk_ids:
        return []
    rows = db.execute(
        text(
            """
            SELECT id, document_id, parent_key
            FROM rag_v5_chunks
            WHERE id = ANY(CAST(:chunk_ids AS uuid[]))
            """
        ),
        {"chunk_ids": list(chunk_ids)},
    ).mappings()
    by_chunk: dict[str, tuple[str, str]] = {}
    for row in rows:
        parent_key = str(row["parent_key"] or "")
        if parent_key:
            by_chunk[str(row["id"])] = (str(row["document_id"]), parent_key)

    # Preserve reranker order. SQL ANY() does not guarantee the input order.
    output: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk_id in chunk_ids:
        key = by_chunk.get(chunk_id)
        if key is not None and key not in seen:
            seen.add(key)
            output.append(key)
    return output


def _section_rows(db: Session, document_id: str, parent_key: str, limit: int) -> list[object]:
    return list(
        db.execute(
            text(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                       c.content_type, c.parent_key, c.section_path, c.heading,
                       c.authority_status, c.text, d.filename
                FROM rag_v5_chunks c
                JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
                JOIN documents d ON d.id=c.document_id AND d.status='ready'
                WHERE c.document_id=CAST(:document_id AS uuid)
                  AND c.parent_key=:parent_key
                ORDER BY c.chunk_index
                LIMIT :limit
                """
            ),
            {"document_id": document_id, "parent_key": parent_key, "limit": limit},
        ).mappings()
    )


def _expand_sections(
    db: Session,
    interpretation: SmartInterpretation,
    ranked: Sequence[RetrievedChunk],
) -> list[RetrievedChunk]:
    if not ranked or not _bool_env("RAG_V51_SECTION_EXPANSION_ENABLED", True):
        return []
    broad_intents = {"list", "procedure", "summary", "requirement", "troubleshooting", "comparison"}
    section_count = _int_env(
        "RAG_V51_SECTION_SEEDS",
        10 if interpretation.intent in broad_intents or interpretation.conversation_act == "navigation" else 4,
        2,
        16,
    )
    seed_ids = [item.chunk.chunk_id for item in ranked[: max(section_count * 2, 8)]]
    keys = _section_keys_for_chunks(db, seed_ids)[:section_count]
    per_section = _int_env("RAG_V51_MAX_SECTION_CHUNKS", 12, 4, 40)
    output: list[RetrievedChunk] = []
    for seed_rank, (document_id, parent_key) in enumerate(keys, 1):
        seed_score = float(ranked[min(seed_rank - 1, len(ranked) - 1)].score)
        section_base = min(seed_score, max(0.62, 0.90 - 0.025 * (seed_rank - 1)))
        for position, row in enumerate(_section_rows(db, document_id, parent_key, per_section)):
            output.append(
                _row_to_result(
                    row,
                    score=max(0.56, section_base - 0.008 * position),
                    method="v5.1-complete-section",
                )
            )
    return output


def _extract_expansions(text_value: str, alias: str) -> list[str]:
    alias_re = re.escape(alias)
    patterns = [
        re.compile(rf"\b([A-Z][A-Za-z&/-]+(?:\s+[A-Z][A-Za-z&/-]+){{1,6}})\s*\(\s*{alias_re}\s*\)"),
        re.compile(rf"\b{alias_re}\s*\(\s*([A-Z][A-Za-z&/-]+(?:\s+[A-Z][A-Za-z&/-]+){{1,6}})\s*\)"),
        re.compile(rf"\b{alias_re}\b\s*(?:means|stands\s+for|[:=\-–—])\s*([A-Z][A-Za-z&/-]+(?:\s+[A-Z][A-Za-z&/-]+){{1,6}})", re.IGNORECASE),
    ]
    output: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text_value):
            candidate = _clean(match.group(1)).strip(" -–—:;,.()")
            key = candidate.casefold()
            if 3 <= len(candidate) <= 100 and key not in seen:
                seen.add(key)
                output.append(candidate)
    return output[:6]


def assistant_terminology_hints(db: Session, question: str, *, limit: int = 20) -> list[str]:
    base = terminology_hints(db, question, limit=limit)
    known_aliases = {
        _norm(item.split("=", 1)[0])
        for item in base
        if "=" in item
    }
    aliases = _unique(_UPPER_ALIAS_RE.findall(question), 10)
    missing = [alias for alias in aliases if _norm(alias) not in known_aliases]
    if not missing:
        return base[:limit]

    output = list(base)
    seen = {item.casefold() for item in output}
    for alias in missing:
        try:
            rows = db.execute(
                text(
                    """
                    SELECT c.text, d.filename, c.page_number
                    FROM rag_v5_chunks c
                    JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
                    JOIN documents d ON d.id=c.document_id AND d.status='ready'
                    WHERE to_tsvector('simple', c.text) @@ plainto_tsquery('simple', :alias)
                    ORDER BY lower(d.filename), c.page_number, c.chunk_index
                    LIMIT 60
                    """
                ),
                {"alias": alias},
            ).mappings()
        except Exception:
            logger.exception("Grounded terminology fallback failed for alias %s", alias)
            continue
        for row in rows:
            for expansion in _extract_expansions(str(row["text"]), alias):
                hint = f"{alias} = {expansion} (PDF: {row['filename']}, page {row['page_number']})"
                key = hint.casefold()
                if key in seen:
                    continue
                seen.add(key)
                output.append(hint)
                if len(output) >= limit:
                    return output
    return output[:limit]


def retrieve_assistant_v51(
    db: Session,
    interpretation: SmartInterpretation,
    *,
    original_question: str,
    extra_queries: Iterable[str] = (),
    prior_results: Sequence[RetrievedChunk] = (),
) -> AssistantRetrievalBundle:
    queries = _query_variants(interpretation, extra_queries)
    baseline_bundle = retrieve_v5(db, interpretation, extra_queries=extra_queries)
    if not _bool_env("RAG_V51_ASSISTANT_ENABLED", True):
        combined = _merge_results([*prior_results, *baseline_bundle.results])
        return AssistantRetrievalBundle(
            results=combined[: _int_env("RAG_V51_FINAL_CANDIDATES", 64, 24, 120)],
            search_queries=_unique([*queries, *baseline_bundle.search_queries], 16),
            candidate_count=len(combined),
            routed_documents=[],
        )

    routes = _route_documents(
        db,
        question=original_question,
        queries=queries,
        baseline=baseline_bundle.results,
    )
    routed_names = [route.filename for route in routes]

    scoped = _scoped_candidates(db, routes, queries)
    headings = _heading_candidates(db, routes, queries, interpretation)
    candidates = _merge_results([
        *prior_results,
        *baseline_bundle.results,
        *headings,
        *scoped,
    ])
    candidate_count = len(candidates)

    reranked = _ai_rerank(interpretation, candidates, routed_names)
    expanded = _expand_sections(db, interpretation, reranked)
    final = _merge_results([*reranked, *expanded])
    final_limit = _int_env("RAG_V51_FINAL_CANDIDATES", 64, 24, 120)
    return AssistantRetrievalBundle(
        results=final[:final_limit],
        search_queries=_unique([*queries, *baseline_bundle.search_queries], 16),
        candidate_count=candidate_count,
        routed_documents=routed_names,
    )
