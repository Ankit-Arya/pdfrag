from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.rag.embeddings import embedding_service
from app.rag.smart_understanding import SmartInterpretation
from app.rag.types import RetrievedChunk, TextChunk
from app.rag.v5.terminology import definition_source_rows

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:#-]*")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "if", "in", "is", "it", "of", "on", "or", "should", "that",
    "the", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "would", "please", "tell", "give",
}
_VALUE_CUES = {
    "amount", "compensation", "payable", "payment", "speed", "limit", "pressure", "voltage",
    "time", "duration", "distance", "penalty", "rate", "value", "threshold", "count", "number",
}


@dataclass(slots=True)
class V5RetrievalBundle:
    results: list[RetrievedChunk]
    search_queries: list[str]
    candidate_count: int


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 500) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _terms(value: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(value.casefold()):
        token = re.sub(r"[^a-z0-9]+", "", raw)
        if not token or token in _STOP:
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
    return output[:40]


def _tsquery(value: str) -> str:
    terms = _terms(value)
    if not terms:
        return "pdfrag_no_match"
    return " | ".join(term if len(term) <= 3 else f"{term}:*" for term in terms)


def _row_to_result(row: object, *, score: float, method: str, vector: float = 0.0, keyword: float = 0.0) -> RetrievedChunk:
    section_path = row["section_path"] or []  # type: ignore[index]
    if not isinstance(section_path, list):
        section_path = []
    page_end = int(row["page_end"] or row["page_number"])  # type: ignore[index]
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=str(row["id"]),  # type: ignore[index]
            filename=str(row["filename"]),  # type: ignore[index]
            page_number=int(row["page_number"]),  # type: ignore[index]
            page_end=page_end,
            text=str(row["text"]),  # type: ignore[index]
            content_type=str(row["content_type"]),  # type: ignore[index]
            section_path=tuple(str(item) for item in section_path),
            heading=str(row["heading"] or ""),  # type: ignore[index]
            document_id=str(row["document_id"]),  # type: ignore[index]
            chunk_index=int(row["chunk_index"]),  # type: ignore[index]
        ),
        score=max(0.0, min(1.0, score)),
        method=method,
        vector_score=max(0.0, min(1.0, vector)),
        keyword_score=max(0.0, min(1.0, keyword)),
    )


def _vector_rows(db: Session, query_vector: list[float], limit: int) -> list[object]:
    return list(
        db.execute(
            text(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                       c.content_type, c.parent_key, c.section_path, c.heading, c.authority_status,
                       c.text, d.filename,
                       GREATEST(0.0, 1 - (c.embedding <=> CAST(:embedding AS vector))) AS vector_score
                FROM rag_v5_chunks c
                JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
                JOIN documents d ON d.id=c.document_id AND d.status='ready'
                ORDER BY c.embedding <=> CAST(:embedding AS vector), c.id
                LIMIT :limit
                """
            ),
            {"embedding": str(query_vector), "limit": limit},
        ).mappings()
    )


def _fts_rows(db: Session, query: str, limit: int, *, table_only: bool = False) -> list[object]:
    tsq = _tsquery(query)
    content_filter = "AND c.content_type='table_row'" if table_only else ""
    return list(
        db.execute(
            text(
                f"""
                WITH q AS (
                    SELECT to_tsquery('simple', :tsq) AS simple_q,
                           to_tsquery('english', :tsq) AS english_q
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
                )
                {content_filter}
                ORDER BY keyword_score DESC, lower(d.filename), c.page_number, c.chunk_index, c.id
                LIMIT :limit
                """
            ),
            {"tsq": tsq, "limit": limit},
        ).mappings()
    )


def _exact_rows(db: Session, query: str, limit: int) -> list[object]:
    priority = [term for term in _terms(query) if len(term) >= 3][:6]
    if not priority:
        return []
    conditions: list[str] = []
    params: dict[str, object] = {"limit": limit}
    for index, term in enumerate(priority):
        key = f"p{index}"
        params[key] = f"%{term}%"
        conditions.append(f"lower(c.text) LIKE :{key}")
    return list(
        db.execute(
            text(
                f"""
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                       c.content_type, c.parent_key, c.section_path, c.heading, c.authority_status,
                       c.text, d.filename, 1.0 AS keyword_score
                FROM rag_v5_chunks c
                JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
                JOIN documents d ON d.id=c.document_id AND d.status='ready'
                WHERE {' AND '.join(conditions)}
                ORDER BY
                    CASE WHEN c.content_type='table_row' THEN 0 ELSE 1 END,
                    lower(d.filename), c.page_number, c.chunk_index
                LIMIT :limit
                """
            ),
            params,
        ).mappings()
    )


def _explicit_year(value: str) -> int | None:
    """Return a year only when the user is intentionally asking for historical state.

    Merely mentioning an amendment year must not disable current-authority precedence.
    """
    years = [int(match.group(0)) for match in re.finditer(r"\b(?:19|20)\d{2}\b", value)]
    if not years:
        return None
    folded = " ".join(value.casefold().split())
    strong_historical = bool(
        re.search(
            r"\b(?:old|older|previous|prior|original|historical|superseded|before\s+(?:the\s+)?amendment)\b",
            folded,
        )
    )
    if strong_historical:
        return min(years) if "before" in folded else max(years)
    if re.search(r"\b(?:current|latest|amended|amendment|revised|replacement)\b", folded):
        return None
    explicit_time = bool(re.search(r"\b(?:in|as\s+of|during)\s+(?:19|20)\d{2}\b", folded))
    return max(years) if explicit_time else None


def _authority_adjust(score: float, authority_status: str, *, authority_sensitive: bool, explicit_year: int | None) -> float:
    status = authority_status.casefold()
    if explicit_year is not None:
        return score
    # Ordinary operational questions mean "what applies now" unless the user explicitly asks
    # for a historical year. Therefore explicit source-derived current/superseded metadata is a
    # default correctness signal, not a feature that depends on the interpreter remembering to
    # set authority_sensitive. The AI flag only makes the current boost slightly stronger.
    current_boost = 0.16 if authority_sensitive else 0.12
    if status.startswith("current"):
        return min(1.0, score + current_boost)
    if status.startswith("superseded"):
        return max(0.0, score - 0.45)
    if status == "historical_appended":
        return max(0.0, score - 0.30)
    return score


def _rrf_merge(
    ranked_lists: list[tuple[str, list[object]]],
    *,
    interpretation: SmartInterpretation,
    explicit_year: int | None,
) -> list[RetrievedChunk]:
    accumulator: dict[str, dict[str, object]] = {}
    k = 60.0
    for method, rows in ranked_lists:
        for rank, row in enumerate(rows, 1):
            chunk_id = str(row["id"])  # type: ignore[index]
            entry = accumulator.setdefault(chunk_id, {"row": row, "rrf": 0.0, "methods": set(), "vector": 0.0, "keyword": 0.0})
            entry["rrf"] = float(entry["rrf"]) + 1.0 / (k + rank)
            methods = entry["methods"]
            assert isinstance(methods, set)
            methods.add(method)
            if "vector_score" in row:
                entry["vector"] = max(float(entry["vector"]), float(row["vector_score"] or 0.0))  # type: ignore[index]
            if "keyword_score" in row:
                entry["keyword"] = max(float(entry["keyword"]), min(1.0, float(row["keyword_score"] or 0.0) * 5.0))  # type: ignore[index]

    if not accumulator:
        return []
    max_rrf = max(float(entry["rrf"]) for entry in accumulator.values()) or 1.0
    results: list[RetrievedChunk] = []
    for entry in accumulator.values():
        row = entry["row"]
        assert row is not None
        score = 0.50 * (float(entry["rrf"]) / max_rrf) + 0.30 * float(entry["vector"]) + 0.20 * float(entry["keyword"])
        content_type = str(row["content_type"])  # type: ignore[index]
        if interpretation.route_strategy == "structured_lookup" and content_type == "table_row":
            score += 0.12
        authority = str(row["authority_status"] or "unknown")  # type: ignore[index]
        score = _authority_adjust(
            score,
            authority,
            authority_sensitive=interpretation.authority_sensitive,
            explicit_year=explicit_year,
        )
        method = "v5-rrf:" + "+".join(sorted(entry["methods"]))
        if authority != "unknown":
            method += f"+authority:{authority}"
        results.append(
            _row_to_result(
                row,
                score=score,
                method=method,
                vector=float(entry["vector"]),
                keyword=float(entry["keyword"]),
            )
        )
    results.sort(key=lambda item: (-float(item.score), item.chunk.filename.casefold(), item.chunk.page_number, item.chunk.chunk_index or -1))
    return results


def _parent_neighbors(db: Session, seeds: list[RetrievedChunk], window: int) -> list[RetrievedChunk]:
    output: list[RetrievedChunk] = []
    for seed in seeds[:16]:
        if not seed.chunk.document_id or seed.chunk.chunk_index is None:
            continue
        row = db.execute(
            text(
                """
                SELECT parent_key FROM rag_v5_chunks
                WHERE id=CAST(:id AS uuid)
                LIMIT 1
                """
            ),
            {"id": seed.chunk.chunk_id},
        ).mappings().first()
        if row is None:
            continue
        parent_key = str(row["parent_key"])
        rows = db.execute(
            text(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                       c.content_type, c.parent_key, c.section_path, c.heading, c.authority_status,
                       c.text, d.filename
                FROM rag_v5_chunks c
                JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
                JOIN documents d ON d.id=c.document_id AND d.status='ready'
                WHERE c.document_id=CAST(:document_id AS uuid)
                  AND c.parent_key=:parent_key
                  AND c.chunk_index BETWEEN :start_idx AND :end_idx
                ORDER BY c.chunk_index
                """
            ),
            {
                "document_id": seed.chunk.document_id,
                "parent_key": parent_key,
                "start_idx": max(0, seed.chunk.chunk_index - window),
                "end_idx": seed.chunk.chunk_index + window,
            },
        ).mappings()
        for candidate in rows:
            distance = abs(int(candidate["chunk_index"]) - seed.chunk.chunk_index)
            output.append(
                _row_to_result(
                    candidate,
                    score=max(0.35, float(seed.score) - 0.07 * distance),
                    method="v5-parent-neighbor",
                )
            )
    return output


def _queries_for_interpretation(interpretation: SmartInterpretation) -> list[str]:
    queries: list[str] = []
    values = [interpretation.resolved_question, *interpretation.search_queries]
    for need in interpretation.evidence_needs:
        values.append(f"{interpretation.resolved_question} {need}")
    if interpretation.authority_sensitive:
        values.append(
            f"{interpretation.resolved_question} current amended amendment revised substituted replacement effective schedule"
        )
    seen: set[str] = set()
    for value in values:
        clean = " ".join(str(value or "").split())
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            queries.append(clean)
        if len(queries) >= 10:
            break
    return queries


def retrieve_v5(
    db: Session,
    interpretation: SmartInterpretation,
    *,
    extra_queries: Iterable[str] = (),
) -> V5RetrievalBundle:
    per_arm = _int_env("RAG_V5_RETRIEVAL_PER_ARM", 48, 12, 160)
    final_limit = _int_env("RAG_V5_FINAL_EVIDENCE", 32, 12, 80)
    parent_window = _int_env("RAG_V5_PARENT_WINDOW", 2, 0, 5)
    queries = _queries_for_interpretation(interpretation)
    for value in extra_queries:
        clean = " ".join(str(value or "").split())
        if clean and clean.casefold() not in {item.casefold() for item in queries}:
            queries.append(clean)
    queries = queries[:12]
    if not queries:
        return V5RetrievalBundle(results=[], search_queries=[], candidate_count=0)

    vectors = embedding_service.encode(queries)
    ranked_lists: list[tuple[str, list[object]]] = []
    if interpretation.intent == "definition":
        definition_rows = definition_source_rows(db, interpretation.resolved_question, limit=max(12, per_arm // 2))
        if definition_rows:
            ranked_lists.append(("terminology-definition", definition_rows))
    structured = interpretation.route_strategy == "structured_lookup" or bool(set(_terms(interpretation.resolved_question)) & _VALUE_CUES)
    for index, (query, vector) in enumerate(zip(queries, vectors, strict=True)):
        ranked_lists.append((f"vector-{index}", _vector_rows(db, vector.tolist(), per_arm)))
        ranked_lists.append((f"fts-{index}", _fts_rows(db, query, per_arm)))
        if index < 4:
            ranked_lists.append((f"exact-{index}", _exact_rows(db, query, max(16, per_arm // 2))))
        if structured and index < 5:
            ranked_lists.append((f"table-{index}", _fts_rows(db, query, per_arm, table_only=True)))

    merged = _rrf_merge(
        ranked_lists,
        interpretation=interpretation,
        explicit_year=_explicit_year(interpretation.resolved_question),
    )
    candidate_count = len(merged)
    seeds = merged[: min(18, final_limit)]
    if parent_window > 0 and seeds:
        neighbors = _parent_neighbors(db, seeds, parent_window)
        by_id = {item.chunk.chunk_id: item for item in merged}
        for item in neighbors:
            old = by_id.get(item.chunk.chunk_id)
            if old is None or item.score > old.score:
                by_id[item.chunk.chunk_id] = item
        merged = sorted(
            by_id.values(),
            key=lambda item: (-float(item.score), item.chunk.filename.casefold(), item.chunk.page_number, item.chunk.chunk_index or -1),
        )
    # Preserve document diversity but do not let an arbitrary per-document quota hide the only table row.
    selected: list[RetrievedChunk] = []
    per_doc: defaultdict[str, int] = defaultdict(int)
    for item in merged:
        doc = item.chunk.document_id or item.chunk.filename
        cap = 14 if item.chunk.content_type == "table_row" else 10
        if per_doc[doc] >= cap and len(selected) >= final_limit // 2:
            continue
        selected.append(item)
        per_doc[doc] += 1
        if len(selected) >= final_limit:
            break
    return V5RetrievalBundle(results=selected, search_queries=queries, candidate_count=candidate_count)
