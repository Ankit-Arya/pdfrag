from __future__ import annotations

# ruff: noqa: E501

import os
import re
from collections import defaultdict
from dataclasses import replace

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.rag.scenario_reasoning import current_scenario, logical_match_score
from app.rag.types import RetrievedChunk, TextChunk

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "give", "how", "if", "in", "is", "it", "of", "on", "or", "please",
    "should", "that", "the", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "would",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 5000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _terms(value: str) -> set[str]:
    result: set[str] = set()
    for token in _TOKEN_RE.findall(value.casefold()):
        if token in _STOP:
            continue
        if len(token) > 6 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 5 and token.endswith("ed") and not token.endswith("eed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        if len(token) > 1 or token.isdigit():
            result.add(token)
    return result


def _tsquery(value: str) -> str:
    safe = sorted(term for term in _terms(value) if re.fullmatch(r"[a-z0-9]+", term))
    return " | ".join(term if len(term) <= 3 or term.isdigit() else f"{term}:*" for term in safe[:48]) or "pdfrag_no_match"


def _lexical(query: str, source: str) -> float:
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    source_terms = _terms(source[:12000])
    return min(1.0, len(query_terms & source_terms) / max(1, len(query_terms)))


def _row_result(row: object, *, score: float, method: str, vector_score: float, keyword_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=str(row["id"]),  # type: ignore[index]
            filename=str(row["filename"]),  # type: ignore[index]
            page_number=int(row["page_number"]),  # type: ignore[index]
            text=str(row["text"]),  # type: ignore[index]
            content_type=str(row["content_type"]),  # type: ignore[index]
            document_id=str(row["document_id"]),  # type: ignore[index]
            chunk_index=int(row["chunk_index"]),  # type: ignore[index]
        ),
        score=max(0.0, min(1.0, score)),
        method=method,
        vector_score=max(0.0, min(1.0, vector_score)),
        keyword_score=max(0.0, min(1.0, keyword_score)),
    )


def route_procedure_documents(
    db: Session,
    query_vector: list[float],
    query_text: str,
) -> dict[str, float]:
    card_k = _int_env("SMART_RAG_PROCEDURE_CARD_K", 12, 1, 100)
    tsq = _tsquery(query_text)
    vector_rows = list(
        db.execute(
            text(
                """
                SELECT pc.document_id,
                       GREATEST(0.0, 1 - (pc.embedding <=> CAST(:embedding AS vector))) AS score
                FROM rag_procedure_cards pc
                JOIN documents d ON d.id = pc.document_id
                WHERE d.status = 'ready' AND pc.embedding IS NOT NULL
                ORDER BY pc.embedding <=> CAST(:embedding AS vector), pc.id
                LIMIT :limit
                """
            ),
            {"embedding": str(query_vector), "limit": card_k},
        ).mappings()
    )
    fts_rows = list(
        db.execute(
            text(
                """
                WITH q AS (SELECT to_tsquery('simple', :tsq) AS query)
                SELECT pc.document_id,
                       LEAST(1.0, ts_rank_cd(to_tsvector('simple', pc.search_text), q.query) * 4.0) AS score
                FROM rag_procedure_cards pc
                JOIN documents d ON d.id = pc.document_id
                CROSS JOIN q
                WHERE d.status = 'ready'
                  AND to_tsvector('simple', pc.search_text) @@ q.query
                ORDER BY score DESC, pc.id
                LIMIT :limit
                """
            ),
            {"tsq": tsq, "limit": card_k},
        ).mappings()
    )
    scores: dict[str, float] = defaultdict(float)
    for row in vector_rows:
        scores[str(row["document_id"])] = max(scores[str(row["document_id"])], float(row["score"] or 0.0))
    for row in fts_rows:
        key = str(row["document_id"])
        scores[key] = max(scores[key], float(row["score"] or 0.0))
    max_docs = _int_env("SMART_RAG_ROUTED_DOCUMENTS", 8, 1, 30)
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max_docs])


def fast_search_chunks(
    db: Session,
    query_vector: list[float],
    query_text: str,
    limit: int,
) -> list[RetrievedChunk]:
    """Bounded hybrid retrieval designed for ~50k+ chunks.

    Unlike the original search, this never asks every document to contribute rows.
    It uses global HNSW/GIN retrieval, then a small procedure-card-routed document set.
    """
    vector_k = _int_env("SMART_RAG_VECTOR_K", 90, 10, 500)
    fts_k = _int_env("SMART_RAG_FTS_K", 90, 10, 500)
    scoped_k = _int_env("SMART_RAG_SCOPED_K", 70, 10, 500)
    return_k = min(_int_env("SMART_RAG_MAX_CANDIDATES", 180, 20, 1000), max(20, limit))
    tsq = _tsquery(query_text)
    scenario = current_scenario()

    routed_docs = route_procedure_documents(db, query_vector, query_text)
    collected: dict[str, dict[str, object]] = {}

    def merge_row(row: object, method: str, vector_score: float = 0.0, keyword_score: float = 0.0) -> None:
        chunk_id = str(row["id"])  # type: ignore[index]
        entry = collected.get(chunk_id)
        if entry is None:
            entry = {
                "row": row,
                "methods": set(),
                "vector": 0.0,
                "keyword": 0.0,
            }
            collected[chunk_id] = entry
        methods = entry["methods"]
        assert isinstance(methods, set)
        methods.add(method)
        entry["vector"] = max(float(entry["vector"]), vector_score)
        entry["keyword"] = max(float(entry["keyword"]), keyword_score)

    # Deterministic rule routing is the bridge for queries such as "4 brakes failed"
    # when the source says "2 or more brakes failed".  It does not depend on the
    # exact user number being present in the source text.
    if scenario is not None and scenario.numeric_facts:
        rule_tokens = sorted({token for fact in scenario.numeric_facts for token in fact.tokens})[:32]
        if rule_tokens:
            token_params: dict[str, object] = {"rule_limit": _int_env("SMART_RAG_RULE_K", 80, 10, 300)}
            token_placeholders: list[str] = []
            for index, token in enumerate(rule_tokens):
                key = f"rt{index}"
                token_params[key] = token
                token_placeholders.append(f":{key}")
            rule_rows = db.execute(
                text(
                    f"""
                    SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename
                    FROM rag_rules r
                    JOIN document_chunks c ON c.id = r.chunk_id
                    JOIN documents d ON d.id = c.document_id
                    WHERE d.status = 'ready'
                      AND r.field_tokens && ARRAY[{', '.join(token_placeholders)}]::text[]
                    ORDER BY r.confidence DESC, c.document_id, c.chunk_index
                    LIMIT :rule_limit
                    """
                ),
                token_params,
            ).mappings()
            for row in rule_rows:
                rule_boost, _ = logical_match_score(scenario, str(row["text"]))
                if rule_boost <= 0:
                    continue
                merge_row(row, "smart-rule-index")
                routed_docs[str(row["document_id"])] = max(1.0, routed_docs.get(str(row["document_id"]), 0.0))

    vector_rows = db.execute(
        text(
            """
            SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename,
                   GREATEST(0.0, 1 - (c.embedding <=> CAST(:embedding AS vector))) AS vector_score
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'ready'
            ORDER BY c.embedding <=> CAST(:embedding AS vector), c.id
            LIMIT :limit
            """
        ),
        {"embedding": str(query_vector), "limit": vector_k},
    ).mappings()
    for row in vector_rows:
        merge_row(row, "smart-hnsw", vector_score=float(row["vector_score"] or 0.0))

    for config in ("simple", "english"):
        rows = db.execute(
            text(
                f"""
                WITH q AS (SELECT to_tsquery('{config}', :tsq) AS query)
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename,
                       LEAST(1.0, ts_rank_cd(to_tsvector('{config}', c.text), q.query) * 4.0) AS keyword_score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                CROSS JOIN q
                WHERE d.status = 'ready'
                  AND to_tsvector('{config}', c.text) @@ q.query
                ORDER BY keyword_score DESC, c.id
                LIMIT :limit
                """
            ),
            {"tsq": tsq, "limit": fts_k},
        ).mappings()
        for row in rows:
            merge_row(row, f"smart-fts-{config}", keyword_score=float(row["keyword_score"] or 0.0))

    if routed_docs:
        doc_params: dict[str, object] = {"embedding": str(query_vector), "tsq": tsq, "limit": scoped_k}
        doc_placeholders: list[str] = []
        for index, document_id in enumerate(routed_docs):
            key = f"doc{index}"
            doc_params[key] = document_id
            doc_placeholders.append(f"CAST(:{key} AS uuid)")
        doc_filter = ", ".join(doc_placeholders)
        rows = db.execute(
            text(
                f"""
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename,
                       GREATEST(0.0, 1 - (c.embedding <=> CAST(:embedding AS vector))) AS vector_score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'ready' AND c.document_id IN ({doc_filter})
                ORDER BY c.embedding <=> CAST(:embedding AS vector), c.id
                LIMIT :limit
                """
            ),
            doc_params,
        ).mappings()
        for row in rows:
            merge_row(row, "smart-routed-hnsw", vector_score=float(row["vector_score"] or 0.0))

        rows = db.execute(
            text(
                f"""
                WITH q AS (SELECT to_tsquery('simple', :tsq) AS query)
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename,
                       LEAST(1.0, ts_rank_cd(to_tsvector('simple', c.text), q.query) * 4.0) AS keyword_score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                CROSS JOIN q
                WHERE d.status = 'ready'
                  AND c.document_id IN ({doc_filter})
                  AND to_tsvector('simple', c.text) @@ q.query
                ORDER BY keyword_score DESC, c.id
                LIMIT :limit
                """
            ),
            doc_params,
        ).mappings()
        for row in rows:
            merge_row(row, "smart-routed-fts", keyword_score=float(row["keyword_score"] or 0.0))

    results: list[RetrievedChunk] = []
    for entry in collected.values():
        row = entry["row"]
        vector_score = float(entry["vector"])
        keyword_score = float(entry["keyword"])
        lexical = _lexical(query_text, str(row["text"]))  # type: ignore[index]
        document_boost = min(0.15, routed_docs.get(str(row["document_id"]), 0.0) * 0.15)  # type: ignore[index]
        logical_boost = 0.0
        if scenario is not None:
            logical_boost, _ = logical_match_score(scenario, str(row["text"]))  # type: ignore[index]
        base = vector_score * 0.50 + keyword_score * 0.31 + lexical * 0.19
        score = max(0.0, min(1.0, base + document_boost + logical_boost))
        methods = sorted(str(value) for value in entry["methods"])
        if logical_boost > 0:
            methods.append("deterministic-rule-match")
        elif logical_boost < 0:
            methods.append("deterministic-rule-mismatch")
        results.append(
            _row_result(
                row,
                score=score,
                method="+".join(methods),
                vector_score=vector_score,
                keyword_score=keyword_score,
            )
        )

    results.sort(
        key=lambda item: (
            -item.score,
            item.chunk.filename.casefold(),
            item.chunk.page_number,
            item.chunk.chunk_index if item.chunk.chunk_index is not None else -1,
            item.chunk.chunk_id,
        )
    )
    return _diverse(results, return_k)


def fast_corpus_scan(
    db: Session,
    query_texts: list[str],
    *,
    focus_terms: list[str] | None = None,
    reference_mode: bool = False,
    limit: int = 10000,
) -> list[RetrievedChunk]:
    """Indexed lexical fallback replacing the old up-to-10k-row corpus pass."""
    cap = _int_env("SMART_RAG_CORPUS_FALLBACK_K", 160 if not reference_mode else 300, 20, 1000)
    combined = " ".join([*query_texts[:3], *(focus_terms or [])[:24]])
    tsq = _tsquery(combined)
    rows = db.execute(
        text(
            """
            WITH q AS (SELECT to_tsquery('simple', :tsq) AS query)
            SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename,
                   LEAST(1.0, ts_rank_cd(to_tsvector('simple', c.text), q.query) * 4.0) AS keyword_score
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            CROSS JOIN q
            WHERE d.status = 'ready'
              AND to_tsvector('simple', c.text) @@ q.query
            ORDER BY keyword_score DESC, c.id
            LIMIT :limit
            """
        ),
        {"tsq": tsq, "limit": min(cap, max(20, limit))},
    ).mappings()
    results = [
        _row_result(
            row,
            score=min(1.0, 0.45 + float(row["keyword_score"] or 0.0) * 0.45),
            method="smart-indexed-corpus-fts",
            vector_score=0.0,
            keyword_score=float(row["keyword_score"] or 0.0),
        )
        for row in rows
    ]
    scenario = current_scenario()
    if scenario is not None:
        adjusted: list[RetrievedChunk] = []
        for item in results:
            boost, _ = logical_match_score(scenario, item.chunk.text)
            method = item.method
            if boost > 0:
                method += "+deterministic-rule-match"
            elif boost < 0:
                method += "+deterministic-rule-mismatch"
            adjusted.append(replace(item, score=max(0.0, min(1.0, item.score + boost)), method=method))
        results = adjusted
    results.sort(key=lambda item: -item.score)
    return _diverse(results, cap)


def rule_notes_for_chunk(chunk_id: str, text_value: str) -> list[str]:
    scenario = current_scenario()
    if scenario is None:
        return []
    _, notes = logical_match_score(scenario, text_value)
    return [note for note in notes if " is true" in note][:3]


def _diverse(results: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    if limit <= 0:
        return []
    selected: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    per_doc: dict[str, int] = defaultdict(int)
    per_doc_soft_cap = max(8, limit // 5)
    for item in results:
        if item.chunk.chunk_id in seen_ids:
            continue
        doc = item.chunk.document_id or item.chunk.filename.casefold()
        if per_doc[doc] >= per_doc_soft_cap:
            continue
        seen_ids.add(item.chunk.chunk_id)
        per_doc[doc] += 1
        selected.append(item)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for item in results:
            if item.chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(item.chunk.chunk_id)
            selected.append(item)
            if len(selected) >= limit:
                break
    return selected
