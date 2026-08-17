from __future__ import annotations

# ruff: noqa: E501

import os
import re
from collections import defaultdict
from dataclasses import replace
from collections.abc import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.rag.authority import normalize_authority_text
from app.rag.scenario_reasoning import current_scenario, logical_match_score
from app.rag.terminology import definition_for_token, definition_request_aliases
from app.rag.types import RetrievedChunk, TextChunk

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "give", "how", "if", "in", "is", "it", "of", "on", "or", "please",
    "should", "that", "the", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "would",
}
_LOW_SIGNAL = {
    "amount", "case", "different", "various", "accident", "metro", "system", "person",
    "people", "given", "give", "much", "applicable", "detail", "details", "information",
    "full", "form", "meaning", "mean", "define", "definition", "expand", "expansion",
}
_MONEY_TERMS = {"compensation", "claim", "payable", "payment"}
_VALUE_DIMENSION_TERMS = {
    "amount", "compensation", "claim", "payable", "payment", "price", "cost", "fee",
    "rate", "limit", "maximum", "minimum", "speed", "pressure", "voltage", "current",
    "temperature", "time", "duration", "distance", "weight", "capacity", "quantity",
    "number", "count", "frequency", "height", "length", "width", "clearance", "allowance",
    "penalty", "fine", "threshold", "value", "range",
}
_STRUCTURED_LIST_CUES = {
    "different", "various", "each", "every", "types", "categories", "classes",
    "cases", "conditions", "modes", "injuries", "items", "table", "schedule", "list",
}
_VALUE_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(?:₹|rs\.?\s*)?\d(?:[\d,]*(?:\.\d+)?)?(?![A-Za-z0-9])", re.IGNORECASE)
_MEASUREMENT_UNIT_RE = re.compile(
    r"\b(?:km/?h|kmph|m/?s|kpa|bar|psi|v|kv|a|ma|hz|kg|g|mm|cm|m|km|sec(?:ond)?s?|min(?:ute)?s?|hours?|days?|rupees?|lakh|crore|percent|%)\b",
    re.IGNORECASE,
)
_VERSION_SENSITIVE_TERMS = {
    "compensation", "claim", "payable", "amount", "rate", "fee", "penalty", "allowance",
    "schedule", "limit", "current", "latest", "revised", "revision", "amendment", "amended",
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


def _priority_terms(value: str) -> set[str]:
    terms = _terms(value)
    priority = {
        term
        for term in terms
        if term not in _LOW_SIGNAL and not term.isdigit() and len(term) > 2
    }
    return priority or {term for term in terms if not term.isdigit()}


def _tsquery(value: str) -> str:
    safe = sorted(term for term in _terms(value) if re.fullmatch(r"[a-z0-9]+", term))
    return " | ".join(term if len(term) <= 3 or term.isdigit() else f"{term}:*" for term in safe[:48]) or "pdfrag_no_match"


def _and_tsquery(terms: Iterable[str], *, limit: int = 5) -> str:
    safe = sorted({term for term in terms if re.fullmatch(r"[a-z0-9]+", term)})[:limit]
    return " & ".join(term if len(term) <= 3 else f"{term}:*" for term in safe) or "pdfrag_no_match"


def _lexical(query: str, source: str) -> float:
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    source_terms = _terms(source[:12000])
    return min(1.0, len(query_terms & source_terms) / max(1, len(query_terms)))


def _weighted_lexical(query: str, source: str) -> float:
    all_score = _lexical(query, source)
    priority = _priority_terms(query)
    if not priority:
        return all_score
    source_terms = _terms(source[:12000])
    priority_score = len(priority & source_terms) / max(1, len(priority))
    return min(1.0, priority_score * 0.74 + all_score * 0.26)


def _money_lookup_query(value: str) -> bool:
    terms = _terms(value)
    return bool(terms & _MONEY_TERMS)


def value_lookup_request(value: str) -> bool:
    """Return True for questions asking for a numeric/measurable value or schedule."""
    terms = _terms(value)
    if terms & _VALUE_DIMENSION_TERMS:
        return True
    lowered = value.casefold()
    return bool(re.search(r"\b(?:how\s+much|how\s+many|at\s+what\s+speed|what\s+speed|what\s+limit|what\s+pressure|what\s+voltage)\b", lowered))


def _version_sensitive_query(value: str) -> bool:
    terms = _terms(value)
    return bool(terms & _VERSION_SENSITIVE_TERMS) or value_lookup_request(value)


def is_amendment_anchor_text(value: str) -> bool:
    folded = " ".join(value.casefold().split())
    return (
        "shall be substituted" in folded
        or "shall be replaced" in folded
        or "shall be omitted" in folded
        or "amendment rules" in folded
        or "amendment regulations" in folded
        or bool(re.search(r"\bamend(?:ed|ment)\b", folded) and re.search(r"\b(?:rule|schedule|section|clause|provision)\b", folded))
    )


def structured_lookup_request(value: str) -> bool:
    """Detect category-to-value/table requests across any measurable dimension."""
    if not value_lookup_request(value):
        return False
    tokens = set(_TOKEN_RE.findall(value.casefold()))
    return bool(tokens & _STRUCTURED_LIST_CUES)


def _has_explicit_value(text_value: str) -> bool:
    return bool(_VALUE_NUMBER_RE.search(text_value) and (_MEASUREMENT_UNIT_RE.search(text_value) or re.search(r"\b\d[\d,]{2,}\b", text_value)))


def retrieval_answerable(query_text: str, results: list[RetrievedChunk]) -> bool:
    """Assess whether the *set* of evidence can answer the requested shape.

    This deliberately reasons across adjacent chunks. PDF tables often split the
    header (``Amount of Compensation``) from the numeric row, so requiring every
    cue in one chunk causes false negatives and favors verbose obsolete prose.
    """
    if not results:
        return False

    aliases = definition_request_aliases(query_text)
    if aliases:
        for alias in aliases:
            if not any(
                "terminology-definition" in item.method
                and definition_for_token(item.chunk.text, alias)
                for item in results[:80]
            ):
                return False
        return True

    if not value_lookup_request(query_text):
        return True

    priority = _priority_terms(query_text)
    grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in results[:120]:
        grouped[item.chunk.document_id or item.chunk.filename.casefold()].append(item)

    for items in grouped.values():
        items.sort(key=lambda item: item.chunk.chunk_index if item.chunk.chunk_index is not None else -1)
        for item in items:
            body = item.chunk.text
            if not _VALUE_NUMBER_RE.search(body):
                continue
            neighborhood = [
                other for other in items
                if abs((other.chunk.chunk_index or 0) - (item.chunk.chunk_index or 0)) <= 2
            ]
            combined = "\n".join(other.chunk.text for other in neighborhood)
            combined_terms = _terms(combined[:24000])
            if not (_MEASUREMENT_UNIT_RE.search(combined) or re.search(r"\b\d[\d,]{2,}\b", body)):
                continue
            dimension_supported = bool(combined_terms & _VALUE_DIMENSION_TERMS) or any(
                "structured-value" in other.method or "answer-shape" in other.method
                for other in neighborhood
            )
            if not dimension_supported:
                continue
            subject_overlap = len(priority & combined_terms)
            semantic_support = max((float(other.vector_score) for other in neighborhood), default=0.0) >= 0.32
            if not priority or subject_overlap >= 1 or semantic_support:
                return True
    return False


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


def definition_evidence_chunks(
    db: Session,
    query_text: str,
    *,
    aliases: Iterable[str] = (),
    limit: int = 12,
) -> list[RetrievedChunk]:
    """Fetch original definition chunks with per-target coverage guarantees.

    A global ORDER BY can otherwise spend the entire limit on one common acronym.
    We query each requested alias independently and keep at least one direct source
    for every resolvable target before adding secondary corroboration.
    """
    requested: list[str] = []
    seen: set[str] = set()
    for alias in (list(aliases) or definition_request_aliases(query_text)):
        clean = " ".join(re.findall(r"[A-Za-z0-9]+", str(alias)))
        norm = clean.casefold()
        if not clean or norm in seen:
            continue
        seen.add(norm)
        requested.append(clean)
    if not requested:
        return []

    primary: list[RetrievedChunk] = []
    secondary: list[RetrievedChunk] = []
    resolved_aliases: set[str] = set()

    for alias in requested[:10]:
        alias_norm = " ".join(re.findall(r"[A-Za-z0-9]+", alias.casefold()))
        rows = list(
            db.execute(
                text(
                    """
                    SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                           c.text, d.filename, t.alias, t.canonical_name, t.confidence, t.verified
                    FROM rag_terminology t
                    JOIN document_chunks c ON c.id = t.chunk_id
                    JOIN documents d ON d.id = c.document_id
                    WHERE d.status = 'ready' AND t.alias_norm = :alias_norm
                    ORDER BY t.verified DESC, t.confidence DESC, lower(d.filename), c.page_number, c.chunk_index
                    LIMIT 4
                    """
                ),
                {"alias_norm": alias_norm},
            ).mappings()
        )
        if rows:
            resolved_aliases.add(alias_norm)
            for position, row in enumerate(rows):
                result = _row_result(
                    row,
                    score=1.0 if bool(row["verified"]) else 0.995,
                    method="direct-terminology-definition",
                    vector_score=0.0,
                    keyword_score=0.99,
                )
                (primary if position == 0 else secondary).append(result)
            continue

        # Backfill may be missing or stale. Search definition-shaped occurrences
        # directly instead of accepting arbitrary usage-only acronym occurrences.
        escaped = re.escape(alias)
        fallback_rows = list(
            db.execute(
                text(
                    """
                    SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                           c.text, d.filename
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE d.status = 'ready'
                      AND (
                        c.text ~* :paren_pattern
                        OR c.text ~* :leading_pattern
                        OR c.text ~* :direct_pattern
                      )
                    ORDER BY
                      CASE WHEN c.text ~* :paren_pattern THEN 0 ELSE 1 END,
                      lower(d.filename), c.page_number, c.chunk_index
                    LIMIT 16
                    """
                ),
                {
                    "paren_pattern": rf"\([[:space:]]*{escaped}[[:space:]]*\)",
                    "leading_pattern": rf"(^|[^A-Za-z0-9]){escaped}[[:space:]]*\(",
                    "direct_pattern": rf"(^|[^A-Za-z0-9]){escaped}[[:space:]]*([-:–—]|means|stands[[:space:]]+for)",
                },
            ).mappings()
        )
        for position, row in enumerate(fallback_rows[:3]):
            result = _row_result(
                row,
                score=0.99,
                method="direct-terminology-definition-fallback",
                vector_score=0.0,
                keyword_score=0.96,
            )
            (primary if position == 0 else secondary).append(result)
            resolved_aliases.add(alias_norm)

    ordered = [*primary, *secondary]
    deduped: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    for item in ordered:
        if item.chunk.chunk_id in seen_ids:
            continue
        seen_ids.add(item.chunk.chunk_id)
        deduped.append(item)
        if len(deduped) >= max(len(primary), min(30, limit)):
            break
    return deduped


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


def _structured_value_rows(
    db: Session,
    query_vector: list[float],
    document_ids: Iterable[str],
    *,
    limit: int = 80,
) -> list[tuple[object, str, float, float]]:
    """Retrieve numeric/table rows semantically inside a small routed document set.

    This is the generic bridge between colloquial wording and formal tables. It does
    not manufacture synonyms: the embedding chooses semantically related source rows,
    then nearby original chunks are included to reconnect split headers and units.
    """
    ids = list(dict.fromkeys(str(value) for value in document_ids if str(value).strip()))[:6]
    if not ids:
        return []
    params: dict[str, object] = {"embedding": str(query_vector), "limit": max(12, min(240, limit))}
    placeholders: list[str] = []
    for index, document_id in enumerate(ids):
        key = f"svdoc{index}"
        params[key] = document_id
        placeholders.append(f"CAST(:{key} AS uuid)")
    doc_filter = ", ".join(placeholders)
    rows = list(
        db.execute(
            text(
                f"""
                WITH ranked AS (
                    SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                           c.text, d.filename,
                           GREATEST(0.0, 1 - (c.embedding <=> CAST(:embedding AS vector))) AS vector_score,
                           ROW_NUMBER() OVER (
                               PARTITION BY c.document_id
                               ORDER BY c.embedding <=> CAST(:embedding AS vector), c.id
                           ) AS doc_rank
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE d.status = 'ready'
                      AND c.document_id IN ({doc_filter})
                      AND (
                        c.content_type = 'table'
                        OR c.text ~ '[0-9][0-9,]*([.][0-9]+)?'
                      )
                )
                SELECT id, document_id, chunk_index, page_number, content_type, text, filename, vector_score
                FROM ranked
                WHERE doc_rank <= 20
                ORDER BY vector_score DESC, id
                LIMIT :limit
                """
            ),
            params,
        ).mappings()
    )
    if not rows:
        return []

    output: list[tuple[object, str, float, float]] = []
    seed_pairs: list[tuple[str, int]] = []
    for row in rows:
        vector_score = float(row["vector_score"] or 0.0)
        body = str(row["text"])
        shape_boost = 0.08 if str(row["content_type"]).casefold() == "table" else 0.0
        if _has_explicit_value(body):
            shape_boost += 0.08
        output.append((row, "smart-structured-value", min(1.0, vector_score + shape_boost), 0.0))
        if vector_score >= 0.24 and len(seed_pairs) < 20:
            seed_pairs.append((str(row["document_id"]), int(row["chunk_index"])))

    # Preserve table headings/units split into neighboring chunks. These neighbors
    # are navigation context; the answer still cites their original PDF text.
    if seed_pairs:
        values_sql = ",".join(f"(CAST(:sd{n} AS uuid), :si{n})" for n, _ in enumerate(seed_pairs))
        neighbor_params: dict[str, object] = {"window": 2}
        for n, (document_id, chunk_index) in enumerate(seed_pairs):
            neighbor_params[f"sd{n}"] = document_id
            neighbor_params[f"si{n}"] = chunk_index
        neighbor_rows = db.execute(
            text(
                f"""
                WITH seeds(document_id, chunk_index) AS (VALUES {values_sql})
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type,
                       c.text, d.filename,
                       MIN(ABS(c.chunk_index - seeds.chunk_index)) AS distance
                FROM seeds
                JOIN document_chunks c ON c.document_id = seeds.document_id
                  AND c.chunk_index BETWEEN seeds.chunk_index - :window AND seeds.chunk_index + :window
                JOIN documents d ON d.id = c.document_id
                GROUP BY c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename
                ORDER BY MIN(ABS(c.chunk_index - seeds.chunk_index)), c.id
                """
            ),
            neighbor_params,
        ).mappings()
        for row in neighbor_rows:
            distance = int(row["distance"] or 0)
            output.append((row, "smart-structured-value-neighbor", max(0.40, 0.72 - 0.10 * distance), 0.0))
    return output


def _authority_directives_for_documents(db: Session, document_ids: Iterable[str]) -> list[object]:
    ids = list(dict.fromkeys(str(value) for value in document_ids if str(value).strip()))[:20]
    if not ids:
        return []
    params: dict[str, object] = {}
    placeholders: list[str] = []
    for index, document_id in enumerate(ids):
        key = f"adoc{index}"
        params[key] = document_id
        placeholders.append(f"CAST(:{key} AS uuid)")
    return list(
        db.execute(
            text(
                f"""
                SELECT ad.*, c.id AS anchor_id, c.content_type AS anchor_content_type,
                       c.text AS anchor_text, d.filename
                FROM rag_authority_directives ad
                JOIN document_chunks c ON c.id = ad.anchor_chunk_id
                JOIN documents d ON d.id = ad.document_id
                WHERE d.status = 'ready' AND ad.document_id IN ({', '.join(placeholders)})
                ORDER BY ad.effective_year DESC NULLS LAST, ad.confidence DESC, ad.id
                """
            ),
            params,
        ).mappings()
    )


def _authority_adjust_results(
    db: Session,
    query_text: str,
    results: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Prefer explicitly current replacement text and quarantine superseded facts.

    Hard precedence is used only for explicit source directives (replace/substitute/
    omit). For section replacement, old-looking rows are penalized only when the
    replacement span itself already contains relevant answer-shaped evidence, so a
    weak heuristic can never erase the only available answer.
    """
    if not results or not _version_sensitive_query(query_text):
        return results
    document_ids = [item.chunk.document_id for item in results if item.chunk.document_id]
    try:
        directives = _authority_directives_for_documents(db, document_ids)
    except Exception:
        return results
    if not directives:
        return results

    by_doc: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in results:
        by_doc[item.chunk.document_id or ""].append(item)

    active_section_directives: set[int] = set()
    current_rows_by_directive: dict[int, list[RetrievedChunk]] = defaultdict(list)
    for directive in directives:
        if str(directive["directive_type"]) != "replace_section":
            continue
        directive_id = int(directive["id"])
        doc_id = str(directive["document_id"])
        start = int(directive["span_start_chunk"])
        end = int(directive["span_end_chunk"])
        for item in by_doc.get(doc_id, []):
            idx = item.chunk.chunk_index if item.chunk.chunk_index is not None else -1
            if not (start <= idx <= end):
                continue
            semantic = float(item.vector_score) >= 0.24 or _weighted_lexical(query_text, item.chunk.text) >= 0.14
            if semantic and (not value_lookup_request(query_text) or _has_explicit_value(item.chunk.text)):
                active_section_directives.add(directive_id)
                current_rows_by_directive[directive_id].append(item)

    relevant_directive_ids: set[int] = set(active_section_directives)
    query_norm = normalize_authority_text(query_text)
    for directive in directives:
        directive_id = int(directive["id"])
        dtype = str(directive["directive_type"])
        doc_id = str(directive["document_id"])
        if dtype == "replace_words":
            old_norm = str(directive["old_norm"] or "")
            new_norm = str(directive["new_norm"] or "")
            if any(
                (old_norm and old_norm in normalize_authority_text(item.chunk.text))
                or (new_norm and new_norm in normalize_authority_text(item.chunk.text))
                for item in by_doc.get(doc_id, [])
            ):
                relevant_directive_ids.add(directive_id)
        elif dtype == "omit":
            target_norm = str(directive["target_norm"] or "")
            if target_norm and (target_norm in query_norm or any(target_norm in normalize_authority_text(item.chunk.text) for item in by_doc.get(doc_id, []))):
                relevant_directive_ids.add(directive_id)

    active_directives = [directive for directive in directives if int(directive["id"]) in relevant_directive_ids]
    if not active_directives:
        return results

    adjusted: list[RetrievedChunk] = []
    anchor_results: list[RetrievedChunk] = []
    for directive in active_directives:
        anchor_results.append(
            _row_result(
                {
                    "id": directive["anchor_id"],
                    "document_id": directive["document_id"],
                    "chunk_index": directive["anchor_chunk_index"],
                    "page_number": directive["page_number"],
                    "content_type": directive["anchor_content_type"],
                    "text": directive["anchor_text"],
                    "filename": directive["filename"],
                },
                score=0.995,
                method="smart-authority-anchor",
                vector_score=0.0,
                keyword_score=0.99,
            )
        )

    for item in results:
        score = float(item.score)
        method = item.method
        doc_id = item.chunk.document_id or ""
        normalized_body = normalize_authority_text(item.chunk.text)
        for directive in active_directives:
            if str(directive["document_id"]) != doc_id:
                continue
            dtype = str(directive["directive_type"])
            if dtype == "replace_words":
                old_norm = str(directive["old_norm"] or "")
                if old_norm and old_norm in normalized_body and item.chunk.chunk_id != str(directive["anchor_id"]):
                    score -= 0.60
                    method += "+superseded-explicit-wording"
                new_norm = str(directive["new_norm"] or "")
                if new_norm and new_norm in normalized_body:
                    score += 0.10
                    method += "+current-explicit-wording"
            elif dtype == "omit":
                target_norm = str(directive["target_norm"] or "")
                if target_norm and target_norm in normalized_body and item.chunk.chunk_id != str(directive["anchor_id"]):
                    score -= 0.55
                    method += "+superseded-omitted-provision"
            elif dtype == "replace_section" and int(directive["id"]) in active_section_directives:
                idx = item.chunk.chunk_index if item.chunk.chunk_index is not None else -1
                start = int(directive["span_start_chunk"])
                end = int(directive["span_end_chunk"])
                if start <= idx <= end:
                    score += 0.20
                    method += "+current-authority-span"
                elif idx > end and _has_explicit_value(item.chunk.text):
                    target_norm = str(directive["target_norm"] or "")
                    in_same_named_section = bool(target_norm and target_norm in normalized_body)
                    competing_similarity = max(
                        (_lexical(current.chunk.text, item.chunk.text) for current in current_rows_by_directive.get(int(directive["id"]), [])),
                        default=0.0,
                    )
                    # Quarantine only a genuinely competing row: either its source
                    # context names the replaced section or it is textually the same
                    # category as a current replacement row but carries another value.
                    if in_same_named_section or competing_similarity >= 0.30:
                        score -= 0.42
                        method += "+superseded-competing-section"
        adjusted.append(replace(item, score=max(0.0, min(1.0, score)), method=method))

    merged: dict[str, RetrievedChunk] = {item.chunk.chunk_id: item for item in adjusted}
    for anchor in anchor_results:
        old = merged.get(anchor.chunk.chunk_id)
        if old is None or anchor.score > old.score:
            merged[anchor.chunk.chunk_id] = anchor
    return list(merged.values())


def _mark_current_amendment_context(
    db: Session,
    collected: dict[str, dict[str, object]],
    merge_row: object,
) -> None:
    """Mark candidates that sit immediately under an explicit amendment/substitution.

    This does not invent revision precedence. It only tags a candidate when the same
    source document contains a nearby preceding chunk that explicitly says amendment
    or that a provision/schedule shall be substituted. The anchor chunk is also added
    so the answer model can cite the actual replacement instruction.
    """
    if not collected:
        return
    chunk_ids = list(collected)[:600]
    params: dict[str, object] = {"window": _int_env("SMART_RAG_AMENDMENT_WINDOW_CHUNKS", 12, 2, 40)}
    placeholders: list[str] = []
    for index, chunk_id in enumerate(chunk_ids):
        key = f"cid{index}"
        params[key] = chunk_id
        placeholders.append(f"CAST(:{key} AS uuid)")
    rows = list(
        db.execute(
            text(
                f"""
                SELECT candidate.id AS candidate_id,
                       anchor.id, anchor.document_id, anchor.chunk_index, anchor.page_number,
                       anchor.content_type, anchor.text, d.filename
                FROM document_chunks candidate
                JOIN documents d ON d.id = candidate.document_id
                JOIN LATERAL (
                    SELECT a.*
                    FROM document_chunks a
                    WHERE a.document_id = candidate.document_id
                      AND a.chunk_index BETWEEN GREATEST(0, candidate.chunk_index - :window) AND candidate.chunk_index
                      AND (
                        lower(a.text) LIKE '%shall be substituted%'
                        OR lower(a.text) LIKE '%amendment rules%'
                        OR lower(a.text) LIKE '%rules, 2025%amendment%'
                      )
                    ORDER BY candidate.chunk_index - a.chunk_index, a.chunk_index DESC
                    LIMIT 1
                ) anchor ON true
                WHERE candidate.id IN ({', '.join(placeholders)})
                """
            ),
            params,
        ).mappings()
    )
    for row in rows:
        candidate_id = str(row["candidate_id"])
        entry = collected.get(candidate_id)
        if entry is None:
            continue
        methods = entry["methods"]
        assert isinstance(methods, set)
        methods.add("current-amendment-context")
        # ``merge_row`` is a local function in fast_search_chunks. Type is kept
        # generic here to avoid introducing a callable protocol only for runtime QA.
        merge_row(row, "smart-amendment-authority-anchor", keyword_score=0.99)  # type: ignore[operator]


def fast_search_chunks(
    db: Session,
    query_vector: list[float],
    query_text: str,
    limit: int,
) -> list[RetrievedChunk]:
    """Bounded hybrid retrieval designed for ~50k+ chunks.

    Global ANN/GIN retrieval is supplemented by a high-signal AND query and by
    amendment-context anchors for version-sensitive lookups. This protects recall
    without restoring the old per-document exhaustive scans.
    """
    vector_k = _int_env("SMART_RAG_VECTOR_K", 90, 10, 500)
    fts_k = _int_env("SMART_RAG_FTS_K", 90, 10, 500)
    scoped_k = _int_env("SMART_RAG_SCOPED_K", 70, 10, 500)
    priority_k = _int_env("SMART_RAG_PRIORITY_FTS_K", 80, 10, 300)
    return_k = min(_int_env("SMART_RAG_MAX_CANDIDATES", 180, 20, 1000), max(20, limit))
    tsq = _tsquery(query_text)
    priority_terms = _priority_terms(query_text)
    priority_tsq = _and_tsquery(priority_terms)
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
    # when the source says "2 or more brakes failed". It does not depend on the
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

    # High-signal conjunction: generic words such as "accident" cannot swamp a
    # rare subject such as "compensation" merely because the broad FTS query is OR.
    if priority_terms:
        rows = db.execute(
            text(
                """
                WITH q AS (SELECT to_tsquery('simple', :tsq) AS query)
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename,
                       LEAST(1.0, ts_rank_cd(to_tsvector('simple', c.text), q.query) * 5.0) AS keyword_score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                CROSS JOIN q
                WHERE d.status = 'ready'
                  AND to_tsvector('simple', c.text) @@ q.query
                ORDER BY keyword_score DESC, c.id
                LIMIT :limit
                """
            ),
            {"tsq": priority_tsq, "limit": priority_k},
        ).mappings()
        for row in rows:
            merge_row(row, "smart-priority-fts", keyword_score=max(0.70, float(row["keyword_score"] or 0.0)))
            routed_docs[str(row["document_id"])] = max(0.92, routed_docs.get(str(row["document_id"]), 0.0))

    # Monetary/table-shaped evidence is explicitly preferred for compensation
    # questions. This is answer-shape retrieval, not an inferred amount.
    if _money_lookup_query(query_text):
        rows = db.execute(
            text(
                """
                WITH q AS (SELECT to_tsquery('simple', :tsq) AS query)
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename,
                       LEAST(1.0, ts_rank_cd(to_tsvector('simple', c.text), q.query) * 5.0) AS keyword_score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                CROSS JOIN q
                WHERE d.status = 'ready'
                  AND to_tsvector('simple', c.text) @@ q.query
                  AND lower(c.text) ~ '(compensation|payable|claim)'
                  AND c.text ~ '[0-9][0-9,]{3,}'
                ORDER BY keyword_score DESC, c.id
                LIMIT :limit
                """
            ),
            {"tsq": _tsquery(query_text), "limit": _int_env("SMART_RAG_AMOUNT_TABLE_K", 80, 10, 250)},
        ).mappings()
        for row in rows:
            merge_row(row, "smart-answer-shape-money", keyword_score=max(0.82, float(row["keyword_score"] or 0.0)))
            routed_docs[str(row["document_id"])] = max(0.96, routed_docs.get(str(row["document_id"]), 0.0))

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

    if value_lookup_request(query_text):
        # Add semantically matched numeric/table rows from the small set of likely
        # documents. This is intentionally independent of exact user vocabulary.
        candidate_docs = list(routed_docs)
        if len(candidate_docs) < 6:
            ranked_docs: dict[str, float] = defaultdict(float)
            for entry in collected.values():
                row = entry["row"]
                doc_id = str(row["document_id"])
                ranked_docs[doc_id] = max(
                    ranked_docs[doc_id],
                    float(entry["vector"]) * 0.65 + float(entry["keyword"]) * 0.35,
                )
            for doc_id, _score in sorted(ranked_docs.items(), key=lambda pair: (-pair[1], pair[0])):
                if doc_id not in candidate_docs:
                    candidate_docs.append(doc_id)
                if len(candidate_docs) >= 6:
                    break
        for row, method, vector_score, keyword_score in _structured_value_rows(
            db,
            query_vector,
            candidate_docs,
            limit=_int_env("SMART_RAG_STRUCTURED_VALUE_K", 96, 24, 240),
        ):
            merge_row(row, method, vector_score=vector_score, keyword_score=keyword_score)
            routed_docs[str(row["document_id"])] = max(0.88, routed_docs.get(str(row["document_id"]), 0.0))

    results: list[RetrievedChunk] = []
    for entry in collected.values():
        row = entry["row"]
        vector_score = float(entry["vector"])
        keyword_score = float(entry["keyword"])
        lexical = _weighted_lexical(query_text, str(row["text"]))  # type: ignore[index]
        document_boost = min(0.15, routed_docs.get(str(row["document_id"]), 0.0) * 0.15)  # type: ignore[index]
        logical_boost = 0.0
        if scenario is not None:
            logical_boost, _ = logical_match_score(scenario, str(row["text"]))  # type: ignore[index]
        methods = sorted(str(value) for value in entry["methods"])
        priority_boost = 0.10 if "smart-priority-fts" in methods else 0.0
        answer_shape_boost = 0.12 if "smart-answer-shape-money" in methods else 0.0
        structured_value_boost = 0.14 if "smart-structured-value" in methods else 0.0
        if "smart-structured-value-neighbor" in methods:
            structured_value_boost = max(structured_value_boost, 0.06)
        amendment_boost = 0.18 if "current-amendment-context" in methods else 0.0
        anchor_boost = 0.18 if "smart-amendment-authority-anchor" in methods else 0.0
        base = vector_score * 0.46 + keyword_score * 0.30 + lexical * 0.24
        score = max(
            0.0,
            min(
                1.0,
                base
                + document_boost
                + logical_boost
                + priority_boost
                + answer_shape_boost
                + structured_value_boost
                + amendment_boost
                + anchor_boost,
            ),
        )
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

    results = _authority_adjust_results(db, query_text, results)

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
    rows = list(
        db.execute(
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
    )
    results = [
        _row_result(
            row,
            score=min(1.0, 0.42 + float(row["keyword_score"] or 0.0) * 0.38 + _weighted_lexical(combined, str(row["text"])) * 0.20),
            method="smart-indexed-corpus-fts",
            vector_score=0.0,
            keyword_score=float(row["keyword_score"] or 0.0),
        )
        for row in rows
    ]

    # Guarantee a small high-signal conjunction set before diversity/capping.
    priority = _priority_terms(combined)
    if priority and not reference_mode:
        priority_rows = db.execute(
            text(
                """
                WITH q AS (SELECT to_tsquery('simple', :tsq) AS query)
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content_type, c.text, d.filename,
                       LEAST(1.0, ts_rank_cd(to_tsvector('simple', c.text), q.query) * 5.0) AS keyword_score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                CROSS JOIN q
                WHERE d.status = 'ready'
                  AND to_tsvector('simple', c.text) @@ q.query
                ORDER BY keyword_score DESC, c.id
                LIMIT :limit
                """
            ),
            {"tsq": _and_tsquery(priority), "limit": min(80, cap)},
        ).mappings()
        for row in priority_rows:
            results.append(
                _row_result(
                    row,
                    score=min(1.0, 0.82 + float(row["keyword_score"] or 0.0) * 0.16),
                    method="smart-indexed-corpus-priority-fts",
                    vector_score=0.0,
                    keyword_score=float(row["keyword_score"] or 0.0),
                )
            )

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
    if not reference_mode:
        results = _authority_adjust_results(db, combined, results)
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
