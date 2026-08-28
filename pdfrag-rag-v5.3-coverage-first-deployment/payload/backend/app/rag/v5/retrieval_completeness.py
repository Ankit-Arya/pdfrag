# IMS_RAG_V53_COVERAGE_FIRST
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.rag.embeddings import embedding_service
from app.rag.types import RetrievedChunk, TextChunk
from app.rag.v5.assistant_retrieval import DocumentRoute, _merge_results, _row_to_result, _vector_rows

_STRUCTURE_RE = re.compile(r"\[PDF STRUCTURE\].*?\[/PDF STRUCTURE\]\s*", re.IGNORECASE | re.DOTALL)
_ALIAS_UPPER_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9/-]{1,11}(?![A-Za-z0-9])")
_ALIAS_EXPLICIT_RE = re.compile(
    r"\b(?:full\s+form|expansion|abbreviation|meaning)\s+(?:of\s+)?(?P<alias>[A-Za-z][A-Za-z0-9/-]{1,11})\b",
    re.IGNORECASE,
)
_WHAT_IS_ALIAS_RE = re.compile(
    r"\bwhat\s+is\s+(?P<alias>[A-Za-z][A-Za-z0-9/-]{1,11})\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_ALIAS_STANDS_RE = re.compile(
    r"\bwhat\s+does\s+(?P<alias>[A-Za-z][A-Za-z0-9/-]{1,11})\s+stand\s+for\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9/&.'-]*")
_IGNORED_INITIAL_WORDS = {"and", "or", "of", "the", "for", "to", "in", "on", "a", "an"}


# v5.3 coverage-first policy helpers. These patterns are intentionally generic and
# source-family based; no rolling-stock number, line number, procedure or acronym is hard-coded.
_RS_SCOPE_RE = re.compile(r"(?<![A-Za-z0-9])RS\s*[-_. ]?\s*(?P<value>\d{1,3}[A-Za-z]?)(?![A-Za-z0-9])", re.IGNORECASE)
_LINE_SCOPE_RE = re.compile(r"(?<![A-Za-z0-9])LINE\s*[-_. ]?\s*(?P<value>\d{1,3}[A-Za-z]?)(?![A-Za-z0-9])", re.IGNORECASE)
_CONDITIONAL_PROCEDURE_RE = re.compile(
    r"\b(?:what\s+(?:will\s+|would\s+)?happens?\s+if|what\s+if|what\s+to\s+do\s+(?:if|when)|"
    r"what\s+should\b.{0,80}\b(?:if|when)|how\s+to|procedure\s+(?:if|when|for)|"
    r"steps?\s+(?:if|when|for)|action\s+(?:if|when|required)|when\s+should|"
    r"if\b.{0,120}\b(?:fail|failed|failure|isolat|unable|not\s+work|not\s+release|not\s+apply|fault))\b",
    re.IGNORECASE | re.DOTALL,
)
_SCOPE_ONLY_RE = re.compile(r"\b(?:only|just|solely|specifically)\b", re.IGNORECASE)
_DEFINITION_CONTEXT_RE = re.compile(
    r"\b(?:abbreviations?|definitions?|glossary|cut[- ]?out\s+cocks?|components?|full\s+form)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class _DocumentState:
    document_id: str
    filename: str
    vector_score: float = 0.0
    keyword_score: float = 0.0
    query_hits: set[int] = field(default_factory=set)
    methods: set[str] = field(default_factory=set)
    candidates: list[RetrievedChunk] = field(default_factory=list)
    best_page: int | None = None
    best_heading: str = ""


@dataclass(slots=True)
class UnifiedDiscovery:
    routes: list[DocumentRoute]
    candidates: list[RetrievedChunk]
    diagnostics: list[dict[str, object]]
    summary: dict[str, object]
    candidate_count: int


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 10000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _float_env(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _norm_alias(value: object) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


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


def aliases_in_question(question: str, *, limit: int = 8) -> list[str]:
    value = _clean(question)
    output: list[str] = []
    seen: set[str] = set()
    for pattern in (_ALIAS_EXPLICIT_RE, _WHAT_IS_ALIAS_RE, _ALIAS_STANDS_RE):
        for match in pattern.finditer(value):
            alias = match.group("alias").upper()
            if len(_norm_alias(alias)) >= 2 and alias not in seen:
                seen.add(alias)
                output.append(alias)
    for match in _ALIAS_UPPER_RE.finditer(value):
        alias = match.group(0).upper()
        if alias not in seen:
            seen.add(alias)
            output.append(alias)
    return output[:limit]


def _initials_match(alias: str, canonical: str) -> bool:
    letters = "".join(ch for ch in alias.upper() if ch.isalpha())
    words = _WORD_RE.findall(canonical)
    if not letters or len(words) < 2:
        return False
    strict = "".join(word[0].upper() for word in words)
    semantic = "".join(word[0].upper() for word in words if word.casefold() not in _IGNORED_INITIAL_WORDS)
    return letters in {strict, semantic}


def _clean_canonical(value: object) -> str:
    cleaned = _clean(value).strip(" -:;,|()[]")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned[:160]


def _matching_prefix(alias: str, value: str) -> str:
    words = _WORD_RE.findall(value)
    for size in range(2, min(9, len(words)) + 1):
        candidate = _clean_canonical(" ".join(words[:size]))
        if _initials_match(alias, candidate):
            return candidate
    return ""


def _extract_alias_expansions(text_value: str, alias: str) -> list[str]:
    body = _STRUCTURE_RE.sub("", str(text_value or ""))
    compact = re.sub(r"\s+", " ", body)
    alias_re = re.escape(alias)
    word = r"[A-Za-z][A-Za-z0-9/&.'-]*"
    long_form = rf"{word}(?:\s+{word}){{1,8}}"
    patterns = [
        re.compile(rf"(?<![A-Za-z0-9]){alias_re}(?![A-Za-z0-9])\s*\(\s*(?P<long>{long_form})\s*\)", re.IGNORECASE),
        re.compile(rf"(?P<long>{long_form})\s*\(\s*{alias_re}\s*\)", re.IGNORECASE),
        re.compile(rf"(?<![A-Za-z0-9]){alias_re}(?![A-Za-z0-9])\s*(?:means|stands\s+for|[:=–—-])\s*(?P<long>{long_form})", re.IGNORECASE),
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(compact):
            canonical = _clean_canonical(match.group("long"))
            if canonical and _initials_match(alias, canonical):
                found.append(canonical)
    token_pattern = re.compile(rf"(?<![A-Za-z0-9]){alias_re}(?![A-Za-z0-9])", re.IGNORECASE)
    for match in token_pattern.finditer(compact):
        tail = compact[match.end(): match.end() + 220].lstrip(" \t:=-–—|")
        if tail.startswith("("):
            continue
        candidate = _matching_prefix(alias, tail)
        if candidate:
            found.append(candidate)
    return _unique(found, 12)


def _metadata(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _cell_expansions(metadata: dict[str, object], alias: str) -> list[str]:
    cells = metadata.get("cells")
    if not isinstance(cells, list):
        return []
    output: list[str] = []
    for index, raw in enumerate(cells):
        cell = _clean(raw)
        if not cell:
            continue
        output.extend(_extract_alias_expansions(cell, alias))
        if _norm_alias(cell) == _norm_alias(alias):
            for neighbor in (index + 1, index - 1):
                if 0 <= neighbor < len(cells):
                    canonical = _clean_canonical(cells[neighbor])
                    if canonical and _initials_match(alias, canonical):
                        output.append(canonical)
    return _unique(output, 12)



def _canonical_scope_value(value: str) -> str:
    clean = _clean(value).upper()
    match = re.fullmatch(r"0*(\d+)([A-Z]?)", clean)
    if not match:
        return clean
    return f"{int(match.group(1))}{match.group(2)}"


def scope_from_text(value: str) -> tuple[str, str]:
    """Return (scope_type, normalized_label) from a filename/question fragment."""
    rs = _RS_SCOPE_RE.search(str(value or ""))
    if rs:
        token = _canonical_scope_value(rs.group("value"))
        return "rs", f"RS-{token}"
    line = _LINE_SCOPE_RE.search(str(value or ""))
    if line:
        token = _canonical_scope_value(line.group("value"))
        return "line", f"Line-{token}"
    return "common", "Common/Other"


def explicit_scope_labels(question: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    value = str(question or "")
    for pattern, prefix in ((_RS_SCOPE_RE, "RS-"), (_LINE_SCOPE_RE, "Line-")):
        for match in pattern.finditer(value):
            label = prefix + _canonical_scope_value(match.group("value"))
            if label.casefold() not in seen:
                seen.add(label.casefold())
                output.append(label)
    return output


def explicit_scope_only(question: str) -> bool:
    return bool(explicit_scope_labels(question) and _SCOPE_ONLY_RE.search(str(question or "")))


def is_conditional_procedure_query(question: str, interpretation: object | None = None) -> bool:
    if _CONDITIONAL_PROCEDURE_RE.search(str(question or "")):
        return True
    intent = str(getattr(interpretation, "intent", "") or "").casefold()
    return intent in {"procedure", "troubleshooting", "requirement", "comparison", "list"}


def _active_documents(db: Session) -> list[dict[str, str]]:
    rows = db.execute(text("""
        SELECT d.id, d.filename
        FROM documents d
        WHERE d.status='ready'
          AND EXISTS (
              SELECT 1 FROM rag_v5_processing_runs r
              WHERE r.document_id=d.id AND r.is_active=true AND r.status='ready'
          )
        ORDER BY lower(d.filename), d.id
    """)).mappings()
    return [{"document_id": str(row["id"]), "filename": str(row["filename"])} for row in rows]


def _indexed_alias_canonicals(db: Session, alias: str) -> list[str]:
    try:
        rows = db.execute(text("""
            SELECT DISTINCT t.canonical_name
            FROM rag_v5_terminology t
            JOIN rag_v5_processing_runs r
              ON r.id=t.run_id AND r.is_active=true AND r.status='ready'
            WHERE t.alias_norm=:alias_norm
              AND COALESCE(t.canonical_name,'') <> ''
            ORDER BY t.canonical_name
        """), {"alias_norm": _norm_alias(alias)}).scalars().all()
    except Exception:
        return []
    return _unique((str(value) for value in rows if value), 40)


def _ordered_tokens_present(tokens: list[str], wanted: list[str], *, max_span: int = 90) -> bool:
    if not wanted or not tokens:
        return False
    start = -1
    cursor = 0
    for wanted_token in wanted:
        found = -1
        for index in range(cursor, len(tokens)):
            if tokens[index] == wanted_token:
                found = index
                break
        if found < 0:
            return False
        if start < 0:
            start = found
        cursor = found + 1
    return start >= 0 and (cursor - 1 - start) <= max_span


def _fuzzy_known_expansions(
    text_value: str,
    alias: str,
    canonicals: Sequence[str],
    *,
    context: str = "",
) -> list[str]:
    """Recover a known expansion from a table-flattened definition.

    Example: an OCR/table flattening can yield `Bic ... (Brake ... Isolation Cock)`.
    We only accept a canonical that is already source-grounded elsewhere or in the
    terminology index, whose initials match the alias, and whose words occur in order
    close to the alias in a definition-like context. This avoids inventing expansions.
    """
    body = _STRUCTURE_RE.sub("", str(text_value or ""))
    alias_re = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
    context_ok = bool(_DEFINITION_CONTEXT_RE.search(context))
    output: list[str] = []
    for match in alias_re.finditer(body):
        window = body[max(0, match.start() - 180): match.end() + 900]
        punctuation_hint = "(" in window or ":" in window or "=" in window
        if not (context_ok or punctuation_hint):
            continue
        tokens = [token.casefold() for token in _WORD_RE.findall(window)]
        for canonical in canonicals:
            clean = _clean_canonical(canonical)
            if not clean or not _initials_match(alias, clean):
                continue
            wanted = [token.casefold() for token in _WORD_RE.findall(clean)]
            if _ordered_tokens_present(tokens, wanted, max_span=80):
                output.append(clean)
    return _unique(output, 20)

def _active_document_count(db: Session) -> int:
    return int(db.execute(text("""
        SELECT COUNT(*) FROM documents d
        WHERE d.status='ready' AND EXISTS (
            SELECT 1 FROM rag_v5_processing_runs r
            WHERE r.document_id=d.id AND r.is_active=true AND r.status='ready'
        )
    """)).scalar_one() or 0)


def _alias_rows(db: Session, alias: str, *, per_document: int, total_limit: int) -> list[object]:
    return list(db.execute(text("""
        WITH hits AS (
            SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                   c.content_type, c.parent_key, c.section_path, c.heading,
                   c.authority_status, c.metadata, c.text, d.filename,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.document_id
                       ORDER BY CASE WHEN c.content_type='table_row' THEN 0 ELSE 1 END,
                                CASE WHEN lower(COALESCE(c.heading,'')) LIKE '%abbrev%' THEN 0
                                     WHEN lower(COALESCE(c.heading,'')) LIKE '%cut-out%' THEN 0
                                     WHEN lower(COALESCE(c.heading,'')) LIKE '%cut out%' THEN 0
                                     ELSE 1 END,
                                c.page_number, c.chunk_index
                   ) AS doc_rank
            FROM rag_v5_chunks c
            JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
            JOIN documents d ON d.id=c.document_id AND d.status='ready'
            WHERE to_tsvector('simple', c.text) @@ plainto_tsquery('simple', :alias)
        )
        SELECT id, document_id, chunk_index, page_number, page_end, content_type,
               parent_key, section_path, heading, authority_status, metadata, text, filename
        FROM hits WHERE doc_rank <= :per_document
        ORDER BY lower(filename), page_number, chunk_index
        LIMIT :total_limit
    """), {"alias": alias, "per_document": per_document, "total_limit": total_limit}).mappings())


def _definition_rows(db: Session, question: str) -> list[tuple[str, object, list[str]]]:
    aliases = aliases_in_question(question)
    if not aliases:
        return []
    per_document = _int_env("RAG_V53_ALIAS_ROWS_PER_DOCUMENT", 80, 12, 200)
    total_scan = _int_env("RAG_V53_ALIAS_SCAN_ROWS", 7000, 500, 20000)
    output: list[tuple[str, object, list[str]]] = []

    for alias in aliases:
        rows = _alias_rows(db, alias, per_document=per_document, total_limit=total_scan)
        clean_by_row: list[tuple[object, list[str]]] = []
        known: list[str] = list(_indexed_alias_canonicals(db, alias))

        # Pass 1: collect clean explicit definitions from every source occurrence.
        for row in rows:
            metadata = _metadata(row["metadata"])
            meanings = _unique(
                [*_extract_alias_expansions(str(row["text"]), alias), *_cell_expansions(metadata, alias)],
                20,
            )
            meanings = [value for value in meanings if _initials_match(alias, value)]
            clean_by_row.append((row, meanings))
            known.extend(meanings)
        known = _unique(known, 60)

        # Pass 2: use only already source-grounded canonical meanings to recover
        # table-flattened occurrences (words may be interleaved by neighboring cells).
        seen_occurrences: set[tuple[str, str, str, int]] = set()
        for row, meanings in clean_by_row:
            metadata = _metadata(row["metadata"])
            section_path = row["section_path"] or []
            if not isinstance(section_path, list):
                section_path = []
            context = " ".join([
                str(row["heading"] or ""),
                " ".join(str(item) for item in section_path),
                str(metadata.get("table_title") or ""),
            ])
            recovered = _fuzzy_known_expansions(
                str(row["text"]), alias, known, context=context
            )
            accepted = _unique([*meanings, *recovered], 20)
            accepted = [value for value in accepted if _initials_match(alias, value)]
            if not accepted:
                continue

            # Preserve one occurrence per meaning/document/page. Identical meanings in
            # different PDFs are intentionally NOT collapsed; the answer layer must cite
            # every applicable source location.
            page_number = int(row["page_number"])
            per_row: list[str] = []
            for canonical in accepted:
                key = (
                    _norm_alias(alias),
                    canonical.casefold(),
                    str(row["document_id"]),
                    page_number,
                )
                if key in seen_occurrences:
                    continue
                seen_occurrences.add(key)
                per_row.append(canonical)
            if per_row:
                output.append((alias, row, per_row))
    return output


def definition_inventory(
    db: Session,
    question: str,
    results: Sequence[RetrievedChunk] = (),
) -> list[dict[str, object]]:
    """Return distinct grounded meanings and all definition source locations.

    When retrieval results are supplied, inventory is built from those already-retrieved
    chunks so diagnostics do not trigger a second corpus alias scan.
    """
    aliases = aliases_in_question(question)
    if not aliases:
        return []

    grouped: dict[tuple[str, str], dict[str, object]] = {}
    seen_sources: defaultdict[tuple[str, str], set[tuple[str, int, int]]] = defaultdict(set)

    if results:
        known_by_alias: dict[str, list[str]] = {
            alias: list(_indexed_alias_canonicals(db, alias)) for alias in aliases
        }
        # First pass collects clean definitions from the already-retrieved chunks.
        for item in results:
            for alias in aliases:
                known_by_alias[alias].extend(
                    _extract_alias_expansions(item.chunk.text, alias)
                )
        for alias in aliases:
            known_by_alias[alias] = _unique(known_by_alias[alias], 80)

        # Second pass maps every clean/fuzzy-known meaning back to its source chunk.
        for result in results:
            chunk = result.chunk
            context = " ".join([
                chunk.heading or "",
                " ".join(chunk.section_path),
                chunk.content_type or "",
            ])
            for alias in aliases:
                meanings = _unique(
                    [
                        *_extract_alias_expansions(chunk.text, alias),
                        *_fuzzy_known_expansions(
                            chunk.text, alias, known_by_alias[alias], context=context
                        ),
                    ],
                    20,
                )
                for canonical in meanings:
                    if not _initials_match(alias, canonical):
                        continue
                    key = (_norm_alias(alias), canonical.casefold())
                    definition = grouped.setdefault(
                        key, {"alias": alias, "meaning": canonical, "sources": []}
                    )
                    page_start = int(chunk.page_number)
                    page_end = int(chunk.page_end or page_start)
                    source_key = (chunk.filename.casefold(), page_start, page_end)
                    if source_key in seen_sources[key]:
                        continue
                    seen_sources[key].add(source_key)
                    source_list = definition["sources"]
                    if isinstance(source_list, list):
                        source_list.append({
                            "filename": chunk.filename,
                            "page_start": page_start,
                            "page_end": page_end,
                            "heading": chunk.heading or "",
                        })
    else:
        # Fallback for diagnostics callers that do not already have exact results.
        for alias, row, meanings in _definition_rows(db, question):
            page_start = int(row["page_number"])
            page_end = int(row["page_end"] or page_start)
            filename = str(row["filename"])
            heading = str(row["heading"] or "")
            for canonical in meanings:
                key = (_norm_alias(alias), canonical.casefold())
                definition = grouped.setdefault(
                    key, {"alias": alias, "meaning": canonical, "sources": []}
                )
                source_key = (filename.casefold(), page_start, page_end)
                if source_key in seen_sources[key]:
                    continue
                seen_sources[key].add(source_key)
                source_list = definition["sources"]
                if isinstance(source_list, list):
                    source_list.append({
                        "filename": filename,
                        "page_start": page_start,
                        "page_end": page_end,
                        "heading": heading,
                    })

    output = list(grouped.values())
    output.sort(
        key=lambda item: (
            str(item.get("alias") or "").casefold(),
            str(item.get("meaning") or "").casefold(),
        )
    )
    return output[:120]


def exact_alias_results(db: Session, question: str, *, limit: int = 120) -> list[RetrievedChunk]:
    aliases = aliases_in_question(question)
    if not aliases:
        return []

    output: list[RetrievedChunk] = []
    # Existing source-derived terminology index remains a first-class arm.
    try:
        from app.rag.v5.terminology import definition_source_rows
        for row in definition_source_rows(db, question, limit=limit):
            output.append(_row_to_result(
                row,
                score=1.0,
                method="v5.3-indexed-alias-definition",
                keyword=1.0,
            ))
    except Exception:
        pass

    # Independent runtime scan recovers definitions that indexing missed because of
    # table layout/OCR case degradation such as BIC -> Bic.
    seen_chunks = {item.chunk.chunk_id for item in output}
    for alias, row, meanings in _definition_rows(db, question):
        chunk_id = str(row["id"])
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        section_path = row["section_path"] or []
        if not isinstance(section_path, list):
            section_path = []
        page_number = int(row["page_number"])
        output.append(RetrievedChunk(
            chunk=TextChunk(
                chunk_id=chunk_id, filename=str(row["filename"]), page_number=page_number,
                page_end=int(row["page_end"] or page_number), text=str(row["text"]),
                content_type=str(row["content_type"]), section_path=tuple(str(item) for item in section_path),
                heading=str(row["heading"] or ""), document_id=str(row["document_id"]),
                chunk_index=int(row["chunk_index"]),
            ),
            score=1.0, method="v5.3-exact-alias-definition", vector_score=0.0, keyword_score=1.0,
        ))
        if len(output) >= limit:
            break
    return _merge_results(output)[:limit]


def complete_terminology_hints(db: Session, question: str, *, limit: int = 120) -> list[str]:
    from app.rag.v5.terminology import terminology_hints
    exact: list[str] = []
    for alias, row, meanings in _definition_rows(db, question):
        for canonical in meanings:
            exact.append(f"{alias} = {canonical} (PDF: {row['filename']}, page {row['page_number']})")
    return _unique([*terminology_hints(db, question, limit=max(limit, 120)), *exact], limit)

def _per_document_fts_rows(db: Session, query: str, *, per_document: int, max_rows: int) -> list[object]:
    return list(db.execute(text("""
        WITH q AS (
            SELECT websearch_to_tsquery('simple', :query) AS simple_q,
                   websearch_to_tsquery('english', :query) AS english_q
        ), hits AS (
            SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                   c.content_type, c.parent_key, c.section_path, c.heading,
                   c.authority_status, c.text, d.filename,
                   GREATEST(ts_rank_cd(to_tsvector('simple', c.text), q.simple_q),
                            ts_rank_cd(to_tsvector('english', c.text), q.english_q) * 0.9) AS keyword_score
            FROM rag_v5_chunks c
            JOIN rag_v5_processing_runs r ON r.id=c.run_id AND r.is_active=true AND r.status='ready'
            JOIN documents d ON d.id=c.document_id AND d.status='ready'
            CROSS JOIN q
            WHERE to_tsvector('simple', c.text) @@ q.simple_q
               OR to_tsvector('english', c.text) @@ q.english_q
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY document_id ORDER BY keyword_score DESC, page_number, chunk_index
            ) AS doc_rank
            FROM hits
        )
        SELECT id, document_id, chunk_index, page_number, page_end, content_type,
               parent_key, section_path, heading, authority_status, text, filename, keyword_score
        FROM ranked WHERE doc_rank <= :per_document
        ORDER BY keyword_score DESC, lower(filename), page_number, chunk_index
        LIMIT :max_rows
    """), {"query": query, "per_document": per_document, "max_rows": max_rows}).mappings())


def _state_for(states: dict[str, _DocumentState], document_id: str, filename: str) -> _DocumentState:
    state = states.get(document_id)
    if state is None:
        state = _DocumentState(document_id=document_id, filename=filename)
        states[document_id] = state
    return state


def _record_candidate(state: _DocumentState, item: RetrievedChunk, *, method: str, query_index: int) -> None:
    state.methods.add(method)
    state.candidates.append(item)
    current_best = max((candidate.score for candidate in state.candidates[:-1]), default=-1.0)
    if state.best_page is None or item.score > current_best:
        state.best_page = item.chunk.page_number
        state.best_heading = item.chunk.heading or (item.chunk.section_path[-1] if item.chunk.section_path else "")


def _route_score(state: _DocumentState, query_count: int) -> float:
    signal = max(state.vector_score, state.keyword_score)
    coverage = min(1.0, len(state.query_hits) / max(1, min(4, query_count)))
    arm_agreement = 1.0 if state.vector_score > 0 and state.keyword_score > 0 else 0.0
    return min(1.0, 0.58 * signal + 0.30 * coverage + 0.12 * arm_agreement)


def unified_corpus_discovery(
    db: Session,
    *,
    queries: Sequence[str],
    prior_results: Sequence[RetrievedChunk] = (),
    question: str = "",
    interpretation: object | None = None,
    coverage_mode: bool | None = None,
) -> UnifiedDiscovery:
    """One corpus-discovery stage with per-document and per-scope coverage.

    In conditional/procedural mode the ordinary global route cap may limit extra
    common documents, but it may not silently eliminate an RS/Line scope that has
    materially relevant evidence. Explicitly named RS/Line scopes are pinned.
    """
    query_limit = _int_env("RAG_V53_DISCOVERY_QUERY_COUNT", 8, 3, 12)
    route_queries = [_clean(query) for query in queries[:query_limit] if _clean(query)]
    eligible_documents = _active_document_count(db)
    coverage_mode = is_conditional_procedure_query(question, interpretation) if coverage_mode is None else bool(coverage_mode)
    explicit_scopes = explicit_scope_labels(question)
    only_explicit = explicit_scope_only(question)

    empty_summary = {
        "eligible_documents": eligible_documents,
        "documents_with_signal": 0,
        "routed_documents": 0,
        "corpus_discovery_stages": 1,
        "query_variants": len(route_queries),
        "no_signal_documents": eligible_documents,
        "diagnostics_truncated": False,
        "coverage_mode": coverage_mode,
        "explicit_scopes": explicit_scopes,
        "scope_count_with_signal": 0,
        "scope_count_routed": 0,
        "scope_promoted_documents": 0,
        "global_route_cap": _int_env("RAG_V53_MAX_RELEVANT_DOCUMENTS", 24, 4, 40),
    }
    if not route_queries:
        return UnifiedDiscovery([], list(prior_results), [], empty_summary, len(prior_results))

    states: dict[str, _DocumentState] = {}
    all_candidates: list[RetrievedChunk] = list(prior_results)
    for prior in prior_results:
        doc_id = prior.chunk.document_id or prior.chunk.filename
        state = _state_for(states, doc_id, prior.chunk.filename)
        state.vector_score = max(state.vector_score, float(prior.vector_score))
        state.keyword_score = max(state.keyword_score, float(prior.keyword_score))
        state.query_hits.add(0)
        state.methods.add("prior_round")
        state.candidates.append(prior)
        state.best_page = prior.chunk.page_number
        state.best_heading = prior.chunk.heading or (prior.chunk.section_path[-1] if prior.chunk.section_path else "")

    try:
        vectors = embedding_service.encode(route_queries)
    except Exception:
        vectors = []

    vector_per_query = _int_env("RAG_V53_DISCOVERY_VECTOR_PER_QUERY", 180, 60, 500)
    fts_per_document = _int_env("RAG_V53_DISCOVERY_FTS_PER_DOCUMENT", 2, 1, 6)
    fts_max_rows = _int_env("RAG_V53_DISCOVERY_FTS_MAX_ROWS", 2200, 200, 7000)

    for query_index, query in enumerate(route_queries):
        if query_index < len(vectors):
            try:
                vector_rows = _vector_rows(db, vectors[query_index].tolist(), vector_per_query)
            except Exception:
                vector_rows = []
            for rank, row in enumerate(vector_rows, 1):
                signal = max(0.0, min(1.0, float(row["vector_score"] or 0.0)))
                item = _row_to_result(
                    row,
                    score=min(1.0, 0.82 * signal + 0.18 / (1.0 + rank / 20.0)),
                    method=f"v5.3-discovery-vector-{query_index}",
                    vector=signal,
                )
                doc_id = str(row["document_id"])
                state = _state_for(states, doc_id, str(row["filename"]))
                state.vector_score = max(state.vector_score, signal)
                if signal >= _float_env("RAG_V53_VECTOR_HIT", 0.36, 0.12, 0.90):
                    state.query_hits.add(query_index)
                _record_candidate(state, item, method="vector", query_index=query_index)
                all_candidates.append(item)

        try:
            fts_rows = _per_document_fts_rows(
                db, query, per_document=fts_per_document, max_rows=fts_max_rows
            )
        except Exception:
            fts_rows = []
        for rank, row in enumerate(fts_rows, 1):
            raw = max(0.0, float(row["keyword_score"] or 0.0))
            signal = min(1.0, raw * 4.5)
            item = _row_to_result(
                row,
                score=min(1.0, 0.86 * signal + 0.14 / (1.0 + rank / 50.0)),
                method=f"v5.3-discovery-fts-{query_index}",
                keyword=signal,
            )
            doc_id = str(row["document_id"])
            state = _state_for(states, doc_id, str(row["filename"]))
            state.keyword_score = max(state.keyword_score, signal)
            if signal >= _float_env("RAG_V53_KEYWORD_HIT", 0.12, 0.01, 0.90):
                state.query_hits.add(query_index)
            _record_candidate(state, item, method="fts", query_index=query_index)
            all_candidates.append(item)

    # Explicitly named scopes are resolved against the active document census even if
    # the first global vector/FTS arms did not surface them. This prevents a route cap
    # or crowd-out from excluding the user's named rolling stock/line before deep search.
    forced_explicit_ids: set[str] = set()
    if explicit_scopes:
        explicit_set = {label.casefold() for label in explicit_scopes}
        for row in _active_documents(db):
            scope_type, scope_label = scope_from_text(row["filename"])
            if scope_label.casefold() not in explicit_set:
                continue
            doc_id = row["document_id"]
            state = _state_for(states, doc_id, row["filename"])
            state.methods.add("explicit_scope")
            forced_explicit_ids.add(doc_id)

    merged_all = _merge_results(all_candidates)
    if not states:
        return UnifiedDiscovery([], merged_all, [], empty_summary, len(merged_all))

    scored = {doc_id: _route_score(state, len(route_queries)) for doc_id, state in states.items()}
    for doc_id in forced_explicit_ids:
        scored[doc_id] = max(scored.get(doc_id, 0.0), 1.0)

    best_score = max(scored.values(), default=1.0) or 1.0
    min_route = _float_env("RAG_V53_DOCUMENT_MIN_SCORE", 0.23, 0.05, 0.80)
    relative = _float_env("RAG_V53_DOCUMENT_RELATIVE_THRESHOLD", 0.18, 0.05, 0.80)
    min_signal = _float_env("RAG_V53_DOCUMENT_MIN_SIGNAL", 0.20, 0.03, 0.90)
    max_docs = _int_env("RAG_V53_MAX_RELEVANT_DOCUMENTS", 24, 4, 40)
    max_coverage_docs = _int_env("RAG_V53_MAX_COVERAGE_DOCUMENTS", 48, 8, 80)
    docs_per_scope = _int_env("RAG_V53_DOCS_PER_SCOPE", 2, 1, 5)
    scope_min_score = _float_env("RAG_V53_SCOPE_MIN_SCORE", 0.14, 0.03, 0.75)
    scope_min_signal = _float_env("RAG_V53_SCOPE_MIN_SIGNAL", 0.10, 0.01, 0.80)

    ordered_ids = sorted(
        states,
        key=lambda doc_id: (-scored[doc_id], states[doc_id].filename.casefold(), doc_id),
    )
    route_rank = {doc_id: index for index, doc_id in enumerate(ordered_ids, 1)}

    regular_selected: list[str] = []
    for doc_id in ordered_ids:
        state = states[doc_id]
        score = scored[doc_id]
        ratio = score / best_score
        strongest = max(state.vector_score, state.keyword_score)
        if doc_id in forced_explicit_ids or (
            (score >= min_route or ratio >= relative)
            and (strongest >= min_signal or len(state.query_hits) >= 2)
        ):
            regular_selected.append(doc_id)
        if len(regular_selected) >= max_docs:
            break

    promoted_scope_ids: set[str] = set()
    scope_signal_labels: set[str] = set()
    if coverage_mode:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for doc_id in ordered_ids:
            scope_type, scope_label = scope_from_text(states[doc_id].filename)
            if scope_type not in {"rs", "line"}:
                continue
            strongest = max(states[doc_id].vector_score, states[doc_id].keyword_score)
            score = scored[doc_id]
            has_signal = strongest >= scope_min_signal or bool(states[doc_id].query_hits)
            if has_signal:
                scope_signal_labels.add(scope_label)
            if doc_id in forced_explicit_ids or (score >= scope_min_score and has_signal):
                grouped[scope_label].append(doc_id)

        for scope_label in sorted(grouped, key=str.casefold):
            candidates = sorted(
                grouped[scope_label],
                key=lambda doc_id: (-scored[doc_id], states[doc_id].filename.casefold()),
            )
            for doc_id in candidates[:docs_per_scope]:
                promoted_scope_ids.add(doc_id)

    selected_ids: list[str] = []
    # Named scopes first, then coverage-promoted RS/Line scopes, then ordinary common/extra routes.
    for doc_id in ordered_ids:
        if doc_id in forced_explicit_ids and doc_id not in selected_ids:
            selected_ids.append(doc_id)
    if not only_explicit:
        for doc_id in ordered_ids:
            if doc_id in promoted_scope_ids and doc_id not in selected_ids:
                selected_ids.append(doc_id)
        for doc_id in regular_selected:
            if doc_id not in selected_ids:
                selected_ids.append(doc_id)

    if only_explicit and forced_explicit_ids:
        selected_ids = [doc_id for doc_id in selected_ids if doc_id in forced_explicit_ids]

    if len(selected_ids) > max_coverage_docs:
        explicit_part = [doc_id for doc_id in selected_ids if doc_id in forced_explicit_ids]
        remaining = [doc_id for doc_id in selected_ids if doc_id not in forced_explicit_ids]
        selected_ids = [*explicit_part, *remaining[: max(0, max_coverage_docs - len(explicit_part))]]

    if not selected_ids and ordered_ids:
        selected_ids.append(ordered_ids[0])

    selected_set = set(selected_ids)
    routes = [
        DocumentRoute(
            document_id=doc_id,
            filename=states[doc_id].filename,
            score=max(0.0, min(1.0, scored[doc_id])),
        )
        for doc_id in selected_ids
    ]
    routed_candidates = _merge_results(
        item for doc_id in selected_ids for item in states[doc_id].candidates
    )

    diagnostics: list[dict[str, object]] = []
    for doc_id in ordered_ids:
        state = states[doc_id]
        score = scored[doc_id]
        routed = doc_id in selected_set
        strongest = max(state.vector_score, state.keyword_score)
        scope_type, scope_label = scope_from_text(state.filename)
        explicit_pinned = doc_id in forced_explicit_ids
        scope_pinned = doc_id in promoted_scope_ids

        if explicit_pinned and routed:
            decision = "EXPLICIT_SCOPE_PINNED"
            reason = "The user named this RS/Line scope, so it is exempt from the ordinary global route cap and is deep-searched."
        elif scope_pinned and routed and doc_id not in regular_selected:
            decision = "SCOPE_COVERAGE_PINNED"
            reason = "This RS/Line scope has relevant evidence and is retained for cross-scope coverage even though it fell outside the ordinary global route cap."
        elif routed and len(state.query_hits) >= 2:
            decision = "ROUTED_MULTI_DIMENSION"
            reason = "Passed unified discovery with matching evidence across multiple query dimensions."
        elif routed:
            decision = "ROUTED_STRONG_SIGNAL"
            reason = "Passed unified discovery using the document's strongest lexical/semantic evidence."
        elif strongest < min_signal:
            decision = "BELOW_MIN_SIGNAL"
            reason = "Retrieval signal was present but below the minimum evidence-signal threshold."
        elif score < min_route and score / best_score < relative:
            decision = "BELOW_ROUTE_THRESHOLD"
            reason = "Evidence did not pass the absolute or relative document-routing threshold."
        else:
            decision = "ROUTE_LIMIT"
            reason = "Document had a usable signal but was not required for RS/Line coverage and fell outside the ordinary global route cap."

        diagnostics.append({
            "document_id": doc_id,
            "filename": state.filename,
            "discovery_score": round(score, 4),
            "vector_score": round(state.vector_score, 4),
            "keyword_score": round(state.keyword_score, 4),
            "dimension_hits": len(state.query_hits),
            "signals": sorted(state.methods),
            "routed": routed,
            "deep_searched": False,
            "rerank_role": "",
            "final_evidence": False,
            "contributing": False,
            "decision": decision,
            "reason": reason,
            "best_page": state.best_page,
            "best_heading": state.best_heading,
            "scope_type": scope_type,
            "scope_label": scope_label,
            "scope_pinned": bool(explicit_pinned or scope_pinned),
            "explicit_scope": explicit_pinned,
            "route_rank": route_rank.get(doc_id),
            "global_route_cap": max_docs,
        })

    diagnostic_limit = _int_env("RAG_V53_DIAGNOSTIC_MAX_DOCS", 300, 20, 1500)
    total_signal = len(diagnostics)
    truncated = total_signal > diagnostic_limit
    diagnostics = diagnostics[:diagnostic_limit]
    routed_scope_labels = {
        scope_from_text(states[doc_id].filename)[1]
        for doc_id in selected_ids
        if scope_from_text(states[doc_id].filename)[0] in {"rs", "line"}
    }
    summary = {
        "eligible_documents": eligible_documents,
        "documents_with_signal": total_signal,
        "routed_documents": len(routes),
        "corpus_discovery_stages": 1,
        "query_variants": len(route_queries),
        "no_signal_documents": max(0, eligible_documents - total_signal),
        "diagnostics_truncated": truncated,
        "diagnostic_documents_included": len(diagnostics),
        "coverage_mode": coverage_mode,
        "explicit_scopes": explicit_scopes,
        "scope_count_with_signal": len(scope_signal_labels),
        "scope_count_routed": len(routed_scope_labels),
        "scope_promoted_documents": len(promoted_scope_ids),
        "global_route_cap": max_docs,
    }
    return UnifiedDiscovery(routes, routed_candidates, diagnostics, summary, len(merged_all))


def direct_retrieval_diagnostics(
    db: Session,
    *,
    results: Sequence[RetrievedChunk],
    routed_documents: Sequence[str],
    question: str = "",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    eligible = _active_document_count(db)
    routed = {name.casefold() for name in routed_documents}
    by_filename: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in results:
        by_filename[item.chunk.filename].append(item)
    diagnostics: list[dict[str, object]] = []
    for filename, items in sorted(by_filename.items(), key=lambda pair: (-max(item.score for item in pair[1]), pair[0].casefold())):
        best = max(items, key=lambda item: item.score)
        exact = any("alias-definition" in item.method for item in items)
        diagnostics.append({
            "document_id": best.chunk.document_id or "",
            "filename": filename,
            "discovery_score": round(float(best.score), 4),
            "vector_score": round(max(float(item.vector_score) for item in items), 4),
            "keyword_score": round(max(float(item.keyword_score) for item in items), 4),
            "dimension_hits": 1,
            "signals": ["exact_alias"] if exact else ["focused_retrieval"],
            "routed": filename.casefold() in routed or exact,
            "deep_searched": filename.casefold() in routed,
            "rerank_role": "definition" if exact else "",
            "final_evidence": True,
            "contributing": False,
            "decision": "EXACT_ALIAS_DEFINITION" if exact else "FOCUSED_EVIDENCE",
            "reason": "Explicit source-grounded alias definition." if exact else "Selected by focused direct-lookup retrieval.",
            "best_page": best.chunk.page_number,
            "best_heading": best.chunk.heading or (best.chunk.section_path[-1] if best.chunk.section_path else ""),
            "scope_type": scope_from_text(filename)[0],
            "scope_label": scope_from_text(filename)[1],
            "scope_pinned": exact,
            "explicit_scope": False,
            "route_rank": None,
            "global_route_cap": None,
        })
    definition_documents = _unique(
        (str(item["filename"]) for item in diagnostics if item["decision"] == "EXACT_ALIAS_DEFINITION"),
        120,
    )
    inventory = definition_inventory(db, question, results) if question and definition_documents else []
    definition_meanings = _unique(
        (str(item.get("meaning") or "") for item in inventory),
        120,
    )
    definition_locations: list[dict[str, object]] = []
    for item in inventory:
        alias = str(item.get("alias") or "")
        meaning = str(item.get("meaning") or "")
        raw_sources = item.get("sources")
        if not isinstance(raw_sources, list):
            continue
        for source in raw_sources:
            if not isinstance(source, dict):
                continue
            definition_locations.append({
                "alias": alias,
                "meaning": meaning,
                "filename": str(source.get("filename") or ""),
                "page_start": int(source.get("page_start") or 0),
                "page_end": int(source.get("page_end") or source.get("page_start") or 0),
                "heading": str(source.get("heading") or ""),
            })
    summary = {
        "eligible_documents": eligible,
        "documents_with_signal": len(diagnostics),
        "routed_documents": sum(1 for item in diagnostics if item["routed"]),
        "corpus_discovery_stages": 1,
        "query_variants": 1,
        "no_signal_documents": max(0, eligible - len(diagnostics)),
        "diagnostics_truncated": False,
        "diagnostic_documents_included": len(diagnostics),
        "definition_enumeration": bool(definition_documents),
        "definition_documents": definition_documents,
        "definition_meanings": definition_meanings,
        "definition_inventory": inventory,
        "definition_locations": definition_locations,
        "definition_source_count": len(definition_locations) or sum(
            1 for item in diagnostics if item["decision"] == "EXACT_ALIAS_DEFINITION"
        ),
    }
    return diagnostics, summary


def finalize_retrieval_diagnostics(
    diagnostics: Sequence[dict[str, object]],
    *,
    routes: Sequence[DocumentRoute],
    final_results: Sequence[RetrievedChunk],
) -> list[dict[str, object]]:
    route_names = {route.filename.casefold() for route in routes}
    by_filename: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in final_results:
        by_filename[item.chunk.filename].append(item)
    priority = {"governing": 8, "definition": 7, "authority": 6, "exception": 5, "restriction": 5, "applicability": 4, "conflict": 4, "supporting": 3}
    output: list[dict[str, object]] = []
    for raw in diagnostics:
        item = dict(raw)
        filename = str(item.get("filename") or "")
        evidence = by_filename.get(filename, [])
        role = ""
        best_role_score = -1
        for result in evidence:
            match = re.search(r"v5\.2-synthesis:([a-z_]+)", result.method, re.IGNORECASE)
            if not match:
                continue
            candidate_role = match.group(1).casefold()
            candidate_score = priority.get(candidate_role, 0)
            if candidate_score > best_role_score:
                best_role_score, role = candidate_score, candidate_role
        deep = filename.casefold() in route_names
        item["deep_searched"] = deep
        item["final_evidence"] = bool(evidence)
        if role:
            item["rerank_role"] = role
        if deep and evidence:
            item["decision"] = "FINAL_EVIDENCE"
            item["reason"] = f"Routed for deep search and retained as {role or 'supporting'} evidence."
        elif deep:
            item["decision"] = "ROUTED_NO_FINAL_EVIDENCE"
            item["reason"] = "Routed for deep search, but no candidate survived rerank/section/final-context selection."
        output.append(item)
    return output



def ensure_routed_document_evidence(
    selected: Sequence[RetrievedChunk],
    candidates: Sequence[RetrievedChunk],
    routes: Sequence[DocumentRoute],
    limit: int,
) -> list[RetrievedChunk]:
    """Ensure every routed document surviving rerank has at least one final candidate."""
    by_doc: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in candidates:
        by_doc[item.chunk.document_id or item.chunk.filename].append(item)
    for values in by_doc.values():
        values.sort(key=lambda item: -float(item.score))

    output: list[RetrievedChunk] = []
    seen: set[str] = set()
    for route in routes:
        values = by_doc.get(route.document_id, [])
        if not values:
            continue
        item = values[0]
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        output.append(item)
        if len(output) >= limit:
            return output

    for item in selected:
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        output.append(item)
        if len(output) >= limit:
            return output
    return output


def required_coverage_documents(
    diagnostics: Sequence[dict[str, object]],
    *,
    include_definitions: bool = True,
) -> list[str]:
    values: list[str] = []
    for item in diagnostics:
        filename = _clean(item.get("filename"))
        if not filename:
            continue
        if include_definitions and item.get("decision") == "EXACT_ALIAS_DEFINITION":
            values.append(filename)
            continue
        if (
            bool(item.get("final_evidence"))
            and str(item.get("scope_type") or "") in {"rs", "line"}
        ):
            values.append(filename)
    return _unique(values, 120)


def enrich_retrieval_summary(
    summary: dict[str, object],
    diagnostics: Sequence[dict[str, object]],
) -> dict[str, object]:
    updated = dict(summary)
    required = required_coverage_documents(diagnostics)
    updated["required_scope_documents"] = [
        value for value in required
        if any(
            str(item.get("filename") or "").casefold() == value.casefold()
            and str(item.get("scope_type") or "") in {"rs", "line"}
            for item in diagnostics
        )
    ]
    updated["required_answer_documents"] = required
    updated["required_scope_labels"] = _unique(
        (
            str(item.get("scope_label") or "")
            for item in diagnostics
            if bool(item.get("final_evidence"))
            and str(item.get("scope_type") or "") in {"rs", "line"}
        ),
        120,
    )
    updated["scope_final_evidence_documents"] = len(updated["required_scope_documents"])
    return updated


def response_diagnostics(
    diagnostics: Sequence[dict[str, object]],
    summary: dict[str, object],
    contributing_documents: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    contributing = {name.casefold() for name in contributing_documents}
    output: list[dict[str, object]] = []
    for raw in diagnostics:
        item = dict(raw)
        is_contributing = str(item.get("filename") or "").casefold() in contributing
        item["contributing"] = is_contributing
        if is_contributing:
            item["decision"] = "CONTRIBUTING_EVIDENCE"
            item["reason"] = "Evidence from this document materially contributes to the coverage-reviewed answer."
        output.append(item)
    updated = enrich_retrieval_summary(dict(summary), output)
    updated["deep_searched_documents"] = sum(1 for item in output if bool(item.get("deep_searched")))
    updated["final_evidence_documents"] = sum(1 for item in output if bool(item.get("final_evidence")))
    updated["contributing_documents"] = sum(1 for item in output if bool(item.get("contributing")))
    return output, updated
