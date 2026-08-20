from __future__ import annotations

import re
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9/-]{1,15}")


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+", value.casefold()))


def _aliases_in_question(value: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(value):
        # Internal abbreviations are normally uppercase/coded. Avoid treating ordinary words as
        # terminology aliases merely because they happen to exist in a noisy OCR definition table.
        if not (token.isupper() or re.search(r"\d", token)):
            continue
        norm = _norm(token)
        if norm and norm not in seen:
            seen.add(norm)
            output.append(token)
    return output[:12]


def terminology_hints(db: Session, question: str, *, limit: int = 20) -> list[str]:
    aliases = _aliases_in_question(question)
    if not aliases:
        return []
    norms = [_norm(value) for value in aliases]
    rows = db.execute(
        text(
            """
            SELECT t.alias, t.canonical_name, t.confidence, d.filename, t.page_number
            FROM rag_v5_terminology t
            JOIN rag_v5_processing_runs r ON r.id=t.run_id AND r.is_active=true AND r.status='ready'
            JOIN documents d ON d.id=t.document_id AND d.status='ready'
            WHERE t.alias_norm = ANY(:norms)
            ORDER BY t.confidence DESC, lower(d.filename), t.page_number
            LIMIT :limit
            """
        ),
        {"norms": norms, "limit": max(1, min(60, int(limit)))},
    ).mappings()
    output: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        alias = str(row["alias"])
        canonical = str(row["canonical_name"])
        key = (_norm(alias), _norm(canonical))
        if key in seen:
            continue
        seen.add(key)
        output.append(
            f"{alias} = {canonical} (PDF: {row['filename']}, page {row['page_number']})"
        )
    return output


def definition_source_rows(db: Session, question: str, *, limit: int = 16) -> list[object]:
    aliases = _aliases_in_question(question)
    if not aliases:
        return []
    norms = [_norm(value) for value in aliases]
    return list(
        db.execute(
            text(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.page_end,
                       c.content_type, c.parent_key, c.section_path, c.heading,
                       c.authority_status, c.text, d.filename,
                       1.0 AS keyword_score
                FROM rag_v5_terminology t
                JOIN rag_v5_processing_runs r ON r.id=t.run_id AND r.is_active=true AND r.status='ready'
                JOIN rag_v5_chunks c ON c.id=t.chunk_id AND c.run_id=t.run_id
                JOIN documents d ON d.id=t.document_id AND d.status='ready'
                WHERE t.alias_norm = ANY(:norms)
                ORDER BY t.confidence DESC, lower(d.filename), t.page_number, c.chunk_index
                LIMIT :limit
                """
            ),
            {"norms": norms, "limit": max(1, min(60, int(limit)))},
        ).mappings()
    )
