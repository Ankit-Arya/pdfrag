from __future__ import annotations

# ruff: noqa: E501

import json
import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.rag.scenario_reasoning import (
    extract_numeric_rules,
    parse_chunk_context,
    strip_chunk_context,
)

logger = logging.getLogger(__name__)

_ABBR_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")
_LONG_FORM = r"[A-Za-z][A-Za-z0-9/&.'-]*(?:\s+[A-Za-z][A-Za-z0-9/&.'-]*){1,8}"
_LONG_THEN_ABBR_RE = re.compile(rf"(?P<long>{_LONG_FORM})\s*\(\s*(?P<abbr>[A-Z][A-Z0-9]{{1,9}})\s*\)")
_ABBR_THEN_LONG_PAREN_RE = re.compile(rf"\b(?P<abbr>[A-Z][A-Z0-9]{{1,9}})\s*\(\s*(?P<long>{_LONG_FORM})\s*\)")
_ABBR_DEF_RE = re.compile(
    rf"(?im)^\s*(?P<abbr>[A-Z][A-Z0-9]{{1,9}})\s*(?:[-:–—]|means\b|stands\s+for\b)\s*(?P<long>{_LONG_FORM})\s*$"
)
_TABLE_DEF_RE = re.compile(
    rf"(?im)^\s*\|?\s*(?P<abbr>[A-Z][A-Z0-9]{{1,9}})\s*\|\s*(?P<long>{_LONG_FORM})\s*\|?\s*$"
)
_LINE_RE = re.compile(r"\bLine\s*(?P<line>\d{1,2}|AEL)\b", re.IGNORECASE)


@dataclass(slots=True)
class _ChunkRow:
    id: str
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    filename: str


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+", value.casefold()))


def _clean_long_form(value: str) -> str:
    cleaned = " ".join(value.split()).strip(" -:;,.")
    # Avoid swallowing synthetic chunk metadata.
    for marker in ("Pages", "Section path", "Rolling stock", "Procedure context", "Important tags"):
        if marker.casefold() in cleaned.casefold():
            cleaned = cleaned.split(marker, 1)[0].strip(" -:;,.")
    return cleaned[:160]


def _best_long_form(alias: str, value: str, *, require_initials: bool) -> str:
    cleaned = _clean_long_form(value)
    if not cleaned:
        return ""
    words = re.findall(r"[A-Za-z][A-Za-z0-9/&.'-]*", cleaned)
    letters = "".join(char for char in alias.upper() if char.isalpha())
    if require_initials and letters and words:
        for start in range(len(words)):
            suffix = words[start:]
            initials = "".join(word[0].upper() for word in suffix if word)
            if initials == letters:
                return " ".join(suffix)[:160]
        # Parenthetical forms are noisy in OCR. Reject a long form whose initials
        # do not support the abbreviation instead of poisoning the global alias map.
        return ""
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned[:160]

def _concept_type(canonical: str) -> str:
    lowered = canonical.casefold()
    if any(word in lowered for word in ("controller", "operator", "manager", "officer", "engineer", "supervisor", "staff")):
        return "role"
    if any(word in lowered for word in ("centre", "center", "system", "door", "circuit", "signal", "train", "vehicle", "equipment")):
        return "system_or_equipment"
    return "term"


def extract_terminology(text_value: str) -> list[tuple[str, str, str, float]]:
    body = strip_chunk_context(text_value)
    found: list[tuple[str, str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for pattern, confidence, require_initials in (
        (_LONG_THEN_ABBR_RE, 0.98, True),
        (_ABBR_THEN_LONG_PAREN_RE, 0.98, True),
        (_ABBR_DEF_RE, 0.96, False),
        (_TABLE_DEF_RE, 0.97, False),
    ):
        for match in pattern.finditer(body):
            alias = match.group("abbr").upper()
            canonical = _best_long_form(alias, match.group("long"), require_initials=require_initials)
            if len(alias) < 2 or len(canonical) < 4:
                continue
            if canonical.casefold() == alias.casefold():
                continue
            # SC-06 and similar identifiers are document codes, not terminology definitions.
            if re.search(rf"\b{re.escape(alias)}[-_/]\d", body):
                # Do not reject a genuine definition elsewhere in the same chunk; reject only
                # if the captured long form itself looks like a code description.
                if re.search(r"\b(?:circular|sop|instruction|manual|document)\b", canonical, re.IGNORECASE):
                    continue
            key = (_norm(alias), _norm(canonical))
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            found.append((alias, canonical, _concept_type(canonical), confidence))
    return found


def _section_key(row: _ChunkRow) -> tuple[str, str, dict[str, str]]:
    metadata = parse_chunk_context(row.text)
    section = metadata.get("section", "").strip()
    if section:
        title = section.split(">")[-1].strip() or section
        return _norm(section)[:500], title[:500], metadata
    # Keep unheaded material searchable without creating a card for every chunk.
    block = row.chunk_index // 8
    return f"unheaded-{block}", f"Document section around page {row.page_number}", metadata


def _applicability(rows: list[_ChunkRow], metadata_values: list[dict[str, str]]) -> dict[str, object]:
    lines: set[str] = set()
    stocks: set[str] = set()
    procedures: set[str] = set()
    tags: set[str] = set()
    sample = "\n".join(strip_chunk_context(row.text)[:1500] for row in rows[:5])
    for match in _LINE_RE.finditer(sample):
        lines.add(match.group("line").upper())
    for metadata in metadata_values:
        stock = metadata.get("stock", "").strip()
        if stock:
            stocks.add(stock)
        procedure = metadata.get("procedure", "").strip()
        if procedure:
            procedures.add(procedure)
        for tag in metadata.get("tags", "").split(","):
            tag = tag.strip()
            if tag:
                tags.add(tag)
    return {
        "lines": sorted(lines),
        "rolling_stock": sorted(stocks),
        "procedures": sorted(procedures),
        "tags": sorted(tags)[:20],
    }


def index_document(db: Session, document_id: str | uuid.UUID) -> dict[str, int]:
    """Build terminology, section cards and deterministic numeric rules for one ready document."""
    document_id_text = str(document_id)
    rows = [
        _ChunkRow(
            id=str(row["id"]),
            document_id=str(row["document_id"]),
            chunk_index=int(row["chunk_index"]),
            page_number=int(row["page_number"]),
            text=str(row["text"]),
            filename=str(row["filename"]),
        )
        for row in db.execute(
            text(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.text, d.filename
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id = CAST(:document_id AS uuid) AND d.status = 'ready'
                ORDER BY c.chunk_index
                """
            ),
            {"document_id": document_id_text},
        ).mappings()
    ]
    if not rows:
        return {"terminology": 0, "procedure_cards": 0, "rules": 0}

    db.execute(text("DELETE FROM rag_rules WHERE document_id = CAST(:document_id AS uuid)"), {"document_id": document_id_text})
    db.execute(text("DELETE FROM rag_procedure_cards WHERE document_id = CAST(:document_id AS uuid)"), {"document_id": document_id_text})
    db.execute(text("DELETE FROM rag_terminology WHERE document_id = CAST(:document_id AS uuid)"), {"document_id": document_id_text})

    terminology_count = 0
    for row in rows:
        for alias, canonical, concept_type, confidence in extract_terminology(row.text):
            db.execute(
                text(
                    """
                    INSERT INTO rag_terminology(
                        alias, alias_norm, canonical_name, canonical_norm, concept_type,
                        confidence, document_id, chunk_id, page_number, evidence
                    ) VALUES (
                        :alias, :alias_norm, :canonical, :canonical_norm, :concept_type,
                        :confidence, CAST(:document_id AS uuid), CAST(:chunk_id AS uuid), :page, :evidence
                    )
                    ON CONFLICT DO NOTHING
                    """
                ),
                {
                    "alias": alias,
                    "alias_norm": _norm(alias),
                    "canonical": canonical,
                    "canonical_norm": _norm(canonical),
                    "concept_type": concept_type,
                    "confidence": confidence,
                    "document_id": row.document_id,
                    "chunk_id": row.id,
                    "page": row.page_number,
                    "evidence": strip_chunk_context(row.text)[:1000],
                },
            )
            terminology_count += 1

    groups: dict[str, list[_ChunkRow]] = defaultdict(list)
    group_titles: dict[str, str] = {}
    group_metadata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key, title, metadata = _section_key(row)
        groups[key].append(row)
        group_titles.setdefault(key, title)
        group_metadata[key].append(metadata)

    card_payloads: list[dict[str, object]] = []
    for key, group in groups.items():
        title = group_titles[key]
        bodies = [strip_chunk_context(item.text) for item in group]
        # Front, middle and tail samples make a compact semantic representation of
        # long procedures without embedding hundreds of source chunks.
        sample_parts = bodies[:3]
        if len(bodies) > 6:
            sample_parts.append(bodies[len(bodies) // 2])
        if len(bodies) > 3:
            sample_parts.append(bodies[-1])
        applicability = _applicability(group, group_metadata[key])
        applicability_text = " ".join(
            [
                *(f"Line {value}" for value in applicability["lines"]),
                *(str(value) for value in applicability["rolling_stock"]),
                *(str(value) for value in applicability["procedures"]),
                *(str(value) for value in applicability["tags"]),
            ]
        )
        search_text = "\n".join([title, applicability_text, *sample_parts])[:7000]
        card_payloads.append(
            {
                "section_key": key,
                "title": title,
                "search_text": search_text,
                "page_start": min(item.page_number for item in group),
                "page_end": max(item.page_number for item in group),
                "start_chunk": min(item.chunk_index for item in group),
                "end_chunk": max(item.chunk_index for item in group),
                "applicability": applicability,
            }
        )

    vectors = None
    if card_payloads:
        try:
            from app.rag.embeddings import embedding_service

            vectors = embedding_service.encode([str(card["search_text"]) for card in card_payloads])
        except Exception:
            logger.exception("Procedure-card embedding failed for document %s; FTS cards will still be built", document_id_text)

    card_ids_by_key: dict[str, int] = {}
    for index, card in enumerate(card_payloads):
        embedding_value = None if vectors is None else vectors[index].tolist()
        row = db.execute(
            text(
                """
                INSERT INTO rag_procedure_cards(
                    document_id, section_key, title, search_text, page_start, page_end,
                    start_chunk_index, end_chunk_index, applicability, embedding, updated_at
                ) VALUES (
                    CAST(:document_id AS uuid), :section_key, :title, :search_text, :page_start, :page_end,
                    :start_chunk, :end_chunk, CAST(:applicability AS jsonb), CAST(:embedding AS vector), now()
                )
                ON CONFLICT (document_id, section_key) DO UPDATE SET
                    title = EXCLUDED.title,
                    search_text = EXCLUDED.search_text,
                    page_start = EXCLUDED.page_start,
                    page_end = EXCLUDED.page_end,
                    start_chunk_index = EXCLUDED.start_chunk_index,
                    end_chunk_index = EXCLUDED.end_chunk_index,
                    applicability = EXCLUDED.applicability,
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                RETURNING id
                """
            ),
            {
                "document_id": document_id_text,
                "section_key": card["section_key"],
                "title": card["title"],
                "search_text": card["search_text"],
                "page_start": card["page_start"],
                "page_end": card["page_end"],
                "start_chunk": card["start_chunk"],
                "end_chunk": card["end_chunk"],
                "applicability": json.dumps(card["applicability"]),
                "embedding": None if embedding_value is None else str(embedding_value),
            },
        ).scalar_one()
        card_ids_by_key[str(card["section_key"])] = int(row)

    rules_count = 0
    for row in rows:
        key, _, _ = _section_key(row)
        card_id = card_ids_by_key.get(key)
        for rule in extract_numeric_rules(row.text):
            db.execute(
                text(
                    """
                    INSERT INTO rag_rules(
                        procedure_card_id, document_id, chunk_id, page_number,
                        field_tokens, operator, threshold, unit, condition_text, confidence
                    ) VALUES (
                        :card_id, CAST(:document_id AS uuid), CAST(:chunk_id AS uuid), :page,
                        :field_tokens, :operator, :threshold, :unit, :condition_text, 0.95
                    )
                    ON CONFLICT DO NOTHING
                    """
                ),
                {
                    "card_id": card_id,
                    "document_id": row.document_id,
                    "chunk_id": row.id,
                    "page": row.page_number,
                    "field_tokens": sorted(rule.tokens),
                    "operator": rule.operator,
                    "threshold": rule.threshold,
                    "unit": rule.unit,
                    "condition_text": rule.raw,
                },
            )
            rules_count += 1

    db.commit()
    return {
        "terminology": terminology_count,
        "procedure_cards": len(card_payloads),
        "rules": rules_count,
    }


def backfill_all(db: Session) -> dict[str, int]:
    document_ids = [
        str(value)
        for value in db.scalars(text("SELECT id FROM documents WHERE status = 'ready' ORDER BY created_at"))
    ]
    totals = {"documents": 0, "terminology": 0, "procedure_cards": 0, "rules": 0}
    for number, document_id in enumerate(document_ids, start=1):
        try:
            counts = index_document(db, document_id)
        except Exception:
            db.rollback()
            logger.exception("Smart-index backfill failed for document %s", document_id)
            continue
        totals["documents"] += 1
        for key in ("terminology", "procedure_cards", "rules"):
            totals[key] += counts[key]
        logger.info(
            "Smart-indexed document %d/%d: %s cards=%d terms=%d rules=%d",
            number,
            len(document_ids),
            document_id,
            counts["procedure_cards"],
            counts["terminology"],
            counts["rules"],
        )
    return totals
