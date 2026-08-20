from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db_models import Document, DocumentChunk, DocumentStatus
from app.rag.authority import (
    extract_authority_directives,
    looks_like_subdocument_boundary,
    normalize_authority_text,
)
from app.rag.embeddings import embedding_service
from app.rag.smart_index import extract_terminology
from app.rag.v5 import PROCESSING_VERSION
from app.rag.v5.chunking import build_v5_chunks
from app.rag.v5.layout import extract_layout_document
from app.rag.v5.schema import ensure_v5_schema
from app.rag.v5.types import V5Chunk

logger = logging.getLogger(__name__)
_STRUCTURE_RE = re.compile(r"\[PDF STRUCTURE\].*?\[/PDF STRUCTURE\]\s*", re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class V5ProcessingSummary:
    run_id: uuid.UUID
    document_id: uuid.UUID
    pages: int
    chunks: int
    tables: int
    table_rows: int
    ocr_pages: int
    headings: int
    figures: int
    low_quality_pages: int


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _body(value: str) -> str:
    return _STRUCTURE_RE.sub("", value).strip()


def _target_chars() -> int:
    settings = get_settings()
    return int(getattr(settings, "rag_v5_chunk_target_chars", 1000))


def _overlap_chars() -> int:
    settings = get_settings()
    return int(getattr(settings, "rag_v5_chunk_overlap_chars", 120))


def _terminology_initials_match(alias: str, canonical: str) -> bool:
    letters = "".join(char for char in alias.upper() if char.isalpha())
    words = re.findall(r"[A-Za-z][A-Za-z0-9/&.'-]*", canonical)
    if not letters or not words:
        return False
    ignored = {"and", "or", "of", "the", "for", "to", "in", "on", "a", "an"}
    semantic = "".join(word[0].upper() for word in words if word.casefold() not in ignored)
    strict = "".join(word[0].upper() for word in words)
    return letters in {semantic, strict}


def _v5_terminology(chunk: V5Chunk) -> list[tuple[str, str, float]]:
    """Extract only strongly supported acronym definitions from source-derived text."""
    found: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for alias, canonical, _kind, confidence in extract_terminology(_body(chunk.text)):
        if not _terminology_initials_match(alias, canonical):
            continue
        key = (normalize_authority_text(alias), normalize_authority_text(canonical))
        if key in seen:
            continue
        seen.add(key)
        found.append((alias, canonical, float(confidence)))

    cells = chunk.metadata.get("cells") if isinstance(chunk.metadata, dict) else None
    if isinstance(cells, list) and len(cells) >= 2:
        for left, right in zip(cells, cells[1:], strict=False):
            alias = " ".join(str(left).split()).strip(" :-–—|()")
            canonical = " ".join(str(right).split()).strip(" :-–—|()")
            if not re.fullmatch(r"[A-Z][A-Z0-9/-]{1,9}", alias):
                continue
            if len(canonical) < 4 or not _terminology_initials_match(alias, canonical):
                continue
            key = (normalize_authority_text(alias), normalize_authority_text(canonical))
            if key in seen:
                continue
            seen.add(key)
            found.append((alias, canonical, 0.98))
    return found


def _authority_text(chunk: V5Chunk) -> str:
    """Return source-derived semantic text for authority parsing.

    V5 deliberately stores headings/section paths outside the visible body. Amendment
    instructions are often headings (for example, "For the Second Schedule ... shall be
    substituted"), so authority extraction must inspect both structural labels and body text.
    Synthetic field names are omitted; only text that came from the PDF is retained.
    """
    values = [*chunk.section_path]
    if chunk.heading and chunk.heading not in values:
        values.append(chunk.heading)
    values.append(_body(chunk.text))
    return "\n".join(value for value in values if str(value).strip())


def _document_amendment_year(chunks: list[V5Chunk]) -> int | None:
    years: list[int] = []
    for chunk in chunks:
        text_value = _authority_text(chunk)
        if "amend" not in text_value.casefold():
            continue
        years.extend(int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", text_value))
    return max(years) if years else None


def _strong_prior_instrument_boundary(
    chunk: V5Chunk,
    *,
    current_year: int | None,
) -> bool:
    """Identify an appended older/base instrument conservatively.

    A generic occurrence of "principal rules" is not enough: amendment documents themselves
    repeatedly use that phrase. We require a prior year/title/publication signal and only apply
    the boundary *after* the current replacement section has ended.
    """
    if not current_year:
        return False
    body = _authority_text(chunk)
    folded = " ".join(body.casefold().split())
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", folded)]
    older_year = bool(years and min(years) < current_year)
    if not older_year:
        return False
    explicit_title = bool(
        re.search(r"\b(?:rules?|regulations?|manual|code)\s*,?\s*(?:19|20)\d{2}\b", folded)
    )
    gazette_boundary = "published by authority" in folded
    notification = (
        "notification" in folded
        and "amendment" not in folded
        and "shall be substituted" not in folded
        and "shall be replaced" not in folded
    )
    compact_title = (
        explicit_title
        and "amendment" not in folded
        and "shall be substituted" not in folded
        and len(folded) <= 900
    )
    # Do not use a mere note saying that the principal rules "were published" as a
    # boundary: amendment gazettes commonly print that note at the end of the *current*
    # replacement table. A new Gazette/notification/title block is a much stronger signal.
    return gazette_boundary or notification or compact_title


def _section_match(chunk: V5Chunk, target_norm: str, anchor_text_norm: str) -> bool:
    if not target_norm:
        return False
    structure = normalize_authority_text(" ".join([*chunk.section_path, chunk.heading]))
    if target_norm in structure:
        return True
    # Some continuation pages inherit the complete amendment heading rather than the short
    # target name. Preserve them when they share a meaningful part of the anchor heading.
    if anchor_text_norm and len(anchor_text_norm) >= 20 and anchor_text_norm in structure:
        return True
    return False


def _authority_metadata(chunks: list[V5Chunk]) -> list[dict[str, object]]:
    """Build current/superseded authority metadata from explicit PDF language.

    The implementation is intentionally conservative. Explicit substitution/omission wording can
    establish precedence; proximity alone cannot. Section replacements are mapped through v5's
    inherited section paths instead of a fixed chunk window, which keeps multi-page tables attached
    to the amendment that introduced them.
    """
    document_year = _document_amendment_year(chunks)
    directives: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for chunk in chunks:
        authority_text = _authority_text(chunk)
        for directive in extract_authority_directives(
            authority_text,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
        ):
            candidate_years = [year for year in (directive.effective_year, document_year) if year]
            effective_year = max(candidate_years) if candidate_years else None
            key = (
                directive.directive_type,
                normalize_authority_text(directive.target),
                normalize_authority_text(directive.old_text),
                normalize_authority_text(directive.new_text),
                effective_year,
            )
            if key in seen:
                continue
            seen.add(key)
            directives.append(
                {
                    "directive_type": directive.directive_type,
                    "target": directive.target,
                    "target_norm": normalize_authority_text(directive.target),
                    "old_text": directive.old_text,
                    "old_norm": normalize_authority_text(directive.old_text),
                    "new_text": directive.new_text,
                    "new_norm": normalize_authority_text(directive.new_text),
                    "effective_year": effective_year,
                    "anchor_chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "span_start_chunk": chunk.chunk_index,
                    "span_end_chunk": chunk.chunk_index,
                    "confidence": directive.confidence,
                    "evidence": authority_text[:2200],
                }
            )

    if not directives:
        return directives

    by_index = {chunk.chunk_index: chunk for chunk in chunks}
    ordered = sorted(chunks, key=lambda item: (item.page_number, item.chunk_index))

    # Map each explicit section replacement to chunks inheriting that exact structural context.
    # This is robust to character chunking and to tables that continue on later pages.
    current_section_indexes: set[int] = set()
    current_section_last_page = 0
    for item in directives:
        if item["directive_type"] != "replace_section":
            continue
        target_norm = str(item["target_norm"] or "")
        anchor = by_index.get(int(item["anchor_chunk_index"]))
        anchor_structure_norm = (
            normalize_authority_text(" ".join([*anchor.section_path, anchor.heading]))
            if anchor is not None else ""
        )
        # Find an unambiguous appended/base-instrument boundary first. Matching a later
        # section with the same name (for example the old "Second Schedule") must not extend
        # the current replacement span into historical material.
        effective_year = int(item["effective_year"]) if item.get("effective_year") else document_year
        boundary_page: int | None = None
        for candidate in ordered:
            if candidate.page_number <= int(item["page_number"]):
                continue
            if _strong_prior_instrument_boundary(candidate, current_year=effective_year):
                boundary_page = candidate.page_number
                break
        matching = [
            chunk for chunk in ordered
            if chunk.chunk_index >= int(item["anchor_chunk_index"])
            and (boundary_page is None or chunk.page_number < boundary_page)
            and _section_match(chunk, target_norm, anchor_structure_norm)
        ]
        # Always retain the explicit anchor even if heading detection produced an unusual path.
        if anchor is not None and anchor not in matching:
            matching.insert(0, anchor)
        if matching:
            indexes = [chunk.chunk_index for chunk in matching]
            item["span_start_chunk"] = min(indexes)
            item["span_end_chunk"] = max(indexes)
            current_section_indexes.update(indexes)
            current_section_last_page = max(current_section_last_page, max(chunk.page_number for chunk in matching))

    # Apply explicit word substitutions. Do not demote the authority anchor itself: it contains
    # both the old and replacement wording by definition.
    for item in directives:
        if item["directive_type"] != "replace_words":
            continue
        new_norm = str(item["new_norm"] or "")
        old_norm = str(item["old_norm"] or "")
        anchor_index = int(item["anchor_chunk_index"])
        for chunk in chunks:
            normalized = normalize_authority_text(_authority_text(chunk))
            if chunk.chunk_index == anchor_index:
                continue
            if new_norm and new_norm in normalized:
                chunk.authority_status = "current_explicit"
            if old_norm and old_norm in normalized and chunk.authority_status == "unknown":
                chunk.authority_status = "superseded_explicit"

    for chunk in chunks:
        if chunk.chunk_index in current_section_indexes and not chunk.authority_status.startswith("superseded"):
            chunk.authority_status = "current_replacement"

    # Mark an appended older/base instrument only after the *last current replacement page*.
    # This avoids the previous false positive where the current table itself was labelled historic.
    highest_year = max(
        (int(item["effective_year"]) for item in directives if item.get("effective_year")),
        default=document_year or 0,
    )
    if highest_year:
        search_after_page = current_section_last_page or max(int(item["page_number"]) for item in directives)
        boundary_page: int | None = None
        for chunk in ordered:
            if chunk.page_number <= search_after_page:
                continue
            if _strong_prior_instrument_boundary(chunk, current_year=highest_year):
                boundary_page = chunk.page_number
                break
        if boundary_page is not None:
            for chunk in chunks:
                if chunk.page_number >= boundary_page and chunk.authority_status == "unknown":
                    chunk.authority_status = "historical_appended"

    return directives

def _insert_run(db: Session, run_id: uuid.UUID, document: Document) -> None:
    db.execute(
        text(
            """
            INSERT INTO rag_v5_processing_runs(id, document_id, processing_version, status, is_active)
            VALUES (CAST(:id AS uuid), CAST(:document_id AS uuid), :version, 'processing', false)
            """
        ),
        {"id": str(run_id), "document_id": str(document.id), "version": PROCESSING_VERSION},
    )


def process_document_v5(
    db: Session,
    document: Document,
    *,
    publish_document_state: bool = True,
) -> V5ProcessingSummary:
    """Build a complete v5 generation without deleting the existing v4 corpus.

    The new generation is written into rag_v5_* tables and becomes active atomically only after
    extraction, structure reconstruction, chunking and embeddings all succeed.
    """
    ensure_v5_schema()
    run_id = uuid.uuid4()
    if publish_document_state:
        document.status = DocumentStatus.processing
        document.error = None
    _insert_run(db, run_id, document)
    db.commit()

    try:
        layout = extract_layout_document(document.content, document.filename)
        chunks = build_v5_chunks(
            layout,
            target_chars=_target_chars(),
            overlap_chars=_overlap_chars(),
        )
        if not chunks:
            raise ValueError("RAG v5 created no semantic chunks")
        directives = _authority_metadata(chunks)

        settings = get_settings()
        max_chunks = int(getattr(settings, "max_chunks_per_collection", 25000))
        if len(chunks) > max_chunks:
            raise ValueError(f"RAG v5 produced {len(chunks)} chunks, above MAX_CHUNKS_PER_COLLECTION={max_chunks}")

        db.execute(
            text("UPDATE rag_v5_processing_runs SET is_active=false WHERE document_id=CAST(:document_id AS uuid)"),
            {"document_id": str(document.id)},
        )

        for page in layout.pages:
            db.execute(
                text(
                    """
                    INSERT INTO rag_v5_pages(
                        run_id, document_id, page_number, width, height, language, native_chars,
                        ocr_used, quality_score, warnings
                    ) VALUES (
                        CAST(:run_id AS uuid), CAST(:document_id AS uuid), :page_number, :width,
                        :height, :language, :native_chars, :ocr_used, :quality_score,
                        CAST(:warnings AS jsonb)
                    )
                    """
                ),
                {
                    "run_id": str(run_id), "document_id": str(document.id),
                    "page_number": page.page_number, "width": page.width, "height": page.height,
                    "language": page.language, "native_chars": page.native_chars,
                    "ocr_used": page.ocr_used, "quality_score": page.quality_score,
                    "warnings": _json(page.warnings),
                },
            )

        for element in layout.elements:
            db.execute(
                text(
                    """
                    INSERT INTO rag_v5_elements(
                        id, run_id, document_id, page_number, order_index, element_type, parent_key,
                        text, bbox, heading_level, confidence, extraction_source, metadata
                    ) VALUES (
                        CAST(:id AS uuid), CAST(:run_id AS uuid), CAST(:document_id AS uuid),
                        :page_number, :order_index, :element_type, :parent_key, :text,
                        CAST(:bbox AS jsonb), :heading_level, :confidence, :extraction_source,
                        CAST(:metadata AS jsonb)
                    )
                    """
                ),
                {
                    "id": element.element_id, "run_id": str(run_id), "document_id": str(document.id),
                    "page_number": element.page_number, "order_index": element.order_index,
                    "element_type": element.element_type, "parent_key": element.parent_key,
                    "text": element.text, "bbox": _json(list(element.bbox)),
                    "heading_level": element.heading_level, "confidence": element.confidence,
                    "extraction_source": element.extraction_source, "metadata": _json(element.metadata),
                },
            )

        for table in layout.tables:
            db.execute(
                text(
                    """
                    INSERT INTO rag_v5_tables(
                        id, run_id, document_id, table_key, title, page_start, page_end, columns,
                        bbox_by_page, confidence, extraction_source, section_path, metadata
                    ) VALUES (
                        CAST(:id AS uuid), CAST(:run_id AS uuid), CAST(:document_id AS uuid),
                        :table_key, :title, :page_start, :page_end, CAST(:columns AS jsonb),
                        CAST(:bbox_by_page AS jsonb), :confidence, :source,
                        CAST(:section_path AS jsonb), CAST(:metadata AS jsonb)
                    )
                    """
                ),
                {
                    "id": table.table_id, "run_id": str(run_id), "document_id": str(document.id),
                    "table_key": table.table_key, "title": table.title,
                    "page_start": table.page_start, "page_end": table.page_end,
                    "columns": _json(table.columns),
                    "bbox_by_page": _json({str(key): list(value) for key, value in table.bbox_by_page.items()}),
                    "confidence": table.confidence, "source": table.source,
                    "section_path": _json(table.section_path), "metadata": _json(table.metadata),
                },
            )
            for row in table.rows:
                normalized = " | ".join(
                    f"{table.columns[index] if index < len(table.columns) else f'Column {index + 1}'}: {cell}"
                    for index, cell in enumerate(row.cells)
                    if str(cell).strip()
                )
                db.execute(
                    text(
                        """
                        INSERT INTO rag_v5_table_rows(
                            table_id, document_id, row_index, page_number, cells, normalized_text,
                            bbox, confidence, extraction_source
                        ) VALUES (
                            CAST(:table_id AS uuid), CAST(:document_id AS uuid), :row_index,
                            :page_number, CAST(:cells AS jsonb), :normalized_text, CAST(:bbox AS jsonb),
                            :confidence, :source
                        )
                        """
                    ),
                    {
                        "table_id": table.table_id, "document_id": str(document.id),
                        "row_index": row.row_index, "page_number": row.page_number,
                        "cells": _json(row.cells), "normalized_text": normalized,
                        "bbox": _json(list(row.bbox)), "confidence": row.confidence, "source": row.source,
                    },
                )

        # After query cutover, mirror the same structure-preserving child chunks into the legacy
        # document_chunks table without re-parsing/re-embedding the PDF. This keeps rollback possible
        # for newly uploaded/reprocessed documents. Migration runs use publish_document_state=False
        # and therefore never alter the live v4 corpus.
        mirror_legacy = publish_document_state and bool(getattr(settings, "rag_v5_legacy_chunk_mirror", True))
        if mirror_legacy:
            db.execute(
                text("DELETE FROM document_chunks WHERE document_id=CAST(:document_id AS uuid)"),
                {"document_id": str(document.id)},
            )

        # Embed the semantically complete child units. Table rows include inherited headers/columns.
        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embedding_service.encode([chunk.text for chunk in batch])
            for chunk, vector in zip(batch, vectors, strict=True):
                db.execute(
                    text(
                        """
                        INSERT INTO rag_v5_chunks(
                            id, run_id, document_id, chunk_index, page_number, page_end, content_type,
                            parent_key, section_path, heading, table_id, table_row_index,
                            extraction_confidence, authority_status, metadata, text, embedding
                        ) VALUES (
                            CAST(:id AS uuid), CAST(:run_id AS uuid), CAST(:document_id AS uuid),
                            :chunk_index, :page_number, :page_end, :content_type, :parent_key,
                            CAST(:section_path AS jsonb), :heading,
                            CASE WHEN :table_id = '' THEN NULL ELSE CAST(:table_id AS uuid) END,
                            :table_row_index, :confidence, :authority_status, CAST(:metadata AS jsonb),
                            :text, CAST(:embedding AS vector)
                        )
                        """
                    ),
                    {
                        "id": chunk.chunk_id, "run_id": str(run_id), "document_id": str(document.id),
                        "chunk_index": chunk.chunk_index, "page_number": chunk.page_number,
                        "page_end": chunk.page_end, "content_type": chunk.content_type,
                        "parent_key": chunk.parent_key, "section_path": _json(chunk.section_path),
                        "heading": chunk.heading, "table_id": chunk.table_id or "",
                        "table_row_index": chunk.table_row_index, "confidence": chunk.extraction_confidence,
                        "authority_status": chunk.authority_status, "metadata": _json(chunk.metadata),
                        "text": chunk.text, "embedding": str(vector.tolist()),
                    },
                )
                if mirror_legacy:
                    db.add(
                        DocumentChunk(
                            document_id=document.id,
                            chunk_index=chunk.chunk_index,
                            page_number=chunk.page_number,
                            content_type=chunk.content_type,
                            text=chunk.text,
                            embedding=vector.tolist(),
                        )
                    )

        terminology_count = 0
        for chunk in chunks:
            for alias, canonical, confidence in _v5_terminology(chunk):
                db.execute(
                    text(
                        """
                        INSERT INTO rag_v5_terminology(
                            run_id, document_id, chunk_id, page_number, alias, alias_norm,
                            canonical_name, canonical_norm, confidence, evidence
                        ) VALUES (
                            CAST(:run_id AS uuid), CAST(:document_id AS uuid), CAST(:chunk_id AS uuid),
                            :page_number, :alias, :alias_norm, :canonical, :canonical_norm,
                            :confidence, :evidence
                        ) ON CONFLICT DO NOTHING
                        """
                    ),
                    {
                        "run_id": str(run_id), "document_id": str(document.id),
                        "chunk_id": chunk.chunk_id, "page_number": chunk.page_number,
                        "alias": alias, "alias_norm": normalize_authority_text(alias),
                        "canonical": canonical, "canonical_norm": normalize_authority_text(canonical),
                        "confidence": confidence, "evidence": _body(chunk.text)[:1200],
                    },
                )
                terminology_count += 1

        for directive in directives:
            db.execute(
                text(
                    """
                    INSERT INTO rag_v5_authority(
                        run_id, document_id, anchor_chunk_index, page_number, directive_type, target,
                        target_norm, old_text, old_norm, new_text, new_norm, effective_year,
                        span_start_chunk, span_end_chunk, confidence, evidence
                    ) VALUES (
                        CAST(:run_id AS uuid), CAST(:document_id AS uuid), :anchor_chunk_index,
                        :page_number, :directive_type, :target, :target_norm, :old_text, :old_norm,
                        :new_text, :new_norm, :effective_year, :span_start_chunk, :span_end_chunk,
                        :confidence, :evidence
                    )
                    """
                ),
                {"run_id": str(run_id), "document_id": str(document.id), **directive},
            )

        metrics = dict(layout.metrics)
        metrics["chunks"] = len(chunks)
        metrics["authority_directives"] = len(directives)
        metrics["terminology"] = terminology_count
        db.execute(
            text(
                """
                UPDATE rag_v5_processing_runs
                SET status='ready', is_active=true, metrics=CAST(:metrics AS jsonb),
                    warnings=CAST(:warnings AS jsonb), completed_at=now()
                WHERE id=CAST(:run_id AS uuid)
                """
            ),
            {"run_id": str(run_id), "metrics": _json(metrics), "warnings": _json(layout.warnings)},
        )

        if publish_document_state:
            document.page_count = layout.total_pages
            document.chunk_count = len(chunks)
            document.warnings = list(dict.fromkeys([*(document.warnings or []), *layout.warnings]))
            document.status = DocumentStatus.ready
            document.processed_at = datetime.now(UTC)
            document.error = None
        db.commit()
        if publish_document_state:
            db.refresh(document)

        return V5ProcessingSummary(
            run_id=run_id,
            document_id=document.id,
            pages=layout.total_pages,
            chunks=len(chunks),
            tables=len(layout.tables),
            table_rows=sum(len(table.rows) for table in layout.tables),
            ocr_pages=len(layout.metrics.get("ocr_pages", [])),
            headings=int(layout.metrics.get("headings", 0)),
            figures=int(layout.metrics.get("figures", 0)),
            low_quality_pages=len(layout.metrics.get("low_quality_pages", [])),
        )
    except Exception as exc:
        db.rollback()
        try:
            db.execute(
                text(
                    """
                    UPDATE rag_v5_processing_runs
                    SET status='failed', error=:error, completed_at=now(), is_active=false
                    WHERE id=CAST(:run_id AS uuid)
                    """
                ),
                {"run_id": str(run_id), "error": str(exc)[:4000]},
            )
            if publish_document_state:
                refreshed = db.get(Document, document.id)
                if refreshed is not None:
                    refreshed.status = DocumentStatus.failed
                    refreshed.error = str(exc)[:4000]
            db.commit()
        except Exception:
            db.rollback()
        logger.exception("RAG v5 document processing failed for %s", document.filename)
        raise
