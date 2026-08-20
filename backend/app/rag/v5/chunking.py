from __future__ import annotations

import re
import uuid
from collections import defaultdict
from typing import Iterable

from app.rag.v5.types import V5Chunk, V5Element, V5LayoutDocument, V5Table


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _digest(*parts: object) -> str:
    seed = "|".join(str(part) for part in parts)
    return str(uuid.uuid5(uuid.UUID("44cd1a95-e2f8-4481-a4e6-c4fd72b79c30"), seed))


def _context_header(
    *,
    filename: str,
    page_start: int,
    page_end: int,
    section_path: list[str],
    content_type: str,
    heading: str = "",
    table_title: str = "",
    columns: list[str] | None = None,
) -> str:
    pages = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    lines = [
        "[PDF STRUCTURE]",
        f"File: {filename}",
        f"Pages: {pages}",
        f"Section path: {' > '.join(section_path) if section_path else 'Unsectioned content'}",
        f"Content type: {content_type}",
    ]
    if heading:
        lines.append(f"Heading: {heading}")
    if table_title:
        lines.append(f"Table: {table_title}")
    if columns:
        lines.append("Columns: " + " | ".join(columns))
    lines.append("[/PDF STRUCTURE]")
    return "\n".join(lines)


def _split_long_unit(text: str, target: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= target:
        return [text] if text else []
    sentences = [part.strip() for part in re.split(r"(?<=[.;:!?])\s+(?=[A-Z0-9(])|\n+", text) if part.strip()]
    if len(sentences) <= 1:
        step = max(200, target - overlap)
        return [text[start : start + target].strip() for start in range(0, len(text), step) if text[start : start + target].strip()]
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence]).strip()
        if current and len(candidate) > target:
            chunks.append(" ".join(current).strip())
            if overlap:
                tail = " ".join(current)[-overlap:].strip()
                current = [tail, sentence] if tail else [sentence]
            else:
                current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _element_groups(elements: Iterable[V5Element]) -> list[list[V5Element]]:
    groups: list[list[V5Element]] = []
    current: list[V5Element] = []
    current_parent = ""
    for element in sorted(elements, key=lambda item: (item.page_number, item.bbox[1], item.bbox[0], item.order_index)):
        if element.element_type in {"heading"}:
            # Heading becomes context for later elements, not a standalone answer chunk unless no body follows.
            if current:
                groups.append(current)
                current = []
            current_parent = element.parent_key or element.text
            continue
        parent = element.parent_key or current_parent or "Unsectioned content"
        if current and parent != (current[-1].parent_key or current_parent or "Unsectioned content"):
            groups.append(current)
            current = []
        current_parent = parent
        current.append(element)
    if current:
        groups.append(current)
    return groups


def _table_row_text(table: V5Table, cells: list[str]) -> str:
    pairs: list[str] = []
    for index, cell in enumerate(cells):
        cell = _clean(cell)
        if not cell:
            continue
        column = table.columns[index] if index < len(table.columns) else f"Column {index + 1}"
        pairs.append(f"{_clean(column)}: {cell}")
    return " | ".join(pairs)


def build_v5_chunks(
    document: V5LayoutDocument,
    *,
    target_chars: int = 1000,
    overlap_chars: int = 120,
) -> list[V5Chunk]:
    """Create retrieval children that preserve complete semantic evidence units.

    Prose is grouped by section and split only at element/sentence boundaries. Table rows are
    first-class child chunks and always carry their table title, columns and section path. Figures
    remain searchable from captions/OCR labels without pretending that their visual meaning was read.
    """
    target_chars = max(500, min(2400, int(target_chars)))
    overlap_chars = max(0, min(400, int(overlap_chars)))
    chunks: list[V5Chunk] = []
    chunk_index = 0

    # Prose / lists / figure-caption elements.
    for group in _element_groups(document.elements):
        if not group:
            continue
        parent_key = group[0].parent_key or "Unsectioned content"
        section_path = list(group[0].metadata.get("section_path") or [])
        if not section_path and parent_key != "Unsectioned content":
            section_path = [part.strip() for part in parent_key.split(">") if part.strip()]
        heading = section_path[-1] if section_path else ""

        units: list[tuple[V5Element, str]] = []
        for element in group:
            text_value = _clean(element.text)
            if not text_value:
                continue
            if element.element_type == "list_item":
                units.append((element, text_value))
            elif element.element_type == "figure":
                units.append((element, text_value))
            else:
                for part in _split_long_unit(text_value, target_chars, overlap_chars):
                    units.append((element, part))

        current: list[tuple[V5Element, str]] = []
        current_len = 0
        for element, unit in units:
            extra = len(unit) + (2 if current else 0)
            if current and current_len + extra > target_chars:
                page_start = min(item.page_number for item, _ in current)
                page_end = max(item.page_number for item, _ in current)
                body = "\n\n".join(text for _, text in current)
                content_types = {item.element_type for item, _ in current}
                content_type = "figure" if content_types == {"figure"} else ("list" if content_types <= {"list_item"} else "text")
                confidence = min(item.confidence for item, _ in current)
                header = _context_header(
                    filename=document.filename,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                    content_type=content_type,
                    heading=heading,
                )
                chunks.append(
                    V5Chunk(
                        chunk_id=_digest(document.filename, parent_key, chunk_index, body),
                        chunk_index=chunk_index,
                        page_number=page_start,
                        page_end=page_end,
                        content_type=content_type,
                        text=f"{header}\n\n{body}",
                        parent_key=f"section:{parent_key}",
                        section_path=section_path,
                        heading=heading,
                        extraction_confidence=confidence,
                        metadata={"element_ids": [item.element_id for item, _ in current]},
                    )
                )
                chunk_index += 1
                current = []
                current_len = 0
            current.append((element, unit))
            current_len += extra
        if current:
            page_start = min(item.page_number for item, _ in current)
            page_end = max(item.page_number for item, _ in current)
            body = "\n\n".join(text for _, text in current)
            content_types = {item.element_type for item, _ in current}
            content_type = "figure" if content_types == {"figure"} else ("list" if content_types <= {"list_item"} else "text")
            confidence = min(item.confidence for item, _ in current)
            header = _context_header(
                filename=document.filename,
                page_start=page_start,
                page_end=page_end,
                section_path=section_path,
                content_type=content_type,
                heading=heading,
            )
            chunks.append(
                V5Chunk(
                    chunk_id=_digest(document.filename, parent_key, chunk_index, body),
                    chunk_index=chunk_index,
                    page_number=page_start,
                    page_end=page_end,
                    content_type=content_type,
                    text=f"{header}\n\n{body}",
                    parent_key=f"section:{parent_key}",
                    section_path=section_path,
                    heading=heading,
                    extraction_confidence=confidence,
                    metadata={"element_ids": [item.element_id for item, _ in current]},
                )
            )
            chunk_index += 1

    # Table rows: never mix rows from different tables or strip their column relationship.
    for table in document.tables:
        section_path = list(table.section_path)
        table_parent = f"table:{table.table_key}"
        for row in table.rows:
            row_text = _table_row_text(table, row.cells)
            if not row_text:
                continue
            header = _context_header(
                filename=document.filename,
                page_start=row.page_number,
                page_end=row.page_number,
                section_path=section_path,
                content_type="table_row",
                heading=section_path[-1] if section_path else "",
                table_title=table.title,
                columns=table.columns,
            )
            chunks.append(
                V5Chunk(
                    chunk_id=_digest(document.filename, table.table_key, row.row_index, row.page_number, row_text),
                    chunk_index=chunk_index,
                    page_number=row.page_number,
                    page_end=row.page_number,
                    content_type="table_row",
                    text=f"{header}\n\nTable row {row.row_index}: {row_text}",
                    parent_key=table_parent,
                    section_path=section_path,
                    heading=section_path[-1] if section_path else table.title,
                    table_id=table.table_id,
                    table_row_index=row.row_index,
                    extraction_confidence=min(table.confidence, row.confidence),
                    metadata={
                        "table_title": table.title,
                        "columns": table.columns,
                        "cells": row.cells,
                        "table_source": table.source,
                        "multipage": bool(table.metadata.get("multipage")),
                    },
                )
            )
            chunk_index += 1

    chunks.sort(key=lambda chunk: (chunk.page_number, 0 if chunk.content_type != "table_row" else 1, chunk.chunk_index))
    # Re-number after reading-order sort so neighboring expansion is meaningful.
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
    return chunks
