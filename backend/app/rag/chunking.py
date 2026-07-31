from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

from app.rag.types import PageText, TextChunk


_NUMBERED_HEADING = re.compile(
    r"^(?P<num>(?:\d{1,3})(?:\.\d{1,3}){0,8})(?:[.)])?\s+(?P<title>[A-Za-z0-9][^|]{2,180})$"
)
_ALPHA_HEADING = re.compile(r"^(?P<num>[A-Z])\.\s+(?P<title>[A-Za-z0-9][^|]{2,160})$")
_ROMAN_HEADING = re.compile(r"^(?P<num>[IVXLCM]{1,8})\.\s+(?P<title>[A-Za-z0-9][^|]{2,160})$")
_ACRONYM_OR_CODE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9/-]{1,14}(?![A-Za-z0-9])")
_MULTISPACE = re.compile(r"[ \t]+")
_PROCEDURE_WORDS = {
    "procedure",
    "procedures",
    "sop",
    "instruction",
    "instructions",
    "operation",
    "operating",
    "isolation",
    "reset",
    "inspection",
    "maintenance",
    "troubleshooting",
    "fault",
    "failure",
    "emergency",
    "evacuation",
    "recovery",
    "test",
    "check",
    "checks",
}
_STOCK_WORDS = {
    "rolling stock",
    "train type",
    "trainset",
    "train set",
    "metro train",
    "emu",
    "dmrc",
    "rs",
    "vehicle type",
    "coach",
    "car",
}
_WARNING_WORDS = {
    "warning",
    "caution",
    "danger",
    "mandatory",
    "shall",
    "must",
    "prohibited",
    "do not",
    "never",
    "emergency",
    "isolation",
}


@dataclass(slots=True)
class _Heading:
    text: str
    level: int


@dataclass(slots=True)
class _Section:
    filename: str
    page_start: int
    page_end: int
    content_type: str
    section_path: list[str]
    blocks: list[str] = field(default_factory=list)
    rolling_stock: str | None = None
    procedure: str | None = None
    table_index: int | None = None

    @property
    def text(self) -> str:
        return _clean_body("\n".join(self.blocks))


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 900,
    overlap: int = 220,
) -> list[TextChunk]:
    """Create heading-aware chunks with explicit operational context.

    The previous splitter cut each page by character count. For metro procedure
    manuals that is unsafe: the same question can have different answers by
    rolling stock, equipment variant, procedure, mode, or subsection. This
    splitter first builds logical sections from headings/subheadings, carries the
    heading path across page boundaries, then prefixes every stored chunk with a
    machine-readable context header. That header is embedded and full-text
    indexed together with the body so retrieval can match both question text and
    document context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    sections = _build_sections(pages)
    chunks: list[TextChunk] = []

    for section in sections:
        body = section.text
        if not body:
            continue

        body_budget = max(350, chunk_size - len(_context_header(section)) - 2)
        if section.content_type == "table":
            pieces = _split_table_body(body, body_budget)
        else:
            pieces = _split_section_body(body, body_budget, overlap)

        for ordinal, piece in enumerate(pieces, start=1):
            if not piece.strip():
                continue
            header = _context_header(section)
            text = f"{header}\n\n{piece.strip()}".strip()
            tags = tuple(_context_tags(section, piece))
            digest = hashlib.sha1(
                (
                    f"{section.filename}:{section.page_start}:{section.page_end}:"
                    f"{section.content_type}:{section.table_index}:{ordinal}:"
                    f"{' > '.join(section.section_path)}:{piece}"
                ).encode()
            ).hexdigest()[:16]
            chunks.append(
                TextChunk(
                    chunk_id=digest,
                    filename=section.filename,
                    page_number=section.page_start,
                    page_end=section.page_end,
                    text=text,
                    content_type=section.content_type,
                    section_path=tuple(section.section_path),
                    heading=section.section_path[-1] if section.section_path else None,
                    rolling_stock=section.rolling_stock,
                    procedure=section.procedure,
                    context_tags=tags,
                )
            )

    return chunks


def _build_sections(pages: list[PageText]) -> list[_Section]:
    sections: list[_Section] = []
    heading_stack: list[_Heading] = []
    current: _Section | None = None

    def flush() -> None:
        nonlocal current
        if current and current.text:
            sections.append(current)
        current = None

    for page in pages:
        if page.content_type == "table":
            table_path = _path_text(heading_stack) or ["Unsectioned content"]
            table_section = _Section(
                filename=page.filename,
                page_start=page.page_number,
                page_end=page.page_number,
                content_type="table",
                section_path=[*table_path, f"Table {page.table_index or 1}"],
                blocks=[page.text],
                rolling_stock=_infer_rolling_stock(table_path),
                procedure=_infer_procedure(table_path),
                table_index=page.table_index,
            )
            sections.append(table_section)
            continue

        for raw_line in page.text.splitlines():
            line = _normalize_line(raw_line)
            if not line:
                if current and current.blocks and current.blocks[-1] != "":
                    current.blocks.append("")
                continue

            heading = _detect_heading(line)
            if heading:
                flush()
                heading_stack = _push_heading(heading_stack, heading)
                continue

            if current is None:
                path = _path_text(heading_stack) or ["Unsectioned content"]
                current = _Section(
                    filename=page.filename,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    content_type="text",
                    section_path=path,
                    rolling_stock=_infer_rolling_stock(path),
                    procedure=_infer_procedure(path),
                )
            current.page_end = page.page_number
            current.blocks.append(line)

    flush()
    return _merge_tiny_sections(sections)


def _push_heading(stack: list[_Heading], heading: _Heading) -> list[_Heading]:
    kept = [item for item in stack if item.level < heading.level]
    kept.append(heading)
    return kept[-8:]


def _path_text(stack: Iterable[_Heading]) -> list[str]:
    return [item.text for item in stack if item.text]


def _detect_heading(line: str) -> _Heading | None:
    if _looks_like_non_heading(line):
        return None

    numbered = _NUMBERED_HEADING.match(line)
    if numbered:
        number = numbered.group("num")
        title = _clean_heading_title(numbered.group("title"))
        level = min(number.count(".") + 2, 8)
        return _Heading(text=f"{number} {title}".strip(), level=level)

    alpha = _ALPHA_HEADING.match(line)
    if alpha and _heading_case_score(alpha.group("title")) >= 0.35:
        return _Heading(text=line, level=2)

    roman = _ROMAN_HEADING.match(line)
    if roman and _heading_case_score(roman.group("title")) >= 0.35:
        return _Heading(text=line, level=2)

    if _is_uppercase_heading(line):
        return _Heading(text=_clean_heading_title(line), level=1)

    # Common procedure-manual subheading style: "Brake Isolation:".
    if line.endswith(":") and 4 <= len(line) <= 120 and len(line.split()) <= 10:
        lowered = line.casefold()
        if any(word in lowered for word in _PROCEDURE_WORDS | _WARNING_WORDS | _STOCK_WORDS):
            return _Heading(text=_clean_heading_title(line[:-1]), level=3)

    return None


def _looks_like_non_heading(line: str) -> bool:
    if len(line) < 3 or len(line) > 220:
        return True
    if line.startswith("|") or "|---" in line:
        return True
    if re.fullmatch(r"[-–—_=. ]{3,}", line):
        return True
    if re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", line, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\d+", line):
        return True
    # Dense numeric rows are usually table data, not headings.
    tokens = line.split()
    numeric_tokens = sum(bool(re.search(r"\d", token)) for token in tokens)
    if len(tokens) >= 5 and numeric_tokens / max(len(tokens), 1) > 0.55:
        return True
    return False


def _clean_heading_title(value: str) -> str:
    value = _normalize_line(value).strip(" :-–—\t")
    return value[:180]


def _is_uppercase_heading(line: str) -> bool:
    if not (4 <= len(line) <= 140):
        return False
    if line.endswith((".", ";", ",")):
        return False
    words = [word for word in re.split(r"\s+", line) if word]
    if not 1 <= len(words) <= 14:
        return False
    letters = [char for char in line if char.isalpha()]
    if len(letters) < 3:
        return False
    upper_ratio = sum(char.isupper() for char in letters) / len(letters)
    return upper_ratio >= 0.72


def _heading_case_score(value: str) -> float:
    letters = [char for char in value if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char.isupper() for char in letters) / len(letters)


def _merge_tiny_sections(sections: list[_Section]) -> list[_Section]:
    if not sections:
        return []

    merged: list[_Section] = []
    for section in sections:
        if (
            merged
            and section.content_type == "text"
            and merged[-1].content_type == "text"
            and section.filename == merged[-1].filename
            and section.page_start <= merged[-1].page_end + 1
            and len(section.text) < 180
            and _same_parent_path(section.section_path, merged[-1].section_path)
        ):
            parent = merged[-1]
            parent.page_end = max(parent.page_end, section.page_end)
            parent.blocks.extend(["", *section.section_path[-1:], *section.blocks])
            continue
        merged.append(section)
    return merged


def _same_parent_path(left: list[str], right: list[str]) -> bool:
    return left[:-1] == right[:-1]


def _context_header(section: _Section) -> str:
    pages = str(section.page_start) if section.page_start == section.page_end else f"{section.page_start}-{section.page_end}"
    path = " > ".join(section.section_path) if section.section_path else "Unsectioned content"
    lines = [
        "[PDF CHUNK CONTEXT]",
        f"File: {section.filename}",
        f"Pages: {pages}",
        f"Section path: {path}",
        f"Content type: {section.content_type}",
    ]
    if section.rolling_stock:
        lines.append(f"Rolling stock / train context: {section.rolling_stock}")
    if section.procedure:
        lines.append(f"Procedure context: {section.procedure}")
    tags = _context_tags(section, section.text)
    if tags:
        lines.append("Important tags: " + ", ".join(tags[:18]))
    lines.append("[/PDF CHUNK CONTEXT]")
    return "\n".join(lines)


def _context_tags(section: _Section, text: str) -> list[str]:
    values: list[str] = []
    combined = "\n".join([*section.section_path, text[:2500]])
    for match in _ACRONYM_OR_CODE.findall(combined):
        clean = match.strip("-/_")
        if 2 <= len(clean) <= 16 and not clean.isdigit():
            values.append(clean)
    lowered = combined.casefold()
    for word in sorted(_WARNING_WORDS):
        if word in lowered:
            values.append(word.upper())
    if section.rolling_stock:
        values.append(section.rolling_stock)
    if section.procedure:
        values.append(section.procedure)
    return _unique_preserve(values)


def _infer_rolling_stock(path: list[str]) -> str | None:
    for item in reversed(path):
        lowered = item.casefold()
        if any(word in lowered for word in _STOCK_WORDS):
            return _trim_context_value(item)
        match = re.search(
            r"\b(?:RS|TS|EMU|BHEL|BEML|ROTEM|ALSTOM|BOMBARDIER)[ -]?[A-Z0-9/-]{1,20}\b",
            item,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(0).strip()
    return None


def _infer_procedure(path: list[str]) -> str | None:
    for item in reversed(path):
        lowered = item.casefold()
        if any(word in lowered for word in _PROCEDURE_WORDS):
            return _trim_context_value(item)
    return None


def _trim_context_value(value: str) -> str:
    value = re.sub(r"^\d+(?:\.\d+)*\s+", "", value).strip(" :-–—")
    return value[:140]


def _split_section_body(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = _clean_body(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    paragraphs = _paragraphs(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph) + (2 if current else 0)
        if current and current_len + paragraph_len > chunk_size:
            chunks.append("\n\n".join(current).strip())
            current = _overlap_tail(current, overlap)
            current_len = len("\n\n".join(current))

        if len(paragraph) > chunk_size:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(_hard_split(paragraph, chunk_size, overlap))
        else:
            current.append(paragraph)
            current_len += paragraph_len

    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(parts) <= 1:
        # Preserve numbered steps better than a raw character splitter.
        parts = [part.strip() for part in re.split(r"(?=\n?(?:\d+[.)]|[-*•])\s+)", text) if part.strip()]
    return parts or ([text] if text else [])


def _overlap_tail(paragraphs: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for paragraph in reversed(paragraphs):
        tail.insert(0, paragraph)
        total += len(paragraph)
        if total >= overlap:
            break
    return tail[-3:]


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            candidates = [
                text.rfind("\n", start + chunk_size // 2, hard_end),
                text.rfind(". ", start + chunk_size // 2, hard_end),
                text.rfind("; ", start + chunk_size // 2, hard_end),
                text.rfind(" ", start + chunk_size // 2, hard_end),
            ]
            best = max(candidates)
            if best > start:
                end = best + (1 if text[best : best + 2] in {". ", "; "} else 0)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_table_body(text: str, chunk_size: int) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    table_start = next((index for index, line in enumerate(lines) if line.startswith("|")), -1)
    if table_start < 0 or len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    prefix = lines[:table_start]
    table_lines = lines[table_start:]
    if len(table_lines) < 3:
        return _hard_split(text, chunk_size, 0)

    header = table_lines[:2]
    rows = table_lines[2:]
    chunks: list[str] = []
    current = prefix + header

    for row in rows:
        candidate = "\n".join(current + [row])
        if len(candidate) > chunk_size and len(current) > len(prefix) + len(header):
            chunks.append("\n".join(current).strip())
            current = prefix + header + [row]
        else:
            current.append(row)

    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def _normalize_line(line: str) -> str:
    return _MULTISPACE.sub(" ", line.replace("\x00", " ")).strip()


def _clean_body(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_line(line) for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def _unique_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split()).strip(" ,;:")
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
