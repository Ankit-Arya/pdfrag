from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

from app.rag.types import PromptSource

_CONTEXT_BLOCK_RE = re.compile(
    r"\[PDF CHUNK CONTEXT]\s*(?P<header>.*?)\s*\[/PDF CHUNK CONTEXT]\s*",
    re.DOTALL | re.IGNORECASE,
)
_SECTION_RE = re.compile(r"^Section path:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_PAGES_RE = re.compile(r"^Pages:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SOURCE_LABEL_RE = re.compile(r"\[S(\d+)]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:#-]*")
_NEW_UNIT_RE = re.compile(
    r"^(?P<marker>\(?\d+[.)]|\([a-zivxlcdm]+\)|[a-z][.)]|[-*•])(?:\s+|$)",
    re.IGNORECASE,
)
_ARABIC_MARKER_RE = re.compile(r"^\(?\d+[.)](?:\s+|$)")
_SUBITEM_MARKER_RE = re.compile(r"^(?:\([a-zivxlcdm]+\)|[a-z][.)])(?:\s+|$)", re.IGNORECASE)
_TABLE_SEPARATOR_RE = re.compile(r"^\|?(?:\s*:?-+:?\s*\|)+\s*$")
_NOISY_SECTION_RE = re.compile(
    r"(?:THE GAZETTE OF INDIA|\[PART\s+II|DURATIONOF ABSENCE|^\d+$)",
    re.IGNORECASE,
)
_PAGE_HEADER_RE = re.compile(
    r"\b\d{1,4}\s+THE GAZETTE OF INDIA\s*:\s*EXTRAORDINARY.*?(?:\[[^]]+])?",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "process",
    "procedure",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(slots=True)
class _EvidenceUnit:
    source: PromptSource
    source_order: int
    filename: str
    pages: str
    section: str
    text: str


def build_evidence_answer(
    question: str,
    sources: list[PromptSource],
) -> tuple[str, list[PromptSource]]:
    """Render readable broad evidence without LLM paraphrasing or omission."""
    question_terms = _query_terms(question)
    candidates = _collect_units(question_terms, sources)
    candidates = _deduplicate_units(candidates)
    used_sources = _used_sources(candidates)
    source_numbers = {id(source): index for index, source in enumerate(used_sources, 1)}

    grouped: OrderedDict[tuple[str, str, str], list[_EvidenceUnit]] = OrderedDict()
    last_section_by_document: dict[str, tuple[int, str]] = {}
    for unit in candidates:
        section = unit.section
        page_start = _page_start(unit.pages)
        previous = last_section_by_document.get(unit.filename.casefold())
        if section == "Not clearly identified in extracted text" and previous:
            previous_page, previous_section = previous
            if page_start <= previous_page + 1:
                section = f"Continuation of {previous_section}"
        if section != "Not clearly identified in extracted text":
            last_section_by_document[unit.filename.casefold()] = (page_start, section)
        unit.section = section
        grouped.setdefault((unit.filename, unit.pages, section), []).append(unit)

    lines = ["## Information found in the documents"]
    for (filename, pages, section), units in grouped.items():
        page_label = "page" if _single_page(pages) else "pages"
        lines.extend(
            [
                "",
                f"### {filename} — {page_label} {pages}",
                "",
                f"**Heading/subheading:** {section}",
                "",
            ]
        )
        for unit in units:
            source_number = source_numbers[id(unit.source)]
            clean = _SOURCE_LABEL_RE.sub(r"(document reference S\1)", unit.text)
            lines.append(f"- {clean} [S{source_number}]")

    if not candidates:
        return "", []
    return "\n".join(lines).strip(), used_sources


def _collect_units(
    question_terms: set[str],
    sources: list[PromptSource],
) -> list[_EvidenceUnit]:
    collected: list[_EvidenceUnit] = []
    for source_order, source in enumerate(sources):
        chunk = source.result.chunk
        header, body = _split_context(source.excerpt)
        pages = _metadata_value(_PAGES_RE, header) or str(chunk.page_number)
        raw_section = (
            " > ".join(chunk.section_path)
            if chunk.section_path
            else _metadata_value(_SECTION_RE, header) or chunk.heading or ""
        )
        section = _clean_section(raw_section)

        if chunk.content_type == "table":
            units = _readable_table_rows(body, question_terms)
            included = units
        else:
            units = _document_units(body)
            relevant_indexes = {
                index
                for index, unit in enumerate(units)
                if _is_relevant(unit, question_terms)
            }
            included = _expand_structured_context(units, relevant_indexes)

        for unit in included:
            clean = _clean_unit(unit)
            if not clean:
                continue
            collected.append(
                _EvidenceUnit(
                    source=source,
                    source_order=source_order,
                    filename=chunk.filename,
                    pages=pages,
                    section=section,
                    text=clean,
                )
            )
    return collected


def _expand_structured_context(units: list[str], relevant: set[int]) -> list[str]:
    if not relevant:
        return []

    include = set(relevant)
    arabic_count = sum(bool(_ARABIC_MARKER_RE.match(unit)) for unit in units)
    if arabic_count >= 2 and arabic_count >= len(units) * 0.6:
        include.update(range(len(units)))

    for index in sorted(relevant):
        lowered = units[index].casefold()
        if "namely" not in lowered and "following" not in lowered:
            continue
        for following in range(index + 1, len(units)):
            if _ARABIC_MARKER_RE.match(units[following]):
                break
            if _SUBITEM_MARKER_RE.match(units[following]):
                include.add(following)

    return [unit for index, unit in enumerate(units) if index in include]


def _readable_table_rows(value: str, question_terms: set[str]) -> list[str]:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 3:
        return []

    header = _table_cells(table_lines[0])
    if not header or not _TABLE_SEPARATOR_RE.match(table_lines[1]):
        return []

    rows: list[str] = []
    for line in table_lines[2:]:
        cells = _table_cells(line)
        if not cells or not _is_relevant(" ".join(cells), question_terms):
            continue
        fields = [
            f"{header[index]}: {cell}"
            for index, cell in enumerate(cells[: len(header)])
            if cell and index < len(header) and header[index]
        ]
        if fields:
            rows.append("; ".join(fields))
    return rows


def _table_cells(value: str) -> list[str]:
    return [cell.strip() for cell in value.strip().strip("|").split("|")]


def _deduplicate_units(units: list[_EvidenceUnit]) -> list[_EvidenceUnit]:
    selected: list[_EvidenceUnit] = []
    for unit in units:
        normalized = _normalized(unit.text)
        duplicate_index: int | None = None
        for index, old in enumerate(selected):
            if old.filename.casefold() != unit.filename.casefold():
                continue
            old_normalized = _normalized(old.text)
            if normalized in old_normalized or old_normalized in normalized:
                duplicate_index = index
                break
            if _jaccard(_tokens(normalized), _tokens(old_normalized)) >= 0.86:
                duplicate_index = index
                break
        if duplicate_index is None:
            selected.append(unit)
        elif len(normalized) > len(_normalized(selected[duplicate_index].text)):
            selected[duplicate_index] = unit
    selected.sort(key=lambda item: item.source_order)
    return selected


def _used_sources(units: list[_EvidenceUnit]) -> list[PromptSource]:
    seen: set[int] = set()
    result: list[PromptSource] = []
    for unit in units:
        key = id(unit.source)
        if key not in seen:
            seen.add(key)
            result.append(unit.source)
    return result


def _split_context(value: str) -> tuple[str, str]:
    match = _CONTEXT_BLOCK_RE.search(value)
    if not match:
        return "", value.strip()
    return match.group("header"), value[match.end() :].strip()


def _metadata_value(pattern: re.Pattern[str], header: str) -> str:
    match = pattern.search(header)
    return match.group(1).strip() if match else ""


def _clean_section(value: str) -> str:
    parts = [part.strip() for part in value.split(" > ") if part.strip()]
    useful = [
        part
        for part in parts
        if not re.fullmatch(r"Table\s+\d+", part, flags=re.IGNORECASE)
        and not _NOISY_SECTION_RE.search(part)
    ]
    return useful[-1] if useful else "Not clearly identified in extracted text"


def _document_units(value: str) -> list[str]:
    lines = [line.strip() for line in value.replace("\r", "\n").splitlines()]
    units: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            units.append(" ".join(current).strip())
            current.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if _NEW_UNIT_RE.match(line):
            flush()
        current.append(line)
    flush()
    return units


def _clean_unit(value: str) -> str:
    value = _PAGE_HEADER_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip(" |")
    if not value or _TABLE_SEPARATOR_RE.match(value):
        return ""
    return value


def _query_terms(value: str) -> set[str]:
    return {term for term in _tokens(value) if term not in _STOPWORDS}


def _tokens(value: str) -> set[str]:
    result: set[str] = set()
    for raw in _TOKEN_RE.findall(value):
        token = raw.casefold()
        result.add(token)
        result.update(part for part in re.split(r"[._/:#-]+", token) if part)
    return result


def _is_relevant(value: str, question_terms: set[str]) -> bool:
    if not question_terms:
        return True
    value_terms = _tokens(value)
    required = 1 if len(question_terms) == 1 else 2
    return len(question_terms & value_terms) >= min(required, len(question_terms))


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _single_page(value: str) -> bool:
    return bool(re.fullmatch(r"\d+", value.strip()))


def _page_start(value: str) -> int:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else 0
