from __future__ import annotations

import re
from collections import OrderedDict

from app.rag.types import PromptSource

_CONTEXT_BLOCK_RE = re.compile(
    r"\[PDF CHUNK CONTEXT]\s*(?P<header>.*?)\s*\[/PDF CHUNK CONTEXT]\s*",
    re.DOTALL | re.IGNORECASE,
)
_SECTION_RE = re.compile(r"^Section path:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_PAGES_RE = re.compile(r"^Pages:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SOURCE_LABEL_RE = re.compile(r"\[S(\d+)]")
_NEW_UNIT_RE = re.compile(
    r"^(?:\(?\d+[.)]|\([a-zivxlcdm]+\)|[a-z][.)]|[-*•])(?:\s+|$)",
    re.IGNORECASE,
)


def build_evidence_answer(sources: list[PromptSource]) -> str:
    """Render broad evidence lookups without LLM paraphrasing or omission."""
    grouped: OrderedDict[tuple[str, str, str], list[tuple[int, str]]] = OrderedDict()

    for source_number, source in enumerate(sources, start=1):
        chunk = source.result.chunk
        header, body = _split_context(source.excerpt)
        pages = _metadata_value(_PAGES_RE, header) or str(chunk.page_number)
        section = (
            " > ".join(chunk.section_path)
            if chunk.section_path
            else _metadata_value(_SECTION_RE, header) or chunk.heading or "Not stated"
        )
        key = (chunk.filename, pages, section)
        grouped.setdefault(key, []).append((source_number, body))

    lines = ["## Information found in the documents"]
    for (filename, pages, section), entries in grouped.items():
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

        seen_units: set[str] = set()
        for source_number, body in entries:
            units = _document_units(body)
            for unit in units:
                normalized = " ".join(unit.casefold().split())
                if not normalized or normalized in seen_units:
                    continue
                seen_units.add(normalized)
                clean = _SOURCE_LABEL_RE.sub(r"(document reference S\1)", unit)
                lines.append(f"- {clean} [S{source_number}]")

    return "\n".join(lines).strip()


def _split_context(value: str) -> tuple[str, str]:
    match = _CONTEXT_BLOCK_RE.search(value)
    if not match:
        return "", value.strip()
    body = value[match.end() :].strip()
    return match.group("header"), body


def _metadata_value(pattern: re.Pattern[str], header: str) -> str:
    match = pattern.search(header)
    return match.group(1).strip() if match else ""


def _single_page(value: str) -> bool:
    return bool(re.fullmatch(r"\d+", value.strip()))


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
    return units or ["No extractable text was available."]
