from __future__ import annotations

import re
from collections import OrderedDict

from app.rag.types import PromptSource

_CONTEXT_BLOCK_RE = re.compile(
    r"\[PDF CHUNK CONTEXT\]\s*(?P<header>.*?)\s*\[/PDF CHUNK CONTEXT\]\s*",
    re.IGNORECASE | re.DOTALL,
)
_PAGES_RE = re.compile(r"^Pages:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SECTION_RE = re.compile(r"^Section path:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

def clean_display_excerpt(value: str) -> str:
    """Return the human-readable PDF excerpt without synthetic retrieval metadata.

    This is intentionally display-only. The model continues to receive the original
    chunk, including its structural context envelope. Keeping the two representations
    separate prevents UI formatting changes from affecting retrieval or grounding.
    """
    _, body = _split_context(value)
    clean = (body or value).strip()
    # Be defensive about malformed/duplicated envelopes from older indexed chunks.
    clean = _CONTEXT_BLOCK_RE.sub("", clean).strip()
    return clean



def format_prompt_sources_markdown(
    sources: list[PromptSource],
    *,
    source_numbers: set[int] | None = None,
) -> str:
    """Render reviewed/cited chunks as readable document evidence.

    The answer model still receives the original chunk text. This function is only
    for UI transparency: it removes the synthetic [PDF CHUNK CONTEXT] envelope and
    groups excerpts by document/page/section like the earlier evidence presentation.
    No chunk selected for display is silently dropped.
    """
    groups: OrderedDict[tuple[str, str, str], list[tuple[int, PromptSource, str]]] = OrderedDict()
    for index, source in enumerate(sources, 1):
        if source_numbers is not None and index not in source_numbers:
            continue
        header, body = _split_context(source.excerpt)
        chunk = source.result.chunk
        pages = _metadata_value(_PAGES_RE, header) or str(chunk.page_number)
        section = _metadata_value(_SECTION_RE, header)
        if not section and chunk.section_path:
            section = " > ".join(chunk.section_path)
        if not section:
            section = chunk.heading or ""
        clean_body = clean_display_excerpt(source.excerpt)
        groups.setdefault((chunk.filename, pages, section), []).append(
            (index, source, clean_body)
        )

    lines: list[str] = []
    for (filename, pages, section), entries in groups.items():
        page_label = "page" if re.fullmatch(r"\d+", pages.strip()) else "pages"
        if lines:
            lines.append("")
        lines.append(f"### {filename} — {page_label} {pages}")
        if section:
            lines.extend(["", f"#### {section}"])

        for index, source, body in entries:
            lines.append("")
            if source.result.chunk.content_type == "table" or _looks_like_markdown_table(body):
                # Preserve actual PDF table structure, then attach the original
                # source label below it.
                lines.append(body)
                lines.append(f"*Source: [S{index}]*")
            else:
                # Earlier evidence presentation used readable bullets grouped by
                # document/page/section. Collapse extraction line-wraps without
                # exposing the database chunk envelope.
                clean = " ".join(body.split())
                lines.append(f"- {clean} **[S{index}]**")

    return "\n".join(lines).strip()


def _split_context(value: str) -> tuple[str, str]:
    match = _CONTEXT_BLOCK_RE.search(value)
    if not match:
        return "", value.strip()
    return match.group("header"), value[match.end() :].strip()


def _metadata_value(pattern: re.Pattern[str], header: str) -> str:
    match = pattern.search(header)
    return match.group(1).strip() if match else ""


def _looks_like_markdown_table(value: str) -> bool:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return "|" in lines[0] and bool(re.match(r"^\|?(?:\s*:?-{3,}:?\s*\|)+", lines[1]))
