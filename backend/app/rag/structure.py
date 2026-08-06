from __future__ import annotations

import re

from app.rag.normalization import canonical_phrase

_SECTION_RE = re.compile(r"^Section path:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def section_path_from_text(value: str) -> str:
    """Return the machine-readable section path carried by a stored chunk."""
    match = _SECTION_RE.search(value[:4000])
    return match.group(1).strip() if match else ""


def section_match_score(query: str, section_path: str) -> float:
    """Score a query against individual heading levels, not arbitrary body text."""
    normalized_query = normalize_heading(query)
    if len(normalized_query) < 3:
        return 0.0

    query_terms = normalized_query.split()
    best = 0.0
    for part in section_parts(section_path):
        normalized_part = normalize_heading(part)
        if not normalized_part:
            continue
        if normalized_query == normalized_part:
            return 1.0
        if len(query_terms) >= 2 and _contains_phrase(normalized_part, normalized_query):
            best = max(best, 0.88)
        elif len(query_terms) >= 2 and _contains_phrase(normalized_query, normalized_part):
            best = max(best, 0.72)
    return best


def major_section_match_score(query: str, section_path: str) -> float:
    """Match chapter-like headings while rejecting incidental prose mentions."""
    normalized_query = normalize_heading(query)
    if len(normalized_query.split()) < 2:
        return 0.0

    best = 0.0
    for part in section_parts(section_path):
        if not is_major_heading(part):
            continue
        normalized_part = normalize_heading(part)
        if normalized_query == normalized_part:
            return 1.0
        if _contains_phrase(normalized_part, normalized_query):
            best = max(best, 0.9)
    return best


def section_parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(" > ") if part.strip()]


def normalize_heading(value: str) -> str:
    return canonical_phrase(value)


def is_major_heading(value: str) -> bool:
    """Identify visually major headings such as PDF chapter titles."""
    letters = [char for char in value if char.isalpha()]
    words = _WORD_RE.findall(value)
    if len(letters) < 4 or not 2 <= len(words) <= 18:
        return False
    return sum(char.isupper() for char in letters) / len(letters) >= 0.72


def _contains_phrase(container: str, phrase: str) -> bool:
    return f" {phrase} " in f" {container} "
