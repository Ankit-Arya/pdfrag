from __future__ import annotations

# ruff: noqa: E501

import re
from dataclasses import dataclass
from typing import Iterable, Protocol


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_QUOTE = r'["“”\']'
_WORD_SUB_RE = re.compile(
    r"for\s+the\s+words?\s+(?P<old>.{1,120}?)\s*,\s*"
    r"the\s+words?\s+(?P<new>.{1,180}?)\s+shall\s+be\s+substituted",
    re.IGNORECASE | re.DOTALL,
)
_DIRECT_SUB_RE = re.compile(
    rf"{_QUOTE}(?P<old>.{{1,140}}?){_QUOTE}\s+shall\s+be\s+(?:substituted|replaced)\s+by\s+"
    rf"{_QUOTE}(?P<new>.{{1,140}}?){_QUOTE}",
    re.IGNORECASE | re.DOTALL,
)
_SECTION_SUB_RE = re.compile(
    r"(?:for|in)\s+the\s+(?P<target>"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?)\s+schedule"
    r"|schedule\s+[A-Za-z0-9-]+"
    r"|rule\s+\d+[A-Za-z]?"
    r"|section\s+\d+[A-Za-z]?"
    r"|clause\s*\([^)]+\)"
    r"|sub-?rule\s*\([^)]+\)"
    r")"
    r"(?:\s+of\s+the\s+[^,.;]{1,100})?\s*,?\s*"
    r"(?:the\s+following\s+(?:schedule|rule|section|clause|provision)[^.;]{0,80}\s+)?"
    r"shall\s+be\s+(?:substituted|replaced)",
    re.IGNORECASE | re.DOTALL,
)
_OMIT_RE = re.compile(
    r"(?P<target>rule\s+\d+[A-Za-z]?|section\s+\d+[A-Za-z]?|clause\s*\([^)]+\)|sub-?rule\s*\([^)]+\))"
    r"[^.;]{0,120}\s+shall\s+be\s+(?:omitted|deleted)",
    re.IGNORECASE | re.DOTALL,
)
_RULE_CONTEXT_RE = re.compile(r"\b(?:in|ln|of)\s+rule\s+(?P<rule>[0-9lI]{1,3}[A-Za-z]?)\b", re.IGNORECASE)
_AMENDMENT_RE = re.compile(r"\bamend(?:ment|ed|ing)?\b", re.IGNORECASE)
_BOUNDARY_RE = re.compile(
    r"\bprincipal\s+rules?\s+(?:were|was)\s+published\b|"
    r"\bprincipal\s+(?:act|rules?|regulations?)\b.*\bpublished\b",
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"\b(?:rules?|regulations?|manual|code)\s*,?\s*(19\d{2}|20\d{2})\b", re.IGNORECASE)
_NOTIFICATION_RE = re.compile(r"(?im)^\s*(?:MINISTRY\s+OF\s+.+\n)?\s*NOTIFICATION\s*$")


@dataclass(frozen=True, slots=True)
class AuthorityDirective:
    directive_type: str
    target: str
    old_text: str = ""
    new_text: str = ""
    effective_year: int | None = None
    anchor_chunk_index: int = 0
    page_number: int = 1
    span_start_chunk: int = 0
    span_end_chunk: int = 0
    confidence: float = 0.99


class AuthorityRow(Protocol):
    chunk_index: int
    page_number: int
    text: str


def normalize_authority_text(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+", value.casefold()))


def _clean(value: str) -> str:
    return " ".join(value.split()).strip(" \t\r\n-:;,.\"'“”")[:240]


def _clean_substitution_phrase(value: str) -> str:
    cleaned = " ".join(value.split())
    cleaned = re.sub(r"^[\s.:'\"“”]+", "", cleaned)
    cleaned = re.sub(r"[\s,.;:'\"“”]+$", "", cleaned)
    return cleaned[:180]


def _normalize_rule_number(value: str) -> str:
    token = value.strip()
    if re.search(r"\d", token):
        token = token.replace("l", "1").replace("I", "1")
    return token


def effective_year(value: str) -> int | None:
    years = [int(match.group(1)) for match in _YEAR_RE.finditer(value)]
    return max(years) if years else None


def extract_authority_directives(
    text_value: str,
    *,
    chunk_index: int = 0,
    page_number: int = 1,
) -> list[AuthorityDirective]:
    """Extract only explicit replacement/omission instructions from source text.

    These directives are navigation/precedence metadata. They never become answer
    evidence by themselves; the original anchor chunk remains the cited evidence.
    """
    body = " ".join(text_value.split())
    if not body:
        return []
    year = effective_year(body)
    found: list[AuthorityDirective] = []
    seen: set[tuple[str, str, str, str]] = set()

    for pattern in (_WORD_SUB_RE, _DIRECT_SUB_RE):
        for match in pattern.finditer(body):
            old_text = _clean_substitution_phrase(match.group("old"))
            new_text = _clean_substitution_phrase(match.group("new"))
            if not old_text or not new_text or old_text.casefold() == new_text.casefold():
                continue
            # Associate each substitution with the nearest preceding rule reference,
            # not the first rule mentioned in a long OCR chunk.
            context_window = body[max(0, match.start() - 260):match.start()]
            context_matches = list(_RULE_CONTEXT_RE.finditer(context_window))
            if context_matches:
                raw_rule = _normalize_rule_number(context_matches[-1].group("rule"))
                target = f"rule {raw_rule}"
            else:
                target = "wording"
            key = ("replace_words", normalize_authority_text(target), normalize_authority_text(old_text), normalize_authority_text(new_text))
            if key in seen:
                continue
            seen.add(key)
            found.append(
                AuthorityDirective(
                    directive_type="replace_words",
                    target=target,
                    old_text=old_text,
                    new_text=new_text,
                    effective_year=year,
                    anchor_chunk_index=chunk_index,
                    page_number=page_number,
                    span_start_chunk=chunk_index,
                    span_end_chunk=chunk_index,
                    confidence=0.995,
                )
            )

    for match in _SECTION_SUB_RE.finditer(body):
        target = _clean(match.group("target"))
        key = ("replace_section", normalize_authority_text(target), "", "")
        if not target or key in seen:
            continue
        seen.add(key)
        found.append(
            AuthorityDirective(
                directive_type="replace_section",
                target=target,
                effective_year=year,
                anchor_chunk_index=chunk_index,
                page_number=page_number,
                span_start_chunk=chunk_index,
                span_end_chunk=chunk_index,
                confidence=0.995,
            )
        )

    for match in _OMIT_RE.finditer(body):
        target = _clean(match.group("target"))
        key = ("omit", normalize_authority_text(target), "", "")
        if not target or key in seen:
            continue
        seen.add(key)
        found.append(
            AuthorityDirective(
                directive_type="omit",
                target=target,
                effective_year=year,
                anchor_chunk_index=chunk_index,
                page_number=page_number,
                span_start_chunk=chunk_index,
                span_end_chunk=chunk_index,
                confidence=0.99,
            )
        )
    return found


def looks_like_subdocument_boundary(text_value: str, *, anchor_year: int | None = None) -> bool:
    """Conservatively identify the start of another appended/base instrument."""
    body = " ".join(text_value.split())
    if not body:
        return False
    if _BOUNDARY_RE.search(body):
        return True
    title_years = [int(value) for value in _TITLE_RE.findall(body)]
    if anchor_year and title_years and min(title_years) < anchor_year and not _AMENDMENT_RE.search(body):
        return True
    if anchor_year and _NOTIFICATION_RE.search(text_value):
        years = [int(match.group(1)) for match in _YEAR_RE.finditer(body)]
        if years and max(years) < anchor_year:
            return True
    return False


def replacement_span_end(
    rows: Iterable[AuthorityRow],
    *,
    anchor_chunk_index: int,
    anchor_year: int | None,
    max_chunks: int = 48,
) -> int:
    """Return a conservative end for replacement text following an authority anchor."""
    ordered = sorted(rows, key=lambda row: row.chunk_index)
    end = anchor_chunk_index
    seen_after_anchor = 0
    for row in ordered:
        if row.chunk_index < anchor_chunk_index:
            continue
        if row.chunk_index > anchor_chunk_index + max_chunks:
            break
        if row.chunk_index > anchor_chunk_index:
            seen_after_anchor += 1
            if seen_after_anchor >= 2 and looks_like_subdocument_boundary(row.text, anchor_year=anchor_year):
                # OCR/chunking can place the tail of the replacement table and the
                # note introducing the appended principal rules in the same chunk.
                # Keep that chunk when the boundary begins well after useful content;
                # exclude it when a new instrument starts near the beginning.
                compact = " ".join(row.text.split())
                boundary_match = _BOUNDARY_RE.search(compact)
                boundary_at = boundary_match.start() if boundary_match else 0
                if boundary_match and boundary_at >= max(120, int(len(compact) * 0.30)):
                    end = row.chunk_index
                break
        end = row.chunk_index
    return max(anchor_chunk_index, end)
