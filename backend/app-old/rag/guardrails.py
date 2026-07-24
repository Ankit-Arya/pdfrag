import re

from app.rag.prompts import NO_ANSWER

_CITATION_PATTERN = re.compile(r"\[S(\d+)]")
_BULLET_PATTERN = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+")


def validate_grounded_answer(answer: str, source_count: int) -> tuple[str, bool]:
    """Validate citation shape without over-rejecting useful grounded answers.

    The old validator failed closed whenever an intro line, heading, or label lacked
    a citation. This version still requires at least one valid citation and rejects
    out-of-range source labels, but it ignores non-claim structure such as headings
    and short lead-in lines ending with a colon.
    """
    normalized = answer.strip()
    if normalized == NO_ANSWER:
        return normalized, False
    if not normalized:
        return NO_ANSWER, False

    citations = [int(value) for value in _CITATION_PATTERN.findall(normalized)]
    if not citations:
        return normalized, False
    if any(citation < 1 or citation > source_count for citation in citations):
        return normalized, False
    if not _claim_units_are_reasonably_cited(normalized):
        return normalized, False
    return normalized, True


def _claim_units_are_reasonably_cited(answer: str) -> bool:
    for block in re.split(r"\n\s*\n", answer):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        claim_lines = [line for line in lines if _looks_like_claim(line)]
        if not claim_lines:
            continue

        bullet_lines = [line for line in claim_lines if _BULLET_PATTERN.match(line)]
        if bullet_lines:
            if any(_CITATION_PATTERN.search(line) is None for line in bullet_lines):
                return False
            non_bullet_claims = [line for line in claim_lines if not _BULLET_PATTERN.match(line)]
            if non_bullet_claims and _CITATION_PATTERN.search(" ".join(non_bullet_claims)) is None:
                return False
            continue

        if _CITATION_PATTERN.search(" ".join(claim_lines)) is None:
            return False

    return True


def _looks_like_claim(line: str) -> bool:
    if _HEADING_PATTERN.match(line):
        return False
    if _CITATION_PATTERN.fullmatch(line):
        return False
    if len(line) <= 80 and line.endswith(":"):
        return False
    return True
