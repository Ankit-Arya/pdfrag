import re

from app.rag.prompts import NO_ANSWER

_CITATION_PATTERN = re.compile(r"\[S(\d+)]")
_BULLET_PATTERN = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+")
_BOLD_HEADING_PATTERN = re.compile(
    r"^(?:\*\*[^*\n]{1,160}\*\*|__[^_\n]{1,160}__):?\s*$"
)
_TABLE_ROW_PATTERN = re.compile(r"^\|.*\|$")
_TABLE_SEPARATOR_PATTERN = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")


def validate_grounded_answer(answer: str, source_count: int) -> tuple[str, bool]:
    """Validate source labels and citation placement.

    The normalized draft is preserved even when validation fails. The service can
    then attempt repair or return the draft explicitly marked as unverified instead
    of falsely reporting that the PDFs contained no evidence.
    """
    normalized = answer.strip()
    reason = grounding_failure_reason(normalized, source_count)
    return normalized or NO_ANSWER, reason is None


def grounding_failure_reason(answer: str, source_count: int) -> str | None:
    """Return a stable failure reason, or None when the answer is valid."""
    normalized = answer.strip()

    if normalized == NO_ANSWER:
        return "model_reported_no_answer"
    if not normalized:
        return "empty_answer"

    citations = [int(value) for value in _CITATION_PATTERN.findall(normalized)]
    if not citations:
        return "missing_citations"
    if any(citation < 1 or citation > source_count for citation in citations):
        return "citation_out_of_range"
    if not _claim_units_are_reasonably_cited(normalized):
        return "uncited_claim_unit"

    return None


def _claim_units_are_reasonably_cited(answer: str) -> bool:
    for block in re.split(r"\n\s*\n", answer):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        table_rows = [line for line in lines if _TABLE_ROW_PATTERN.match(line)]
        if table_rows:
            factual_rows = [
                line
                for index, line in enumerate(table_rows)
                if index > 0 and not _TABLE_SEPARATOR_PATTERN.match(line)
            ]
            if factual_rows and _CITATION_PATTERN.search(" ".join(lines)) is None:
                return False

            lines = [
                line for line in lines if not _TABLE_ROW_PATTERN.match(line)
            ]

        claim_lines = [line for line in lines if _looks_like_claim(line)]
        if not claim_lines:
            continue

        bullet_lines = [
            line for line in claim_lines if _BULLET_PATTERN.match(line)
        ]
        if bullet_lines:
            if any(_CITATION_PATTERN.search(line) is None for line in bullet_lines):
                return False

            non_bullet_claims = [
                line for line in claim_lines if not _BULLET_PATTERN.match(line)
            ]
            if (
                non_bullet_claims
                and _CITATION_PATTERN.search(" ".join(non_bullet_claims)) is None
            ):
                return False
            continue

        if _CITATION_PATTERN.search(" ".join(claim_lines)) is None:
            return False

    return True


def _looks_like_claim(line: str) -> bool:
    if _HEADING_PATTERN.match(line):
        return False
    if _BOLD_HEADING_PATTERN.match(line):
        return False
    if _CITATION_PATTERN.fullmatch(line):
        return False
    if len(line) <= 80 and line.endswith(":"):
        return False
    return True
