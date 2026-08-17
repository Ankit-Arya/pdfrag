from __future__ import annotations

# ruff: noqa: E501

import re
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]{1,9})(?![A-Za-z0-9])")
_DOC_CODE_RE = re.compile(r"\b(?:SC|SM|SOP|JPO|INST(?:RUCTION)?|MRGR)(?:\s*[-_/]\s*|\s+)\d+[A-Z]?\b", re.IGNORECASE)
_BARE_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
_DEFINITION_CUE_RE = re.compile(
    r"\b(?:full\s+form|stands?\s+for|meaning|means|define|definition|expand|expansion)\b",
    re.IGNORECASE,
)
_DEFINITION_TARGET_PATTERNS = (
    re.compile(r"\bfull\s+form\s+(?:of\s+)?(?P<alias>[A-Za-z][A-Za-z0-9]{1,9})\b", re.IGNORECASE),
    re.compile(r"\b(?P<alias>[A-Za-z][A-Za-z0-9]{1,9})\s+(?:full\s+form|meaning|means|stands?\s+for)\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+is\s+(?P<alias>[A-Za-z][A-Za-z0-9]{1,9})\s*[?.!]*\s*$", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+the\s+full\s+form\s+of\s+(?P<alias>[A-Za-z][A-Za-z0-9]{1,9})\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+(?P<alias>[A-Za-z][A-Za-z0-9]{1,9})\s+mean\b", re.IGNORECASE),
    re.compile(r"\b(?:define|expand)\s+(?P<alias>[A-Za-z][A-Za-z0-9]{1,9})\b", re.IGNORECASE),
)
_STOP = {
    "if", "in", "is", "it", "of", "on", "or", "to", "do", "go", "no", "so", "up",
    "and", "are", "can", "for", "from", "has", "have", "how", "the", "this", "what",
    "when", "where", "which", "with", "would", "should", "train", "line", "status",
    "full", "form", "meaning", "means", "define", "definition", "expand", "expansion",
}


@dataclass(frozen=True, slots=True)
class ResolvedTerm:
    alias: str
    canonical: str
    concept_type: str
    confidence: float
    evidence_count: int
    ambiguous: bool = False


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+", value.casefold()))


def _hint_aliases(hints: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        if "=" not in hint:
            continue
        alias = " ".join(hint.split("=", 1)[0].split()).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,9}", alias):
            continue
        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(alias)
    return result


def candidate_aliases(value: str) -> list[str]:
    without_codes = _DOC_CODE_RE.sub(" ", value)
    result: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(without_codes):
        token = match.group(1)
        lowered = token.casefold()
        if lowered in seen:
            continue
        if lowered in _STOP and not token.isupper():
            continue
        # Prefer explicit uppercase acronyms, while still allowing users to type
        # known short aliases in lowercase ("sc", "occ"). Database lookup is the
        # authority; unknown ordinary words simply return no rows.
        if not (token.isupper() or 2 <= len(token) <= 5):
            continue
        seen.add(lowered)
        result.append(token)
    return result[:12]


def definition_request_aliases(value: str, hints: Iterable[str] = ()) -> list[str]:
    """Return acronym/short-term targets when the user is asking for a definition.

    Bare uppercase acronyms are treated as definition requests. Lowercase bare terms
    are only promoted when corpus-derived abbreviation hints already establish that
    the token is an organisation abbreviation. Explicit cues such as ``full form``
    work in either case.
    """
    normalized = " ".join(value.split()).strip()
    if not normalized or _DOC_CODE_RE.fullmatch(normalized):
        return []

    targets: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        token = raw.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,9}", token):
            return
        lowered = token.casefold()
        if lowered in _STOP or lowered in seen:
            return
        seen.add(lowered)
        targets.append(token.upper() if token.isalpha() else token)

    for pattern in _DEFINITION_TARGET_PATTERNS:
        for match in pattern.finditer(normalized):
            add(match.group("alias"))

    if _BARE_ACRONYM_RE.fullmatch(normalized):
        add(normalized)

    hinted = _hint_aliases(hints)
    if len(normalized.split()) == 1:
        for alias in hinted:
            if alias.casefold() == normalized.casefold():
                add(alias)

    # For phrasing such as ``BIC full form`` the regex above catches the target.
    # If a definition cue is present but unusual word order is used, fall back to
    # explicit uppercase candidates rather than treating "full"/"form" as aliases.
    if _DEFINITION_CUE_RE.search(normalized):
        for token in candidate_aliases(normalized):
            if token.isupper():
                add(token)

    return targets[:6]


def is_definition_request(value: str, hints: Iterable[str] = ()) -> bool:
    return bool(definition_request_aliases(value, hints))


def resolve_terms(db: Session, value: str) -> list[ResolvedTerm]:
    aliases = candidate_aliases(value)
    if not aliases:
        return []
    norms = [_norm(alias) for alias in aliases]
    params: dict[str, object] = {}
    placeholders: list[str] = []
    for index, norm in enumerate(norms):
        key = f"a{index}"
        params[key] = norm
        placeholders.append(f":{key}")

    rows = list(
        db.execute(
            text(
                f"""
                SELECT alias_norm, min(alias) AS alias, canonical_name, concept_type,
                       count(*) AS evidence_count,
                       max(CASE WHEN verified THEN 1 ELSE 0 END) AS verified_rank,
                       avg(confidence) AS confidence
                FROM rag_terminology
                WHERE alias_norm IN ({', '.join(placeholders)})
                GROUP BY alias_norm, canonical_name, concept_type
                ORDER BY alias_norm,
                         max(CASE WHEN verified THEN 1 ELSE 0 END) DESC,
                         count(*) DESC,
                         avg(confidence) DESC,
                         canonical_name
                """
            ),
            params,
        ).mappings()
    )
    grouped: dict[str, list[object]] = {}
    for row in rows:
        grouped.setdefault(str(row["alias_norm"]), []).append(row)

    resolved: list[ResolvedTerm] = []
    for original in aliases:
        candidates = grouped.get(_norm(original), [])
        if not candidates:
            continue
        best = candidates[0]
        best_count = int(best["evidence_count"] or 0)
        second_count = int(candidates[1]["evidence_count"] or 0) if len(candidates) > 1 else 0
        verified = bool(int(best["verified_rank"] or 0))
        second_verified = bool(int(candidates[1]["verified_rank"] or 0)) if len(candidates) > 1 else False
        # Auto-canonicalize only when one corpus meaning clearly dominates. Multiple
        # verified meanings are explicitly ambiguous because verification confirms
        # that both expansions are legitimate in some organisational scope.
        ambiguous = len(candidates) > 1 and (
            (verified and second_verified)
            or (not verified and second_count >= max(1, int(best_count * 0.65)))
        )
        resolved.append(
            ResolvedTerm(
                alias=original.upper() if original.isalpha() else original,
                canonical=str(best["canonical_name"]),
                concept_type=str(best["concept_type"]),
                confidence=float(best["confidence"] or 0.0),
                evidence_count=best_count,
                ambiguous=ambiguous,
            )
        )
        if ambiguous:
            second = candidates[1]
            resolved.append(
                ResolvedTerm(
                    alias=original.upper() if original.isalpha() else original,
                    canonical=str(second["canonical_name"]),
                    concept_type=str(second["concept_type"]),
                    confidence=float(second["confidence"] or 0.0),
                    evidence_count=second_count,
                    ambiguous=True,
                )
            )
    return resolved


def terminology_hints(db: Session, value: str) -> list[str]:
    result: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for term in resolve_terms(db, value):
        suffix = " (ambiguous corpus meaning; use context)" if term.ambiguous else ""
        result.append(
            f"{term.alias} = {term.canonical} — organisation terminology index, "
            f"{term.evidence_count} supporting definition(s){suffix}"
        )
        seen_pairs.add((_norm(term.alias), _norm(term.canonical)))

    # Reverse lookup: if the user writes the full organisational term while an
    # answer document uses only its abbreviation, expose the same alias pair.
    normalized = _norm(value)
    if normalized:
        reverse_rows = db.execute(
            text(
                """
                SELECT min(alias) AS alias, canonical_name, concept_type,
                       count(*) AS evidence_count, avg(confidence) AS confidence
                FROM rag_terminology
                WHERE length(canonical_norm) >= 5
                  AND position(canonical_norm in :normalized) > 0
                GROUP BY canonical_norm, canonical_name, concept_type
                ORDER BY count(*) DESC, avg(confidence) DESC, canonical_name
                LIMIT 12
                """
            ),
            {"normalized": normalized},
        ).mappings()
        for row in reverse_rows:
            alias = str(row["alias"])
            canonical = str(row["canonical_name"])
            pair = (_norm(alias), _norm(canonical))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            result.append(
                f"{alias} = {canonical} — organisation terminology index, "
                f"{int(row['evidence_count'] or 0)} supporting definition(s)"
            )
    return result[:16]


def _definition_for_token(text_value: str, token: str) -> str:
    escaped = re.escape(token)
    word = r"[A-Za-z][A-Za-z0-9/&.'-]*"
    before = re.compile(
        rf"((?:{word}[ \t]+){{1,7}}{word})[ \t]*\([ \t]*{escaped}[ \t]*\)",
        re.IGNORECASE,
    )
    after = re.compile(
        rf"(?<![A-Za-z0-9]){escaped}[ \t]*\([ \t]*((?:{word}[ \t]+){{0,7}}{word})[ \t]*\)",
        re.IGNORECASE,
    )
    direct = re.compile(
        rf"(?im)(?<![A-Za-z0-9]){escaped}[ \t]*(?:[-:–—]|means\b|stands\s+for\b)[ \t]*"
        rf"((?:{word}[ \t]+){{0,7}}{word})"
    )
    table = re.compile(
        rf"(?im)^\s*\|?\s*{escaped}\s*\|\s*((?:{word}[ \t]+){{0,7}}{word})\s*\|?\s*$"
    )
    for pattern in (before, after, direct, table):
        match = pattern.search(text_value[:16000])
        if not match:
            continue
        expansion = re.sub(r"\s+", " ", match.group(1)).strip(" -,:;|")
        if 3 <= len(expansion) <= 120 and expansion.casefold() != token.casefold():
            return expansion
    return ""


def explicit_definition_hints(
    db: Session,
    value: str,
    *,
    existing_hints: Iterable[str] = (),
    max_chunks_per_alias: int = 120,
) -> list[str]:
    """Definition-specific corpus fallback for incomplete terminology backfills.

    The legacy abbreviation scan orders by filename/page and can exhaust its small
    limit on usage-only occurrences (``BIC isolation``) before reaching a definition.
    This path explicitly prioritises parenthetical/definition-shaped occurrences.
    """
    aliases = definition_request_aliases(value, existing_hints)
    if not aliases:
        aliases = [token for token in candidate_aliases(value) if token.isupper()]
    result: list[str] = []
    seen: set[tuple[str, str]] = set()

    sql = text(
        """
        SELECT c.text, c.page_number, d.filename
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'ready'
          AND c.text ~* :token_pattern
        ORDER BY
          CASE
            WHEN c.text ~* :paren_pattern THEN 0
            WHEN c.text ~* :leading_pattern THEN 1
            WHEN c.text ~* :direct_pattern THEN 2
            ELSE 3
          END,
          lower(d.filename), c.page_number, c.chunk_index
        LIMIT :chunk_limit
        """
    )

    for alias in aliases[:6]:
        escaped = re.escape(alias)
        token_pattern = rf"(^|[^A-Za-z0-9]){escaped}([^A-Za-z0-9]|$)"
        params = {
            "token_pattern": token_pattern,
            "paren_pattern": rf"\([[:space:]]*{escaped}[[:space:]]*\)",
            "leading_pattern": rf"(^|[^A-Za-z0-9]){escaped}[[:space:]]*\(",
            "direct_pattern": rf"(^|[^A-Za-z0-9]){escaped}[[:space:]]*([-:–—]|means|stands[[:space:]]+for)",
            "chunk_limit": max_chunks_per_alias,
        }
        for row in db.execute(sql, params).mappings():
            canonical = _definition_for_token(str(row["text"]), alias)
            if not canonical:
                continue
            pair = (_norm(alias), _norm(canonical))
            if pair in seen:
                continue
            seen.add(pair)
            result.append(
                f"{alias.upper()} = {canonical} — explicit PDF definition in "
                f"{row['filename']} p.{int(row['page_number'])}"
            )
            if len(result) >= 12:
                return result
    return result


def canonicalize_query(db: Session, value: str) -> tuple[str, list[ResolvedTerm]]:
    terms = resolve_terms(db, value)
    dominant: list[ResolvedTerm] = []
    seen_alias: set[str] = set()
    for term in terms:
        key = term.alias.casefold()
        if key in seen_alias or term.ambiguous:
            continue
        seen_alias.add(key)
        dominant.append(term)
    if not dominant:
        return value, terms
    expansions = "; ".join(f"{term.alias} means {term.canonical}" for term in dominant[:8])
    return f"{value}. Organisation terminology: {expansions}.", terms
