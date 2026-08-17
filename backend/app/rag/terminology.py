from __future__ import annotations

# ruff: noqa: E501

import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]{1,9})(?![A-Za-z0-9])")
_DOC_CODE_RE = re.compile(r"\b(?:SC|SM|SOP|JPO|INST(?:RUCTION)?|MRGR)(?:\s*[-_/]\s*|\s+)\d+[A-Z]?\b", re.IGNORECASE)
_STOP = {
    "if", "in", "is", "it", "of", "on", "or", "to", "do", "go", "no", "so", "up",
    "and", "are", "can", "for", "from", "has", "have", "how", "the", "this", "what",
    "when", "where", "which", "with", "would", "should", "train", "line", "status",
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
        # known short aliases in lowercase ("sc", "occ").  Database lookup is the
        # authority; unknown ordinary words simply return no rows.
        if not (token.isupper() or 2 <= len(token) <= 5):
            continue
        seen.add(lowered)
        result.append(token)
    return result[:12]


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
