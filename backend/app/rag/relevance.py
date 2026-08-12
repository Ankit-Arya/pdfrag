from __future__ import annotations

import re
from dataclasses import dataclass, replace

from app.config import get_settings
from app.rag.normalization import search_terms
from app.rag.structure import (
    major_section_match_score,
    section_match_score,
    section_path_from_text,
)
from app.rag.types import QueryPlan, RetrievedChunk, TextChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:#-]*")
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,80})[\"']")
_SECTION_RE = re.compile(r"^Section path:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_CONTEXT_LINE_RE = re.compile(
    r"^(?:Rolling stock / train context|Procedure context):\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "should",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}

_INTENT_CUES = {
    "procedure": {
        "action",
        "after",
        "announce",
        "assist",
        "authorization",
        "before",
        "check",
        "confirm",
        "ensure",
        "hand",
        "hold",
        "inform",
        "instruction",
        "isolate",
        "permission",
        "procedure",
        "regulate",
        "remove",
        "report",
        "reset",
        "restore",
        "resume",
        "step",
        "stop",
        "suspend",
        "verify",
        "warning",
    },
    "troubleshooting": {
        "alarm",
        "cause",
        "check",
        "diagnosis",
        "error",
        "failure",
        "fault",
        "remedy",
        "symptom",
        "troubleshooting",
    },
    "comparison": {
        "advantage",
        "compare",
        "difference",
        "disadvantage",
        "versus",
    },
    "requirement": {
        "mandatory",
        "must",
        "permission",
        "prerequisite",
        "prohibited",
        "requirement",
        "required",
        "rule",
        "shall",
    },
    "definition": {"definition", "means", "refers", "term"},
    "summary": {"conclusion", "overview", "scope", "summary"},
    "list": {
        "carry",
        "carried",
        "contain",
        "contents",
        "equipment",
        "include",
        "item",
        "items",
        "keep",
        "kept",
        "list",
        "possession",
        "required",
        "types",
    },
    "fact_lookup": set(),
}


@dataclass(frozen=True, slots=True)
class _ScoredChunk:
    result: RetrievedChunk
    relevance: float
    coverage: float
    anchor_coverage: float
    intent_evidence: float
    section_match: float
    major_section_match: float
    context_key: tuple[str, str]


def select_context_chunks(
    plan: QueryPlan,
    candidates: list[RetrievedChunk],
    max_chunks: int | None = None,
    *,
    preferred_document_ids: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Keep all materially relevant evidence instead of treating top-K as truth."""
    if not candidates:
        return []

    focus_terms = _focus_terms(plan)
    anchor_groups = _anchor_groups(plan)
    preferred_ids = preferred_document_ids or set()
    scored = [
        _score_candidate(plan, candidate, focus_terms, anchor_groups, preferred_ids)
        for candidate in candidates
    ]
    scored.sort(key=_scored_sort_key)

    major_matches = [item for item in scored if item.major_section_match >= 0.9]
    if plan.search_mode == "references" and major_matches:
        scored = major_matches

    best = scored[0].relevance
    if best < 0.08:
        return []

    if plan.search_mode == "references":
        eligible = [
            item
            for item in scored
            if item.coverage > 0
            or item.section_match >= 0.70
            or item.result.keyword_score > 0
            or "corpus-fts" in item.result.method
        ]
    else:
        cutoff = max(0.10, best * 0.32)
        eligible = []
        for item in scored:
            preferred = _is_preferred(item.result.chunk, preferred_ids)
            has_focus = (
                item.coverage >= 0.25
                or item.section_match >= 0.82
                or not focus_terms
                or (preferred and plan.intent == "procedure" and item.intent_evidence >= 0.05)
            )
            required_anchor_coverage = (
                1.0 / len(anchor_groups)
                if anchor_groups and plan.intent == "comparison"
                else 1.0
            )
            has_anchor = item.anchor_coverage >= required_anchor_coverage or not anchor_groups
            has_intent = item.intent_evidence >= 0.08
            constraint_ok = preferred or item.section_match >= 0.82 or (
                has_anchor if anchor_groups else (has_focus or has_intent)
            )
            if item.relevance >= cutoff and has_focus and constraint_ok:
                eligible.append(item)

    if not eligible:
        top = scored[0]
        if top.coverage >= 0.20 or top.anchor_coverage > 0 or top.section_match >= 0.75:
            eligible = [top]
        else:
            return []

    limit = _selection_limit(plan, max_chunks)
    selected = _preferred_then_diverse_selection(
        eligible,
        limit,
        preferred_ids,
        plan.intent,
    )

    # Pull same-section / adjacent continuations into remaining capacity. A
    # continuation may omit the acronym but contain the next procedural steps.
    selected_ids = {item.result.chunk.chunk_id for item in selected}
    for item in scored:
        if len(selected) >= limit:
            break
        if item.result.chunk.chunk_id in selected_ids:
            continue
        if item.coverage < 0.15 and item.intent_evidence < 0.08:
            continue
        if any(_supports_anchor(item, anchor) for anchor in selected):
            selected.append(item)
            selected_ids.add(item.result.chunk.chunk_id)

    selected.sort(key=_scored_sort_key)
    return [
        replace(
            item.result,
            score=round(item.relevance, 6),
            method=_append_method(item.result.method, "intent-rerank"),
        )
        for item in selected[:limit]
    ]


def _score_candidate(
    plan: QueryPlan,
    result: RetrievedChunk,
    focus_terms: set[str],
    anchor_groups: list[set[str]],
    preferred_document_ids: set[str],
) -> _ScoredChunk:
    chunk = result.chunk
    text_terms = _tokens(chunk.text)
    exact_text_terms = _tokens(chunk.text, keep_single=True)
    heading_terms = _tokens(_heading_text(chunk))
    section_path = _context_label(chunk)
    coverage = _coverage(focus_terms, text_terms)
    heading_coverage = _coverage(focus_terms, heading_terms)
    anchor_coverage = _group_coverage(anchor_groups, exact_text_terms)
    question_for_structure = plan.contextual_question or plan.original_question
    structural_match = section_match_score(question_for_structure, section_path)
    major_structural_match = major_section_match_score(
        question_for_structure,
        section_path,
    )

    cues = _INTENT_CUES.get(plan.intent, set())
    intent_matches = len(cues & text_terms)
    intent_evidence = min(1.0, intent_matches / 3.0) if cues else 0.0
    if plan.intent == "fact_lookup":
        intent_evidence = min(1.0, coverage)

    retrieval = max(0.0, min(1.0, float(result.score)))
    relevance = (
        retrieval * 0.36
        + coverage * 0.34
        + heading_coverage * 0.12
        + intent_evidence * 0.08
        + anchor_coverage * 0.10
        + structural_match * 0.24
    )
    if _is_preferred(chunk, preferred_document_ids):
        relevance += 0.16
    if _is_low_information_excerpt(chunk.text):
        relevance *= 0.12

    if focus_terms and coverage < 0.12:
        relevance *= 0.55
    if anchor_groups and anchor_coverage == 0 and structural_match < 0.82:
        relevance *= 0.72
    if _is_generic_heading(chunk.text) and coverage < 0.5:
        relevance *= 0.55

    return _ScoredChunk(
        result=result,
        relevance=max(0.0, min(1.0, relevance)),
        coverage=coverage,
        anchor_coverage=anchor_coverage,
        intent_evidence=intent_evidence,
        section_match=structural_match,
        major_section_match=major_structural_match,
        context_key=(chunk.filename.casefold(), _context_label(chunk).casefold()),
    )


def _supports_anchor(candidate: _ScoredChunk, anchor: _ScoredChunk) -> bool:
    left = candidate.result.chunk
    right = anchor.result.chunk
    if left.filename.casefold() != right.filename.casefold():
        return False
    if candidate.context_key[1] and candidate.context_key == anchor.context_key:
        return True
    if (
        left.document_id
        and left.document_id == right.document_id
        and left.chunk_index is not None
        and right.chunk_index is not None
        and abs(left.chunk_index - right.chunk_index) <= 2
    ):
        return True
    return abs(left.page_number - right.page_number) <= 1


def _selection_limit(plan: QueryPlan, requested: int | None) -> int:
    settings = get_settings()
    configured = (
        settings.reference_evidence_chunk_limit
        if plan.search_mode == "references"
        else settings.answer_evidence_chunk_limit
    )
    if requested is None:
        return configured
    return min(configured, max(1, requested))



def _preferred_then_diverse_selection(
    eligible: list[_ScoredChunk],
    limit: int,
    preferred_document_ids: set[str],
    intent: str,
) -> list[_ScoredChunk]:
    if not preferred_document_ids or intent != "procedure":
        return _strong_then_diverse_selection(eligible, limit)

    settings = get_settings()
    preferred = [item for item in eligible if _is_preferred(item.result.chunk, preferred_document_ids)]
    other = [item for item in eligible if not _is_preferred(item.result.chunk, preferred_document_ids)]
    selected = preferred[:limit]
    selected_ids = {item.result.chunk.chunk_id for item in selected}
    remaining = max(0, limit - len(selected))
    supplement_limit = min(remaining, settings.primary_document_supplement_limit)
    if supplement_limit:
        strong_other = [
            item for item in other
            if item.coverage >= 0.45 or item.anchor_coverage > 0 or item.section_match >= 0.82
        ]
        for item in strong_other[:supplement_limit]:
            if item.result.chunk.chunk_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.result.chunk.chunk_id)
    selected.sort(key=_scored_sort_key)
    return selected[:limit]


def _is_preferred(chunk: TextChunk, preferred_document_ids: set[str]) -> bool:
    return bool(chunk.document_id and chunk.document_id in preferred_document_ids)


def _is_low_information_excerpt(value: str) -> bool:
    body = re.sub(
        r"\[PDF CHUNK CONTEXT\].*?\[/PDF CHUNK CONTEXT\]",
        "",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if not body:
        return True
    alnum = re.sub(r"[^A-Za-z0-9]+", "", body)
    if len(alnum) < 18:
        return True
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) >= 2 and "|" in lines[0] and re.search(r"-{3,}", lines[1]):
        cells = [
            cell.strip(" *_`|")
            for line in lines[2:]
            for cell in line.split("|")
        ]
        if cells and not any(re.search(r"[A-Za-z0-9]", cell) for cell in cells):
            return True
    return False


def _strong_then_diverse_selection(
    eligible: list[_ScoredChunk],
    limit: int,
) -> list[_ScoredChunk]:
    if limit <= 0:
        return []

    selected: list[_ScoredChunk] = []
    selected_ids: set[str] = set()
    seen_documents: set[str] = set()

    # Keep the strongest procedure/context together first, then broaden document
    # coverage. This avoids displacing a multi-chunk answer merely to show more PDFs.
    strength_slots = min(len(eligible), max(1, int(limit * 0.60)))
    for item in eligible[:strength_slots]:
        selected.append(item)
        selected_ids.add(item.result.chunk.chunk_id)
        seen_documents.add(_document_key(item.result.chunk))

    for item in eligible:
        if len(selected) >= limit:
            break
        if item.result.chunk.chunk_id in selected_ids:
            continue
        document_key = _document_key(item.result.chunk)
        if document_key in seen_documents:
            continue
        selected.append(item)
        selected_ids.add(item.result.chunk.chunk_id)
        seen_documents.add(document_key)

    for item in eligible:
        if len(selected) >= limit:
            break
        if item.result.chunk.chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.result.chunk.chunk_id)

    return selected


def _document_key(chunk: TextChunk) -> str:
    return chunk.document_id or chunk.filename.casefold()


def _focus_terms(plan: QueryPlan) -> set[str]:
    supplied = " ".join([*plan.focus_terms, *plan.context_terms])
    base = supplied if supplied.strip() else (plan.contextual_question or plan.original_question)
    return {term for term in _tokens(base) if term not in _STOPWORDS}


def _anchor_groups(plan: QueryPlan) -> list[set[str]]:
    question = plan.contextual_question or plan.original_question
    values = [*plan.context_terms]
    values.extend(_QUOTED_RE.findall(question))
    values.extend(
        token
        for token in _TOKEN_RE.findall(question)
        if any(char.isdigit() for char in token)
        or (len(token) >= 2 and token.upper() == token and any(char.isalpha() for char in token))
    )
    groups: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for value in values:
        if value.casefold() in _STOPWORDS:
            continue
        group = _tokens(value, keep_single=True)
        key = frozenset(group)
        if group and key not in seen:
            groups.append(group)
            seen.add(key)
    return groups


def _heading_text(chunk: TextChunk) -> str:
    values = [chunk.heading or "", *chunk.section_path]
    section = _SECTION_RE.search(chunk.text[:2500])
    if section:
        values.append(section.group(1))
    values.extend(_CONTEXT_LINE_RE.findall(chunk.text[:2500]))
    return " ".join(values)


def _context_label(chunk: TextChunk) -> str:
    if chunk.section_path:
        return " > ".join(chunk.section_path)
    return section_path_from_text(chunk.text)


def _tokens(value: str, *, keep_single: bool = False) -> set[str]:
    return search_terms(value, keep_single=keep_single)


def _coverage(required: set[str], available: set[str]) -> float:
    if not required:
        return 0.0
    return len(required & available) / len(required)


def _group_coverage(required: list[set[str]], available: set[str]) -> float:
    if not required:
        return 0.0
    matches = sum(group.issubset(available) for group in required)
    return matches / len(required)


def _is_generic_heading(value: str) -> bool:
    first_lines = " ".join(value.splitlines()[:7]).casefold()
    generic = {"contents", "operating manual", "table of contents", "introduction"}
    return any(label in first_lines for label in generic)


def _append_method(method: str, value: str) -> str:
    parts = [part for part in method.split("+") if part]
    if value not in parts:
        parts.append(value)
    return "+".join(parts)


def _scored_sort_key(item: _ScoredChunk) -> tuple[float, str, int, int, str]:
    chunk = item.result.chunk
    return (
        -item.relevance,
        chunk.filename.casefold(),
        chunk.page_number,
        chunk.chunk_index if chunk.chunk_index is not None else -1,
        chunk.chunk_id,
    )
