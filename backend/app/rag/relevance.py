from __future__ import annotations

import re
from dataclasses import dataclass, replace

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
        "before",
        "check",
        "ensure",
        "instruction",
        "isolate",
        "procedure",
        "reset",
        "step",
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
        "prerequisite",
        "requirement",
        "required",
        "shall",
    },
    "definition": {"definition", "means", "refers", "term"},
    "summary": {"conclusion", "overview", "scope", "summary"},
    "list": {"include", "item", "list", "types"},
    "fact_lookup": set(),
}


@dataclass(frozen=True, slots=True)
class _ScoredChunk:
    result: RetrievedChunk
    relevance: float
    coverage: float
    anchor_coverage: float
    intent_evidence: float
    context_key: tuple[str, str]


def select_context_chunks(
    plan: QueryPlan,
    candidates: list[RetrievedChunk],
    max_chunks: int | None = None,
) -> list[RetrievedChunk]:
    """Select answerable evidence instead of forwarding broad retrieval output.

    Vector similarity remains one signal, but a chunk must also match the
    question's focus, exact constraints, expected answer shape, or the context
    of a stronger anchor. This prevents generic manual headings and unrelated
    procedures from winning merely because their embeddings are similar.
    """
    if not candidates:
        return []

    focus_terms = _focus_terms(plan)
    anchor_groups = _anchor_groups(plan)
    scored = [
        _score_candidate(plan, candidate, focus_terms, anchor_groups)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item.relevance, reverse=True)

    best = scored[0].relevance
    if best < 0.12:
        return []

    cutoff = max(0.16, best * 0.48)
    eligible: list[_ScoredChunk] = []

    for item in scored:
        has_focus = item.coverage >= 0.50 or not focus_terms
        required_anchor_coverage = (
            1.0 / len(anchor_groups)
            if anchor_groups and plan.intent == "comparison"
            else 1.0
        )
        has_anchor = item.anchor_coverage >= required_anchor_coverage or not anchor_groups
        has_intent = item.intent_evidence >= 0.12
        constraint_ok = has_anchor if anchor_groups else (has_focus or has_intent)
        if item.relevance >= cutoff and has_focus and constraint_ok:
            eligible.append(item)

    if not eligible:
        top = scored[0]
        if top.coverage >= 0.25 or top.anchor_coverage > 0:
            eligible.append(top)
        else:
            return []

    eligible_document_count = len({_document_key(item.result.chunk) for item in eligible})
    limit = _selection_limit(plan, max_chunks, eligible_document_count)
    selected = _document_diverse_selection(eligible, limit)

    # A continuation can omit the acronym or full subject while still containing
    # essential steps. Keep it only when it is close to a selected anchor and has
    # either topical or intent-specific evidence.
    selected_ids = {item.result.chunk.chunk_id for item in selected}
    for item in scored:
        if len(selected) >= limit:
            break
        if item.result.chunk.chunk_id in selected_ids:
            continue
        if item.coverage < 0.22 and item.intent_evidence < 0.12:
            continue
        if any(_supports_anchor(item, anchor) for anchor in selected):
            selected.append(item)
            selected_ids.add(item.result.chunk.chunk_id)

    selected.sort(key=lambda item: item.relevance, reverse=True)
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
) -> _ScoredChunk:
    chunk = result.chunk
    text_terms = _tokens(chunk.text)
    exact_text_terms = _tokens(chunk.text, keep_single=True)
    heading_terms = _tokens(_heading_text(chunk))
    coverage = _coverage(focus_terms, text_terms)
    heading_coverage = _coverage(focus_terms, heading_terms)
    anchor_coverage = _group_coverage(anchor_groups, exact_text_terms)

    cues = _INTENT_CUES.get(plan.intent, set())
    intent_evidence = _coverage(cues, text_terms)
    if plan.intent == "fact_lookup":
        intent_evidence = min(1.0, coverage)

    retrieval = max(0.0, min(1.0, float(result.score)))
    relevance = (
        retrieval * 0.40
        + coverage * 0.32
        + heading_coverage * 0.13
        + intent_evidence * 0.10
        + anchor_coverage * 0.05
    )

    if focus_terms and coverage < 0.18:
        relevance *= 0.42
    if anchor_groups and anchor_coverage < 1.0:
        relevance *= 0.62
    if _is_generic_heading(chunk.text) and coverage < 0.5:
        relevance *= 0.55

    return _ScoredChunk(
        result=result,
        relevance=max(0.0, min(1.0, relevance)),
        coverage=coverage,
        anchor_coverage=anchor_coverage,
        intent_evidence=intent_evidence,
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
        and abs(left.chunk_index - right.chunk_index) <= 1
    ):
        return True
    return abs(left.page_number - right.page_number) <= 1


def _selection_limit(
    plan: QueryPlan,
    requested: int | None,
    relevant_document_count: int,
) -> int:
    question_terms = _tokens(plan.original_question)
    complex_question = (
        plan.intent in {"comparison", "summary", "troubleshooting"}
        or len(question_terms) > 10
        or bool({"and", "versus", "vs"} & question_terms)
    )
    adaptive = 8 if complex_question else 4
    adaptive = max(adaptive, min(relevant_document_count, 12))
    if requested is not None:
        adaptive = min(adaptive, max(1, requested))
    return adaptive


def _document_diverse_selection(
    eligible: list[_ScoredChunk],
    limit: int,
) -> list[_ScoredChunk]:
    selected: list[_ScoredChunk] = []
    selected_ids: set[str] = set()
    seen_documents: set[str] = set()

    for item in eligible:
        document_key = _document_key(item.result.chunk)
        if document_key in seen_documents:
            continue
        selected.append(item)
        selected_ids.add(item.result.chunk.chunk_id)
        seen_documents.add(document_key)
        if len(selected) >= limit:
            return selected

    for item in eligible:
        if item.result.chunk.chunk_id in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _document_key(chunk: TextChunk) -> str:
    return chunk.document_id or chunk.filename.casefold()


def _focus_terms(plan: QueryPlan) -> set[str]:
    supplied = " ".join([*plan.focus_terms, *plan.context_terms])
    terms = _tokens(supplied) if supplied.strip() else _tokens(plan.original_question)
    return {term for term in terms if term not in _STOPWORDS}


def _anchor_groups(plan: QueryPlan) -> list[set[str]]:
    values = [*plan.context_terms]
    values.extend(_QUOTED_RE.findall(plan.original_question))
    values.extend(
        token
        for token in _TOKEN_RE.findall(plan.original_question)
        if any(char.isdigit() for char in token)
        or (len(token) >= 2 and token.upper() == token and any(char.isalpha() for char in token))
    )
    groups: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for value in values:
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
    section = _SECTION_RE.search(chunk.text[:2500])
    if section:
        return section.group(1).strip()
    return ""


def _tokens(value: str, *, keep_single: bool = False) -> set[str]:
    terms: set[str] = set()
    for raw_token in _TOKEN_RE.findall(value):
        token = raw_token.casefold()
        candidates = [token, *re.split(r"[._/:#-]+", token)]
        terms.update(
            candidate
            for candidate in candidates
            if candidate and (keep_single or len(candidate) > 1)
        )
    return terms


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
